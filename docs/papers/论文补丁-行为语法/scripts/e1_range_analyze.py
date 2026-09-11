#!/usr/bin/env python3
"""E1 靶场版分析：重尾时序敌手在全管线下暴露多少

流程：
  1. tracee cap.jsonl → 8-token（parse_raw_tracee 同一规则）
  2. 按 adv_phases.log 时间窗切片（benign 基线 + 各 σ 档）
  3. benign 段校准 slot τ（calibrate_slot_tau 同一方法）
  4. 逐档打分：prior NLL 分槽超阈（context/rarity）+ UNK + 时序 CV
  5. 输出：每档 × 每观测面的暴露率 + 总结论

用法: e1_range_analyze.py <cap.jsonl> <adv_phases.log> <model_dir>
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_prior import TinyGPT, CTX
from parse_raw_tracee import event_to_tokens

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 0.3
CTX_SLOTS = {"PARENT", "DST", "DT"}
RARE_SLOTS = {"PROC", "ARGV"}
CV_LINE = 1.5  # 时序面检测线（论文口径）


def slot_of(tok):
    if ":" in tok:
        return tok.split(":")[0]
    if tok.startswith("ARGV"):
        return "ARGV"
    if tok.startswith("DT"):
        return "DT"
    return tok


def load_cap(path):
    """tracee JSONL → [(ts_ms, tokens)]"""
    events = []
    prev_ts = None
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts_ns = ev.get("timestamp", 0)
        ts_ms = ts_ns // 1_000_000
        delta = 0 if prev_ts is None else max(0, ts_ms - prev_ts)
        prev_ts = ts_ms
        tokens = event_to_tokens(ev, delta)
        proc = ev.get("processName", "?")
        events.append({"ts_ms": ts_ms, "tokens": tokens, "proc": proc})
    events.sort(key=lambda e: e["ts_ms"])
    return events


def load_phases(path):
    """adv_phases.log → {'s0.3': (start_ms, end_ms), ...}"""
    phases = {}
    for line in open(path):
        parts = line.split()
        if len(parts) < 3:
            continue
        t = float(parts[0]) * 1000  # s → ms
        tag, msg = parts[1], parts[2]
        if msg.startswith("PHASE_START"):
            phases.setdefault(tag, [None, None])[0] = t
        elif msg.startswith("PHASE_END"):
            phases.setdefault(tag, [None, None])[1] = t
    return phases


def score_events(model, stoi, evs, slot_tau):
    """对事件序列打分，返回逐事件 dict"""
    out = []
    window, ewma = [], 0.0
    for e in evs:
        ids = [stoi.get(t, 0) for t in e["tokens"]]
        window = (window + ids)[-CTX:]
        n, L = len(ids), len(window)
        if L < 2:
            continue
        start = max(L - n, 1)
        with torch.no_grad():
            x = torch.tensor(window, device=DEVICE).unsqueeze(0)
            lp = torch.log_softmax(model(x), dim=-1)[0]
            tgt = torch.tensor(window[start:L], device=DEVICE)
            nll = -lp[start - 1:L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        s_ev = nll.max().item()
        ewma = ALPHA * s_ev + (1 - ALPHA) * ewma
        fired = []
        for t, v in zip(e["tokens"], nll.tolist()):
            s = slot_of(t)
            if v > slot_tau.get(s, 999):
                fired.append(s)
        n_unk = sum(1 for t in e["tokens"] if t not in stoi)
        ctx = [s for s in fired if s in CTX_SLOTS]
        rare = [s for s in fired if s in RARE_SLOTS]
        out.append({"s_ev": s_ev, "ewma": ewma, "n_unk": n_unk,
                    "ctx": ctx, "rare": rare, "tokens": e["tokens"],
                    "proc": e["proc"]})
    return out


def calibrate_slot_tau(model, stoi, benign_evs):
    """benign 段逐槽 NLL → p995 τ（下限 1.0）"""
    slot_nll = defaultdict(list)
    window = []
    model.eval()
    for e in benign_evs:
        ids = [stoi.get(t, 0) for t in e["tokens"]]
        window = (window + ids)[-CTX:]
        L = len(window)
        if L < 2:
            continue
        x = torch.tensor(window, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            lp = torch.log_softmax(model(x), dim=-1)[0]
        n = len(ids)
        start = max(L - n, 1)
        tgt = window[start:L]
        tgt_t = torch.tensor(tgt, device=DEVICE)
        nll = -lp[start - 1:L - 1].gather(-1, tgt_t.unsqueeze(-1)).squeeze(-1)
        for tok_id, v in zip(tgt, nll.tolist()):
            tok = {i: k for k, i in stoi.items()}.get(tok_id, "<UNK>")
            slot_nll[slot_of(tok)].append(v)
    tau = {}
    for slot in ["ET", "PROC", "ARGV", "PC", "PARENT", "UID", "DST", "DT"]:
        vals = np.array(slot_nll.get(slot, [0]))
        tau[slot] = max(float(np.quantile(vals, 0.995)), 1.0) if len(vals) > 1 else 1.0
    return tau


def temporal_cv(evs, proc_filter=None):
    """敌手进程的事件间隔 CV（时序面复算）"""
    ts = [e["ts_ms"] for e in evs if proc_filter is None or e["proc"] == proc_filter]
    if len(ts) < 10:
        return None, len(ts)
    ts = sorted(ts)
    dts = np.diff(ts) / 1000.0
    dts = dts[dts > 0.001]
    if len(dts) < 5:
        return None, len(ts)
    return float(np.std(dts) / np.mean(dts)), len(ts)


def main():
    cap_path, phases_path, model_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    ckpt = torch.load(os.path.join(model_dir, "prior.pt"),
                      map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    events = load_cap(cap_path)
    phases = load_phases(phases_path)
    adv_start = min(v[0] for v in phases.values() if v[0])
    benign = [e for e in events if e["ts_ms"] < adv_start]
    print(f"事件总数 {len(events)} | benign 基线 {len(benign)} | "
          f"阶段数 {len(phases)}: {sorted(phases.keys())}")

    slot_tau = calibrate_slot_tau(model, stoi, benign)
    print(f"benign 校准 slot τ: { {k: round(v,2) for k,v in slot_tau.items()} }")

    # benign 段误报基线
    b_scores = score_events(model, stoi, benign, slot_tau)
    b_alert = sum(1 for s in b_scores if s["ctx"] or s["rare"] or s["n_unk"] > 0)
    print(f"benign 段: {len(b_scores)} 事件, 告警 {b_alert} "
          f"({b_alert/max(1,len(b_scores)):.1%})")

    print(f"\n{'σ档':<8}{'事件':>6}{'CV':>7}{'CV检出':>7}{'context':>9}"
          f"{'rarity':>8}{'UNK':>6}{'DT超τ':>7}{'语法暴露':>9}")
    print("-" * 75)

    report = {"slot_tau": slot_tau, "benign_fpr": b_alert / max(1, len(b_scores)),
              "phases": {}}
    for tag in sorted(phases.keys(), key=lambda t: float(t[1:])):
        start, end = phases[tag]
        if not start or not end:
            continue
        sigma = float(tag[1:])
        pevs = [e for e in events if start <= e["ts_ms"] <= end]
        # 敌手事件标记：PARENT:sh（adv.py → sh → action 的 action 层）
        adv_evs = [e for e in pevs
                   if any(t == "PARENT:sh" for t in e["tokens"])]
        tag_note = "" if adv_evs else "(全窗)"
        if not adv_evs:
            adv_evs = pevs

        # 时序面：对敌手事件（PARENT:sh）算间隔 CV
        cv, n_cv = temporal_cv(adv_evs)
        cv_det = cv is not None and cv < CV_LINE

        # 语法面
        scored = score_events(model, stoi, adv_evs, slot_tau)
        n = len(scored)
        n_ctx = sum(1 for s in scored if s["ctx"])
        n_rare = sum(1 for s in scored if s["rare"])
        n_unk = sum(1 for s in scored if s["n_unk"] > 0)
        n_dt = sum(1 for s in scored if "DT" in s["ctx"])
        n_grammar = sum(1 for s in scored if s["ctx"] or s["rare"] or s["n_unk"] > 0)

        cv_s = f"{cv:.2f}" if cv is not None else "n/a"
        print(f"{tag}{tag_note:<6}{n:>6}{cv_s:>7}{'是' if cv_det else '否':>7}"
              f"{n_ctx/max(1,n):>9.1%}{n_rare/max(1,n):>8.1%}{n_unk/max(1,n):>6.1%}"
              f"{n_dt/max(1,n):>7.1%}{n_grammar/max(1,n):>9.1%}")

        report["phases"][tag] = {
            "sigma": sigma, "n_events": n, "cv": cv, "cv_detected": cv_det,
            "ctx_rate": n_ctx / max(1, n), "rare_rate": n_rare / max(1, n),
            "unk_rate": n_unk / max(1, n), "dt_rate": n_dt / max(1, n),
            "grammar_exposed": n_grammar / max(1, n),
        }

    out = os.path.join(os.path.dirname(cap_path), "e1_range_report.json")
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告 -> {out}")

    # 结论
    print("\n=== 判定 ===")
    for tag, r in report["phases"].items():
        if not r["cv_detected"]:
            exp = r["grammar_exposed"]
            print(f"σ={r['sigma']}: CV 逃逸，语法面暴露 {exp:.1%} "
                  f"({'被抓' if exp > 0.1 else '未被抓'})")


if __name__ == "__main__":
    main()
