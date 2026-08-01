#!/usr/bin/env python3
"""部署打分守护 v3：五网融合（稀有度 + 上下文 + 模式库 + 时序 + 自适应）
尾随 host_tracee.jsonl，对新增事件打分告警；每轮从上次偏移量续读。

判定逻辑（优先级从高到低，同一事件只出一条告警）：
  P0 自适应行为 —— adaptive_detector 高危（morphological transformation/伪装C2/SUID提权）
  P1 模式命中   —— patterns.jsonl 字段级匹配（已知坏，带 ATT&CK 技术号）
  P2 上下文异常 —— PARENT/DST/DT 槽位超分维度 τ（组合不对）
  P3 稀有度异常 —— PROC/ARGV 槽位超 τ 或 UNK（没见过）
  P4 时序异常   —— temporal_analyzer 机器节奏（CV<1.5）
  P5 自适应低危 —— adaptive_detector 低危（sleep步进/侦察轮换）

用法: deploy_scorer.py <model_dir> [--src 原始jsonl] [--state 状态文件]
       [--alerts 告警文件] [--patterns 模式库]
"""
import json
import os
import sys
import time

import torch

from parse_raw_tracee import event_to_tokens
from pattern_db import PatternDB
from train_prior import TinyGPT, CTX
from temporal_analyzer import TemporalAnalyzer
from adaptive_detector import AdaptiveDetector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 0.3
CTX_SLOTS = {"PARENT", "DST", "DT"}   # 上下文维度
RARE_SLOTS = {"PROC", "ARGV"}          # 稀有度维度（ET 主要是突发噪声，只做参考不报警）


def slot_of(tok):
    if ":" in tok:
        return tok.split(":")[0]
    if tok.startswith("ARGV"):
        return "ARGV"
    if tok.startswith("DT"):
        return "DT"
    return tok


def main():
    model_dir = sys.argv[1]
    args = sys.argv[2:]

    def opt(name, default):
        return args[args.index(name) + 1] if name in args else default

    src = opt("--src", os.path.expanduser("~/siming/telemetry/tracee.jsonl"))
    state_path = opt("--state", os.path.join(model_dir, "scorer_state.json"))
    alerts_path = opt("--alerts", os.path.expanduser("~/siming/data/alerts.jsonl"))
    patterns_path = opt("--patterns", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "patterns.jsonl"))

    ckpt = torch.load(os.path.join(model_dir, "prior.pt"), map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    # τ 优先级：slot_tau_local.json（onboard 标定的）> slot_tau_vm.json > slot_tau.json
    for tau_name in ("slot_tau_local.json", "slot_tau_vm.json", "slot_tau.json"):
        tau_path = os.path.join(model_dir, tau_name)
        if os.path.exists(tau_path):
            slot_tau = json.load(open(tau_path))["slot_tau"]
            break
    db = PatternDB(patterns_path)
    temporal = TemporalAnalyzer(min_samples=20)
    adaptive = AdaptiveDetector(window_size=200, cooldown=50)

    state = {"offset": 0, "window": [], "ewma": 0.0, "prev_ts": None}
    if os.path.exists(state_path):
        state.update(json.load(open(state_path)))

    n_scored = 0
    n_alert = {"P0": 0, "P1": 0, "P2": 0, "P3": 0, "P4": 0, "P5": 0}
    with open(src, errors="replace") as f:
        f.seek(state["offset"])
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = ev.get("timestamp", 0)
            delta = 0 if state["prev_ts"] is None else max(0, (ts - state["prev_ts"]) // 1_000_000)
            state["prev_ts"] = ts
            tokens = event_to_tokens(ev, delta)
            ids = [stoi.get(t, 0) for t in tokens]
            window = (state["window"] + ids)[-CTX:]
            state["window"] = window
            n, L = len(ids), len(window)
            start = max(L - n, 1)
            with torch.no_grad():
                x = torch.tensor(window, device=DEVICE).unsqueeze(0)
                lp = torch.log_softmax(model(x), dim=-1)[0]
                tgt = torch.tensor(window[start:L], device=DEVICE)
                nll = -lp[start - 1:L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            s_ev = nll.max().item()
            state["ewma"] = ALPHA * s_ev + (1 - ALPHA) * state["ewma"]
            n_scored += 1

            # 分维度判定
            fired = []   # (slot, token, nll)
            for t, v in zip(tokens, nll.tolist()):
                s = slot_of(t)
                if v > slot_tau.get(s, 1.0):
                    fired.append((s, t, round(v, 2)))
            n_unk = sum(1 for t in tokens if t not in stoi)
            ctx_fired = [x for x in fired if x[0] in CTX_SLOTS]
            rare_fired = [x for x in fired if x[0] in RARE_SLOTS]

            # 模式匹配
            hits = db.match(tokens)
            patterns = [{"id": h["id"], "technique": h["technique"],
                         "severity": h["severity"], "name": h["name"]} for h in hits]

            # 自适应检测（需要 8-token，补 PC 槽）
            tokens_8 = list(tokens)
            # 在 DST 后插 PC（如果还没有）
            if not any(t.startswith("PC:") for t in tokens_8):
                dst_idx = next((i for i, t in enumerate(tokens_8)
                                if t.startswith("DST:")), 5)
                tokens_8.insert(dst_idx + 1, "PC:NONE")
            adapt_alerts = adaptive.update(tokens_8, ts=ts)
            adapt_p5 = [a for a in adapt_alerts if a["severity"] >= 5]
            adapt_low = [a for a in adapt_alerts if a["severity"] < 5]

            # 时序分析
            proc_tok = next((t for t in tokens if t.startswith("PROC:")), "PROC:?")
            proc_name = proc_tok.split(":", 1)[1] if ":" in proc_tok else "?"
            uid_tok = next((t for t in tokens if t.startswith("UID:")), "UID:?")
            uid_val = uid_tok.split(":", 1)[1] if ":" in uid_tok else "?"
            temporal_result = temporal.update(proc_name, ts, state["ewma"], uid=uid_val)

            # 融合判定：P0自适应高危 > P1模式 > P2上下文 > P3稀有度 > P4时序 > P5自适应低危
            if adapt_p5:
                prio = "P0"
            elif patterns:
                prio = "P1"
            elif ctx_fired:
                prio = "P2"
            elif rare_fired or n_unk > 0:
                prio = "P3"
            elif temporal_result and temporal_result["verdict"] == "anomalous":
                prio = "P4"
            elif adapt_low:
                prio = "P5"
            else:
                continue

            n_alert[prio] += 1
            alert_rec = {
                "ts": ts, "prio": prio,
                "patterns": patterns or None,
                "fired_dims": [f"{t}:{v}" for _, t, v in fired] or None,
                "s_ev": round(s_ev, 3), "ewma": round(state["ewma"], 3),
                "n_unk": n_unk, "tokens": tokens,
            }
            if adapt_p5:
                alert_rec["adaptive"] = adapt_p5
            if adapt_low:
                alert_rec["adaptive_low"] = adapt_low
            if temporal_result and temporal_result["verdict"] != "normal":
                alert_rec["temporal"] = temporal_result
            with open(alerts_path, "a") as af:
                af.write(json.dumps(alert_rec, ensure_ascii=False) + "\n")
        state["offset"] = f.tell()

    json.dump(state, open(state_path, "w"))
    print(f"[{time.strftime('%F %T')}] 打分 {n_scored} 事件 | "
          f"P0={n_alert['P0']} P1={n_alert['P1']} P2={n_alert['P2']} "
          f"P3={n_alert['P3']} P4={n_alert['P4']} P5={n_alert['P5']} "
          f"-> {alerts_path}")


if __name__ == "__main__":
    main()
