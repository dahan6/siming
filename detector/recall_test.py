#!/usr/bin/env python3
"""召回率测试：真实攻击轨迹逐一过检测器（先验分维度 + 模式库）
对 atomic_staging 每轮每技术的 tracee 文件：
  - 基线窗内不应告警（观察窗误报检查）
  - 攻击窗（基线之后）应出现告警，且 P1 模式命中的技术号应正确
输出每技术召回矩阵。

用法: recall_test.py <model_dir> <staging_dir>
"""
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_raw_tracee import event_to_tokens
from pattern_db import PatternDB
from proto_head import embed_sequence
from train_prior import CTX, DEVICE, TinyGPT


def slot_of(tok):
    if ":" in tok:
        return tok.split(":")[0]
    if tok.startswith("ARGV"):
        return "ARGV"
    if tok.startswith("DT"):
        return "DT"
    return tok


def main():
    model_dir, staging = sys.argv[1], sys.argv[2]
    ckpt = torch.load(os.path.join(model_dir, "prior.pt"), map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    slot_tau = json.load(open(os.path.join(model_dir, "slot_tau.json")))["slot_tau"]
    db = PatternDB(os.path.join(os.path.dirname(os.path.abspath(__file__)), "patterns.jsonl"))
    proto = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "prototypes.jsonl")))
    P = {t: (torch.tensor(v["prototypes"]).to(DEVICE), v["radii"][0])
         for t, v in proto["techniques"].items()}

    def proto_hit(toks):
        e = embed_sequence(model, stoi, toks)
        best, bd = None, 1e9
        for t, (ps, r) in P.items():
            d = (ps - e).norm(dim=1).min().item()
            if d < bd:
                best, bd = t, d
        for t, (ps, r) in P.items():
            if (ps - e).norm(dim=1).min().item() <= r:
                return t
        return None

    print(f"{'技术':<14}{'轮':<5}{'基线告警':>8}{'攻击窗事件':>9}{'检出':>5}{'P1技术号':<22}")
    rows = []
    for meta_path in sorted(glob.glob(os.path.join(staging, "**", "*.meta.json"), recursive=True)):
        rnd = os.path.basename(os.path.dirname(meta_path))
        meta = json.load(open(meta_path))
        tid = meta["id"].rstrip("b")
        raw = meta_path.replace(".meta.json", ".jsonl")
        events = []
        for line in open(raw, errors="replace"):
            if line.strip().startswith("{"):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        base_n = meta["baseline_lines"]
        window, prev_ts = [], None
        base_alerts = atk_events = atk_alerts = 0
        p1_hits = set()
        for i, ev in enumerate(events):
            ts = ev.get("timestamp", 0)
            delta = 0 if prev_ts is None else max(0, (ts - prev_ts) // 1_000_000)
            prev_ts = ts
            toks = event_to_tokens(ev, delta)
            ids = [stoi.get(t, 0) for t in toks]
            window = (window + ids)[-CTX:]
            n, L = len(ids), len(window)
            start = max(L - n, 1)
            with torch.no_grad():
                x = torch.tensor(window, device=DEVICE).unsqueeze(0)
                lp = torch.log_softmax(model(x), dim=-1)[0]
                tgt = torch.tensor(window[start:L], device=DEVICE)
                nll = -lp[start - 1:L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            fired = any(v > slot_tau.get(slot_of(t), 1.0)
                        for t, v in zip(toks, nll.tolist()))
            n_unk = any(t not in stoi for t in toks)
            hits = db.match(toks)
            phit = proto_hit(toks)
            if phit:
                p1_hits.add(phit)
            if i < base_n:
                base_alerts += bool(fired or n_unk or hits or phit)
            else:
                atk_events += 1
                if fired or n_unk or hits or phit:
                    atk_alerts += 1
                for h in hits:
                    p1_hits.add(h["technique"])
        recall = atk_alerts / max(1, atk_events)
        rows.append((tid, rnd, base_alerts, atk_events, atk_alerts, recall, sorted(p1_hits)))
        print(f"{tid:<14}{rnd:<5}{base_alerts:>8}{atk_events:>9}"
              f"{atk_alerts:>5}  {','.join(sorted(p1_hits)) or '-':<22}")

    # 汇总：技术级召回（任一轮攻击窗告警率>50% 视为该技术可召回）
    from collections import defaultdict
    by_t = defaultdict(list)
    for tid, rnd, ba, ae, aa, r, p1 in rows:
        by_t[tid].append(r)
    print("\n技术级召回（3轮平均攻击窗告警率）:")
    n_rec = 0
    for t in sorted(by_t):
        avg = sum(by_t[t]) / len(by_t[t])
        ok = avg > 0.5
        n_rec += ok
        print(f"  {t:<14} {avg:6.1%} {'✓' if ok else '✗'}")
    print(f"\n召回 {n_rec}/{len(by_t)} 技术")


if __name__ == "__main__":
    main()
