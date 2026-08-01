#!/usr/bin/env python3
"""通用 prior 模型跨机验证：在 VM 遥测上验证 FPR 和检测率

用法: eval_universal.py <model_dir>
"""
import json
import os
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np

from train_prior import TinyGPT, CTX
from pattern_db import PatternDB
from temporal_analyzer import TemporalAnalyzer
from adaptive_detector import AdaptiveDetector

DET = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 0.3
CTX_SLOTS = {"PARENT", "DST", "DT"}
RARE_SLOTS = {"PROC", "ARGV"}


def slot_of(tok):
    if ":" in tok:
        return tok.split(":")[0]
    if tok.startswith("ARGV"):
        return "ARGV"
    if tok.startswith("DT"):
        return "DT"
    return tok


def convert_7to8(tokens_7):
    out = list(tokens_7)
    if not any(t.startswith("PC:") for t in out):
        dst_idx = next((i for i, t in enumerate(out) if t.startswith("DST:")), 5)
        out.insert(dst_idx + 1, "PC:NONE")
    return out


def evaluate(model_dir, events_path, hour_filter=None, n_max=50000, label=""):
    ckpt = torch.load(os.path.join(model_dir, "prior.pt"),
                      map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 用 slot_tau_local 如果存在，否则 slot_tau
    tau_path = os.path.join(model_dir, "slot_tau_local.json")
    if os.path.exists(tau_path):
        slot_tau = json.load(open(tau_path))["slot_tau"]
    else:
        slot_tau_path = os.path.join(model_dir, "slot_tau.json")
        slot_tau = json.load(open(slot_tau_path))["slot_tau"] if os.path.exists(slot_tau_path) else {}

    db = PatternDB(os.path.join(DET, "patterns.jsonl"))
    temporal = TemporalAnalyzer(min_samples=20)
    adaptive = AdaptiveDetector(window_size=200, cooldown=50)

    window = []
    ewma = 0.0
    n_scored = 0
    n_alert = Counter()

    for line in open(events_path):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except:
            continue
        if hour_filter and e["ts"][11:13] not in hour_filter:
            continue
        if n_scored >= n_max:
            break

        tokens = convert_7to8(e["tokens"])
        from datetime import datetime
        try:
            ts = datetime.fromisoformat(e["ts"]).timestamp() * 1000
        except:
            ts = n_scored * 1000

        n_scored += 1

        # Prior NLL
        ids = [stoi.get(t, 0) for t in tokens]
        window = (window + ids)[-CTX:]
        n_toks, L = len(ids), len(window)
        start = max(L - n_toks, 1)
        with torch.no_grad():
            x = torch.tensor(window, device=DEVICE).unsqueeze(0)
            lp = torch.log_softmax(model(x), dim=-1)[0]
            tgt = torch.tensor(window[start:L], device=DEVICE)
            nll = -lp[start-1:L-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        s_ev = nll.max().item()
        ewma = ALPHA * s_ev + (1 - ALPHA) * ewma

        fired = []
        for t, v in zip(tokens, nll.tolist()):
            s = slot_of(t)
            if v > slot_tau.get(s, 999):
                fired.append((s, t, v))
        n_unk = sum(1 for t in tokens if t not in stoi)
        ctx_fired = [x for x in fired if x[0] in CTX_SLOTS]
        rare_fired = [x for x in fired if x[0] in RARE_SLOTS]

        # Pattern
        hits = db.match(tokens)
        strong_pats = [h for h in hits if h.get("severity", 0) >= 4]
        weak_pats = [h for h in hits if h.get("severity", 0) < 4]

        # Adaptive
        adapt_alerts = adaptive.update(tokens, ts=e.get("ts", ""))
        adapt_p5 = [a for a in adapt_alerts if a["severity"] >= 5]
        adapt_low = [a for a in adapt_alerts if a["severity"] < 5]

        # Temporal
        proc_name = "?"
        uid_val = "?"
        for t in tokens:
            if t.startswith("PROC:"):
                proc_name = t.split(":", 1)[1]
            elif t.startswith("UID:"):
                uid_val = t.split(":", 1)[1]
        temp_res = temporal.update(proc_name, ts, ewma, uid=uid_val)

        # Fusion
        if adapt_p5:
            n_alert["P0"] += 1
        elif strong_pats:
            n_alert["P1"] += 1
        elif ctx_fired:
            n_alert["P2"] += 1
        elif rare_fired or n_unk > 0:
            n_alert["P3"] += 1
        elif temp_res and temp_res["verdict"] == "anomalous":
            n_alert["P4"] += 1
        elif adapt_low or weak_pats:
            n_alert["P5"] += 1

    total = sum(n_alert.values())
    fpr = total / max(n_scored, 1) * 100
    unk_rate = 1.0  # calculated during loop above conceptually

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  事件: {n_scored}  告警: {total} ({fpr:.1f}%)")
    for p in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        if n_alert[p] > 0:
            print(f"  {p}: {n_alert[p]} ({n_alert[p]/max(total,1)*100:.1f}%)")

    return {"n_scored": n_scored, "n_alert": dict(n_alert), "fpr": fpr}


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DET, "model-vm-universal")
    if not os.path.exists(os.path.join(model_dir, "prior.pt")):
        print(f"ERROR: {model_dir}/prior.pt 不存在")
        sys.exit(1)

    clone = os.path.expanduser("~/data/telemetry/clone_events.jsonl")
    regime = os.path.expanduser("~/data/telemetry/regime_events.jsonl")

    # 先标定
    print("=== 自动标定 slot_tau_local ===")
    os.system(f"cd {DET} && python3 onboard_v2.py {model_dir} data/onboard_benign.jsonl 2>&1 | tail -20")

    # 评估
    agent = evaluate(model_dir, regime, hour_filter={"13"}, n_max=50000,
                   label="adaptive agent活跃期")
    benign = evaluate(model_dir, clone, hour_filter={"02", "03", "04", "05"}, n_max=20000,
                      label="良性夜间")

    # 对比
    print(f"\n{'='*50}")
    print(f"  跨机验证汇总")
    print(f"{'='*50}")
    print(f"{'指标':<20}{'adaptive agent':>12}{'良性':>12}")
    print(f"{'-'*44}")
    print(f"{'FPR':<20}{agent['fpr']:>11.1f}%{benign['fpr']:>11.1f}%")
    for p in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        b = agent["n_alert"].get(p, 0)
        g = benign["n_alert"].get(p, 0)
        if b + g > 0:
            print(f"  {p:<18}{b:>12}{g:>12}")

    # 良性 P2+P3（核心跨机指标）
    benign_p23 = benign["n_alert"].get("P2", 0) + benign["n_alert"].get("P3", 0)
    benign_p23_pct = benign_p23 / max(benign["n_scored"], 1) * 100
    print(f"\n  良性 P2+P3 (跨机误报核心): {benign_p23} ({benign_p23_pct:.1f}%)")
    if benign_p23_pct < 20:
        print("  ✅ 跨机误报可控 (<20%)")
    elif benign_p23_pct < 40:
        print("  ⚠️  跨机误报偏高 (20-40%)")
    else:
        print("  ❌ 跨机误报过高 (>40%)，需继续标定")


if __name__ == "__main__":
    main()
