#!/usr/bin/env python3
"""跨机迁移实验：宿主机先验 + 三种适配策略在 VM 数据上的 FPR 对比
策略 A: 零适配（宿主机 slot_tau 直接打分）——预期复现 ~92% 误报
策略 B: 方言重校准（仅用 VM 校准段重标定各槽位 τ，不重训）——验证手术刀
策略 C: 参考线（VM 自训先验的槽位 τ，由 onboard 管线产出后回填）

用法: transfer_experiment.py <host_model_dir> <vm_tokens.jsonl> [--vm_model_dir 可选]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_prior import CTX, DEVICE, TinyGPT


def slot_of(tok):
    if ":" in tok:
        return tok.split(":")[0]
    if tok.startswith("ARGV"):
        return "ARGV"
    if tok.startswith("DT"):
        return "DT"
    return tok


def score_events(model, stoi, events, calib_events=None):
    """返回每个事件的 [(slot, nll), ...] 列表"""
    window = []
    out = []
    for ev in events:
        ids = [stoi.get(t, 0) for t in ev["tokens"]]
        window = (window + ids)[-CTX:]
        n, L = len(ids), len(window)
        start = max(L - n, 1)
        x = torch.tensor(window, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            lp = torch.log_softmax(model(x), dim=-1)[0]
            tgt = torch.tensor(window[start:L], device=DEVICE)
            nll = -lp[start - 1:L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        out.append([(slot_of(t), t, float(v)) for t, v in zip(ev["tokens"], nll.tolist())])
    return out


def main():
    host_dir, vm_tokens = sys.argv[1], sys.argv[2]
    ckpt = torch.load(os.path.join(host_dir, "prior.pt"), map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    host_tau = json.load(open(os.path.join(host_dir, "slot_tau.json")))["slot_tau"]

    events = [json.loads(l) for l in open(vm_tokens)]
    n_calib = min(10000, len(events) // 2)
    calib, holdout = events[:n_calib], events[n_calib:]
    print(f"VM 事件 {len(events)}（校准 {len(calib)} / 留出 {len(holdout)}）")

    calib_sc = score_events(model, stoi, calib)
    hold_sc = score_events(model, stoi, holdout)

    # 策略 A：宿主机 τ 直接判
    def fpr(scores, taus, vocab):
        from collections import Counter
        hits = Counter()
        n_fp = 0
        for ev_sc in scores:
            fired = [s for s, t, v in ev_sc if v > taus.get(s, 1.0)]
            if fired or any(t not in vocab for _, t, _ in ev_sc):
                n_fp += 1
                for s in fired:
                    hits[s] += 1
        return n_fp / len(scores), hits

    fpr_a, hits_a = fpr(hold_sc, host_tau, stoi)
    print(f"\n策略A 零适配: FPR={fpr_a:.1%}  命中维度 Top={hits_a.most_common(4)}")

    # 策略 B：用 VM 校准段重标定（p995, floor 1.0），在留出段判
    from collections import defaultdict
    slot_vals = defaultdict(list)
    for ev_sc in calib_sc:
        for s, t, v in ev_sc:
            slot_vals[s].append(v)
    tau_b = {s: max(float(np.quantile(vs, 0.995)), 1.0) for s, vs in slot_vals.items()}
    fpr_b, hits_b = fpr(hold_sc, tau_b, stoi)
    print(f"策略B 方言重校准: FPR={fpr_b:.1%}  命中维度 Top={hits_b.most_common(4)}")
    print(f"  槽位 τ 对比（宿主机→VM重校准）: " +
          ", ".join(f"{s} {host_tau.get(s,1.0):.1f}→{tau_b.get(s,1.0):.1f}"
                    for s in ["UID", "DST", "PARENT", "PROC"]))

    # 策略 C：VM 自训模型（若提供）
    if len(sys.argv) > 3 and sys.argv[3] != "--":
        vm_dir = sys.argv[3]
        ck = torch.load(os.path.join(vm_dir, "prior.pt"), map_location=DEVICE, weights_only=False)
        stoi_v = ck["stoi"]
        model_v = TinyGPT(len(stoi_v)).to(DEVICE)
        model_v.load_state_dict(ck["model"])
        model_v.eval()
        tau_c = json.load(open(os.path.join(vm_dir, "slot_tau.json")))["slot_tau"]
        hold_sc_c = score_events(model_v, stoi_v, holdout)
        fpr_c, hits_c = fpr(hold_sc_c, tau_c, stoi_v)
        print(f"策略C VM自训: FPR={fpr_c:.1%}  命中维度 Top={hits_c.most_common(4)}")


if __name__ == "__main__":
    main()
