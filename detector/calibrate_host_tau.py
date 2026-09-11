#!/usr/bin/env python3
"""标定宿主机真实数据的 slot_tau_local.json

逻辑：在良性数据上计算 per-slot NLL 分布，
τ = max(p99.5, min_tau)，min_tau 保底 2.0 nats。
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np

from train_prior import TinyGPT, CTX

DET = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_TAU = 2.0


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
    has_pc = any(t.startswith("PC:") for t in out)
    if not has_pc:
        dst_idx = next((i for i, t in enumerate(out) if t.startswith("DST:")), 5)
        out.insert(dst_idx + 1, "PC:NONE")
    return out


def calibrate(model_dir, benign_jsonl, percentile=99.5):
    ckpt = torch.load(os.path.join(model_dir, "prior.pt"),
                      map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    slot_nlls = defaultdict(list)
    window = []
    n = 0

    for line in open(benign_jsonl):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        tokens = convert_7to8(e.get("tokens", e.get("event", [])))
        ids = [stoi.get(t, 0) for t in tokens]
        window = (window + ids)[-CTX:]
        n_toks, L = len(ids), len(window)
        start = max(L - n_toks, 1)

        with torch.no_grad():
            x = torch.tensor(window, device=DEVICE).unsqueeze(0)
            lp = torch.log_softmax(model(x), dim=-1)[0]
            tgt = torch.tensor(window[start:L], device=DEVICE)
            nll = -lp[start-1:L-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

        for t, v in zip(tokens, nll.tolist()):
            s = slot_of(t)
            slot_nlls[s].append(v)
        n += 1

    # 标定 τ = max(p99.5 * margin, MIN_TAU)
    # margin=1.05 补偿小样本 p99.5 的方差（n=217 时 p99.5≈最大值，不够稳定）
    MARGIN = 1.05
    new_tau = {}
    print(f"标定样本: {n} 事件, margin={MARGIN}\n")
    print(f"{'Slot':<10}{'n':>8}{'mean':>8}{'p50':>8}{'p90':>8}"
          f"{'p95':>8}{'p99':>8}{'p995':>8}{'τ':>8}")
    print("-" * 74)
    for slot in sorted(slot_nlls.keys()):
        vals = np.array(slot_nlls[slot])
        p50 = np.percentile(vals, 50)
        p90 = np.percentile(vals, 90)
        p95 = np.percentile(vals, 95)
        p99 = np.percentile(vals, 99)
        p995 = np.percentile(vals, 99.5)
        tau = max(float(p995) * MARGIN, MIN_TAU)
        new_tau[slot] = round(tau, 3)
        print(f"{slot:<10}{len(vals):>8}{np.mean(vals):>8.2f}{p50:>8.2f}"
              f"{p90:>8.2f}{p95:>8.2f}{p99:>8.2f}{p995:>8.2f}{tau:>8.2f}")

    out_path = os.path.join(model_dir, "slot_tau_local.json")
    with open(out_path, "w") as f:
        json.dump({
            "slot_tau": new_tau,
            "source": "host_real_benign",
            "percentile": percentile,
            "min_tau": MIN_TAU,
            "n_calib": n,
        }, f, indent=2)
    print(f"\nslot_tau 已保存 → {out_path}")
    return new_tau


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        DET, "model-host-real-v2")
    benign_jsonl = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        DET, "data/host_real_benign.jsonl")
    print(f"=== 标定 slot_tau (模型: {os.path.basename(model_dir)}) ===\n")
    calibrate(model_dir, benign_jsonl)


if __name__ == "__main__":
    main()
