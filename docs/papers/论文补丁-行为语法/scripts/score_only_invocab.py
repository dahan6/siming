#!/usr/bin/env python3
"""用已训练的 TinyGPT 生产模型做 score-only 硬子集分析（不训练）

直接加载 model-host-r3-clean/prior.pt（正式 host 基线，vocab 167），
对 rove/bee 轨迹打分并按 UNK 拆分词表内/外子集。
比子采样重训更权威（全量良性词表）。

用法: score_only_invocab.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_prior import TinyGPT, CTX

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 0.3
DET = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(DET, "model-host-r3-clean", "prior.pt")
ROVE = os.path.join(DET, "data", "rove_attacks.jsonl")
BEE = os.path.join(DET, "data", "bee_active.jsonl")


def score_stream(model, stoi, seqs, max_n=40000):
    model.eval()
    window, ewma = [], 0.0
    out = []
    for i, ev in enumerate(seqs):
        if i >= max_n:
            break
        ids = [stoi.get(t, 0) for t in ev]
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
        s = nll.max().item()
        ewma = ALPHA * s + (1 - ALPHA) * ewma
        unk = sum(1 for t in ev if t not in stoi)
        out.append((s, ewma, unk))
    return np.array([o[0] for o in out]), np.array([o[1] for o in out]), np.array([o[2] for o in out])


def load(path):
    seqs = []
    for line in open(path):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("tokens"):
            seqs.append(e["tokens"])
    return seqs


def main():
    ckpt = torch.load(MODEL, map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tau = ckpt["baseline_nll"]["p995"]
    print(f"生产模型 vocab={len(stoi)} τ(p995)={tau:.3f}")

    report = {"model": "model-host-r3-clean", "tau": tau, "vocab": len(stoi)}
    for tag, path in [("rove", ROVE), ("bee_active", BEE)]:
        seqs = load(path)
        s, e, u = score_stream(model, stoi, seqs)
        all_det = float(((s > tau) | (u > 0)).mean())
        inv = u == 0
        inv_share = float(inv.mean())
        inv_det = float((s[inv] > tau).mean()) if inv.sum() else None
        inv_det_e = float((e[inv] > tau).mean()) if inv.sum() else None
        report[tag] = {"n": int(len(s)), "all_detect": all_det,
                       "invocab_share": inv_share, "invocab_detect": inv_det,
                       "invocab_detect_ewma": inv_det_e,
                       "score_p50": float(np.percentile(s, 50))}
        print(f"{tag}: 全量 {all_det:.1%} | 词表内 {inv_share:.1%} (n={inv.sum()}) "
              f"检出 {inv_det:.1%} (EWMA {inv_det_e:.1%}) | p50={report[tag]['score_p50']:.2f}")

    out = os.path.join(DET, "results", "invocab_production_model.json")
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
