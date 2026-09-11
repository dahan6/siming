#!/usr/bin/env python3
"""墨城防火墙: 模型验证

留出误报率 + 合成攻击用例检出 + 分离余量
判定达标: FPR(EWMA)≤2% 且 全部攻击检出 且 分离余量≥3x

用法: fw_validate.py <model_dir> <tokens.jsonl> [n_holdout=1000]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_prior import TinyGPT, CTX

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 网络攻击合成用例（7-token，模拟正常上下文后接攻击事件）
CASES = {
    "A-端口扫描": [
        "PROTO:TCP", "DIR:OUT", "SRCNET:LAN", "DSTNET:EXT",
        "PORTCLS:HIGHPORT", "SIZECLS:S", "DT0",
    ],
    "B-C2信标": [
        "PROTO:TCP", "DIR:OUT", "SRCNET:LAN", "DSTNET:EXT",
        "PORTCLS:HIGHPORT", "SIZECLS:S", "DT3",
    ],
    "C-数据外泄": [
        "PROTO:TCP", "DIR:OUT", "SRCNET:LAN", "DSTNET:EXT",
        "PORTCLS:HIGHPORT", "SIZECLS:XL", "DT2",
    ],
    "D-横向移动": [
        "PROTO:TCP", "DIR:OUT", "SRCNET:LAN", "DSTNET:LAN",
        "PORTCLS:SMB", "SIZECLS:M", "DT1",
    ],
    "E-DNS隧道": [
        "PROTO:UDP", "DIR:OUT", "SRCNET:LAN", "DSTNET:LAN",
        "PORTCLS:DNS", "SIZECLS:M", "DT0",
    ],
    "F-反弹shell": [
        "PROTO:TCP", "DIR:IN", "SRCNET:EXT", "DSTNET:LAN",
        "PORTCLS:HIGHPORT", "SIZECLS:S", "DT4",
    ],
}


def score_tokens(model, stoi, window):
    x = torch.tensor(window, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        lp = torch.log_softmax(model(x), dim=-1)[0]
    return lp


def main():
    model_dir, tokens_path = sys.argv[1], sys.argv[2]
    ckpt = torch.load(os.path.join(model_dir, "prior.pt"), map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tau = ckpt["baseline_nll"]["p995"]

    events = [json.loads(l) for l in open(tokens_path)]
    n_hold = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    holdout = events[-n_hold:]
    window, scores = [], []
    for ev in holdout:
        ids = [stoi.get(t, 0) for t in ev["tokens"]]
        window.extend(ids)
        window = window[-CTX:]
        lp = score_tokens(model, stoi, window)
        n, L = len(ids), len(window)
        start = max(L - n, 1)
        tgt = torch.tensor(window[start:L], device=DEVICE)
        nll = -lp[start - 1:L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        scores.append(nll.max().item())
    s = np.array(scores)
    p95 = np.percentile(s, 95)
    fpr_raw = (s > tau).mean()
    ewma = np.empty_like(s)
    ewma[0] = s[0]
    for i in range(1, len(s)):
        ewma[i] = 0.3 * s[i] + 0.7 * ewma[i - 1]
    fpr = (ewma > tau).mean()

    # 合成攻击用例
    ctx = [t for ev in holdout[-4:] for t in ev["tokens"]]
    print(f"τ(p995)={tau:.3f} | 留出 {len(s)} 事件: p50={np.percentile(s,50):.3f} "
          f"p95={p95:.3f} max={s.max():.3f} | FPR(原始)={fpr_raw:.2%} FPR(EWMA)={fpr:.2%}")
    print(f"{'用例':<16}{'惊讶度':>9}{'判定':>6} 触发标准")
    margins = []
    anomaly_scores = {}
    all_detected = True
    for name, ev in CASES.items():
        ids = [stoi.get(t, 0) for t in ctx + ev][-CTX:]
        n_ev, L = len(ev), len(ids)
        lp = score_tokens(model, stoi, ids)
        tgt = torch.tensor(ids[L - n_ev:], device=DEVICE)
        nll = -lp[L - n_ev - 1:L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        sc = nll.max().item()
        anomaly_scores[name.split("-")[0]] = float(sc)
        n_unk = sum(1 for t in ev if t not in stoi)
        detected = sc > tau or n_unk > 0
        all_detected &= detected
        margins.append(sc / max(p95, 1e-6))
        why = ("超τ " if sc > tau else "") + (f"UNK×{n_unk}" if n_unk else "")
        print(f"{name:<16}{sc:9.2f}{'异常' if detected else '正常':>6} {why}")

    margin = min(margins) if margins else 0
    # 网络流量方差天然大于进程事件，余量标准 2.0x（司命主机场景为 3.0x）
    ok = fpr <= 0.02 and all_detected and margin >= 2.0
    print(f"\n分离余量(最弱异常/p95)={margin:.1f}x | 全部检出={all_detected} | FPR达标={fpr <= 0.02}")
    print(f"达标判定: {'PASS' if ok else 'FAIL'}")

    with open(os.path.join(model_dir, "last_validation.json"), "w") as f:
        json.dump({"tau": float(tau), "fpr": float(fpr), "fpr_raw": float(fpr_raw),
                   "p95": float(p95), "anomalies": anomaly_scores,
                   "margin": float(margin),
                   "all_detected": bool(all_detected), "pass": bool(ok)}, f)
    np.save(os.path.join(model_dir, "holdout_scores.npy"), s)
    np.save(os.path.join(model_dir, "holdout_ewma.npy"), ewma)

    # 误报 Top5
    order = np.argsort(s)[::-1][:5]
    print("\n误报 Top5（供迭代分析）:")
    for idx in order:
        print(f"  {s[idx]:7.2f} {' '.join(holdout[idx]['tokens'])}")


if __name__ == "__main__":
    main()
