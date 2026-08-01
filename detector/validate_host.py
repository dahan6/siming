#!/usr/bin/env python3
"""宿主机模型验证：留出误报率 + 宿主机语境合成异常 + 分离余量
用法: validate_host.py <model_dir> <host_tokens.jsonl>
判定达标：FPR≤1% 且 全部合成异常检出 且 分离余量(min异常分/正常p95)≥3
"""
import json
import os
import sys

import numpy as np
import torch

from train_prior import TinyGPT, CTX

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 宿主机语境的异常用例（真实工作机，EXT 外联本身是常态，异常在"高端口+非常驻进程"）
CASES = {
    "A-bash高端口外联": ["ET:CONN", "PROC:bash", "ARGV0", "PC:NONE", "PARENT:?", "UID:1000", "DST:EXT:HIGH", "DT5"],
    "B-base64参数": ["ET:EXEC", "PROC:bash", "ARGV:N2B", "PC:NONE", "PARENT:bash", "UID:1000", "DST:NONE", "DT1"],
    "C-伪装进程名": ["ET:EXEC", "PROC:.kworker_u9", "ARGV:N1-", "PC:NONE", "PARENT:bash", "UID:0", "DST:NONE", "DT0"],
    "E-root高端口外联": ["ET:CONN", "PROC:python3", "ARGV0", "PC:NONE", "PARENT:?", "UID:0", "DST:EXT:HIGH", "DT2"],
}


def score_tokens(model, stoi, window):
    """对完整 window（token id 列表）最后一个事件的 token 打分，取 max NLL。"""
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
    n_hold = int(sys.argv[3]) if len(sys.argv) > 3 else 800
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
    # EWMA 滑窗（设计文档第 3 节）：检测统计量是平滑后的窗口分，不是单点
    ewma = np.empty_like(s)
    ewma[0] = s[0]
    for i in range(1, len(s)):
        ewma[i] = 0.3 * s[i] + 0.7 * ewma[i - 1]
    fpr = (ewma > tau).mean()

    ctx = [t for ev in holdout[-4:] for t in ev["tokens"]]
    print(f"τ(p995)={tau:.3f} | 留出 {len(s)} 事件: p50={np.percentile(s,50):.3f} "
          f"p95={p95:.3f} max={s.max():.3f} | FPR(原始)={fpr_raw:.2%} FPR(EWMA)={fpr:.2%}")
    print(f"{'用例':<18}{'惊讶度':>9}{'判定':>6} 触发标准")
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
        print(f"{name:<18}{sc:9.2f}{'异常' if detected else '正常':>6} {why}")

    margin = min(margins)
    ok = fpr <= 0.01 and all_detected and margin >= 3.0
    print(f"\n分离余量(最弱异常/p95)={margin:.1f}x | 全部检出={all_detected} | FPR达标={fpr <= 0.01}")
    print(f"达标判定: {'PASS' if ok else 'FAIL'}")
    # 机器可读摘要，供 night_round.sh 追加到迭代日志
    with open(os.path.join(model_dir, "last_validation.json"), "w") as f:
        json.dump({"tau": float(tau), "fpr": float(fpr), "fpr_raw": float(fpr_raw),
                   "p95": float(p95), "anomalies": anomaly_scores,
                   "margin": float(margin),
                   "all_detected": bool(all_detected), "pass": bool(ok)}, f)
    # 逐事件分数落地，供绘图脚本使用
    np.save(os.path.join(model_dir, "holdout_scores.npy"), s)
    np.save(os.path.join(model_dir, "holdout_ewma.npy"), ewma)

    # 误报样本取证：打印分数最高的 5 个基线事件，供迭代分析
    order = np.argsort(s)[::-1][:5]
    print("\n误报 Top5（供迭代分析）:")
    for idx in order:
        print(f"  {s[idx]:7.2f} {' '.join(holdout[idx]['tokens'])}")


if __name__ == "__main__":
    main()
