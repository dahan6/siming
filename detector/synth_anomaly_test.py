#!/usr/bin/env python3
"""离线合成异常验证（不触碰任何 VM/遥测）
构造三类典型异常行为的 token 序列，用先验模型打分，
与基线分位数对比，验证"判定标准一（超阈）+ 标准二（UNK）"。

用法: synth_anomaly_test.py [model_path]
"""
import json
import os
import sys

import torch

from train_prior import TinyGPT, CTX

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 正常背景事件（从基线里常见的模式拼的上下文）
NORMAL_CTX = [
    ["ET:CONN", "PROC:systemd-resolve", "ARGV0", "PARENT:?", "UID:991", "DST:LAN:WELL", "DT3"],
    ["ET:EXEC", "PROC:systemctl", "ARGV:N2-", "PARENT:bash", "UID:0", "DST:NONE", "DT0"],
    ["ET:EXEC", "PROC:journalctl", "ARGV:N1-", "PARENT:bash", "UID:0", "DST:NONE", "DT4"],
    ["ET:CONN", "PROC:rsyslogd", "ARGV0", "PARENT:?", "UID:0", "DST:LAN:WELL", "DT2"],
]

CASES = {
    "A-高端口外联": ["ET:CONN", "PROC:bash", "ARGV0", "PARENT:?", "UID:1000", "DST:LAN:HIGH", "DT5"],
    "B-base64参数": ["ET:EXEC", "PROC:bash", "ARGV:N2B", "PARENT:bash", "UID:1000", "DST:NONE", "DT1"],
    "C-陌生进程伪装": ["ET:EXEC", "PROC:.kworker_u9", "ARGV:N1-", "PARENT:bash", "UID:0", "DST:NONE", "DT0"],
    "D-正常对照": ["ET:EXEC", "PROC:ps", "ARGV:N2-", "PARENT:bash", "UID:1000", "DST:NONE", "DT3"],
}


def score_event(model, stoi, ctx_tokens, ev_tokens):
    """把 ctx+event 拼成窗口，返回事件 token 的最大 NLL 和 UNK 数。"""
    ids = [stoi.get(t, 0) for t in ctx_tokens + ev_tokens][-CTX:]
    n_ev = len(ev_tokens)
    L = len(ids)
    with torch.no_grad():
        x = torch.tensor(ids, device=DEVICE).unsqueeze(0)
        lp = torch.log_softmax(model(x), dim=-1)[0]
        # 对齐：事件 token 位于 [L-n_ev, L)，各自由前一位 logits 预测
        tgt = torch.tensor(ids[L - n_ev:], device=DEVICE)
        nll = -lp[L - n_ev - 1:L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    n_unk = sum(1 for t in ev_tokens if t not in stoi)
    return nll.max().item(), n_unk


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/defense-lab/detector/model/prior.pt")
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    bl = ckpt["baseline_nll"]
    print(f"基线分位数: p50={bl['p50']:.2f} p95={bl['p95']:.2f} p99={bl['p99']:.2f} τ(p995)={bl['p995']:.2f}\n")

    ctx = [t for ev in NORMAL_CTX for t in ev]
    print(f"{'用例':<16} {'惊讶度':>8} {'判定':<6} 触发标准")
    for name, ev in CASES.items():
        s, n_unk = score_event(model, stoi, ctx, ev)
        verdict = "异常" if (s > bl["p995"] or n_unk > 0) else "正常"
        why = []
        if s > bl["p995"]:
            why.append("超τ")
        if n_unk:
            why.append(f"UNK×{n_unk}")
        print(f"{name:<16} {s:8.2f} {verdict:<6} {','.join(why) or '-'}")


if __name__ == "__main__":
    main()
