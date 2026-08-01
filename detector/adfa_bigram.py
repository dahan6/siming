#!/usr/bin/env python3
"""ADFA-LD bigram 变体：syscall 对作为 token，罕见转移触发 UNK
动机（诊断结论）：神经模型泛化=宽恕，常见 syscall 的非常规排序打不出惊讶度；
把词表具体化到 bigram，让攻击的罕见转移掉出词表（UNK 机制复活）。

用法: adfa_bigram.py <adfa_root> [--epochs 3]
"""
import glob
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_prior import TinyGPT, CTX

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 3
BATCH, LR = 256, 3e-4


def load_traces(root):
    def rd(pattern):
        out = []
        for f in sorted(glob.glob(pattern)):
            nums = open(f).read().split()
            if len(nums) >= 16:
                out.append([int(x) for x in nums])
        return out
    train = rd(os.path.join(root, "Training_Data_Master", "*"))
    valid = rd(os.path.join(root, "Validation_Data_Master", "*"))
    attacks = {}
    for d in sorted(glob.glob(os.path.join(root, "Attack_Data_Master", "*"))):
        cls = os.path.basename(d.rstrip("/")).rsplit("_", 1)[0]
        attacks.setdefault(cls, []).extend(rd(os.path.join(d, "*")))
    return train, valid, attacks


def main():
    root = sys.argv[1]
    train, valid, attacks = load_traces(root)

    # 自训练模式：良性数据全用于训练（攻击永不见），验证集尾部做阈值
    # 动机：833 条训练集盖不住良性 bigram 空间（验证集 UNK 率 23%）
    n_val = int(sys.argv[sys.argv.index("--valsplit") + 1]) if "--valsplit" in sys.argv else 0
    if n_val > 0:
        train = train + valid[:n_val]
        valid = valid[n_val:]
        print(f"自训练: 训练 {len(train)} 阈值验证 {len(valid)}")

    # bigram 词表：训练集里 min_freq>=3 的 syscall 对
    cnt = Counter()
    for t in train:
        cnt.update(zip(t, t[1:]))
    bstoi = {b: i + 1 for i, (b, c) in enumerate(cnt.most_common()) if c >= 5}
    print(f"bigram 词表 {len(bstoi)}（min_freq=5，训练集共 {sum(cnt.values())} 对）")

    def encode(t):
        return [bstoi.get(b, 0) for b in zip(t, t[1:])]

    # 分批编码避免一次性大 tensor OOM
    encoded = []
    for t in train:
        encoded.extend(encode(t))
    ids = torch.tensor(encoded, dtype=torch.long)
    del encoded
    vocab = len(bstoi) + 1
    split = int(len(ids) * 0.9)
    tr, va = ids[:split], ids[split:]
    model = TinyGPT(vocab).to(DEVICE)
    print(f"token 总数 {len(ids)} 参数量 {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()

    def batches(data, starts, bs=BATCH):
        for i in range(0, len(starts) - bs + 1, bs):
            cs = starts[i:i + bs]
            yield (torch.stack([data[s:s + CTX] for s in cs]).to(DEVICE),
                   torch.stack([data[s + 1:s + CTX + 1] for s in cs]).to(DEVICE))

    t0 = time.time()
    for ep in range(EPOCHS):
        starts = torch.randperm(len(tr) - CTX - 1).tolist()
        tot, nb = 0.0, 0
        for x, y in batches(tr, starts):
            loss = lossf(model(x).view(-1, vocab), y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vs = list(range(0, len(va) - CTX, CTX))
            vl = np.mean([lossf(model(x).view(-1, vocab), y.view(-1)).item()
                          for x, y in batches(va, vs, min(BATCH, max(1, len(vs))))])
        model.train()
        print(f"epoch {ep+1}/{EPOCHS} train {tot/nb:.4f} val {vl:.4f} {time.time()-t0:.0f}s")

    def trace_score(trace):
        enc = encode(trace)
        unk = sum(1 for i in enc if i == 0) / max(1, len(enc))
        t = torch.tensor(enc, dtype=torch.long, device=DEVICE)
        if len(t) > 5000:
            t = t[:5000]
        scores = []
        with torch.no_grad():
            for s in range(0, max(1, len(t) - CTX), CTX):
                x = t[s:s + CTX].unsqueeze(0)
                if x.size(1) < 2:
                    continue
                lp = torch.log_softmax(model(x), dim=-1)
                nll = -lp[0, :-1].gather(-1, x[0, 1:].unsqueeze(-1)).squeeze(-1)
                scores.append(nll.mean().item())
        q = float(np.quantile(scores, 0.98)) if scores else 0.0
        return q, unk

    val_sc = np.array([trace_score(t) for t in valid], dtype=object)
    # 双信号打分：q98 窗口分 + UNK 率（标准化后取 max）
    vq = np.array([s[0] for s in val_sc]); vu = np.array([s[1] for s in val_sc])
    thr_q, thr_u = np.quantile(vq, 0.99), np.quantile(vu, 0.99)

    def det(q, u):
        return float((q > thr_q).mean() + 0)  # 仅 q 通道
    def det_fused(qs, us):
        zq = qs / thr_q; zu = us / np.maximum(thr_u, 1e-9)
        return float((np.maximum(zq, zu) > 1.0).mean())

    fpr_q = float((vq > thr_q).mean())
    fpr_f = det_fused(vq, vu)
    print(f"\n阈值: q98={thr_q:.3f} UNK率={thr_u:.4f} | 验证 FPR: q通道={fpr_q:.2%} 融合={fpr_f:.2%}")
    print(f"{'攻击类':<20}{'条数':>6}{'q通道':>8}{'融合':>8}{'平均UNK率':>10}")
    dq, df = [], []
    for cls, traces in sorted(attacks.items()):
        sc = np.array([trace_score(t) for t in traces], dtype=object)
        q = np.array([s[0] for s in sc]); u = np.array([s[1] for s in sc])
        d1, d2 = det(q, u), det_fused(q, u)
        dq.append(d1); df.append(d2)
        print(f"{cls:<20}{len(traces):>6}{d1:>8.1%}{d2:>8.1%}{u.mean():>10.3f}")
    print(f"\n宏平均: q通道 {np.mean(dq):.1%} | 融合 {np.mean(df):.1%} @ FPR≈1%")


if __name__ == "__main__":
    main()
