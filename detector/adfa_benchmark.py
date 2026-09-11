#!/usr/bin/env python3
"""ADFA-LD 外部基准：行为语法先验 vs 公开攻击数据集
训练：833 条良性 syscall 序列 → TinyGPT 先验
评测：4372 条良性验证 + 746 条攻击（6 类），按 trace 打分（窗口 max NLL 的均值）
阈值：良性验证集 p99（FPR≈1%），报各类检出率 + ROC 近似点

用法: adfa_benchmark.py <adfa_root> [--epochs 2]
"""
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_prior import TinyGPT, CTX

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 2
BATCH, LR = 256, 3e-4


def load_traces(root):
    def rd(pattern):
        out = []
        for f in sorted(glob.glob(pattern)):
            nums = open(f).read().split()
            if len(nums) >= 16:
                out.append((f, torch.tensor([int(x) for x in nums], dtype=torch.long)))
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
    print(f"良性训练 {len(train)} 验证 {len(valid)} 攻击类 { {k: len(v) for k, v in attacks.items()} }")

    ids = torch.cat([t for _, t in train])
    vocab = int(ids.max().item()) + 2  # syscall 号 + UNK 余量
    split = int(len(ids) * 0.9)
    tr, va = ids[:split], ids[split:]
    model = TinyGPT(vocab).to(DEVICE)
    print(f"token 总数 {len(ids)} 词表 {vocab} 参数量 {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()

    def batches(data, starts, bs=BATCH):
        for i in range(0, len(starts) - bs + 1, bs):
            cs = starts[i:i + bs]
            xs = torch.stack([data[s:s + CTX] for s in cs]).to(DEVICE)
            ys = torch.stack([data[s + 1:s + CTX + 1] for s in cs]).to(DEVICE)
            yield xs, ys

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

    # 训练完立即存档（打分阶段再崩也不丢模型）
    torch.save({"model": model.state_dict(), "vocab": vocab},
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "adfa_prior.pt"))

    def trace_score(trace):
        """整条 trace 打分：滑窗逐 token NLL，取窗口均值"""
        t = trace.to(DEVICE)
        if len(t) > 20000:
            t = t[:20000]
        xs = [t[max(0, i - CTX + 1):i + 1] for i in range(0, len(t), 16)]
        scores = []
        with torch.no_grad():
            for i in range(0, len(xs), BATCH):
                chunk = xs[i:i + BATCH]
                L = max(len(c) for c in chunk)
                x = torch.zeros(len(chunk), L, dtype=torch.long, device=DEVICE)
                for j, c in enumerate(chunk):
                    x[j, :len(c)] = c
                    if len(c) < L:
                        x[j, len(c):] = c[0]
                lp = torch.log_softmax(model(x), dim=-1)
                for j, c in enumerate(chunk):
                    lc = len(c)
                    if lc < 2:
                        continue
                    nll = -lp[j, :lc - 1].gather(-1, x[j, 1:lc].unsqueeze(-1)).squeeze(-1)
        scores.append(nll.mean().item())
        if not scores:
            return 0.0, np.array([])
        # 攻击是长正常轨迹里的短片段：取窗口分的 98 分位（抗稀释），而非均值
        return float(np.quantile(scores, 0.98)), np.array(scores)

    val_pairs = [trace_score(t) for _, t in valid]
    val_scores = np.array([s for s, _ in val_pairs])
    thr = np.quantile(val_scores, 0.99)
    fpr = float((val_scores > thr).mean())
    print(f"\n阈值(p99 良性)={thr:.3f}  验证集 FPR={fpr:.2%}")
    print(f"{'攻击类':<20}{'条数':>6}{'检出率':>8}{'平均分':>9}")
    det_all = []
    atk_scores = {}
    for cls, traces in sorted(attacks.items()):
        pairs = [trace_score(t) for _, t in traces]
        ss = np.array([s for s, _ in pairs])
        atk_scores[cls] = ss
        det = float((ss > thr).mean())
        det_all.append(det)
        print(f"{cls:<20}{len(traces):>6}{det:>8.1%}{ss.mean():>9.2f}")
    print(f"\n宏平均检出率 {np.mean(det_all):.1%} @ FPR≈1%")

    # 缓存：模型 + 全部分数（后续分析免重训）
    torch.save({"model": model.state_dict(), "vocab": vocab},
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "adfa_prior.pt"))
    np.savez("/tmp/adfa_scores.npz", val=val_scores,
             **{f"atk_{k}": v for k, v in atk_scores.items()})
    np.save("/tmp/adfa_val_windows.npy", np.array([w for _, w in val_pairs], dtype=object))
    print("模型与分数已缓存（adfa_prior.pt, /tmp/adfa_scores.npz）")


if __name__ == "__main__":
    main()
