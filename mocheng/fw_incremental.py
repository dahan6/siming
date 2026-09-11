#!/usr/bin/env python3
"""墨城防火墙: 增量训练 + 自适应阈值

特性:
  1. 增量训练: 新采集的数据追加到训练集，重训模型（不需从零开始）
  2. 自适应阈值: τ 随实时流量模式滑动调整（EWMA 跟踪 p995）
  3. 时序增强: 从原始事件生成时序变体（shift/jitter），让模型更鲁棒

用法:
  python fw_incremental.py model/ data/new_tokens.jsonl
"""
import json
import os
import sys
import time
import math

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fw_paths import MODEL_DIR, DATA_DIR
from train_prior import TinyGPT, CTX, D_MODEL, N_LAYER, N_HEAD, EPOCHS, BATCH, LR, DEVICE, UNK


def augment_sequences(seqs, n_shift=2, n_jitter=2):
    """时序数据增强

    1. DT shift: 把 DT 桶随机 ±1 档（模拟不同时间段的真实间隔波动）
    2. DT jitter: 把 DT0/DT6（边界值）随机变为相邻桶

    这让模型不再对特定 DT 序列过拟合——
    司命的教训：真实流量 DT 分布随时间/负载波动很大。
    """
    augmented = list(seqs)  # 保留原始
    rng = np.random.default_rng(42)

    for seq in seqs:
        for _ in range(n_shift):
            new_seq = list(seq)
            # 找到 DT token 的位置（最后一个 token）
            for i, tok in enumerate(new_seq):
                if tok.startswith("DT") and len(tok) == 3:
                    dt_val = int(tok[2])
                    # 随机偏移 ±1
                    shift = rng.choice([-1, 0, 1])
                    new_dt = max(0, min(6, dt_val + shift))
                    new_seq[i] = f"DT{new_dt}"
            augmented.append(new_seq)

        for _ in range(n_jitter):
            new_seq = list(seq)
            # 随机替换一个 DT token 为另一个合理值
            dt_indices = [i for i, t in enumerate(new_seq) if t.startswith("DT")]
            if dt_indices:
                idx = rng.choice(dt_indices)
                new_seq[idx] = f"DT{rng.integers(0, 7)}"
            augmented.append(new_seq)

    return augmented


def load_all_tokens(model_dir, new_tokens_path=None):
    """加载累积训练数据（旧基线 + 新数据）"""
    baseline_path = os.path.join(model_dir, "training_tokens.jsonl")
    all_seqs = []

    # 加载旧基线
    if os.path.exists(baseline_path):
        for line in open(baseline_path):
            all_seqs.append(json.loads(line)["tokens"])

    # 追加新数据
    if new_tokens_path and os.path.exists(new_tokens_path):
        for line in open(new_tokens_path):
            all_seqs.append(json.loads(line)["tokens"])

    return all_seqs


def incremental_train(model_dir, new_tokens_path, epochs=2, augment=True):
    """增量训练模型

    1. 合并旧基线 + 新数据
    2. 时序增强（可选）
    3. 在旧权重基础上微调（不是从零开始）
    """
    from collections import Counter
    seqs = load_all_tokens(model_dir, new_tokens_path)
    if not seqs:
        print("[ERROR] 无训练数据")
        return False

    # 时序增强
    if augment:
        before = len(seqs)
        seqs = augment_sequences(seqs)
        print(f"时序增强: {before} → {len(seqs)} 序列")

    # 词表
    c = Counter(t for ev in seqs for t in ev)
    vocab = [UNK] + sorted(t for t, n in c.items() if n >= 2)
    stoi = {t: i for i, t in enumerate(vocab)}

    ids = torch.tensor([stoi.get(t, 0) for ev in seqs for t in ev], dtype=torch.long)
    split = int(len(ids) * 0.85)
    train_ids, val_ids = ids[:split], ids[split:]
    print(f"token 总数 {len(ids)}, 词表 {len(stoi)}, 训练 {len(train_ids)} / 验证 {len(val_ids)}")

    # 加载旧权重（如果存在）或新建模型
    prior_path = os.path.join(model_dir, "prior.pt")
    model = TinyGPT(len(stoi)).to(DEVICE)
    if os.path.exists(prior_path):
        old_ckpt = torch.load(prior_path, map_location=DEVICE, weights_only=False)
        try:
            model.load_state_dict(old_ckpt["model"])
            print("从旧权重微调")
        except RuntimeError:
            print("词表变化，从头训练")
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR * 0.5)  # 微调用更小的 LR
    lossf = torch.nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(epochs):
        starts = torch.randperm(max(1, len(train_ids) - CTX - 1)).tolist()
        tot, nb = 0.0, 0
        for i in range(0, len(starts) - BATCH + 1, BATCH):
            chunk = starts[i:i + BATCH]
            xs = torch.stack([train_ids[s:s + CTX] for s in chunk]).to(DEVICE)
            ys = torch.stack([train_ids[s + 1:s + CTX + 1] for s in chunk]).to(DEVICE)
            loss = lossf(model(xs).view(-1, len(stoi)), ys.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1

        # 验证
        model.eval()
        vlosses = []
        with torch.no_grad():
            vstarts = list(range(0, max(1, len(val_ids) - CTX), CTX))
            for i in range(0, len(vstarts) - BATCH + 1, BATCH):
                chunk = vstarts[i:i + BATCH]
                xs = torch.stack([val_ids[s:s + CTX] for s in chunk]).to(DEVICE)
                ys = torch.stack([val_ids[s + 1:s + CTX + 1] for s in chunk]).to(DEVICE)
                vlosses.append(lossf(model(xs).view(-1, len(stoi)), ys.view(-1)).item())
        vl = sum(vlosses) / max(1, len(vlosses))
        model.train()
        print(f"epoch {ep+1}/{epochs} train_loss {tot/max(1,nb):.4f} val_loss {vl:.4f} "
              f"(val ppl {math.exp(min(vl, 20)):.1f}) {time.time()-t0:.0f}s")

    # 计算 baseline NLL
    model.eval()
    nlls = []
    with torch.no_grad():
        starts = list(range(0, max(1, len(val_ids) - CTX), CTX // 2))
        for i in range(0, len(starts) - BATCH + 1, BATCH):
            chunk = starts[i:i + BATCH]
            xs = torch.stack([val_ids[s:s + CTX] for s in chunk]).to(DEVICE)
            ys = torch.stack([val_ids[s + 1:s + CTX + 1] for s in chunk]).to(DEVICE)
            lp = torch.log_softmax(model(xs), dim=-1)
            nll = -lp.gather(-1, ys.unsqueeze(-1)).squeeze(-1)
            nlls.extend(nll.view(-1).tolist())
    if not nlls:
        # 验证集太小，用训练集尾部估算
        with torch.no_grad():
            for i in range(0, min(len(train_ids) - CTX, CTX * 10), CTX):
                xs = train_ids[i:i + CTX].unsqueeze(0).to(DEVICE)
                ys = train_ids[i + 1:i + CTX + 1].unsqueeze(0).to(DEVICE)
                lp = torch.log_softmax(model(xs), dim=-1)
                nll = -lp.gather(-1, ys.unsqueeze(-1)).squeeze(-1)
                nlls.extend(nll.view(-1).tolist())
    nlls_t = torch.tensor(nlls)
    qs = torch.quantile(nlls_t, torch.tensor([0.5, 0.95, 0.99, 0.995]))
    stats = {"mean": nlls_t.mean().item(), "p50": qs[0].item(),
             "p95": qs[1].item(), "p99": qs[2].item(), "p995": qs[3].item()}
    print("基线 NLL:", {k: round(v, 3) for k, v in stats.items()})

    # 保存模型 + 训练数据
    torch.save({"model": model.state_dict(), "stoi": stoi,
                "config": {"d_model": D_MODEL, "n_layer": N_LAYER,
                           "n_head": N_HEAD, "ctx": CTX},
                "baseline_nll": stats},
               prior_path)

    # 累积训练数据（用于下次增量）
    train_log = os.path.join(model_dir, "training_tokens.jsonl")
    if new_tokens_path and new_tokens_path != train_log:
        with open(train_log, "a") as out:
            for line in open(new_tokens_path):
                out.write(line)
    print(f"模型已保存 -> {prior_path}")
    print(f"训练数据已累积 -> {train_log}")
    return True


class AdaptiveTau:
    """自适应阈值：用 EWMA 跟实时 p995 滑动调整 τ

    司命的教训：流量高峰/低谷用同一个阈值不合理。
    τ 在 ±20% 范围内随实时惊讶度分布自适应。
    """

    def __init__(self, initial_tau, alpha=0.01, min_ratio=0.8, max_ratio=1.2):
        self.base_tau = initial_tau
        self.current_tau = initial_tau
        self.alpha = alpha
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.ewma_p95 = initial_tau * 0.6  # p95 通常 < p995

    def update(self, scores):
        """用最近一批打分更新自适应阈值

        Args:
            scores: 最近一批事件的 s_ev 列表
        """
        if not scores:
            return self.current_tau
        arr = np.array(scores)
        p95 = float(np.percentile(arr, 95))
        # EWMA 平滑 p95
        self.ewma_p95 = self.alpha * p95 + (1 - self.alpha) * self.ewma_p95
        # τ = base * ratio，ratio 由 p95 的当前位置决定
        if self.base_tau > 0:
            ratio = self.ewma_p95 / (self.base_tau * 0.6)  # 归一化
            ratio = max(self.min_ratio, min(self.max_ratio, ratio))
            self.current_tau = self.base_tau * ratio
        return self.current_tau


def main():
    if len(sys.argv) < 3:
        print("用法: fw_incremental.py <model_dir> <new_tokens.jsonl> [--no-augment]")
        sys.exit(1)
    model_dir = sys.argv[1]
    new_tokens = sys.argv[2]
    augment = "--no-augment" not in sys.argv
    success = incremental_train(model_dir, new_tokens, augment=augment)
    if success:
        print("\n增量训练完成。运行 fw_calibrate.py 更新分维度阈值。")


if __name__ == "__main__":
    main()
