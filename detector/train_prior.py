#!/usr/bin/env python3
"""Siming M2: 序列先验训练
小型因果 Transformer 学习正常行为 token 流的条件分布 p(x_t | x_<t)。
异常分 = 交叉熵（惊讶度）。设计见 docs/检测模型架构设计.md 第 2-3 节。

用法: train_prior.py <tokens.jsonl> [out_dir]
"""
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

from functools import partial
print = partial(print, flush=True)

D_MODEL, N_LAYER, N_HEAD, CTX = 128, 4, 4, 128
EPOCHS, BATCH, LR = 2, 256, 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UNK = "<UNK>"


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, D_MODEL)
        self.pos = nn.Embedding(CTX, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=512,
            batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=N_LAYER,
                                            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size)
        self.register_buffer("causal_mask", torch.triu(
            torch.full((CTX, CTX), float("-inf")), diagonal=1))

    def forward(self, idx):
        t = idx.size(1)
        h = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        h = self.blocks(h, mask=self.causal_mask[:t, :t])
        return self.head(self.norm(h))


def load_stream(path):
    seqs = []
    for line in open(path):
        seqs.append(json.loads(line)["tokens"])
    return seqs  # list of events, each event = list of 7 tokens


def build_vocab(seqs, min_freq=2):
    from collections import Counter
    c = Counter(t for ev in seqs for t in ev)
    vocab = [UNK] + sorted(t for t, n in c.items() if n >= min_freq)
    return {t: i for i, t in enumerate(vocab)}, c


def encode(seqs, stoi):
    ids = [stoi.get(t, 0) for ev in seqs for t in ev]
    return torch.tensor(ids, dtype=torch.long)


def batches(data, starts, batch_size=BATCH):
    """批量产出 (B, CTX) 训练对。修复 v1 batch=1 的效率问题。"""
    for i in range(0, len(starts) - batch_size + 1, batch_size):
        chunk_starts = starts[i:i + batch_size]
        xs = torch.stack([data[s:s + CTX] for s in chunk_starts])
        ys = torch.stack([data[s + 1:s + CTX + 1] for s in chunk_starts])
        yield xs.to(DEVICE), ys.to(DEVICE)


def main():
    path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
        "~/siming/models")
    os.makedirs(out_dir, exist_ok=True)

    seqs = load_stream(path)
    stoi, counter = build_vocab(seqs)
    ids = encode(seqs, stoi)
    split = int(len(ids) * 0.85)
    train_ids, val_ids = ids[:split], ids[split:]
    print(f"token 总数 {len(ids)}, 词表 {len(stoi)}, 训练 {len(train_ids)} / 验证 {len(val_ids)}, 设备 {DEVICE}")

    model = TinyGPT(len(stoi)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()

    def run_eval(data):
        model.eval()
        losses = []
        with torch.no_grad():
            starts = list(range(0, max(1, len(data) - CTX), CTX))
            for x, y in batches(data, starts, batch_size=min(BATCH, max(1, len(starts)))):
                losses.append(lossf(model(x).view(-1, len(stoi)), y.view(-1)).item())
        model.train()
        return sum(losses) / max(1, len(losses))

    t0 = time.time()
    for ep in range(EPOCHS):
        starts = torch.randperm(max(1, len(train_ids) - CTX - 1)).tolist()
        tot, nb = 0.0, 0
        for x, y in batches(train_ids, starts):
            loss = lossf(model(x).view(-1, len(stoi)), y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        vl = run_eval(val_ids)
        print(f"epoch {ep+1}/{EPOCHS} train_loss {tot/max(1,nb):.4f} val_loss {vl:.4f} "
              f"(val ppl {math.exp(min(vl, 20)):.1f}) {time.time()-t0:.0f}s")

    # 基线 NLL 分布（验证集逐 token 惊讶度）
    model.eval()
    nlls = []
    with torch.no_grad():
        starts = list(range(0, max(1, len(val_ids) - CTX), CTX // 2))
        for x, y in batches(val_ids, starts, batch_size=min(BATCH, max(1, len(starts)))):
            lp = torch.log_softmax(model(x), dim=-1)
            nll = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            nlls.extend(nll.view(-1).tolist())
    nlls_t = torch.tensor(nlls)
    qs = torch.quantile(nlls_t, torch.tensor([0.5, 0.95, 0.99, 0.995]))
    stats = {"mean": nlls_t.mean().item(),
             "p50": qs[0].item(), "p95": qs[1].item(),
             "p99": qs[2].item(), "p995": qs[3].item()}
    print("基线 NLL 分布:", {k: round(v, 3) for k, v in stats.items()})

    torch.save({"model": model.state_dict(), "stoi": stoi,
                "config": {"d_model": D_MODEL, "n_layer": N_LAYER,
                           "n_head": N_HEAD, "ctx": CTX},
                "baseline_nll": stats},
               os.path.join(out_dir, "prior.pt"))
    print(f"模型已保存 -> {out_dir}/prior.pt")


if __name__ == "__main__":
    main()
