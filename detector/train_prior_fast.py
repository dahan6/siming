"""Fast batched training for defender prior (TinyGPT).
Drop-in replacement for train_prior.py with proper batching.
"""
import json
import math
import os
import sys
import time
from collections import Counter

import torch
import torch.nn as nn

D_MODEL, N_LAYER, N_HEAD, CTX = 128, 4, 4, 128
EPOCHS, BATCH, LR = 8, 64, 3e-4
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
        self.blocks = nn.TransformerEncoder(layer, num_layers=N_LAYER)
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
    return seqs


def build_vocab(seqs, min_freq=2):
    c = Counter(t for ev in seqs for t in ev)
    vocab = [UNK] + sorted(t for t, n in c.items() if n >= min_freq)
    return {t: i for i, t in enumerate(vocab)}, c


def encode(seqs, stoi):
    ids = [stoi.get(t, 0) for ev in seqs for t in ev]
    return torch.tensor(ids, dtype=torch.long)


def make_batches(data, batch_size=BATCH):
    """Yield (B, CTX+1) batches by sliding window."""
    n = len(data)
    starts = list(range(0, max(1, n - CTX), CTX))
    # Shuffle starts
    perm = torch.randperm(len(starts)).tolist()
    for i in perm:
        s = starts[i]
        chunk = data[s:s + CTX + 1]
        if len(chunk) < CTX + 1:
            pad = torch.zeros(CTX + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, pad])
        yield chunk


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
            chunks = list(make_batches(data, BATCH))
            for i in range(0, len(chunks), BATCH):
                batch = torch.stack(chunks[i:i+BATCH]).to(DEVICE)
                x, y = batch[:, :-1], batch[:, 1:].contiguous().view(-1)
                logits = model(x).view(-1, len(stoi))
                losses.append(lossf(logits, y).item())
        model.train()
        return sum(losses) / max(1, len(losses))

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        chunks = list(make_batches(train_ids, BATCH))
        tot, nb = 0.0, 0
        for i in range(0, len(chunks), BATCH):
            batch = torch.stack(chunks[i:i+BATCH]).to(DEVICE)
            x, y = batch[:, :-1], batch[:, 1:].contiguous().view(-1)
            loss = lossf(model(x).view(-1, len(stoi)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        vl = run_eval(val_ids)
        print(f"epoch {ep+1}/{EPOCHS} train_loss {tot/max(1,nb):.4f} val_loss {vl:.4f} "
              f"(val ppl {math.exp(min(vl, 20)):.1f}) {time.time()-t0:.0f}s")

    # Baseline NLL distribution
    model.eval()
    nlls = []
    with torch.no_grad():
        chunks = list(make_batches(val_ids, BATCH))
        for i in range(0, len(chunks), BATCH):
            batch = torch.stack(chunks[i:i+BATCH]).to(DEVICE)
            x, y = batch[:, :-1], batch[:, 1:].contiguous().view(-1)
            lp = torch.log_softmax(model(x).view(-1, len(stoi)), dim=-1)
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
    print(f"模型已保存 -> {out_dir}/prior.pt ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
