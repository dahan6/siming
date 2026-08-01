#!/usr/bin/env python3
"""快速 VM 通用 prior 训练（限 2000 batch，~7分钟 CPU）"""
import json, math, os, time
import torch, torch.nn as nn
from functools import partial
from collections import Counter
print = partial(print, flush=True)

D_MODEL, N_LAYER, N_HEAD, CTX = 128, 4, 4, 128
BATCH, LR = 128, 3e-4
MAX_BATCHES = 1000
DEVICE = 'cpu'

class TinyGPT(nn.Module):
    def __init__(self, vs):
        super().__init__()
        self.tok = nn.Embedding(vs, D_MODEL)
        self.pos = nn.Embedding(CTX, D_MODEL)
        layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=512, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=N_LAYER, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vs)
        self.register_buffer("causal_mask", torch.triu(torch.full((CTX, CTX), float("-inf")), diagonal=1))
    def forward(self, idx):
        t = idx.size(1)
        h = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        h = self.blocks(h, mask=self.causal_mask[:t, :t])
        return self.head(self.norm(h))

DET = os.path.dirname(os.path.abspath(__file__))
seqs = [json.loads(l)["tokens"] for l in open(os.path.join(DET, "data/vm_multi_train.jsonl"))]
c = Counter(t for ev in seqs for t in ev)
vocab = ["<UNK>"] + sorted(t for t, n in c.items() if n >= 2)
stoi = {t: i for i, t in enumerate(vocab)}
ids = torch.tensor([stoi.get(t, 0) for ev in seqs for t in ev], dtype=torch.long)
split = int(len(ids) * 0.85)
train_ids, val_ids = ids[:split], ids[split:]
print(f"tokens={len(ids)} vocab={len(stoi)} train={len(train_ids)} val={len(val_ids)}")

model = TinyGPT(len(stoi))
print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
opt = torch.optim.AdamW(model.parameters(), lr=LR)
lossf = nn.CrossEntropyLoss()

import random
random.seed(42)
starts = list(range(0, len(train_ids) - CTX - 1))
random.shuffle(starts)
starts = starts[:MAX_BATCHES * BATCH]

t0 = time.time()
tot, nb = 0.0, 0
for i in range(0, len(starts) - BATCH + 1, BATCH):
    chunk = starts[i:i+BATCH]
    xs = torch.stack([train_ids[s:s+CTX] for s in chunk])
    ys = torch.stack([train_ids[s+1:s+CTX+1] for s in chunk])
    loss = lossf(model(xs).view(-1, len(stoi)), ys.view(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    tot += loss.item(); nb += 1
    if nb % 200 == 0:
        print(f"  batch {nb}/{MAX_BATCHES} loss={tot/nb:.4f} {time.time()-t0:.0f}s")

# Eval
model.eval()
with torch.no_grad():
    vs = list(range(0, max(1, len(val_ids) - CTX), CTX))
    vl = 0.0; vn = 0
    for i in range(0, min(len(vs)-BATCH+1, 5000), BATCH):
        chunk = vs[i:i+BATCH]
        xs = torch.stack([val_ids[s:s+CTX] for s in chunk])
        ys = torch.stack([val_ids[s+1:s+CTX+1] for s in chunk])
        vl += lossf(model(xs).view(-1, len(stoi)), ys.view(-1)).item(); vn += 1
    vl /= max(1, vn)
print(f"val_loss={vl:.4f} ppl={math.exp(min(vl,20)):.1f}")

# NLL stats
nlls = []
with torch.no_grad():
    for i in range(0, min(len(vs)-BATCH+1, 5000), BATCH):
        chunk = vs[i:i+BATCH]
        xs = torch.stack([val_ids[s:s+CTX] for s in chunk])
        ys = torch.stack([val_ids[s+1:s+CTX+1] for s in chunk])
        lp = torch.log_softmax(model(xs), dim=-1)
        nll = -lp.gather(-1, ys.unsqueeze(-1)).squeeze(-1)
        nlls.extend(nll.view(-1).tolist())
nt = torch.tensor(nlls)
qs = torch.quantile(nt, torch.tensor([0.5, 0.95, 0.99]))
stats = {"mean": nt.mean().item(), "p50": qs[0].item(), "p95": qs[1].item(), "p99": qs[2].item()}
print(f"NLL stats: mean={stats['mean']:.3f} p50={stats['p50']:.3f} p95={stats['p95']:.3f} p99={stats['p99']:.3f}")

out_dir = os.path.join(DET, "model-vm-universal")
os.makedirs(out_dir, exist_ok=True)
torch.save({"model": model.state_dict(), "stoi": stoi,
            "config": {"d_model": D_MODEL, "n_layer": N_LAYER, "n_head": N_HEAD, "ctx": CTX},
            "baseline_nll": stats}, os.path.join(out_dir, "prior.pt"))
print(f"DONE -> {out_dir}/prior.pt ({os.path.getsize(os.path.join(out_dir, 'prior.pt'))} bytes)")
