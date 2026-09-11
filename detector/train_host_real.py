#!/usr/bin/env python3
"""在宿主机真实 procfs 数据上训练 prior 模型
输出到 model-host-real-v2/
"""
import json, math, os, time, sys
import torch, torch.nn as nn
from functools import partial
from collections import Counter
print = partial(print, flush=True)

D_MODEL, N_LAYER, N_HEAD, CTX = 128, 4, 4, 128
BATCH, LR, EPOCHS = 128, 3e-4, 10
MAX_BATCHES_PER_EPOCH = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UNK = "<UNK>"

DET = os.path.dirname(os.path.abspath(__file__))

torch.manual_seed(42)
import random
random.seed(42)
np_seed = 42


class TinyGPT(nn.Module):
    def __init__(self, vs):
        super().__init__()
        self.tok = nn.Embedding(vs, D_MODEL)
        self.pos = nn.Embedding(CTX, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=512,
            batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=N_LAYER, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vs)
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
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        tokens = e.get("tokens", e.get("event", []))
        # 确保 8 token
        if len(tokens) == 7:
            dst_idx = next((i for i, t in enumerate(tokens) if t.startswith("DST:")), 5)
            tokens.insert(dst_idx + 1, "PC:NONE")
        seqs.append(tokens)
    return seqs


def build_vocab(seqs, min_freq=1):
    """对宿主机真实数据用 min_freq=1（词表小，保留全部 token）。"""
    c = Counter(t for ev in seqs for t in ev)
    vocab = [UNK] + sorted(t for t, n in c.items() if n >= min_freq)
    return {t: i for i, t in enumerate(vocab)}, c


def encode(seqs, stoi):
    ids = [stoi.get(t, 0) for ev in seqs for t in ev]
    return torch.tensor(ids, dtype=torch.long)


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        DET, "data/host_real_benign.jsonl")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        DET, "model-host-real-v2")
    os.makedirs(out_dir, exist_ok=True)

    seqs = load_stream(in_path)
    print(f"加载 {len(seqs)} 事件 from {in_path}")

    stoi, counter = build_vocab(seqs)
    ids = encode(seqs, stoi)
    split = int(len(ids) * 0.85)
    train_ids, val_ids = ids[:split], ids[split:]
    print(f"token 总数 {len(ids)}, 词表 {len(stoi)}, "
          f"训练 {len(train_ids)} / 验证 {len(val_ids)}, 设备 {DEVICE}")

    model = TinyGPT(len(stoi)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()

    import random
    random.seed(42)

    def run_eval(data):
        model.eval()
        losses = []
        with torch.no_grad():
            starts = list(range(0, max(1, len(data) - CTX), CTX))
            bs = min(BATCH, max(1, len(starts)))
            for i in range(0, len(starts) - bs + 1, bs):
                chunk = starts[i:i+bs]
                xs = torch.stack([data[s:s+CTX] for s in chunk]).to(DEVICE)
                ys = torch.stack([data[s+1:s+CTX+1] for s in chunk]).to(DEVICE)
                losses.append(lossf(model(xs).view(-1, len(stoi)),
                                    ys.view(-1)).item())
        model.train()
        return sum(losses) / max(1, len(losses))

    t0 = time.time()
    for ep in range(EPOCHS):
        starts = list(range(0, max(1, len(train_ids) - CTX - 1)))
        random.shuffle(starts)
        starts = starts[:MAX_BATCHES_PER_EPOCH * BATCH]
        tot, nb = 0.0, 0
        for i in range(0, len(starts) - BATCH + 1, BATCH):
            chunk = starts[i:i+BATCH]
            xs = torch.stack([train_ids[s:s+CTX] for s in chunk]).to(DEVICE)
            ys = torch.stack([train_ids[s+1:s+CTX+1] for s in chunk]).to(DEVICE)
            loss = lossf(model(xs).view(-1, len(stoi)), ys.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
            if nb % 200 == 0:
                print(f"  epoch {ep+1} batch {nb} loss={tot/nb:.4f} {time.time()-t0:.0f}s")
        vl = run_eval(val_ids)
        print(f"epoch {ep+1}/{EPOCHS} train_loss {tot/max(1,nb):.4f} "
              f"val_loss {vl:.4f} (val ppl {math.exp(min(vl, 20)):.1f}) "
              f"{time.time()-t0:.0f}s")

    # NLL 分布
    model.eval()
    nlls = []
    with torch.no_grad():
        starts = list(range(0, max(1, len(val_ids) - CTX), CTX // 2))
        bs = min(BATCH, max(1, len(starts)))
        for i in range(0, len(starts) - bs + 1, bs):
            chunk = starts[i:i+bs]
            xs = torch.stack([val_ids[s:s+CTX] for s in chunk]).to(DEVICE)
            ys = torch.stack([val_ids[s+1:s+CTX+1] for s in chunk]).to(DEVICE)
            lp = torch.log_softmax(model(xs), dim=-1)
            nll = -lp.gather(-1, ys.unsqueeze(-1)).squeeze(-1)
            nlls.extend(nll.view(-1).tolist())
    nlls_t = torch.tensor(nlls) if nlls else torch.tensor([0.0])
    qs = torch.quantile(nlls_t, torch.tensor([0.5, 0.95, 0.99, 0.995]))
    stats = {
        "mean": nlls_t.mean().item(),
        "p50": qs[0].item(), "p95": qs[1].item(),
        "p99": qs[2].item(), "p995": qs[3].item(),
    }
    print("基线 NLL 分布:", {k: round(v, 3) for k, v in stats.items()})

    torch.save({
        "model": model.state_dict(),
        "stoi": stoi,
        "config": {"d_model": D_MODEL, "n_layer": N_LAYER,
                   "n_head": N_HEAD, "ctx": CTX},
        "baseline_nll": stats,
    }, os.path.join(out_dir, "prior.pt"))
    print(f"模型已保存 -> {out_dir}/prior.pt "
          f"({os.path.getsize(os.path.join(out_dir, 'prior.pt'))} bytes)")
    print(f"\n词表大小: {len(stoi)}")
    print(f"最终 val_loss: {vl:.4f}")


if __name__ == "__main__":
    main()
