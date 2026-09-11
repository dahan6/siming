#!/usr/bin/env python3
"""OpTC 基准评测 v2（选择性打分，适配大文件）

攻击日 client-day 有 300 万+ 事件，逐事件 GPU 前向不可行。
策略：流式维护 token 窗口（廉价），但只对
  (a) GT 恶意 pid 的攻击事件（数百个）做前向打分
  (b) 良性事件每 N 个抽样一个做前向（FPR 样本）

协议：
  - 训练：OpTC 良性期 sep16 12 client 池化 → TinyGPT + τ(p995)
  - 检出率：GT 恶意 pid 事件超阈比例
  - FPR：良性抽样事件超阈比例

用法: optc_eval.py <tok_dir> <gt_dir> [--sample-every 50]
"""
import glob
import gzip
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CTX, D_MODEL, N_LAYER, N_HEAD = 128, 128, 4, 4
EPOCHS, BATCH, LR = 2, 256, 3e-4
ALPHA = 0.3
UNK = "<UNK>"


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, D_MODEL)
        self.pos = nn.Embedding(CTX, D_MODEL)
        layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD,
                                           dim_feedforward=512, batch_first=True,
                                           norm_first=True)
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


def load_tokens(paths, limit_per_file=None):
    rows = []
    for p in paths:
        n = 0
        for line in open(p, errors="replace"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            n += 1
            if limit_per_file and n >= limit_per_file:
                break
    return rows


def build_vocab(seqs, min_freq=3):
    c = Counter(t for r in seqs for t in r["tokens"])
    vocab = [UNK] + sorted(t for t, n in c.items() if n >= min_freq)
    return {t: i for i, t in enumerate(vocab)}


def encode(seqs, stoi):
    return torch.tensor([stoi.get(t, 0) for r in seqs for t in r["tokens"]],
                        dtype=torch.long)


def batches(data, starts, bs=BATCH):
    for i in range(0, len(starts) - bs + 1, bs):
        cs = starts[i:i + bs]
        yield (torch.stack([data[s:s + CTX] for s in cs]).to(DEVICE),
               torch.stack([data[s + 1:s + CTX + 1] for s in cs]).to(DEVICE))


def train_prior(seqs, seed=42):
    torch.manual_seed(seed)
    stoi = build_vocab(seqs)
    ids = encode(seqs, stoi)
    split = int(len(ids) * 0.9)
    tr, va = ids[:split], ids[split:]
    model = TinyGPT(len(stoi)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        starts = torch.randperm(max(1, len(tr) - CTX - 1)).tolist()
        for x, y in batches(tr, starts):
            loss = lossf(model(x).view(-1, len(stoi)), y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    vls = []
    with torch.no_grad():
        vs = list(range(0, max(1, len(va) - CTX), CTX))
        for x, y in batches(va, vs, min(BATCH, max(1, len(vs)))):
            vls.append(lossf(model(x).view(-1, len(stoi)), y.view(-1)).item())
    return model, stoi, float(np.mean(vls)) if vls else 0.0


def score_selective(model, stoi, rows, score_fn):
    """流式维护窗口，只对 score_fn(i,row) 为真的索引做前向。
    返回 {i: (s_ev, ewma, n_unk)}"""
    model.eval()
    out = {}
    window, ewma = [], 0.0
    for i, r in enumerate(rows):
        ids = [stoi.get(t, 0) for t in r["tokens"]]
        window = (window + ids)[-CTX:]
        want = score_fn(i, r)
        if not want:
            continue
        n, L = len(ids), len(window)
        if L < 2:
            continue
        start = max(L - n, 1)
        with torch.no_grad():
            x = torch.tensor(window, device=DEVICE).unsqueeze(0)
            lp = torch.log_softmax(model(x), dim=-1)[0]
            tgt = torch.tensor(window[start:L], device=DEVICE)
            nll = -lp[start - 1:L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        s = nll.max().item()
        ewma = ALPHA * s + (1 - ALPHA) * ewma
        out[i] = (s, ewma, sum(1 for t in r["tokens"] if t not in stoi))
    return out


def load_gt_pids(gt_path):
    pids = set()
    with gzip.open(gt_path, "rt", errors="replace") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = e.get("pid")
            if p is not None and p == p:
                pids.add(int(p))
    return pids


def main():
    tok_dir, gt_dir = sys.argv[1], sys.argv[2]
    sample_every = 50
    if "--sample-every" in sys.argv:
        sample_every = int(sys.argv[sys.argv.index("--sample-every") + 1])

    print("=== 训练（OpTC 良性期 sep16, 12 client 池化） ===")
    benign_paths = sorted(glob.glob(os.path.join(tok_dir, "benign", "sep16", "*.jsonl")))
    benign = load_tokens(benign_paths, limit_per_file=30000)
    print(f"良性事件: {len(benign)}")
    model_cache = os.path.join(tok_dir, "optc_prior.pt")
    if os.path.exists(model_cache):
        ckpt = torch.load(model_cache, map_location=DEVICE, weights_only=False)
        stoi = ckpt["stoi"]
        model = TinyGPT(len(stoi)).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()
        vl = ckpt.get("val_loss", 0.0)
        print(f"词表 {len(stoi)} (cached)")
    else:
        model, stoi, vl = train_prior(benign)
        torch.save({"model": model.state_dict(), "stoi": stoi, "val_loss": vl},
                   model_cache)
        print(f"模型已缓存 -> {model_cache}")
    ppl = math.exp(min(vl, 20))
    print(f"词表 {len(stoi)}, val_ppl={ppl:.2f}")

    calib = benign[-len(benign) // 8:]
    cs = score_selective(model, stoi, calib, lambda i, r: True)
    tau = float(np.quantile([v[0] for v in cs.values()], 0.995)) if cs else 1.0
    print(f"τ(p995)={tau:.3f}")

    # 良性留出 FPR
    bh = benign[-20000:]
    bs = score_selective(model, stoi, bh, lambda i, r: True)
    b_fpr = np.mean([1 if (v[1] > tau or v[2] > 0) else 0 for v in bs.values()])
    print(f"良性留出 FPR={b_fpr:.2%}")

    SCEN = [
        ("sc1", "sep23", {"SysClient0201": "gt_sc1_0201.json.gz",
                          "SysClient0402": "gt_sc1_0402.json.gz"}),
        ("sc2", "sep24", {"SysClient0501": "gt_sc2_0501.json.gz",
                          "SysClient0005": "gt_sc2_0005.json.gz"}),
        ("sc3", "sep25", {"SysClient0051": "gt_sc3_0051.json.gz",
                          "SysClient0351": "gt_sc3_0351.json.gz"}),
    ]

    def eval_scenario_stream(tok_path, pids, sample_every):
        """流式评测：不 hold 全量，边读边打分。返回 (n_atk, n_det, n_ben, n_ben_fp)"""
        model.eval()
        n_atk = n_det = n_ben = n_ben_fp = 0
        window = []
        with open(tok_path, errors="replace") as f:
            for i, line in enumerate(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                is_atk = r.get("pid", -1) in pids
                is_sample = (i % sample_every == 0)
                ids = [stoi.get(t, 0) for t in r["tokens"]]
                window = (window + ids)[-CTX:]
                if not (is_atk or is_sample):
                    continue
                n, L = len(ids), len(window)
                if L < 2:
                    continue
                start = max(L - n, 1)
                with torch.no_grad():
                    x = torch.tensor(window, device=DEVICE).unsqueeze(0)
                    lp = torch.log_softmax(model(x), dim=-1)[0]
                    tgt = torch.tensor(window[start:L], device=DEVICE)
                    nll = -lp[start - 1:L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                s = nll.max().item()
                u = sum(1 for t in r["tokens"] if t not in stoi)
                flagged = (s > tau) or (u > 0)
                if is_atk:
                    n_atk += 1
                    n_det += 1 if flagged else 0
                else:
                    n_ben += 1
                    n_ben_fp += 1 if flagged else 0
        return n_atk, n_det, n_ben, n_ben_fp

    report = {"val_ppl": ppl, "tau": tau, "benign_holdout_fpr": float(b_fpr)}
    print(f"\n{'场景':<6}{'client':<16}{'攻击':>6}{'检出':>6}{'检出率':>8}{'良性样本':>8}{'FPR':>7}")
    print("-" * 62)
    for sc, day, targets in SCEN:
        for host, gtf in targets.items():
            tok_path = os.path.join(tok_dir, sc, day, f"{host.lower()}.jsonl")
            gt_path = os.path.join(gt_dir, gtf)
            if not os.path.exists(tok_path) or not os.path.exists(gt_path):
                print(f"{sc:<6}{host:<16} 数据缺失")
                continue
            pids = load_gt_pids(gt_path)
            n_atk, n_det, n_ben, n_ben_fp = eval_scenario_stream(
                tok_path, pids, sample_every)
            det = n_det / n_atk if n_atk else 0.0
            fpr = n_ben_fp / n_ben if n_ben else 0.0
            report[f"{sc}_{host}"] = {"n_attack": n_atk, "n_detect": n_det,
                                      "detect_rate": det, "n_benign_sample": n_ben,
                                      "fpr": fpr}
            print(f"{sc:<6}{host:<16}{n_atk:>6}{n_det:>6}{det:>8.1%}{n_ben:>8}{fpr:>7.2%}")

    out = os.path.join(tok_dir, "optc_eval.json")
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
