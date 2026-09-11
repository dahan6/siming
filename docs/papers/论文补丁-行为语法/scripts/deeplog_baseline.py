#!/usr/bin/env python3
"""司命 vs DeepLog 外部基线对比实验

三种配置，同一数据、同一打分口径（事件级 max NLL + EWMA + 基线 p995 阈值）：
  DL-keys : LSTM 吃事件键序列（ET|PROC 复合键，经典 DeepLog 输入形态）
  DL-full : LSTM 吃完整 8-token 语法流（同司命表示，隔离架构差异）
  TinyGPT : 因果 Transformer 吃完整 8-token 语法流（司命先验）

指标：
  良性留出 FPR（原始 + EWMA）
  AAA 攻击轨迹检出率（按事件超阈比例）
  val_loss / val_ppl

用法:
  deeplog_baseline.py --arch {lstm_keys,lstm_full,tinygpt} --seed N --out results/dl_seed{N}_{arch}.json
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from functools import partial
print = partial(print, flush=True)

D_MODEL, CTX = 128, 128
EPOCHS, BATCH, LR = 2, 256, 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UNK = "<UNK>"
ALPHA = 0.3

DET = os.path.dirname(os.path.abspath(__file__))
BENIGN_PATH = os.environ.get("BENIGN_PATH",
                             os.path.join(DET, "data", "host_tokens_clean.jsonl"))
ATTACK_PATH = os.environ.get("ATTACK_PATH",
                             os.path.join(DET, "data", "rove_attacks.jsonl"))
ATTACK_PATH2 = os.environ.get("ATTACK_PATH2",
                              os.path.join(DET, "data", "bee_active.jsonl"))


# ── 模型 ──

class DeepLogLSTM(nn.Module):
    """DeepLog 风格 LSTM：embedding → 2 层 LSTM → 线性头"""

    def __init__(self, vocab_size, d_model=D_MODEL, n_layer=2):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layer,
                            batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        h, _ = self.lstm(self.tok(idx))
        return self.head(self.norm(h))


class TinyGPT(nn.Module):
    """司命先验（同 train_prior.py）"""

    def __init__(self, vocab_size):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, D_MODEL)
        self.pos = nn.Embedding(CTX, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=4, dim_feedforward=512,
            batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=4,
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


# ── 数据 ──

def key_of(tokens):
    """ET|PROC 复合键（DeepLog 经典输入形态）"""
    et = tokens[0] if tokens else "ET:?"
    proc = tokens[1] if len(tokens) > 1 else "PROC:?"
    return f"{et}|{proc}"


def load_benign(mode):
    """加载良性数据。mode: 'keys' 或 'full'"""
    seqs = []
    with open(BENIGN_PATH) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            toks = e.get("tokens", [])
            if mode == "keys":
                seqs.append([key_of(toks)])
            else:
                seqs.append(toks)
    return seqs


def load_attacks(mode, path):
    """加载攻击轨迹。返回 [(label, seq), ...]"""
    seqs = []
    if not os.path.exists(path):
        return seqs
    with open(path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            toks = e.get("tokens", [])
            if not toks:
                continue
            label = e.get("label", "attack")
            seqs.append((label, [key_of(toks)] if mode == "keys" else toks))
    return seqs


def build_vocab(seqs, min_freq=2):
    from collections import Counter
    c = Counter(t for ev in seqs for t in ev)
    vocab = [UNK] + sorted(t for t, n in c.items() if n >= min_freq)
    return {t: i for i, t in enumerate(vocab)}


def encode(seqs, stoi):
    ids = [stoi.get(t, 0) for ev in seqs for t in ev]
    return torch.tensor(ids, dtype=torch.long)


def batches(data, starts, batch_size=BATCH):
    for i in range(0, len(starts) - batch_size + 1, batch_size):
        cs = starts[i:i + batch_size]
        xs = torch.stack([data[s:s + CTX] for s in cs])
        ys = torch.stack([data[s + 1:s + CTX + 1] for s in cs])
        yield xs.to(DEVICE), ys.to(DEVICE)


# ── 训练 ──

def train_model(model, train_ids, val_ids, stoi):
    lossf = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    def eval_loss(data):
        model.eval()
        losses = []
        with torch.no_grad():
            starts = list(range(0, max(1, len(data) - CTX), CTX))
            bs = min(BATCH, max(1, len(starts)))
            for x, y in batches(data, starts, batch_size=bs):
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
        vl = eval_loss(val_ids)
        print(f"  epoch {ep+1}/{EPOCHS} train {tot/max(1,nb):.4f} "
              f"val {vl:.4f} (ppl {math.exp(min(vl, 20)):.1f}) {time.time()-t0:.0f}s")
    return vl


# ── 打分（与司命同口径）──

def score_events(model, stoi, seqs, max_events=None):
    """逐事件打分，返回 (scores, ewma_scores, unk_counts)"""
    model.eval()
    scores, ewmas, unks = [], [], []
    window, ewma = [], 0.0
    for i, ev in enumerate(seqs):
        if max_events and i >= max_events:
            break
        ids = [stoi.get(t, 0) for t in ev]
        window = (window + ids)[-CTX:]
        n, L = len(ids), len(window)
        if L < 2:
            continue
        start = max(L - n, 1)
        with torch.no_grad():
            x = torch.tensor(window, device=DEVICE).unsqueeze(0)
            lp = torch.log_softmax(model(x), dim=-1)[0]
            tgt = torch.tensor(window[start:L], device=DEVICE)
            nll = -lp[start - 1:L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        s_ev = nll.max().item()
        ewma = ALPHA * s_ev + (1 - ALPHA) * ewma
        scores.append(s_ev)
        ewmas.append(ewma)
        unks.append(sum(1 for t in ev if t not in stoi))
    return np.array(scores), np.array(ewmas), np.array(unks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["lstm_keys", "lstm_full", "tinygpt"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--benign-holdout", type=int, default=8000)
    ap.add_argument("--attack-max", type=int, default=30000)
    ap.add_argument("--benign-max", type=int, default=0,
                    help="良性训练数据子采样上限（0=全量）")
    ap.add_argument("--dump-scores", default=None,
                    help="保存分数数组到 npz（用于分布图）")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    mode = "keys" if a.arch == "lstm_keys" else "full"
    print(f"=== arch={a.arch} seed={a.seed} mode={mode} ===")

    # 数据
    seqs = load_benign(mode)
    if a.benign_max and len(seqs) > a.benign_max:
        seqs = seqs[:a.benign_max]
    print(f"良性事件: {len(seqs)}")
    stoi = build_vocab(seqs)
    ids = encode(seqs, stoi)
    split = int(len(ids) * 0.85)
    train_ids, val_ids = ids[:split], ids[split:]
    print(f"token {len(ids)}, 词表 {len(stoi)}, 训练 {len(train_ids)} / 验证 {len(val_ids)}")

    # 模型
    if a.arch == "tinygpt":
        model = TinyGPT(len(stoi)).to(DEVICE)
    else:
        model = DeepLogLSTM(len(stoi)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params/1e6:.2f}M")

    # 训练
    vl = train_model(model, train_ids, val_ids, stoi)

    # τ 用验证段前半校准，FPR 用验证段后半留出（两个窗口不重叠，避免构造性 0.5%）
    val_seqs = seqs[int(len(seqs) * 0.85):]
    half = len(val_seqs) // 2
    calib_seqs = val_seqs[:half]
    holdout_seqs = val_seqs[half:][:a.benign_holdout]

    base_scores, _, _ = score_events(model, stoi, calib_seqs)
    tau = float(np.quantile(base_scores, 0.995))
    print(f"τ(p995)={tau:.3f} (校准窗 {len(calib_seqs)} 事件)")

    bs, be, bu = score_events(model, stoi, holdout_seqs)
    fpr_raw = float((bs > tau).mean())
    fpr_ewma = float((be > tau).mean())
    unk_rate = float((bu > 0).mean())
    print(f"良性留出: FPR(原始)={fpr_raw:.2%} FPR(EWMA)={fpr_ewma:.2%} UNK率={unk_rate:.2%}")

    # 攻击检出
    attack_results = {}
    dumped = {}
    for tag, path in [("rove", ATTACK_PATH), ("bee_active", ATTACK_PATH2)]:
        atk = load_attacks(mode, path)
        if not atk:
            continue
        labels = sorted({l for l, _ in atk})
        seqs_atk = [s for _, s in atk]
        asc, aew, au = score_events(model, stoi, seqs_atk, max_events=a.attack_max)
        det = float(((asc > tau) | (au > 0)).mean())
        det_ewma = float(((aew > tau) | (au > 0)).mean())
        attack_results[tag] = {
            "n": len(asc), "labels": labels,
            "detect_raw": det, "detect_ewma": det_ewma,
            "score_p50": float(np.percentile(asc, 50)),
            "score_p95": float(np.percentile(asc, 95)),
        }
        dumped[f"{tag}_scores"] = asc
        dumped[f"{tag}_ewma"] = aew
        dumped[f"{tag}_unk"] = au
        print(f"攻击[{tag}]: n={len(asc)} 检出率(原始)={det:.2%} (EWMA)={det_ewma:.2%} "
              f"分数 p50={attack_results[tag]['score_p50']:.2f}")

    if a.dump_scores:
        np.savez(a.dump_scores,
                 benign_scores=bs, benign_ewma=be, benign_unk=bu, tau=tau, **dumped)
        print(f"分数数组 -> {a.dump_scores}")

    result = {
        "arch": a.arch, "seed": a.seed, "mode": mode,
        "n_params": n_params, "vocab": len(stoi),
        "val_loss": vl, "val_ppl": math.exp(min(vl, 20)),
        "tau": tau,
        "fpr_raw": fpr_raw, "fpr_ewma": fpr_ewma, "unk_rate": unk_rate,
        "attacks": attack_results,
    }
    out = a.out or os.path.join(DET, "..", "results",
                                f"deeplog_{a.arch}_seed{a.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果 -> {out}")


if __name__ == "__main__":
    main()
