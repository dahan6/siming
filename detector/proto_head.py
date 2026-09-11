#!/usr/bin/env python3
"""原型头：从标注 sequence 样本学习 ATT&CK 技术原型（对比学习）
流程：冻结 TinyGPT 出事件嵌入(128d, mean-pool) → 每技术 K 原型对比训练 →
      留一验证接住率 → 逐原型标定告警半径(命中样本距离 p99) → prototypes.jsonl

用法: proto_head.py <model_dir> [--k 3] [--epochs 200] [--out prototypes.jsonl]
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from train_prior import CTX, DEVICE, TinyGPT

DET = os.path.dirname(os.path.abspath(__file__))


def load_encoder(model_dir):
    ckpt = torch.load(os.path.join(model_dir, "prior.pt"), map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, stoi


def embed_sequence(model, stoi, tokens):
    """一条 token 序列 -> 128d 嵌入（最后一层隐状态 mean-pool）"""
    ids = [stoi.get(t, 0) for t in tokens][-CTX:]
    if len(ids) < 2:
        ids = ids * 2
    x = torch.tensor(ids, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        t = x.size(1)
        h = model.tok(x) + model.pos(torch.arange(t, device=DEVICE))
        h = model.blocks(h, mask=model.causal_mask[:t, :t])
        h = model.norm(h)
        return h.mean(dim=1).squeeze(0)


def load_samples():
    """patterns.jsonl 里 review=approved 的 sequence 条目 + 良性对照（留出的随机事件）"""
    seqs = []
    for line in open(os.path.join(DET, "patterns.jsonl")):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        e = json.loads(line)
        if e.get("type") == "sequence" and e.get("review") == "approved":
            seqs.append((e["technique"], e["sequence"]))
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(DET, "prototypes.jsonl"))
    a = ap.parse_args()

    model, stoi = load_encoder(a.model_dir)
    samples = load_samples()
    if not samples:
        raise SystemExit("没有 approved 的 sequence 样本，先跑采集+过审")

    techs = sorted({t for t, _ in samples})
    print(f"样本 {len(samples)} 条，技术 {len(techs)} 个: {techs}")

    # 嵌入（存盘复用）
    t0 = time.time()
    emb = [(t, embed_sequence(model, stoi, s)) for t, s in samples]
    print(f"嵌入完成 {time.time()-t0:.0f}s")

    # 初始化：每技术 K 原型 = 类内随机样本嵌入
    rng = np.random.default_rng(42)
    protos = {}
    for t in techs:
        cands = [e for tt, e in emb if tt == t]
        idx = rng.choice(len(cands), size=min(a.k, len(cands)), replace=False)
        protos[t] = nn.Parameter(torch.stack([cands[i] for i in idx]).clone())

    opt = torch.optim.Adam([p for p in protos.values()], lr=1e-2)
    E = torch.stack([e for _, e in emb])
    T = [t for t, _ in emb]

    for ep in range(a.epochs):
        total = 0.0
        for i, (t, e) in enumerate(emb):
            pos = min(((protos[t][k] - e).norm() for k in range(protos[t].shape[0])),
                      key=lambda d: d.item())
            negs = []
            for tt in techs:
                if tt == t:
                    continue
                negs.extend((protos[tt][k] - e).norm() for k in range(protos[tt].shape[0]))
            neg = torch.stack(negs).min() if negs else torch.tensor(a.margin * 2, device=DEVICE)
            loss = torch.relu(pos - neg + a.margin)
            if loss.item() > 0:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if (ep + 1) % 50 == 0:
            print(f"epoch {ep+1}/{a.epochs} loss={total:.3f}")

    # 留一验证：每条样本能否被本技术最近原型接住（距离 < 半径）
    # 半径标定：留一距离分布的 p99（每原型）
    P = {t: protos[t].detach() for t in techs}
    hits_all = {}
    radii = {}
    for t in techs:
        ds = [min((P[t][k] - e).norm().item() for k in range(P[t].shape[0]))
              for tt, e in emb if tt == t]
        hits_all[t] = ds
        radii[t] = [float(np.quantile(ds, 0.99))] * P[t].shape[0]

    n_hit = n_tot = 0
    for t in techs:
        r = radii[t][0]
        h = sum(1 for d in hits_all[t] if d <= r)
        n_hit += h; n_tot += len(hits_all[t])
        print(f"{t}: 留一接住 {h}/{len(hits_all[t])} 半径={r:.3f}")
    print(f"总接住率 {n_hit}/{n_tot} = {n_hit/n_tot:.1%}")

    meta = {"version": int(time.time()), "k": a.k, "margin": a.margin,
            "model_dir": a.model_dir, "techniques": {}}
    for t in techs:
        meta["techniques"][t] = {
            "prototypes": P[t].tolist(),
            "radii": radii[t],
            "n_samples": sum(1 for tt, _ in emb if tt == t),
        }
    with open(a.out, "w") as f:
        json.dump(meta, f)
    print(f"码本已保存 -> {a.out} ({os.path.getsize(a.out)} 字节)")


if __name__ == "__main__":
    main()
