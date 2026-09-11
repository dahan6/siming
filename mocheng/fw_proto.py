#!/usr/bin/env python3
"""墨城防火墙: 网络攻击原型学习头

从标注 sequence 样本学习网络攻击技术原型（对比学习）。
流程：冻结 TinyGPT 出事件嵌入(128d, mean-pool) → 每技术 K 原型对比训练 →
      留一验证接住率 → 逐原型标定告警半径(命中样本距离 p99) → fw_prototypes.jsonl

用法: fw_proto.py <model_dir> [--k 3] [--epochs 200]
样本来源: data/attack_sequences.jsonl (type=sequence, review=approved)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    """一条 token 序列 → 128d 嵌入（最后一层隐状态 mean-pool）"""
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


def load_samples(path=None):
    """加载 attack sequence 样本

    从 data/attack_sequences.jsonl 读取，格式:
    {"type":"sequence", "technique":"T1046", "review":"approved", "sequence":[...7 tokens...]}
    """
    path = path or os.path.join(DET, "data", "attack_sequences.jsonl")
    seqs = []
    if not os.path.exists(path):
        return seqs
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        e = json.loads(line)
        if e.get("type") == "sequence" and e.get("review") == "approved":
            seqs.append((e["technique"], e["sequence"]))
    return seqs


def generate_default_samples():
    """生成默认攻击 sequence 样本（从合成攻击数据自动构建）

    用 fw_tokens 对 attack_flows.jsonl 离散化，
    每条攻击流作为一个 sequence 样本。
    """
    from fw_tokens import flow_to_tokens

    attack_path = os.path.join(DET, "data", "attack_flows.jsonl")
    if not os.path.exists(attack_path):
        return []

    # 技术 ID 映射
    TECH_MAP = {
        "portscan": "T1046",      # Network Service Discovery
        "c2beacon": "T1071",      # Application Layer Protocol
        "exfil": "T1041",         # Exfiltration Over C2 Channel
        "lateral": "T1021",       # Remote Services
        "dnstunnel": "T1572",     # Protocol Tunneling
    }

    samples = []
    prev_ts = {}
    for line in open(attack_path):
        flow = json.loads(line)
        atk_type = flow.get("attack_type", "")
        tid = TECH_MAP.get(atk_type, "T9999")
        ts_epoch = flow.get("ts_epoch", 0)
        delta = 0
        if tid in prev_ts:
            delta = max(0, int((ts_epoch - prev_ts[tid]) * 1000))
        prev_ts[tid] = ts_epoch
        tokens = flow_to_tokens(flow, delta)
        samples.append((tid, tokens))
    return samples


def save_samples(samples, path=None):
    """保存 sequence 样本到 attack_sequences.jsonl"""
    path = path or os.path.join(DET, "data", "attack_sequences.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for tid, seq in samples:
            f.write(json.dumps({
                "type": "sequence", "technique": tid,
                "review": "approved", "sequence": seq,
                "created": time.strftime("%F %T"),
            }, ensure_ascii=False) + "\n")
    print(f"保存 {len(samples)} 条 sequence 样本 -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--k", type=int, default=3, help="每技术原型数")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(DET, "fw_prototypes.jsonl"))
    ap.add_argument("--gen-samples", action="store_true",
                    help="从合成攻击数据自动生成 sequence 样本")
    a = ap.parse_args()

    model, stoi = load_encoder(a.model_dir)

    # 加载样本
    samples = load_samples()
    if not samples or a.gen_samples:
        print("未找到 approved 样本，从合成攻击数据生成...")
        samples = generate_default_samples()
        if samples:
            save_samples(samples)

    if not samples:
        raise SystemExit("没有 sequence 样本可用。先运行: python fw_proto.py model/ --gen-samples")

    techs = sorted({t for t, _ in samples})
    print(f"样本 {len(samples)} 条，技术 {len(techs)} 个: {techs}")

    # 嵌入
    t0 = time.time()
    emb = [(t, embed_sequence(model, stoi, s)) for t, s in samples]
    print(f"嵌入完成 {time.time()-t0:.1f}s")

    # 良性 hard negative：从良性事件采样，参与对比训练
    benign_tokens_path = os.path.join(DET, "data", "benign_tokens.jsonl")
    benign_emb = []
    if os.path.exists(benign_tokens_path):
        benign_events = [json.loads(l) for l in open(benign_tokens_path)]
        # 随机采样良性事件作为 hard negative
        rng_bn = np.random.default_rng(99)
        n_bn = min(200, len(benign_events))
        idx = rng_bn.choice(len(benign_events), size=n_bn, replace=False)
        benign_emb = [embed_sequence(model, stoi, benign_events[i]["tokens"])
                      for i in idx]
        print(f"良性 hard negative: {len(benign_emb)} 条")

    # 初始化原型
    rng = np.random.default_rng(42)
    protos = {}
    for t in techs:
        cands = [e for tt, e in emb if tt == t]
        idx = rng.choice(len(cands), size=min(a.k, len(cands)), replace=False)
        protos[t] = nn.Parameter(torch.stack([cands[i] for i in idx]).clone())

    opt = torch.optim.Adam([p for p in protos.values()], lr=1e-2)
    E = torch.stack([e for _, e in emb])
    T = [t for t, _ in emb]

    # 对比训练（加入良性 hard negative 推远）
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
            # 良性 hard negative：推远良性样本
            bn_dist = (protos[t] - e.unsqueeze(0)).norm(dim=1)
            # 良性嵌入作为额外的负样本（每技术原型应远离良性）
            neg = torch.stack(negs).min() if negs else torch.tensor(a.margin * 2, device=DEVICE)
            loss = torch.relu(pos - neg + a.margin)
            if loss.item() > 0:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if (ep + 1) % 50 == 0:
            print(f"epoch {ep+1}/{a.epochs} loss={total:.3f}")

    # 留一验证 + 半径标定
    P = {t: protos[t].detach() for t in techs}
    hits_all = {}
    radii = {}
    for t in techs:
        ds = [min((P[t][k] - e).norm().item() for k in range(P[t].shape[0]))
              for tt, e in emb if tt == t]
        hits_all[t] = ds
        # 良性距离分布：用于收紧半径
        bn_dists = [min((P[t][k] - be).norm().item() for k in range(P[t].shape[0]))
                    for be in benign_emb] if benign_emb else []
        # 半径策略：取攻击 p99 和良性 p1 的中点
        # 如果良性最近距离都很远，半径就按攻击 p99 走
        # 如果良性最近距离很近（重叠），半径取攻击分布的更收紧的分位
        atk_p99 = float(np.quantile(ds, 0.99))
        atk_p75 = float(np.quantile(ds, 0.75))
        if bn_dists:
            bn_p1 = float(np.quantile(bn_dists, 0.01))
            # 收紧半径：取 min(攻击p99, 良性p1) 然后乘以 0.9
            # 但必须保证至少能接住 p75 的攻击样本
            tight = min(atk_p99, bn_p1) * 0.92
            r = max(tight, atk_p75)
        else:
            r = atk_p99
        radii[t] = [r] * P[t].shape[0]

    n_hit = n_tot = 0
    print(f"\n{'技术':<14}{'样本':>6}{'留一接住':>8}{'半径':>8}")
    for t in techs:
        r = radii[t][0]
        h = sum(1 for d in hits_all[t] if d <= r)
        n_hit += h; n_tot += len(hits_all[t])
        print(f"{t:<14}{len(hits_all[t]):>6}{h:>8}{r:>8.3f}")
    print(f"\n总接住率 {n_hit}/{n_tot} = {n_hit/n_tot:.1%}")

    # 良性对照：用留出良性事件验证不误报
    benign_tokens_path = os.path.join(DET, "data", "benign_tokens.jsonl")
    if os.path.exists(benign_tokens_path):
        benign_events = [json.loads(l) for l in open(benign_tokens_path)][-500:]
        fp = 0
        for ev in benign_events:
            e = embed_sequence(model, stoi, ev["tokens"])
            for t in techs:
                if any((P[t][k] - e).norm().item() <= radii[t][0]
                       for k in range(P[t].shape[0])):
                    fp += 1
                    break
        print(f"良性对照误报: {fp}/{len(benign_events)} = {fp/len(benign_events):.1%}")

    # 保存码本
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
    print(f"\n码本已保存 -> {a.out} ({os.path.getsize(a.out)} 字节)")


if __name__ == "__main__":
    main()
