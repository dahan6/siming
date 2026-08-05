#!/usr/bin/env python3
"""对比学习 + 分类头联合训练

阶段1: 对比学习预训练 — 让 embedding 空间按行为类别聚类
阶段2: 分类头微调 — 6 类行为意图判别

用法: python3 train_semantic.py [--epochs1 30] [--epochs2 50]
"""
import json, os, sys, random, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_prior import TinyGPT, CTX, DEVICE

DET = os.path.dirname(os.path.abspath(__file__))

D_MODEL = 128
N_LABELS = 6
LABELS = ["benign", "recon", "persist", "exfil", "privesc", "lateral"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}


class ContrastiveProjection(nn.Module):
    """128d → 128d 投影 + L2 归一化"""
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x):
        return F.normalize(self.bn(self.proj(x)), dim=-1)


class ClassificationHead(nn.Module):
    """128d → 6 类"""
    def __init__(self, d=D_MODEL, n=N_LABELS):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(d, n)

    def forward(self, x):
        return self.fc2(self.drop(F.relu(self.fc1(x))))


def embed_sequence(model, stoi, tokens):
    """序列 → 128d mean-pool embedding"""
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


def load_classifier_data():
    """加载分类头训练数据"""
    path = os.path.join(DET, "data", "classifier_train.jsonl")
    data = []
    for line in open(path):
        e = json.loads(line)
        data.append((e["tokens"], LABEL_TO_IDX[e["label"]]))
    return data


def load_contrastive_data():
    """加载对比学习对"""
    path = os.path.join(DET, "data", "contrastive_pairs.jsonl")
    pairs = []
    for line in open(path):
        e = json.loads(line)
        pairs.append(e)
    return pairs


def train_contrastive(model, stoi, pairs, epochs=30):
    """阶段1：对比学习预训练"""
    proj = ContrastiveProjection().to(DEVICE)
    # 只训练投影头 + 微调 embedding（backbone 冻结大部分）
    params = list(proj.parameters())
    # 解冻最后 1 层 + embedding（TransformerEncoder 的 layer 在 .layers 里）
    if hasattr(model.blocks, 'layers'):
        last_layer = model.blocks.layers[-1]
        params += list(last_layer.parameters())
    params += list(model.norm.parameters())
    params += list(model.tok.parameters())

    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-5)
    temp = 0.1  # InfoNCE 温度

    print(f"\n=== 阶段1: 对比学习 ({len(pairs)} 对, {epochs} epochs) ===")

    for ep in range(epochs):
        random.shuffle(pairs)
        total_loss = 0
        n_batches = 0

        # 按 batch 处理
        batch_size = 32
        for i in range(0, len(pairs) - batch_size, batch_size):
            batch = pairs[i:i+batch_size]
            pos_anchors = []
            pos_positives = []
            neg_anchors = []
            neg_negatives = []

            for p in batch:
                if p["type"] == "pos":
                    pos_anchors.append(p["anchor"])
                    pos_positives.append(p["positive"])
                else:
                    neg_anchors.append(p["anchor"])
                    neg_negatives.append(p.get("negative", []))

            if not pos_anchors:
                continue

            # 编码
            anchor_embs = torch.stack([embed_sequence(model, stoi, t) for t in pos_anchors])
            positive_embs = torch.stack([embed_sequence(model, stoi, t) for t in pos_positives])

            # 投影
            z_a = proj(anchor_embs)
            z_p = proj(positive_embs)

            # InfoNCE loss
            sim = torch.mm(z_a, z_p.t()) / temp
            labels = torch.arange(z_a.size(0), device=DEVICE)
            loss = F.cross_entropy(sim, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        if (ep + 1) % 5 == 0:
            print(f"  ep{ep+1}: loss={total_loss/max(n_batches,1):.4f}")

    return proj


def train_classifier(model, proj, stoi, data, epochs=50):
    """阶段2：分类头微调"""
    clf = ClassificationHead().to(DEVICE)

    # 冻结 backbone，只训练分类头
    for p in model.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    # 预计算所有 embedding
    print(f"\n=== 阶段2: 分类头 ({len(data)} 样本, {epochs} epochs) ===")
    print("  预计算 embedding...")
    embeddings = []
    labels = []
    for tokens, label in data:
        emb = embed_sequence(model, stoi, tokens)
        embeddings.append(emb.detach())
        labels.append(label)

    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels, device=DEVICE)

    # 如果有投影头，用它
    if proj is not None:
        with torch.no_grad():
            embeddings = proj(embeddings).detach()

    print(f"  embedding 形状: {embeddings.shape}")

    for ep in range(epochs):
        perm = torch.randperm(len(embeddings))
        total_loss = 0
        correct = 0
        n_batches = 0

        batch_size = 64
        for i in range(0, len(perm) - batch_size, batch_size):
            idx = perm[i:i+batch_size]
            x = embeddings[idx].to(DEVICE)
            y = labels[idx]

            out = clf(x)
            loss = lossf(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            correct += (out.argmax(-1) == y).sum().item()
            n_batches += 1

        if (ep + 1) % 10 == 0:
            acc = correct / (n_batches * batch_size) * 100
            print(f"  ep{ep+1}: loss={total_loss/max(n_batches,1):.4f} acc={acc:.1f}%")

    # 留一验证
    clf.eval()
    with torch.no_grad():
        preds = clf(embeddings.to(DEVICE)).argmax(-1)
    acc = (preds == labels).float().mean().item() * 100

    # 按类别报告
    print(f"\n  训练集准确率: {acc:.1f}%")
    for i, label in enumerate(LABELS):
        mask = labels == i
        if mask.sum() > 0:
            cls_acc = (preds[mask] == labels[mask]).float().mean().item() * 100
            print(f"    {label:10s}: {cls_acc:.1f}% ({mask.sum().item()} 样本)")

    return clf, acc


def main():
    # 加载 prior
    model_dir = os.path.join(DET, "model-host-real-v2")
    if not os.path.exists(os.path.join(model_dir, "prior.pt")):
        model_dir = os.path.join(DET, "model-vm-universal")

    ckpt = torch.load(os.path.join(model_dir, "prior.pt"),
                      map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    print(f"Prior 模型: {model_dir} (词表 {len(stoi)})")

    # 阶段1: 对比学习
    pairs = load_contrastive_data()
    proj = train_contrastive(model, stoi, pairs, epochs=30)

    # 阶段2: 分类头
    data = load_classifier_data()
    clf, acc = train_classifier(model, proj, stoi, data, epochs=50)

    # 保存
    out_dir = os.path.join(DET, "model-semantic")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "stoi": stoi,
        "prior_config": ckpt["config"],
        "proj": proj.state_dict(),
        "classifier": clf.state_dict(),
        "labels": LABELS,
        "train_acc": acc,
    }, os.path.join(out_dir, "semantic.pt"))
    print(f"\n保存 → {out_dir}/semantic.pt")

    # 也保存带 backbone 的完整模型
    torch.save({
        "model": model.state_dict(),
        "stoi": stoi,
        "config": ckpt["config"],
        "proj": proj.state_dict(),
        "classifier": clf.state_dict(),
        "labels": LABELS,
    }, os.path.join(out_dir, "full_model.pt"))
    print(f"保存 → {out_dir}/full_model.pt")


if __name__ == "__main__":
    main()
