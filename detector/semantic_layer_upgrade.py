#!/usr/bin/env python3
"""语义层复合升级：窗口级分类 + focal loss

升级内容：
1. 窗口级分类：不用单事件 embedding，用 16 事件窗口的序列 embedding
   - 一个 sudo 单独看像 privesc，但在 16 事件上下文里很正常
   - 窗口 embedding = 16 个事件 embedding 的 mean-pool + attention 加权
2. Focal loss：对难分样本（persist vs benign 边界）加大权重
   - FL(pt) = -α_t(1-pt)^γ * log(pt)
   - γ=2 时，conf=0.5 的样本 loss 权重是 conf=0.9 的 25 倍

用法:
  python3 semantic_layer_upgrade.py --train --eval
"""
import json, os, sys, random, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_prior import TinyGPT, CTX, D_MODEL
from train_semantic import ContrastiveProjection, ClassificationHead, LABELS

DET = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_LABELS = len(LABELS)
WINDOW_SIZE = 16


class FocalLoss(nn.Module):
    """Focal Loss: 对难分样本加大权重

    FL(pt) = -α_t * (1-pt)^γ * log(pt)
    当 γ=2 时，conf=0.5 的样本权重是 conf=0.9 的 25 倍
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 可选：每类的权重
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


class WindowClassifier(nn.Module):
    """窗口级分类器：16 事件窗口 → 6 类

    架构：
    1. 每个事件通过 backbone + proj → 128d
    2. 窗口内 16 个事件 embedding 做 self-attention
    3. mean-pool + max-pool 拼接 → 256d
    4. FC → 6 类
    """
    def __init__(self, d=D_MODEL, n=N_LABELS, window=WINDOW_SIZE):
        super().__init__()
        self.window = window

        # 窗口 self-attention（轻量）
        self.window_attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.window_norm = nn.LayerNorm(d)

        # 分类头
        self.fc1 = nn.Linear(d * 2, d)  # mean + max pool
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(d, n)

    def forward(self, window_embeddings):
        """window_embeddings: (B, W, D) — B 个窗口，每个 W 个事件的 embedding"""
        # Self-attention 让窗口内事件互相关注
        attn_out, _ = self.window_attn(window_embeddings, window_embeddings, window_embeddings)
        h = self.window_norm(window_embeddings + attn_out)

        # Dual pooling
        mean_pool = h.mean(dim=1)  # (B, D)
        max_pool = h.max(dim=1).values  # (B, D)

        combined = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 2D)
        x = F.relu(self.fc1(combined))
        x = self.drop(x)
        return self.fc2(x)


def build_window_dataset(events, stoi, model, proj, window=WINDOW_SIZE, stride=4):
    """把事件流切成窗口，每个窗口生成一条训练样本"""
    from train_semantic import embed_sequence

    # 预计算每个事件的 embedding
    event_embs = []
    event_procs = []
    for ev in events:
        tokens = ev["tokens"]
        emb = embed_sequence(model, stoi, tokens)
        event_embs.append(emb)
        proc = "unknown"
        for t in tokens:
            if t.startswith("PROC:"):
                proc = t.split(":", 1)[1]
                break
        event_procs.append(proc)

    # 切窗口
    windows = []
    for i in range(0, len(event_embs) - window, stride):
        window_embs = event_embs[i:i+window]
        window_procs = event_procs[i:i+window]
        windows.append({
            "embeddings": window_embs,
            "procs": window_procs,
            "label": events[min(i+window-1, len(events)-1)].get("label", "benign"),
        })

    return windows


def load_events_with_labels(path):
    """加载带 label 的事件数据"""
    events = []
    for line in open(path):
        e = json.loads(line)
        if "tokens" in e:
            if "label" not in e:
                e["label"] = "benign"
            events.append(e)
    return events


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()

    # 加载 backbone + proj
    sem_ckpt = torch.load(os.path.join(DET, "model-semantic-v5/full_model.pt"),
                          map_location=DEVICE, weights_only=False)
    stoi = sem_ckpt["stoi"]
    backbone = TinyGPT(len(stoi)).to(DEVICE)
    backbone.load_state_dict(sem_ckpt["model"])
    backbone.causal_mask = torch.triu(torch.full((CTX,CTX),float('-inf')),diagonal=1).to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    proj = ContrastiveProjection().to(DEVICE)
    proj.load_state_dict(sem_ckpt["proj"])
    proj.eval()
    for p in proj.parameters():
        p.requires_grad_(False)

    print(f"Backbone 词表: {len(stoi)}")

    if args.train:
        # 加载训练数据
        train_events = load_events_with_labels(os.path.join(DET, "data/classifier_train_v5.jsonl"))

        # 加攻击数据
        attack_path = os.path.join(DET, "data/synth_attacks_v4.jsonl")
        if os.path.exists(attack_path):
            for line in open(attack_path):
                e = json.loads(line)
                if e.get("label") != "benign":
                    train_events.append(e)

        print(f"训练事件: {len(train_events)}")

        # 预计算 embedding
        print("预计算事件 embedding...")
        from train_semantic import embed_sequence

        event_embs = []
        labels = []
        for ev in train_events:
            with torch.no_grad():
                emb = embed_sequence(backbone, stoi, ev["tokens"])
                projected = proj(emb.unsqueeze(0)).squeeze(0)
            event_embs.append(projected.cpu())
            labels.append(LABELS.index(ev["label"]))

        event_embs = torch.stack(event_embs)
        labels = torch.tensor(labels)
        print(f"Embedding: {event_embs.shape}, 分布: {dict(Counter(LABELS[l] for l in labels.tolist()).most_common())}")

        # 构建窗口训练集
        # 按标签分组，每组内构建窗口
        print("构建窗口训练集...")
        window_data = []
        window_labels = []

        by_label_events = defaultdict(list)
        for i, (emb, label) in enumerate(zip(event_embs, labels)):
            by_label_events[label.item()].append(emb)

        for label_idx, embs in by_label_events.items():
            # 在同类事件内切窗口
            if len(embs) < WINDOW_SIZE:
                # 不足一个窗口，补齐
                while len(embs) < WINDOW_SIZE:
                    embs.append(embs[-1])
                window_data.append(torch.stack(embs[:WINDOW_SIZE]))
                window_labels.append(label_idx)
            else:
                for i in range(0, len(embs) - WINDOW_SIZE + 1, max(1, WINDOW_SIZE // 2)):
                    window_data.append(torch.stack(embs[i:i+WINDOW_SIZE]))
                    window_labels.append(label_idx)

        window_data = torch.stack(window_data).to(DEVICE)
        window_labels = torch.tensor(window_labels).to(DEVICE)
        print(f"窗口训练集: {window_data.shape} 分布: {dict(Counter(LABELS[l] for l in window_labels.tolist()).most_common())}")

        # 训练窗口分类器 + focal loss
        clf = WindowClassifier().to(DEVICE)

        # 类别权重（平衡）
        class_counts = Counter(window_labels.tolist())
        total = len(window_labels)
        class_weights = torch.tensor([total / (N_LABELS * class_counts.get(i, 1)) for i in range(N_LABELS)],
                                     dtype=torch.float32).to(DEVICE)

        focal = FocalLoss(gamma=2.0, alpha=class_weights)
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)

        print(f"\n训练窗口分类器 + focal loss (γ=2.0)...")
        EPOCHS = 80
        BATCH = 32
        for ep in range(EPOCHS):
            perm = torch.randperm(len(window_data))
            tot = 0; correct = 0; nb = 0
            for i in range(0, len(perm) - BATCH, BATCH):
                idx = perm[i:i+BATCH]
                out = clf(window_data[idx])
                loss = focal(out, window_labels[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); correct += (out.argmax(-1)==window_labels[idx]).sum().item(); nb += 1
            if (ep+1) % 20 == 0:
                acc = correct/(nb*BATCH)*100
                print(f"  ep{ep+1}: loss={tot/nb:.4f} acc={acc:.1f}%")

        # 评估
        clf.eval()
        with torch.no_grad():
            preds = clf(window_data).argmax(-1)
        acc = (preds == window_labels).float().mean() * 100
        print(f"\n训练准确率: {acc:.1f}%")
        for i, label in enumerate(LABELS):
            m = window_labels == i
            if m.sum() > 0:
                cls_acc = (preds[m] == window_labels[m]).float().mean() * 100
                print(f"  {label:10s}: {cls_acc:.1f}% ({m.sum().item()})")

        # 保存
        out_dir = os.path.join(DET, "model-window-cls")
        os.makedirs(out_dir, exist_ok=True)
        torch.save({
            "classifier": clf.state_dict(),
            "stoi": stoi,
            "labels": LABELS,
            "config": {"window": WINDOW_SIZE, "focal_gamma": 2.0},
        }, os.path.join(out_dir, "window_classifier.pt"))
        print(f"保存 → {out_dir}/window_classifier.pt")

    if args.eval:
        # 在真实 audit 数据上评估 FPR
        clf_path = os.path.join(DET, "model-window-cls/window_classifier.pt")
        if not os.path.exists(clf_path):
            print("请先 --train")
            return

        clf_ckpt = torch.load(clf_path, map_location=DEVICE, weights_only=False)
        clf = WindowClassifier().to(DEVICE)
        clf.load_state_dict(clf_ckpt["classifier"])
        clf.eval()

        # 加载真实数据
        audit_events = load_events_with_labels(os.path.join(DET, "data/audit_all.jsonl"))
        for ev in audit_events:
            ev["label"] = "benign"

        print(f"\n=== 真实 audit FPR ({len(audit_events)} 事件) ===")
        from train_semantic import embed_sequence

        # 预计算 embedding
        event_embs = []
        for ev in audit_events:
            with torch.no_grad():
                emb = embed_sequence(backbone, stoi, ev["tokens"])
                projected = proj(emb.unsqueeze(0)).squeeze(0)
            event_embs.append(projected.cpu())

        # 切窗口评估
        correct = total = 0
        mis = Counter()

        for i in range(0, len(event_embs) - WINDOW_SIZE, 4):
            window_emb = torch.stack(event_embs[i:i+WINDOW_SIZE]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = clf(window_emb)
                probs = F.softmax(out, dim=-1)[0]
            pred = out.argmax(-1).item()
            conf = probs[pred].item()
            total += 1

            # 置信度阈值
            if conf < 0.7 or LABELS[pred] == "benign":
                correct += 1
            else:
                mis[LABELS[pred]] += 1

        fpr = (total - correct) / max(total, 1) * 100
        print(f"FPR: {fpr:.1f}% ({total-correct}/{total})")
        if mis:
            print(f"误分类: {dict(mis.most_common())}")


if __name__ == "__main__":
    main()
