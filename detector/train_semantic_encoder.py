#!/usr/bin/env python3
"""syscall 语义嵌入：mini-encoder 预训练 + 融合到 TinyGPT

阶段1: 文本描述 → mini-encoder → 128d 语义向量
阶段2: 语义向量注入 TinyGPT embedding 层（替代随机初始化）
阶段3: 端到端微调（可选）

mini-encoder：2 层 Transformer，用对比学习训练
  正样本：同类别进程（如 cat ↔ head，都是文件读取）
  负样本：不同类别（如 cat ↔ ss）
"""
import json, os, sys, random, re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from semantic_descriptions import PROC_DESCRIPTIONS, SLOT_DESCRIPTIONS, PARENT_DESCRIPTIONS, build_description_corpus
from train_prior import TinyGPT, CTX, D_MODEL

DET = os.path.dirname(os.path.abspath(__file__))


# ═══ Token 类别分组（用于对比学习正样本对）═══
TOKEN_CATEGORIES = {
    "file_read": {"PROC:cat", "PROC:head", "PROC:tail", "PROC:less", "PROC:more", "PROC:dd"},
    "file_search": {"PROC:find", "PROC:locate", "PROC:which", "PROC:whereis"},
    "net_recon": {"PROC:ss", "PROC:netstat", "PROC:lsof", "PROC:ip", "PROC:ifconfig", "PROC:arp", "PROC:nmap"},
    "proc_recon": {"PROC:ps", "PROC:top", "PROC:htop", "PROC:pgrep", "PROC:pidof"},
    "persist": {"PROC:crontab", "PROC:at", "PROC:atrm", "PROC:atq", "PROC:batch",
                "PROC:systemctl", "PROC:tee", "PROC:setsid", "PROC:nohup"},
    "privesc": {"PROC:sudo", "PROC:su", "PROC:pkexec", "PROC:chmod", "PROC:chown",
                "PROC:setcap", "PROC:getcap"},
    "net_tool": {"PROC:curl", "PROC:wget", "PROC:nc", "PROC:ncat",
                 "PROC:ssh", "PROC:scp", "PROC:sftp", "PROC:rsync"},
    "encode": {"PROC:base64", "PROC:openssl", "PROC:xxd"},
    "file_op": {"PROC:cp", "PROC:mv", "PROC:rm", "PROC:mkdir", "PROC:ln"},
    "sys_info": {"PROC:uname", "PROC:hostname", "PROC:whoami", "PROC:id", "PROC:who",
                 "PROC:env", "PROC:date", "PROC:uptime", "PROC:free", "PROC:df", "PROC:du",
                 "PROC:pwd", "PROC:ls", "PROC:echo", "PROC:wc"},
    "log": {"PROC:journalctl", "PROC:dmesg", "PROC:last"},
    "parser": {"PROC:grep", "PROC:awk", "PROC:sed", "PROC:sort", "PROC:cut",
               "PROC:tr", "PROC:readlink", "PROC:dirname", "PROC:md5sum"},
    "system": {"PROC:suricata", "PROC:snap", "PROC:snapctl", "PROC:apt", "PROC:dpkg",
               "PROC:getent", "PROC:modprobe", "PROC:unix_chkpwd", "PROC:systemd-executor"},
    "lang": {"PROC:python3", "PROC:python", "PROC:perl", "PROC:ruby", "PROC:node",
             "PROC:bash", "PROC:sh", "PROC:dash"},
    "vcs": {"PROC:git", "PROC:vim"},
    "kill": {"PROC:kill", "PROC:pkill"},
}


class TextEncoder(nn.Module):
    """轻量文本编码器：char-level CNN + projection"""
    def __init__(self, embed_dim=128, vocab_size=128):
        super().__init__()
        # Char embedding
        self.char_emb = nn.Embedding(vocab_size, 64)
        # 1D CNN over chars
        self.conv1 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Linear(128, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, char_ids):
        """char_ids: (B, L)"""
        x = self.char_emb(char_ids)  # (B, L, 64)
        x = x.transpose(1, 2)  # (B, 64, L)
        x = F.relu(self.conv1(x))  # (B, 128, L)
        x = F.relu(self.conv2(x))  # (B, 128, L)
        x = self.pool(x).squeeze(-1)  # (B, 128)
        x = self.proj(x)  # (B, embed_dim)
        return self.norm(x)


def text_to_chars(text, max_len=100, vocab_size=128):
    """文本 → char IDs"""
    ids = [min(ord(c), vocab_size - 1) for c in text[:max_len]]
    while len(ids) < max_len:
        ids.append(0)
    return ids


def build_semantic_pairs(corpus):
    """构建对比学习对：同类 = 正，异类 = 负"""
    # token → category
    token_to_cat = {}
    for cat, tokens in TOKEN_CATEGORIES.items():
        for t in tokens:
            token_to_cat[t] = cat

    # 添加未分类 token 到 "other"
    for token in corpus:
        if token not in token_to_cat:
            token_to_cat[token] = "other"

    # 正样本对：同类
    pos_pairs = []
    cat_tokens = defaultdict(list)
    for token, cat in token_to_cat.items():
        if token in corpus:
            cat_tokens[cat].append(token)

    for cat, tokens in cat_tokens.items():
        if len(tokens) < 2:
            continue
        for _ in range(50):
            a, b = random.sample(tokens, 2)
            pos_pairs.append((a, b))

    # 负样本对：不同类
    neg_pairs = []
    cats = list(cat_tokens.keys())
    for _ in range(len(pos_pairs)):
        ca, cb = random.sample(cats, 2)
        if cat_tokens[ca] and cat_tokens[cb]:
            a = random.choice(cat_tokens[ca])
            b = random.choice(cat_tokens[cb])
            neg_pairs.append((a, b))

    return pos_pairs, neg_pairs, token_to_cat


def train_semantic_encoder(corpus, epochs=100):
    """训练语义文本编码器"""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = TextEncoder(embed_dim=D_MODEL).to(DEVICE)

    # 准备数据
    pos_pairs, neg_pairs, token_to_cat = build_semantic_pairs(corpus)
    print(f"对比对: pos={len(pos_pairs)} neg={len(neg_pairs)}")

    # 预计算 char IDs
    char_cache = {}
    for token, desc in corpus.items():
        char_cache[token] = torch.tensor(text_to_chars(desc)).to(DEVICE)

    opt = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-5)
    temp = 0.1

    for ep in range(epochs):
        random.shuffle(pos_pairs)
        total_loss = 0
        n = 0

        for a, b in pos_pairs:
            if a not in char_cache or b not in char_cache:
                continue
            za = encoder(char_cache[a].unsqueeze(0))
            zb = encoder(char_cache[b].unsqueeze(0))

            # 正样本拉近
            pos_loss = 1 - F.cosine_similarity(za, zb).mean()

            # 随机负样本
            neg_idx = random.randint(0, len(neg_pairs) - 1)
            neg_a, neg_b = neg_pairs[neg_idx]
            if neg_a in char_cache and neg_b in char_cache:
                zn = encoder(char_cache[neg_b].unsqueeze(0))
                neg_loss = F.relu(F.cosine_similarity(za, zn).mean() - 0.0).clamp(min=0)
            else:
                neg_loss = torch.tensor(0.0, device=DEVICE)

            loss = pos_loss + neg_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n += 1

        if (ep + 1) % 20 == 0:
            print(f"  ep{ep+1}: loss={total_loss/max(n,1):.4f}")

    return encoder, char_cache


def extract_embeddings(encoder, char_cache, stoi):
    """提取所有 token 的语义 embedding，对齐到 prior 模型的词表"""
    encoder.eval()
    n_tokens = len(stoi)
    semantic_emb = torch.zeros(n_tokens, D_MODEL)

    matched = 0
    with torch.no_grad():
        for token, idx in stoi.items():
            if token in char_cache:
                emb = encoder(char_cache[token].unsqueeze(0)).squeeze(0).cpu()
                semantic_emb[idx] = emb
                matched += 1
            # else: 保持零向量（UNK 等）

    print(f"语义 embedding 覆盖: {matched}/{n_tokens} tokens")
    return semantic_emb


def main():
    corpus = build_description_corpus()
    print(f"描述库: {len(corpus)} tokens")

    # 阶段1: 训练语义编码器
    print("\n=== 阶段1: 训练语义编码器 ===")
    encoder, char_cache = train_semantic_encoder(corpus, epochs=100)

    # 阶段2: 提取 embedding
    print("\n=== 阶段2: 提取语义 embedding ===")

    # 加载 prior 模型获取词表
    model_dir = os.path.join(DET, "model-host-real-v2")
    if not os.path.exists(os.path.join(model_dir, "prior.pt")):
        model_dir = os.path.join(DET, "model-vm-universal")

    ckpt = torch.load(os.path.join(model_dir, "prior.pt"), map_location="cpu", weights_only=False)
    stoi = ckpt["stoi"]
    print(f"Prior 词表: {len(stoi)} tokens")

    semantic_emb = extract_embeddings(encoder, char_cache, stoi)

    # 阶段3: 验证语义聚类质量
    print("\n=== 阶段3: 验证语义聚类 ===")
    from collections import defaultdict
    cat_centroids = defaultdict(list)
    token_to_cat = {}
    for cat, tokens in TOKEN_CATEGORIES.items():
        for t in tokens:
            token_to_cat[t] = cat

    for token, idx in stoi.items():
        cat = token_to_cat.get(token, "other")
        cat_centroids[cat].append(semantic_emb[idx])

    # 计算类内平均距离 vs 类间平均距离
    for cat in ["file_read", "net_recon", "persist", "privesc", "net_tool"]:
        embs = cat_centroids.get(cat, [])
        if len(embs) < 2:
            continue
        embs = torch.stack(embs)
        centroid = embs.mean(0)
        intra_dist = (embs - centroid).norm(dim=1).mean().item()

        # 和 net_recon 的类间距离
        other = "file_read" if cat != "file_read" else "net_recon"
        other_embs = cat_centroids.get(other, [])
        if other_embs:
            other_centroid = torch.stack(other_embs).mean(0)
            inter_dist = (centroid - other_centroid).norm().item()
        else:
            inter_dist = 0

        ratio = inter_dist / max(intra_dist, 0.01)
        print(f"  {cat:12s}: 类内距离={intra_dist:.3f} 类间距离={inter_dist:.3f} 比值={ratio:.1f}")

    # 保存
    out_dir = os.path.join(DET, "model-semantic-embed")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "encoder": encoder.state_dict(),
        "semantic_embeddings": semantic_emb,
        "stoi": stoi,
        "corpus": corpus,
    }, os.path.join(out_dir, "semantic_encoder.pt"))
    print(f"\n保存 → {out_dir}/semantic_encoder.pt")

    # 也保存为可注入 prior 的格式
    torch.save(semantic_emb, os.path.join(out_dir, "token_embeddings.pt"))
    print(f"保存 → {out_dir}/token_embeddings.pt ({semantic_emb.shape})")


if __name__ == "__main__":
    main()
