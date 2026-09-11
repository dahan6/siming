#!/usr/bin/env python3
"""攻击行为自学习模式提取器

从标注/未标注的攻击序列中自动发现行为模式：
1. 用现有 prior 模型对每条序列做嵌入（128d mean-pool）
2. K-means 聚类发现自然行为组
3. 每个聚类提取代表性模式（高频 token 组合 + 序列模板）
4. 输出候选 pattern 供人工审核

流程：攻击数据 → 嵌入 → 聚类 → 提取 → 审核 → 入库

用法:
  # 从 patterns.jsonl 的 sequence 条目提取模式
  python3 auto_pattern.py <model_dir> --from-patterns

  # 从 atomic_attacks.jsonl 提取模式
  python3 auto_pattern.py <model_dir> --from-atomic atomic_attacks.jsonl

  # 从任意 token 序列文件提取
  python3 auto_pattern.py <model_dir> --from-jsonl data/attacks.jsonl
"""
import json
import os
import sys
import argparse
import time
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_prior import TinyGPT, CTX

DET = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(model_dir):
    ckpt = torch.load(os.path.join(model_dir, "prior.pt"),
                      map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, stoi


def embed_sequence(model, stoi, tokens):
    """一条 token 序列 → 128d 嵌入"""
    ids = [stoi.get(t, 0) for t in tokens][-CTX:]
    if len(ids) < 2:
        ids = ids * 2
    x = torch.tensor(ids, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        t = x.size(1)
        h = model.tok(x) + model.pos(torch.arange(t, device=DEVICE))
        h = model.blocks(h, mask=model.causal_mask[:t, :t])
        h = model.norm(h)
        return h.mean(dim=1).squeeze(0).cpu().numpy()


def load_from_patterns():
    """从 patterns.jsonl 加载 sequence 条目"""
    samples = []
    path = os.path.join(DET, "patterns.jsonl")
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        e = json.loads(line)
        if e.get("type") == "sequence":
            samples.append({
                "tokens": e["sequence"],
                "technique": e.get("technique", "?"),
                "source": e.get("source", "patterns"),
                "id": e.get("id", "?"),
            })
    return samples


def load_from_jsonl(path):
    """从任意 JSONL 加载 token 序列"""
    samples = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            tokens = e.get("tokens", [])
            if len(tokens) >= 4:
                samples.append({
                    "tokens": tokens,
                    "technique": e.get("technique", "?"),
                    "source": e.get("source", "jsonl"),
                    "id": e.get("id", "?"),
                })
        except Exception:
            continue
    return samples


def cluster_samples(embeddings, n_clusters=None, max_clusters=30):
    """K-means 聚类"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    emb = np.array(embeddings)
    n = len(emb)

    if n_clusters is None:
        # 自动选最优 K（轮廓系数）
        max_k = min(max_clusters, n - 1)
        if max_k < 2:
            return np.zeros(n, dtype=int), 1
        best_k, best_score = 2, -1
        for k in range(2, max_k + 1):
            try:
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(emb)
                score = silhouette_score(emb, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception:
                continue
        n_clusters = best_k
        print(f"  最优聚类数 K={n_clusters} (轮廓系数={best_score:.3f})")

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(emb)
    return labels, n_clusters


def extract_patterns(samples, labels, n_clusters):
    """从每个聚类提取代表性模式"""
    patterns = []

    for c in range(n_clusters):
        members = [i for i in range(len(samples)) if labels[i] == c]
        if len(members) < 2:
            continue

        # 聚类内统计
        techs = Counter(samples[i]["technique"] for i in members)
        procs = Counter()
        parents = Counter()
        etypes = Counter()
        argvs = Counter()
        dsts = Counter()

        for i in members:
            toks = samples[i]["tokens"]
            for t in toks:
                if t.startswith("PROC:"):
                    procs[t.split(":", 1)[1]] += 1
                elif t.startswith("PARENT:"):
                    parents[t.split(":", 1)[1]] += 1
                elif t.startswith("ET:"):
                    etypes[t.split(":", 1)[1]] += 1
                elif t.startswith("ARGV:"):
                    argvs[t.split(":", 1)[1]] += 1
                elif t.startswith("DST:"):
                    dsts[t.split(":", 1)[1]] += 1

        # 提取高频 token 组合（≥50% 出现率）
        threshold = len(members) * 0.5
        common_procs = [p for p, c in procs.most_common(5) if c >= threshold]
        common_parents = [p for p, c in parents.most_common(5) if c >= threshold]
        common_etypes = [e for e, c in etypes.most_common(3) if c >= threshold]

        # 共现 token 模式（2-gram）
        bigrams = Counter()
        for i in members:
            toks = samples[i]["tokens"]
            for j in range(len(toks) - 1):
                bigrams[(toks[j], toks[j+1])] += 1
        common_bigrams = [(bg, cnt) for bg, cnt in bigrams.most_common(10)
                         if cnt >= max(threshold, 2)]

        # 主技术
        main_tech = techs.most_common(1)[0][0] if techs else "?"

        pattern = {
            "cluster_id": c,
            "n_samples": len(members),
            "techniques": dict(techs.most_common(5)),
            "main_technique": main_tech,
            "common_procs": common_procs,
            "common_parents": common_parents,
            "common_etypes": common_etypes,
            "common_bigrams": [{"pair": list(bg), "count": cnt}
                              for bg, cnt in common_bigrams],
            "top_procs": dict(procs.most_common(10)),
            "top_parents": dict(parents.most_common(5)),
        }
        patterns.append(pattern)

    return patterns


def main():
    ap = argparse.ArgumentParser(description="攻击行为自学习模式提取")
    ap.add_argument("model_dir", help="模型目录")
    ap.add_argument("--from-patterns", action="store_true",
                    help="从 patterns.jsonl 加载")
    ap.add_argument("--from-atomic", metavar="PATH",
                    help="从 atomic_attacks.jsonl 加载")
    ap.add_argument("--from-jsonl", metavar="PATH",
                    help="从任意 JSONL 加载")
    ap.add_argument("--k", type=int, default=None,
                    help="聚类数（不指定则自动选）")
    ap.add_argument("--out", default=os.path.join(DET, "auto_patterns.json"))
    a = ap.parse_args()

    # 加载样本
    samples = []
    if a.from_patterns:
        samples.extend(load_from_patterns())
        print(f"从 patterns.jsonl 加载 {len(samples)} 条")
    if a.from_atomic:
        if os.path.exists(a.from_atomic):
            n = load_from_jsonl(a.from_atomic)
            samples.extend(n)
            print(f"从 {a.from_atomic} 加载 {len(n)} 条")
    if a.from_jsonl:
        n = load_from_jsonl(a.from_jsonl)
        samples.extend(n)
        print(f"从 {a.from_jsonl} 加载 {len(n)} 条")

    if not a.from_patterns and not a.from_atomic and not a.from_jsonl:
        # 默认从 patterns.jsonl
        samples = load_from_patterns()
        print(f"从 patterns.jsonl 加载 {len(samples)} 条")

    if len(samples) < 5:
        print(f"样本太少 ({len(samples)}), 至少需要 5 条")
        sys.exit(1)

    # 加载模型
    model, stoi = load_encoder(a.model_dir)
    print(f"模型: {a.model_dir} (词表 {len(stoi)})")

    # 嵌入
    print(f"\n=== 嵌入 {len(samples)} 条序列 ===")
    embeddings = []
    for i, s in enumerate(samples):
        emb = embed_sequence(model, stoi, s["tokens"])
        embeddings.append(emb)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(samples)}")

    # 聚类
    print(f"\n=== 聚类 ===")
    labels, n_clusters = cluster_samples(embeddings, a.k)

    # 统计
    for c in range(n_clusters):
        members = [i for i in range(len(samples)) if labels[i] == c]
        techs = Counter(samples[i]["technique"] for i in members)
        print(f"  聚类 {c}: {len(members)} 条, 技术={dict(techs.most_common(3))}")

    # 提取模式
    print(f"\n=== 模式提取 ===")
    patterns = extract_patterns(samples, labels, n_clusters)

    for p in patterns:
        print(f"\n--- 聚类 {p['cluster_id']} ({p['n_samples']} 条, "
              f"主技术={p['main_technique']}) ---")
        if p["common_procs"]:
            print(f"  高频进程: {p['common_procs']}")
        if p["common_parents"]:
            print(f"  高频父进程: {p['common_parents']}")
        if p["common_bigrams"]:
            print(f"  高频 2-gram:")
            for bg in p["common_bigrams"][:5]:
                print(f"    {bg['pair'][0]} → {bg['pair'][1]} ({bg['count']})")

    # 保存
    meta = {
        "generated_at": time.strftime("%F %T"),
        "n_samples": len(samples),
        "n_clusters": n_clusters,
        "model_dir": a.model_dir,
        "patterns": patterns,
    }
    with open(a.out, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n模式已保存 → {a.out}")

    # 生成候选 pattern 条目（供人工审核）
    cand_path = a.out.replace(".json", "_candidates.jsonl")
    with open(cand_path, "w") as f:
        for p in patterns:
            if not p["common_procs"]:
                continue
            proc_list = p["common_procs"][:3]
            parent_list = p["common_parents"][:3] if p["common_parents"] else None
            cand = {
                "id": f"AUTO-C{p['cluster_id']}-{p['main_technique']}",
                "type": "pattern",
                "technique": p["main_technique"],
                "name": f"自动发现模式 聚类{p['cluster_id']} "
                        f"({p['n_samples']}样本)",
                "severity": 3,
                "match": {"et": p["common_etypes"][0] if p["common_etypes"] else "EXEC",
                          "proc_in": proc_list},
                "precision": "category",
                "context_required": True,
                "source": "auto_cluster",
                "n_samples": p["n_samples"],
                "review": "pending",
                "notes": f"自动聚类提取，需人工审核。"
                         f"高频父进程: {parent_list}",
            }
            if parent_list:
                cand["match"]["parent_in"] = parent_list
            f.write(json.dumps(cand, ensure_ascii=False) + "\n")
    print(f"候选模式 → {cand_path} (待人工审核)")


if __name__ == "__main__":
    main()
