#!/usr/bin/env python3
"""E3 异质工作负载基线分析

回答审稿人"良性同质"质疑：结论在负载多样性下是否成立？

流程：
  1. 各 VM token 化（3 角色 × ~2 VM）
  2. 逐角色：token 分布（证异质性）+ 训练先验 → val_ppl、留出 FPR
  3. LOVO（leave-one-VM-out）：7 折交叉，跨机泛化 FPR
  4. 跨角色：角色 A 训练 → 角色 B 测试，FPR 矩阵
  5. 逐角色模型：合成攻击检出率

用法: e3_analyze.py <data_dir> [--skip-train]
  data_dir 下需有 <vm>_tokens.jsonl（e3-dev-1_tokens.jsonl 等）
"""
import glob
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_prior import TinyGPT, CTX

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH, LR = 2, 256, 3e-4
ALPHA = 0.3
UNK = "<UNK>"

ROLE_OF = {"dev": ["e3-dev-1", "e3-dev-2", "e3-dev-3"],
           "web": ["e3-web-1", "e3-web-2"],
           "desktop": ["e3-desktop-1", "e3-desktop-2"]}
ALL_VMS = [v for vs in ROLE_OF.values() for v in vs]


def load_tokens(path):
    seqs = []
    for line in open(path):
        try:
            seqs.append(json.loads(line)["tokens"])
        except (json.JSONDecodeError, KeyError):
            continue
    return seqs


def build_vocab(seqs, min_freq=2):
    c = Counter(t for ev in seqs for t in ev)
    vocab = [UNK] + sorted(t for t, n in c.items() if n >= min_freq)
    return {t: i for i, t in enumerate(vocab)}


def encode(seqs, stoi):
    return torch.tensor([stoi.get(t, 0) for ev in seqs for t in ev], dtype=torch.long)


def batches(data, starts, bs=BATCH):
    for i in range(0, len(starts) - bs + 1, bs):
        cs = starts[i:i + bs]
        yield (torch.stack([data[s:s + CTX] for s in cs]).to(DEVICE),
               torch.stack([data[s + 1:s + CTX + 1] for s in cs]).to(DEVICE))


def train_prior_model(seqs, seed=42):
    torch.manual_seed(seed)
    stoi = build_vocab(seqs)
    ids = encode(seqs, stoi)
    split = int(len(ids) * 0.85)
    tr, va = ids[:split], ids[split:]
    model = TinyGPT(len(stoi)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        starts = torch.randperm(max(1, len(tr) - CTX - 1)).tolist()
        for x, y in batches(tr, starts):
            loss = lossf(model(x).view(-1, len(stoi)), y.view(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    # val ppl
    model.eval()
    vls = []
    with torch.no_grad():
        vs = list(range(0, max(1, len(va) - CTX), CTX))
        for x, y in batches(va, vs, min(BATCH, max(1, len(vs)))):
            vls.append(lossf(model(x).view(-1, len(stoi)), y.view(-1)).item())
    vl = float(np.mean(vls)) if vls else 0.0
    return model, stoi, vl


def score_seqs(model, stoi, seqs):
    """逐事件惊讶度（max NLL）+ EWMA + UNK"""
    model.eval()
    scores, ewmas, unks = [], [], []
    window, ewma = [], 0.0
    for ev in seqs:
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
        s = nll.max().item()
        ewma = ALPHA * s + (1 - ALPHA) * ewma
        scores.append(s)
        ewmas.append(ewma)
        unks.append(sum(1 for t in ev if t not in stoi))
    return np.array(scores), np.array(ewmas), np.array(unks)


def eval_pair(train_seqs, test_seqs, seed=42):
    """训练→校准→测 FPR + 合成攻击检出。返回 dict"""
    model, stoi, vl = train_prior_model(train_seqs, seed)
    # 校准 τ：训练集尾部
    calib = train_seqs[-len(train_seqs) // 8:]
    cs, _, _ = score_seqs(model, stoi, calib)
    tau = float(np.quantile(cs, 0.995)) if len(cs) else 1.0
    # 测试集 FPR
    ts, te, tu = score_seqs(model, stoi, test_seqs)
    fpr = float((te > tau).mean()) if len(te) else 0.0
    unk_rate = float((tu > 0).mean()) if len(tu) else 0.0
    # 合成攻击（固定用例，在测试上下文中评估）
    attacks = {
        "高端口外联": ["ET:CONN", "PROC:bash", "ARGV0", "PC:NONE", "PARENT:?", "UID:1000", "DST:EXT:HIGH", "DT3"],
        "伪装进程": ["ET:EXEC", "PROC:.kworker", "ARGV:N1-", "PC:NONE", "PARENT:bash", "UID:0", "DST:NONE", "DT1"],
        "base64载荷": ["ET:EXEC", "PROC:bash", "ARGV:N2B", "PC:NONE", "PARENT:bash", "UID:1000", "DST:NONE", "DT1"],
    }
    det = 0
    ctx = [t for ev in test_seqs[-4:] for t in ev]
    for name, atk in attacks.items():
        ids = [stoi.get(t, 0) for t in ctx + atk][-CTX:]
        n, L = len(atk), len(ids)
        with torch.no_grad():
            x = torch.tensor(ids, device=DEVICE).unsqueeze(0)
            lp = torch.log_softmax(model(x), dim=-1)[0]
            tgt = torch.tensor(ids[L - n:], device=DEVICE)
            nll = -lp[L - n - 1:L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        s = nll.max().item()
        n_unk = sum(1 for t in atk if t not in stoi)
        if s > tau or n_unk > 0:
            det += 1
    return {"val_ppl": float(math.exp(min(vl, 20))), "tau": tau,
            "fpr": fpr, "unk_rate": unk_rate,
            "attack_detect": det, "attack_total": len(attacks)}


def main():
    data_dir = sys.argv[1]
    tokens = {}
    for vm in ALL_VMS:
        p = os.path.join(data_dir, f"{vm}_tokens.jsonl")
        if os.path.exists(p):
            tokens[vm] = load_tokens(p)
    have = [v for v in ALL_VMS if v in tokens and len(tokens[v]) > 100]
    print(f"可用 VM: {have}")

    # 1. token 分布异质性
    print("\n=== 1. 逐角色 token 分布（证异质性） ===")
    for role, vms in ROLE_OF.items():
        pool = [t for v in vms if v in tokens for t in tokens[v]]
        procs = Counter(t for ev in pool for t in ev if t.startswith("PROC:"))
        ets = Counter(t for ev in pool for t in ev if t.startswith("ET:"))
        print(f"{role:<9} n={len(pool):<6} PROC top3={procs.most_common(3)}  ET={dict(ets)}")

    results = {}
    # 2. 逐角色：训练 + 留出 FPR + 检出
    print("\n=== 2. 逐角色：val_ppl / 留出 FPR / 攻击检出 ===")
    for role, vms in ROLE_OF.items():
        pool = [t for v in vms if v in tokens for t in tokens[v]]
        if len(pool) < 500:
            print(f"{role}: 数据不足 {len(pool)}")
            continue
        split = int(len(pool) * 0.8)
        r = eval_pair(pool[:split], pool[split:], seed=42)
        results[f"role_{role}"] = r
        print(f"{role:<9} val_ppl={r['val_ppl']:.2f} τ={r['tau']:.2f} "
              f"留出FPR={r['fpr']:.2%} UNK={r['unk_rate']:.2%} "
              f"攻击检出={r['attack_detect']}/{r['attack_total']}")

    # 3. LOVO（leave-one-VM-out）
    print("\n=== 3. LOVO 跨机泛化（7 折） ===")
    lovo_fprs = []
    for hold in have:
        train_pool = [t for v in have if v != hold for t in tokens[v]]
        r = eval_pair(train_pool, tokens[hold], seed=42)
        lovo_fprs.append(r["fpr"])
        results[f"lovo_{hold}"] = r
        print(f"留出 {hold:<14} FPR={r['fpr']:.2%} UNK={r['unk_rate']:.2%}")
    if lovo_fprs:
        results["lovo_mean"] = {"fpr": float(np.mean(lovo_fprs)),
                                "fpr_std": float(np.std(lovo_fprs))}
        print(f"LOVO 平均 FPR = {np.mean(lovo_fprs):.2%} ±{np.std(lovo_fprs):.2%}")

    # 4. 跨角色 FPR 矩阵
    print("\n=== 4. 跨角色 FPR（A 训 → B 测） ===")
    roles = [r for r in ROLE_OF if any(v in tokens for v in ROLE_OF[r])]
    matrix = {}
    print(f"{'训↓测→':<10}" + "".join(f"{r:>10}" for r in roles))
    for ra in roles:
        pool_a = [t for v in ROLE_OF[ra] if v in tokens for t in tokens[v]]
        row = []
        for rb in roles:
            pool_b = [t for v in ROLE_OF[rb] if v in tokens for t in tokens[v]]
            if not pool_a or not pool_b:
                row.append(float("nan")); continue
            r = eval_pair(pool_a[:int(len(pool_a)*0.8)], pool_b, seed=42)
            row.append(r["fpr"])
            matrix[f"{ra}_to_{rb}"] = r["fpr"]
        matrix.update({f"cross_{ra}_{rb}": row[j] for j, rb in enumerate(roles)})
        print(f"{ra:<10}" + "".join(f"{x:>9.1%}" for x in row))
    results["cross_matrix"] = matrix

    out = os.path.join(data_dir, "e3_analysis.json")
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
