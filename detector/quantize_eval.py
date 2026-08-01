#!/usr/bin/env python3
"""量化评估：先验模型 FP32 → INT8 动态量化，测掉点与体积
- val_loss 对比（留出段）
- 打分稳定性：同一批事件 fp32 vs int8 的 NLL Spearman 相关
- 体积对比
掉点 ≤2% 则 PTQ 直接可用，无需 QAT。

用法: quantize_eval.py <model_dir> <tokens.jsonl> [n_eval=5000]
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from train_prior import CTX, DEVICE, TinyGPT


def load(model_dir):
    ck = torch.load(os.path.join(model_dir, "prior.pt"), map_location="cpu", weights_only=False)
    stoi = ck["stoi"]
    m = TinyGPT(len(stoi))
    m.load_state_dict(ck["model"])
    m.eval()
    return m, stoi, ck


def val_loss(model, stoi, ids, vocab):
    lossf = nn.CrossEntropyLoss()
    losses = []
    with torch.no_grad():
        starts = list(range(0, len(ids) - CTX, CTX * 4))[:400]
        for i in range(0, len(starts), 64):
            cs = starts[i:i + 64]
            x = torch.stack([ids[s:s + CTX] for s in cs])
            y = torch.stack([ids[s + 1:s + CTX + 1] for s in cs])
            losses.append(lossf(model(x).view(-1, vocab), y.view(-1)).item())
    return float(np.mean(losses))


def main():
    model_dir, tokens_path = sys.argv[1], sys.argv[2]
    n_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    m32, stoi, ck = load(model_dir)
    vocab = len(stoi)

    events = [json.loads(l) for l in open(tokens_path)][-n_eval:]
    ids = torch.tensor([stoi.get(t, 0) for ev in events for t in ev["tokens"]], dtype=torch.long)
    split = int(len(ids) * 0.85)
    va = ids[split:]

    l32 = val_loss(m32, stoi, va, vocab)
    # 选择性量化（torchao Int8WeightOnly）：self-attention 保持 fp32，
    # FFN + 输出头 int8（体积大头；torch2.9 旧 eager 路径对裸 Linear 静默无效，须用 torchao）
    import copy
    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    m8 = copy.deepcopy(m32)
    for layer in m8.blocks.layers:
        quantize_(layer.linear1, Int8WeightOnlyConfig())
        quantize_(layer.linear2, Int8WeightOnlyConfig())
    quantize_(m8.head, Int8WeightOnlyConfig())
    l8 = val_loss(m8, stoi, va, vocab)

    # 打分稳定性：随机 200 个窗口的逐 token NLL 相关
    rng = np.random.default_rng(0)
    nll32, nll8 = [], []
    with torch.no_grad():
        for s in rng.choice(len(va) - CTX - 1, 200, replace=False):
            x = va[s:s + CTX].unsqueeze(0)
            y = va[s + 1:s + CTX + 1]
            lp32 = torch.log_softmax(m32(x), dim=-1)
            lp8 = torch.log_softmax(m8(x), dim=-1)
            nll32.extend((-lp32[0].gather(-1, y.unsqueeze(-1)).squeeze(-1)).tolist())
            nll8.extend((-lp8[0].gather(-1, y.unsqueeze(-1)).squeeze(-1)).tolist())
    from scipy.stats import spearmanr
    rho = spearmanr(nll32, nll8).statistic

    p32 = os.path.join(model_dir, "prior.pt")
    p8 = os.path.join(model_dir, "prior-int8.pt")
    # state_dict 可序列化量化张量（整模块 pickle 会撞 partial bug）
    torch.save({"model": m8.state_dict(), "stoi": stoi, "config": ck["config"],
                "baseline_nll": ck["baseline_nll"], "quant": "torchao-int8w-ffn-head"}, p8)
    s32, s8 = os.path.getsize(p32), os.path.getsize(p8)
    print(f"val_loss: fp32={l32:.4f} int8={l8:.4f}  变化 {(l8-l32)/l32:+.2%}")
    print(f"打分 Spearman ρ={rho:.4f}")
    print(f"体积: {s32/1e6:.2f}MB -> {s8/1e6:.2f}MB ({s8/s32:.0%})")
    verdict = "PTQ 可直接部署" if abs((l8 - l32) / l32) <= 0.02 and rho > 0.99 else "建议 QAT 微调"
    print(f"判定（掉点≤2% 且 ρ>0.99）: {verdict}")


if __name__ == "__main__":
    main()
