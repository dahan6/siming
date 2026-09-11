#!/usr/bin/env python3
"""墨城防火墙: 分维度阈值校准

按 token 槽位（PROTO/DIR/SRCNET/DSTNET/PORTCLS/SIZECLS/DT）分别标定 p995，
叠加分层 min_tau 保底。解决小样本下 p995 过低导致正常流量全报的问题。

保底策略（类似司命 p99.5 + min_tau 方案）:
  - 全局保底: 不低于 baseline p995（保证槽位阈值不弱于全局阈值）
  - 上下文槽位 (DSTNET/DT/PORTCLS): 额外 +0.5（这些触发 P2，需更严格）
  - 绝对下限: 3.0（≈ -ln(5%)，低于此说明模型对该 token 仍有 >5% 置信度）

用法: fw_calibrate.py <model_dir> <tokens.jsonl> [n_calib=20000]
输出: <model_dir>/slot_tau.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_prior import CTX, DEVICE, TinyGPT

# 墨城槽位（对应 7-token schema）
SLOTS = ["PROTO", "DIR", "SRCNET", "DSTNET", "PORTCLS", "SIZECLS", "DT"]

# 槽位分类
CTX_SLOTS = {"DSTNET", "DT", "PORTCLS"}
RARE_SLOTS = {"PROTO", "SRCNET", "SIZECLS", "DIR"}

# 绝对下限: ≈ -ln(0.05) = 2.996，模型置信度 >5% 的 token 不应告警
ABSOLUTE_MIN_TAU = 3.0
# 上下文槽位额外提升（触发 P2，需更严格）
CTX_BONUS = 0.5


def slot_of(tok):
    """token → 槽位名"""
    if ":" in tok:
        return tok.split(":")[0]
    if tok.startswith("DT"):
        return "DT"
    return tok


def compute_min_tau(slot, global_p995):
    """计算槽位的保底阈值"""
    base = max(ABSOLUTE_MIN_TAU, global_p995)
    if slot in CTX_SLOTS:
        return base + CTX_BONUS
    return base


def main():
    model_dir, tokens_path = sys.argv[1], sys.argv[2]
    n_calib = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    ckpt = torch.load(os.path.join(model_dir, "prior.pt"), map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    global_p995 = ckpt["baseline_nll"]["p995"]

    itos = {i: k for k, i in stoi.items()}
    events = [json.loads(l) for l in open(tokens_path)][-n_calib:]
    slot_nll = defaultdict(list)
    window = []
    for ev in events:
        ids = [stoi.get(t, 0) for t in ev["tokens"]]
        window.extend(ids)
        window = window[-CTX:]
        L = len(window)
        if L < 2:
            continue
        x = torch.tensor(window, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            lp = torch.log_softmax(model(x), dim=-1)[0]
        n = len(ids)
        start = max(L - n, 1)
        tgt = window[start:L]
        tgt_t = torch.tensor(tgt, device=DEVICE)
        nll = -lp[start - 1:L - 1].gather(-1, tgt_t.unsqueeze(-1)).squeeze(-1)
        for tok_id, v in zip(tgt, nll.tolist()):
            tok = itos.get(tok_id, "<UNK>")
            slot = slot_of(tok)
            slot_nll[slot].append(v)

    slot_tau = {}
    print(f"校准集 {len(events)} 事件 | 全局 τ(p995)={global_p995:.3f}")
    print(f"保底策略: 绝对下限={ABSOLUTE_MIN_TAU}, 上下文槽位额外+{CTX_BONUS}")
    print(f"{'槽位':<12}{'样本数':>8}{'p50':>8}{'p995':>8}{'min_τ':>8}{'τ(最终)':>10}")
    for slot in SLOTS:
        vals = np.array(slot_nll.get(slot, [0]))
        p995 = float(np.quantile(vals, 0.995)) if len(vals) > 1 else 0.0
        min_tau = compute_min_tau(slot, global_p995)
        slot_tau[slot] = max(p995, min_tau)
        med = float(np.median(vals)) if len(vals) > 1 else 0.0
        print(f"{slot:<12}{len(vals):>8}{med:>8.3f}{p995:>8.3f}{min_tau:>8.3f}{slot_tau[slot]:>10.3f}")

    out = os.path.join(model_dir, "slot_tau.json")
    json.dump({"n_calib": len(events), "slot_tau": slot_tau,
               "global_p995": global_p995,
               "min_tau_config": {"absolute": ABSOLUTE_MIN_TAU,
                                  "ctx_bonus": CTX_BONUS}},
              open(out, "w"), ensure_ascii=False, indent=2)
    print(f"已保存 -> {out}")


if __name__ == "__main__":
    main()
