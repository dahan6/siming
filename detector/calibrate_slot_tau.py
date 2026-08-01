#!/usr/bin/env python3
"""分维度阈值校准：按 token 槽位（ET/PROC/ARGV/PARENT/UID/DST/DT）分别标定 p995。
解决 max 池化下单个稀有 token 掩盖上下文信号的问题。

用法: calibrate_slot_tau.py <model_dir> <tokens.jsonl> [n_calib=20000]
输出: <model_dir>/slot_tau.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

from train_prior import CTX, DEVICE, TinyGPT


def main():
    model_dir, tokens_path = sys.argv[1], sys.argv[2]
    n_calib = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    ckpt = torch.load(os.path.join(model_dir, "prior.pt"), map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

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
            if ":" in tok:
                slot = tok.split(":")[0]
            elif tok.startswith("ARGV"):
                slot = "ARGV"
            elif tok.startswith("DT"):
                slot = "DT"
            else:
                slot = tok
            slot_nll[slot].append(v)

    slot_tau = {}
    print(f"校准集 {len(events)} 事件 | 全局 τ(p995)={ckpt['baseline_nll']['p995']:.3f}")
    print(f"{'槽位':<8}{'样本数':>8}{'p50':>8}{'p995':>8}{'τ(下限1.0)':>10}")
    for slot in ["ET", "PROC", "ARGV", "PC", "PARENT", "UID", "DST", "DT"]:
        vals = np.array(slot_nll[slot])
        p995 = float(np.quantile(vals, 0.995))
        # 下限 1.0：p995 塌成 0 的维度（PARENT/UID）需要抗噪下限，否则任何微小惊讶都开火
        slot_tau[slot] = max(p995, 1.0)
        print(f"{slot:<8}{len(vals):>8}{np.median(vals):>8.3f}{p995:>8.3f}{slot_tau[slot]:>10.3f}")

    out = os.path.join(model_dir, "slot_tau.json")
    json.dump({"n_calib": len(events), "slot_tau": slot_tau}, open(out, "w"),
              ensure_ascii=False, indent=2)
    print(f"已保存 -> {out}")


if __name__ == "__main__":
    main()
