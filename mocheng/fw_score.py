#!/usr/bin/env python3
"""墨城防火墙 M2 配套: 网络流事件打分

加载 prior.pt，对 token 流逐事件计算惊讶度（事件内 token NLL 取 max），
EWMA 聚合为窗口分，按基线 p995 阈值判定异常。

用法: fw_score.py <tokens.jsonl> [--top N] [--model model/]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fw_paths import PRIOR_PT
from train_prior import TinyGPT, CTX, UNK

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 0.3  # EWMA 平滑系数


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--model", default=PRIOR_PT)
    a = ap.parse_args()

    ckpt = torch.load(a.model, map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tau = ckpt["baseline_nll"]["p995"]
    print(f"基线 p995 阈值 τ={tau:.3f}")

    events = [json.loads(l) for l in open(a.tokens)]
    results = []
    window = []
    ewma = 0.0
    for i, ev in enumerate(events):
        ids = [stoi.get(t, 0) for t in ev["tokens"]]
        window.extend(ids)
        window = window[-CTX:]
        with torch.no_grad():
            x = torch.tensor(window, device=DEVICE).unsqueeze(0)
            lp = torch.log_softmax(model(x), dim=-1)[0]
            n = len(ids)
            L = len(window)
            start = max(L - n, 1)
            tgt = torch.tensor(window[start:L], device=DEVICE)
            nll = -lp[start - 1:L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        s_ev = nll.max().item()
        ewma = ALPHA * s_ev + (1 - ALPHA) * ewma
        n_unk = sum(1 for t in ev["tokens"] if t not in stoi)
        label = "异常" if (s_ev > tau or n_unk > 0) else "正常"
        results.append({"i": i, "ts": ev["ts"], "host": ev.get("host", ""),
                        "tokens": ev["tokens"], "s_ev": s_ev, "ewma": ewma,
                        "n_unk": n_unk, "label": label,
                        "raw": ev.get("raw", {})})

    out_path = a.tokens.replace(".jsonl", "") + ".labeled.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_abn = sum(1 for r in results if r["label"] == "异常")
    print(f"\n== 全量标注结果: 共 {len(results)} 事件 ==")
    print(f"正常: {len(results) - n_abn}  异常: {n_abn}  (异常率 {n_abn/max(1,len(results)):.1%})")
    print(f"明细 -> {out_path}")

    print(f"\n== 异常事件清单（时间序）==")
    for r in sorted((r for r in results if r["label"] == "异常"), key=lambda r: r["i"])[:a.top]:
        unk = f" [UNK×{r['n_unk']}]" if r["n_unk"] else ""
        thr = " 超τ" if r["s_ev"] > tau else ""
        raw = r.get("raw", {})
        ip_info = f" {raw.get('src_ip','')}→{raw.get('dst_ip','')}:{raw.get('dst_port','')}"
        print(f"[异常]{thr} {r['s_ev']:7.2f} ewma={r['ewma']:5.2f} {r['ts']}{ip_info} "
              f"{' '.join(r['tokens'])}{unk}")
    if n_abn > a.top:
        print(f"... 其余 {n_abn - a.top} 条见明细文件")


if __name__ == "__main__":
    main()
