#!/usr/bin/env python3
"""词表内（in-vocab）硬子集检出率分析

审稿风险点：rove/bee 轨迹含词表外事件，全量检出率被 UNK 机制抬高。
本脚本回答：只看"全部 token 都在良性词表内"的攻击事件，检出率是多少？
这是基线对比的诚实硬指标（对应论文 condition i 的近似）。

用法: invocab_analysis.py [results_dir]
输入: results/scores_{lstm_keys,lstm_full,tinygpt}.npz（含 *_unk 数组）
输出: results/invocab_analysis.json + 终端表格
"""
import json
import os
import sys

import numpy as np

DET = os.path.dirname(os.path.abspath(__file__))


def main():
    res_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DET, "results")
    report = {}
    print(f"{'架构':<12}{'轨迹':<12}{'全量检出':>10}{'词表内占比':>12}{'词表内检出':>12}")
    print("-" * 60)
    for arch in ["lstm_keys", "lstm_full", "tinygpt"]:
        path = os.path.join(res_dir, f"scores_{arch}.npz")
        if not os.path.exists(path):
            print(f"{arch:<12} 无数据（{path}）")
            continue
        d = np.load(path)
        tau = float(d["tau"])
        report[arch] = {"tau": tau}
        for tag in ["rove", "bee_active"]:
            key = f"{tag}_scores"
            if key not in d:
                continue
            s, u = d[f"{tag}_scores"], d[f"{tag}_unk"]
            all_det = float(((s > tau) | (u > 0)).mean())
            inv = u == 0
            inv_share = float(inv.mean())
            inv_det = float((s[inv] > tau).mean()) if inv.sum() > 0 else None
            report[arch][tag] = {
                "n": int(len(s)), "all_detect": all_det,
                "invocab_share": inv_share, "invocab_detect": inv_det,
            }
            inv_det_s = f"{inv_det:.1%}" if inv_det is not None else "n/a"
            print(f"{arch:<12}{tag:<12}{all_det:>10.1%}{inv_share:>12.1%}{inv_det_s:>12}")

    out = os.path.join(res_dir, "invocab_analysis.json")
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
