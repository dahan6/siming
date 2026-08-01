#!/usr/bin/env python3
"""τ 敏感性扫描：给定某轮模型目录，用已保存的留出 EWMA 分数和异常用例分数，
找同时满足 FPR≤1% 与 余量≥3x 的 τ 区间。判定迭代方向的决策工具。

用法: tau_sweep.py <model_dir>
"""
import glob
import json
import sys

import numpy as np

d = sys.argv[1]
ewma = np.load(f"{d}/holdout_ewma.npy")
lv = json.load(open(f"{d}/last_validation.json"))
anom = lv.get("anomalies", {})
anom_vals = list(anom.values()) if isinstance(anom, dict) else []
lo_anom = min(anom_vals) if anom_vals else None

print(f"模型目录: {d}")
print(f"留出点数 {len(ewma)}，异常用例分数 {anom}")
print(f"{'τ':>8} {'EWMA-FPR':>9} {'余量(min异常/τ)':>14}")
feasible = []
for tau in np.concatenate([np.arange(1.0, 12.01, 0.5)]):
    fpr = float((ewma > tau).mean() * 100)
    margin = (lo_anom / tau) if lo_anom else float("nan")
    ok = fpr <= 1.0 and margin >= 3.0
    if ok:
        feasible.append(tau)
    mark = "  <== 双达标" if ok else ""
    print(f"{tau:8.2f} {fpr:8.2f}% {margin:13.2f}x{mark}")

if feasible:
    print(f"\n可行 τ 区间: [{min(feasible):.2f}, {max(feasible):.2f}]，建议取中位附近")
else:
    print("\n无可行 τ：双判据交集为空 → 需继续压缩正常分布（加数据/减污染），调阈无用")
