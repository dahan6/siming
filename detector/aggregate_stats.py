#!/usr/bin/env python3
"""实验结果聚合：多 seed 均值 ± 标准差 + 95% 置信区间

汇总两类实验产物：
  1. deeplog_*.json  — DeepLog/TinyGPT 对比矩阵（3 架构 × 5 seed）
  2. adfa_*.json     — ADFA-LD 多 seed 矩阵（3 skip 变体 × 5 seed）

输出：
  - 终端表格（mean ± std, 95% CI）
  - results/aggregate_summary.json
"""
import glob
import json
import os
import sys

import numpy as np

DET = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(DET, "results")


def ci95(arr):
    """95% 置信区间（t 分布近似，小样本用 2.776 for n=5）"""
    n = len(arr)
    if n < 2:
        return 0.0
    t_crit = {2: 12.7, 3: 4.30, 4: 3.18, 5: 2.776, 6: 2.571}.get(n, 1.96)
    return float(t_crit * np.std(arr, ddof=1) / np.sqrt(n))


def agg(files, keys):
    """聚合一组结果文件的指定指标"""
    rows = []
    for f in sorted(files):
        with open(f) as fh:
            rows.append(json.load(fh))
    out = {"n_runs": len(rows)}
    for k in keys:
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        if not vals:
            continue
        vals = [v for v in vals if isinstance(v, (int, float))]
        if not vals:
            continue
        out[k] = {"mean": float(np.mean(vals)),
                  "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                  "ci95": ci95(vals),
                  "min": float(np.min(vals)), "max": float(np.max(vals)),
                  "n": len(vals)}
    return out, rows


def fmt(v, pct=True):
    if not isinstance(v, dict):
        return str(v)
    if pct:
        return f"{v['mean']*100:.2f}% ±{v['std']*100:.2f}% (CI95 ±{v['ci95']*100:.2f}%)"
    return f"{v['mean']:.3f} ±{v['std']:.3f} (CI95 ±{v['ci95']:.3f})"


def main():
    summary = {}

    # ── DeepLog 对比矩阵 ──
    print("=" * 72)
    print("  司命 vs DeepLog 对比矩阵（多 seed 聚合）")
    print("=" * 72)
    for arch in ["lstm_keys", "lstm_full", "tinygpt"]:
        files = glob.glob(os.path.join(RES, f"deeplog_{arch}_seed*.json"))
        if not files:
            print(f"\n[{arch}] 无结果")
            continue
        s, rows = agg(files, ["val_loss", "tau", "fpr_raw", "fpr_ewma"])
        # 攻击检出率（rove / bee_active，原始与 EWMA）
        for tag in ["rove", "bee_active"]:
            det_r = [r["attacks"][tag]["detect_raw"] for r in rows
                     if tag in r.get("attacks", {})]
            det_e = [r["attacks"][tag]["detect_ewma"] for r in rows
                     if tag in r.get("attacks", {})]
            if det_r:
                s[f"{tag}_detect_raw"] = {
                    "mean": float(np.mean(det_r)),
                    "std": float(np.std(det_r, ddof=1)) if len(det_r) > 1 else 0,
                    "ci95": ci95(det_r), "n": len(det_r)}
            if det_e:
                s[f"{tag}_detect_ewma"] = {
                    "mean": float(np.mean(det_e)),
                    "std": float(np.std(det_e, ddof=1)) if len(det_e) > 1 else 0,
                    "ci95": ci95(det_e), "n": len(det_e)}
        summary[f"deeplog_{arch}"] = s

        print(f"\n[{arch}] n={s['n_runs']} runs, "
              f"params={rows[0]['n_params']/1e6:.2f}M, vocab={rows[0]['vocab']}")
        print(f"  val_loss:  {fmt(s.get('val_loss'), pct=False)}")
        print(f"  FPR(原始): {fmt(s.get('fpr_raw'))}")
        print(f"  FPR(EWMA): {fmt(s.get('fpr_ewma'))}")
        print(f"  AAA rove 检出(原始):  {fmt(s.get('rove_detect_raw'))}")
        print(f"  AAA rove 检出(EWMA):  {fmt(s.get('rove_detect_ewma'))}")
        print(f"  AAA bee  检出(EWMA):  {fmt(s.get('bee_active_detect_ewma'))}")

    # ── ADFA 矩阵 ──
    print("\n" + "=" * 72)
    print("  ADFA-LD 外部基准（多 seed 聚合）")
    print("=" * 72)
    for skip in [0, 1, 2]:
        files = glob.glob(os.path.join(RES, f"adfa_skip{skip}_seed*.json"))
        if not files:
            print(f"\n[skip={skip}] 无结果")
            continue
        s, rows = agg(files, ["fpr_q", "fpr_f", "macro_q", "macro_f"])
        # 分类别检出率
        classes = sorted(rows[0].get("per_class", {}).keys())
        pc = {}
        for c in classes:
            vals = [r["per_class"][c]["fused_det"] for r in rows
                    if c in r.get("per_class", {})]
            pc[c] = {"mean": float(np.mean(vals)),
                     "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0,
                     "ci95": ci95(vals), "n": len(vals)}
        s["per_class_agg"] = pc
        summary[f"adfa_skip{skip}"] = s

        tag = {0: "bigram", 1: "skip-1", 2: "skip-2"}[skip]
        print(f"\n[{tag}] n={s['n_runs']} runs")
        print(f"  宏检出(q通道): {fmt(s.get('macro_q'))}")
        print(f"  宏检出(融合):  {fmt(s.get('macro_f'))}")
        print(f"  FPR(融合):     {fmt(s.get('fpr_f'))}")
        for c, v in pc.items():
            print(f"    {c:<20} {fmt(v)}")

    # ── ADFA 上的 LSTM（DeepLog 风格）参照 ──
    lstm_files = glob.glob(os.path.join(RES, "adfa_lstm_skip*_seed*.json"))
    if lstm_files:
        s, rows = agg(lstm_files, ["fpr_q", "fpr_f", "macro_q", "macro_f"])
        classes = sorted(rows[0].get("per_class", {}).keys())
        pc = {}
        for c in classes:
            vals = [r["per_class"][c]["fused_det"] for r in rows
                    if c in r.get("per_class", {})]
            pc[c] = {"mean": float(np.mean(vals)),
                     "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0,
                     "ci95": ci95(vals), "n": len(vals)}
        s["per_class_agg"] = pc
        summary["adfa_lstm_bigram"] = s
        print(f"\n[ADFA DeepLog-LSTM 参照] n={s['n_runs']} runs")
        print(f"  宏检出(q通道): {fmt(s.get('macro_q'))}")
        print(f"  宏检出(融合):  {fmt(s.get('macro_f'))}")
        print(f"  FPR(融合):     {fmt(s.get('fpr_f'))}")
        for c, v in pc.items():
            print(f"    {c:<20} {fmt(v)}")

    out = os.path.join(RES, "aggregate_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n聚合结果 -> {out}")


if __name__ == "__main__":
    main()
