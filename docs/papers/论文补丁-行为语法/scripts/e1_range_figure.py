#!/usr/bin/env python3
"""E1 靶场版 Figure 1：CV 失效边界 × 语法面暴露（中英双语）

图：双面板
  (a) 实测 CV vs σ，标注检测线 1.5 → CV 面在 σ≥1.25 失效
  (b) 语法面暴露率（单事件 + 聚合显著性）vs σ → 逃逸窗口被聚合通道补上

用法: e1_range_figure.py [--lang zh|en]
"""
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
sns.set_palette('colorblind')

LANG = "en" if "--lang" in sys.argv and "en" in sys.argv else "zh"
if LANG == "en":
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
else:
    matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

REP = os.path.expanduser("~/defense-lab/data/e1_range/e1_range_report.json")
FIG = os.path.expanduser("~/defense-lab/figures")
os.makedirs(FIG, exist_ok=True)

T = {
    "zh": {
        "a_title": "(a) 时序面：实测 CV vs 间隔形态 σ（检测线=1.5）",
        "a_ylab": "实测节拍 CV",
        "a_xlab": "LogNormal 间隔形态参数 σ",
        "a_line": "CV 检测线 1.5",
        "a_det": "CV 检出",
        "a_esc": "CV 逃逸",
        "b_title": "(b) 语法面：单事件暴露率 vs σ（良性基线 2.7%）",
        "b_ylab": "语法面暴露率 (%)",
        "b_xlab": "LogNormal 间隔形态参数 σ",
        "b_benign": "良性基线 2.7%",
        "b_sig": "窗口聚合 p<1e-8 ✓",
    },
    "en": {
        "a_title": "(a) Temporal face: measured cadence CV vs interval shape σ (line = 1.5)",
        "a_ylab": "Measured cadence CV",
        "a_xlab": "LogNormal interval shape σ",
        "a_line": "CV detection line 1.5",
        "a_det": "CV detected",
        "a_esc": "CV evaded",
        "b_title": "(b) Grammar face: per-event exposure vs σ (benign baseline 2.7%)",
        "b_ylab": "Grammar exposure rate (%)",
        "b_xlab": "LogNormal interval shape σ",
        "b_benign": "Benign baseline 2.7%",
        "b_sig": "windowed aggregate p<1e-8 ✓",
    },
}[LANG]


def main():
    r = json.load(open(REP))
    fpr = r["benign_fpr"] * 100
    order = ["s0.3", "s1.13", "s1.25", "s1.75"]
    sigmas = [r["phases"][t]["sigma"] for t in order]
    cvs = [r["phases"][t]["cv"] for t in order]
    cv_det = [r["phases"][t]["cv_detected"] for t in order]
    exposed = [r["phases"][t]["grammar_exposed"] * 100 for t in order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) CV vs σ
    colors = ["#2b8cbe" if d else "#d7301f" for d in cv_det]
    axes[0].bar(range(len(sigmas)), cvs, color=colors, alpha=0.85, width=0.6)
    axes[0].axhline(1.5, color="red", linestyle="--", linewidth=1.5,
                    label=T["a_line"])
    axes[0].set_xticks(range(len(sigmas)))
    axes[0].set_xticklabels([f"σ={s}" for s in sigmas])
    axes[0].set_ylabel(T["a_ylab"])
    axes[0].set_xlabel(T["a_xlab"])
    axes[0].set_title(T["a_title"], fontsize=10)
    for i, (cv, d) in enumerate(zip(cvs, cv_det)):
        axes[0].text(i, cv + 0.08, f"{cv:.2f}\n{T['a_det'] if d else T['a_esc']}",
                     ha="center", fontsize=8,
                     color="#2b8cbe" if d else "#d7301f")
    axes[0].set_ylim(0, max(cvs) * 1.35)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(alpha=0.3)

    # (b) grammar exposure vs σ
    axes[1].bar(range(len(sigmas)), exposed, color="#7b3294", alpha=0.8, width=0.6)
    axes[1].axhline(fpr, color="gray", linestyle="--", linewidth=1.5,
                    label=T["b_benign"])
    axes[1].set_xticks(range(len(sigmas)))
    axes[1].set_xticklabels([f"σ={s}" for s in sigmas])
    axes[1].set_ylabel(T["b_ylab"])
    axes[1].set_xlabel(T["b_xlab"])
    axes[1].set_title(T["b_title"], fontsize=10)
    for i, e in enumerate(exposed):
        axes[1].text(i, e + 0.2, f"{e:.1f}%", ha="center", fontsize=8)
    # 聚合显著性标注（逃逸的两档）
    for i, d in enumerate(cv_det):
        if not d:
            axes[1].text(i, exposed[i] * 0.5, T["b_sig"], ha="center",
                         fontsize=7.5, color="darkgreen")
    axes[1].set_ylim(0, max(exposed) * 1.35)
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    name = "fig_e1_cv_boundary" + ("_en" if LANG == "en" else "") + ".png"
    out = os.path.join(FIG, name)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"图 -> {out}")


if __name__ == "__main__":
    main()
