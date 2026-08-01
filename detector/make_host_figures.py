#!/usr/bin/env python3
"""正式轮图表：基于 host 模型验证结果出三张图（遵循 viz-chart skill 规范）
用法: make_host_figures.py <model_dir> [--tag TAG]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "AR PL UMing CN", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("colorblind")
# 必须在 seaborn 设置之后再指定字体，否则被重置为方块
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "AR PL UMing CN", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = os.path.expanduser("~/siming/figures")
os.makedirs(OUT, exist_ok=True)

# 合成异常分数（validate_host 运行时的打印值，由 last_validation.json 的 anomalies 字段提供）
ANOM_NAMES = {"A": "bash高端口外联", "B": "base64参数", "C": "伪装进程名", "E": "root高端口外联"}


def main():
    model_dir = sys.argv[1]
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "host"
    v = json.load(open(os.path.join(model_dir, "last_validation.json")))
    tau = v["tau"]
    s = np.load(os.path.join(model_dir, "holdout_scores.npy"))
    ewma = np.load(os.path.join(model_dir, "holdout_ewma.npy"))
    anomalies = v.get("anomalies", {})  # name -> score

    # fig-h1: 留出事件分数分布 + τ
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(s, bins=60, alpha=0.6, density=True, label="原始事件分")
    ax.hist(ewma, bins=60, alpha=0.6, density=True, label="EWMA 窗口分")
    ax.axvline(tau, color="red", ls="--", label=f"τ = {tau:.2f}")
    for name, sc in anomalies.items():
        ax.axvline(sc, color="orange", ls=":", alpha=0.8)
    ax.set_xlabel("惊讶度 (NLL)")
    ax.set_ylabel("密度")
    ax.set_title(f"留出基线分数分布（{tag}）")
    ax.legend()
    fig.savefig(f"{OUT}/fig-h1_{tag}_score_dist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # fig-h2: 阈值-误报率曲线 + 工作点 + 异常水位
    taus = np.linspace(0.1, max(25, s.max() * 1.05), 200)
    fpr = [(s > t).mean() for t in taus]
    fpr_e = [(ewma > t).mean() for t in taus]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(taus, np.array(fpr) * 100, label="FPR (原始)")
    ax.plot(taus, np.array(fpr_e) * 100, label="FPR (EWMA)")
    ax.axvline(tau, color="red", ls="--", label=f"工作点 τ={tau:.2f}")
    for name, sc in anomalies.items():
        ax.axvline(sc, color="orange", ls=":", alpha=0.8, label=f"异常 {name}" if name == list(anomalies)[0] else None)
    ax.axhline(1.0, color="green", ls="-.", alpha=0.6, label="目标 FPR=1%")
    ax.set_xlabel("阈值 τ")
    ax.set_ylabel("留出误报率 (%)")
    ax.set_yscale("log")
    ax.set_title(f"阈值-误报率曲线（{tag}）")
    ax.legend(fontsize=8)
    fig.savefig(f"{OUT}/fig-h2_{tag}_threshold_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # fig-h3: 分数时序（事件序）
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s, lw=0.5, alpha=0.5, label="原始事件分")
    ax.plot(ewma, lw=1.0, label="EWMA 窗口分")
    ax.axhline(tau, color="red", ls="--", label=f"τ = {tau:.2f}")
    ax.set_xlabel("留出事件序号")
    ax.set_ylabel("惊讶度 (NLL)")
    ax.set_title(f"留出期分数走势（{tag}）")
    ax.legend()
    fig.savefig(f"{OUT}/fig-h3_{tag}_timeseries.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"3 张图已生成 -> {OUT}/fig-h{{1,2,3}}_{tag}_*.png")


if __name__ == "__main__":
    main()
