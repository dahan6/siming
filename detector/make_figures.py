#!/usr/bin/env python3
"""生成验证报告三张图：基线 s_ev 分布、惊讶度时序、阈值-误报率曲线。
数据: data/tokens.labeled.jsonl (score_events.py 全量标注产物)
"""
import json
import os

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC',
                                          'Droid Sans Fallback', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style('whitegrid')
sns.set_palette('colorblind')
# 注意：sns.set_style 会重置字体 rcParams，字体必须在其后设置
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC',
                                          'Droid Sans Fallback', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
LABELED = os.path.join(BASE, "data", "tokens.labeled.jsonl")
FIGDIR = os.path.expanduser("~/siming/figures")
os.makedirs(FIGDIR, exist_ok=True)

TAU = 3.141
SYNTH = {"A-高端口外联": 19.39, "B-base64参数": 17.61, "C-陌生进程伪装": 9.71}

recs = [json.loads(l) for l in open(LABELED)]
s_all = np.array([r["s_ev"] for r in recs])
abn = np.array([r["label"] == "异常" for r in recs])
s_norm = s_all[~abn]
print(f"事件 {len(recs)}，正常 {(~abn).sum()}，异常 {abn.sum()}")

# ---------- fig1: 正常事件 s_ev 分布 ----------
p50, p95 = np.percentile(s_norm, [50, 95])
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(s_norm, bins=80, density=True, alpha=0.55, label="直方图")
xs = np.linspace(0, s_norm.max(), 400)
from scipy.stats import gaussian_kde
kde = gaussian_kde(s_norm)
ax.plot(xs, kde(xs), lw=1.8, label="KDE")
for v, name, c, ls in [(p50, f"p50={p50:.2f}", "C1", "--"),
                       (p95, f"p95={p95:.2f}", "C2", "--"),
                       (TAU, f"τ(p995)={TAU:.2f}", "red", "-")]:
    ax.axvline(v, color=c, ls=ls, lw=1.6, label=name)
ax.set_xlabel("事件惊讶度 s_ev (max token NLL)")
ax.set_ylabel("密度")
ax.set_title("基线正常事件惊讶度分布（全量标注后判正常的事件）")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig(os.path.join(FIGDIR, "fig1_baseline_nll_dist.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- fig2: 惊讶度时序 ----------
idx = np.arange(len(recs))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(idx, s_all, lw=0.4, color="C0", alpha=0.6, label="s_ev")
ax.scatter(idx[abn], s_all[abn], s=14, color="red", zorder=3,
           label=f"判异常事件 (n={abn.sum()})")
ax.axhline(TAU, color="red", ls="--", lw=1.2, label=f"τ={TAU:.2f}")
ax.set_xlabel("事件序号（时间序）")
ax.set_ylabel("事件惊讶度 s_ev")
ax.set_title("事件惊讶度走势与异常事件高亮")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig(os.path.join(FIGDIR, "fig2_score_timeseries.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- fig3: 阈值-误报率曲线 ----------
taus = np.linspace(0.5, 25, 200)
fpr = np.array([(s_all > t).mean() for t in taus])  # 全量数据近似真实流量误报率
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(taus, fpr * 100, lw=1.8, color="C0", label="误报率（labeled 全量近似）")
fpr_tau = (s_all > TAU).mean()
ax.scatter([TAU], [fpr_tau * 100], color="red", zorder=5)
ax.annotate(f"当前工作点 τ={TAU:.2f}\nFPR≈{fpr_tau*100:.2f}%",
            xy=(TAU, fpr_tau * 100), xytext=(TAU + 2.5, max(fpr * 100) * 0.55),
            arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=9)
for name, sc in SYNTH.items():
    ax.axvline(sc, ls=":", lw=1.4, alpha=0.85,
               label=f"{name} s_ev={sc:.2f}")
ax.set_xlabel("阈值 τ")
ax.set_ylabel("误报率 (%)")
ax.set_title("阈值扫描：误报率曲线与合成异常分数")
ax.set_yscale("log")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3, which="both")
fig.savefig(os.path.join(FIGDIR, "fig3_threshold_curve.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print("figs ->", FIGDIR)
print(f"FPR@τ={TAU}: {fpr_tau:.4%}")
for t in [1, 2, 3.141, 5, 10, 20, 25]:
    print(f"  τ={t:6.3f}  FPR={(s_all > t).mean():.4%}")
