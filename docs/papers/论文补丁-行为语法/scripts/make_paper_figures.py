#!/usr/bin/env python3
"""论文级图表生成（中英双语）

用法:
  make_paper_figures.py [results_dir] [--lang zh|en]
  中文输出 → figures/；英文输出 → figures/en/
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

DET = os.path.dirname(os.path.abspath(__file__))

# ── 双语标签表 ──
LANG = "zh"
T = {}


def setup_lang(lang):
    global LANG, T
    LANG = lang
    if lang == "en":
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        T.update({
            "arch_keys": "DeepLog-keys\n(LSTM, event keys)",
            "arch_full": "DeepLog-full\n(LSTM, 8-token)",
            "arch_tiny": "TinyGPT (ours)\n(Transformer, 8-token)",
            "det_rate": "AAA event detection rate (%)",
            "fpr": "Benign holdout FPR (%)",
            "f1a_title": "(a) AAA detection rate (5 seeds, mean ± 95% CI)",
            "f1b_title": "(b) Benign FPR (5 seeds, mean ± 95% CI)",
            "rove": "AAA traces (rove)",
            "bee": "AAA traces (bee)",
            "fpr_raw": "FPR (raw)",
            "fpr_ewma": "FPR (EWMA)",
            "adfa_title": "ADFA-LD per-class detection (threshold = benign q99, 5 seeds, mean ± 95% CI)",
            "adfa_macro": "{label} macro {m:.1f}±{ci:.1f}%",
            "adfa_fpr_note": "Measured FPR: ",
            "det_pct": "Detection rate (%)",
            "fig3_title": "Surprise score distributions: benign vs AAA (by architecture)",
            "benign": "Benign holdout",
            "aaa_rove": "AAA (rove)",
            "score_x": "Event surprise (max NLL)",
            "density": "Density",
            "fig4a_title": "(a) Inter-event interval distribution",
            "fig4b_title": "(b) Per-field distribution distance (lower = more benign-like)",
            "share_pct": "Share of events (%)",
            "dt_x": "Inter-event interval bucket (log ms)",
            "kl_y": "KL divergence vs benign",
            "benign_host": "Benign host",
            "aaa_rove_l": "AAA (rove)",
            "aaa_bee_l": "AAA (bee active)",
        })
    else:
        matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'AR PL UMing CN', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        T.update({
            "arch_keys": "DeepLog-keys\n(LSTM, 事件键)",
            "arch_full": "DeepLog-full\n(LSTM, 8-token)",
            "arch_tiny": "司命 TinyGPT\n(Transformer, 8-token)",
            "det_rate": "AAA 事件检出率 (%)",
            "fpr": "良性留出误报率 (%)",
            "f1a_title": "(a) AAA 攻击检出率（5 seeds, mean ± 95% CI）",
            "f1b_title": "(b) 良性误报率（5 seeds, mean ± 95% CI）",
            "rove": "AAA 轨迹 (rove)",
            "bee": "AAA 轨迹 (bee)",
            "fpr_raw": "FPR (原始)",
            "fpr_ewma": "FPR (EWMA)",
            "adfa_title": "ADFA-LD 分类别检出率（阈值按良性 q99 标定，5 seeds, mean ± 95% CI）",
            "adfa_macro": "{label} 宏平均 {m:.1f}±{ci:.1f}%",
            "adfa_fpr_note": "实测 FPR: ",
            "det_pct": "检出率 (%)",
            "fig3_title": "良性 vs AAA 攻击的惊讶度分布（按架构）",
            "benign": "良性留出",
            "aaa_rove": "AAA (rove)",
            "score_x": "事件惊讶度（max NLL）",
            "density": "密度",
            "fig4a_title": "(a) 事件间隔分布对比",
            "fig4b_title": "(b) 逐字段分布距离（越小越像良性）",
            "share_pct": "事件占比 (%)",
            "dt_x": "事件间隔桶（毫秒对数级）",
            "kl_y": "KL 散度 vs 良性分布",
            "benign_host": "良性主机",
            "aaa_rove_l": "AAA (rove)",
            "aaa_bee_l": "AAA (bee 活跃期)",
        })


def fig_dir():
    d = os.path.expanduser("~/defense-lab/figures") if LANG == "zh" \
        else os.path.expanduser("~/defense-lab/figures/en")
    os.makedirs(d, exist_ok=True)
    return d


def load_agg(results_dir):
    with open(os.path.join(results_dir, "aggregate_summary.json")) as f:
        return json.load(f)


def fig1_baseline_comparison(agg):
    archs = ["lstm_keys", "lstm_full", "tinygpt"]
    archs = [a for a in archs if f"deeplog_{a}" in agg]
    if not archs:
        print("图1: 无基线数据，跳过")
        return
    labels = {"lstm_keys": T["arch_keys"], "lstm_full": T["arch_full"],
              "tinygpt": T["arch_tiny"]}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(archs))
    w = 0.35
    for i, (tag, label) in enumerate([("rove_detect_ewma", T["rove"]),
                                      ("bee_active_detect_ewma", T["bee"])]):
        means = [agg[f"deeplog_{a}"].get(tag, {}).get("mean", 0) * 100 for a in archs]
        cis = [agg[f"deeplog_{a}"].get(tag, {}).get("ci95", 0) * 100 for a in archs]
        axes[0].bar(x + (i - 0.5) * w, means, w, yerr=cis, capsize=4,
                    label=label, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[a] for a in archs], fontsize=9)
    axes[0].set_ylabel(T["det_rate"])
    axes[0].set_ylim(0, 118)
    axes[0].set_title(T["f1a_title"])
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].grid(alpha=0.3)

    means = [agg[f"deeplog_{a}"].get("fpr_ewma", {}).get("mean", 0) * 100 for a in archs]
    cis = [agg[f"deeplog_{a}"].get("fpr_ewma", {}).get("ci95", 0) * 100 for a in archs]
    means_raw = [agg[f"deeplog_{a}"].get("fpr_raw", {}).get("mean", 0) * 100 for a in archs]
    axes[1].bar(x - w/2, means_raw, w, label=T["fpr_raw"], alpha=0.85)
    axes[1].bar(x + w/2, means, w, yerr=cis, capsize=4, label=T["fpr_ewma"], alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels[a] for a in archs], fontsize=9)
    axes[1].set_ylabel(T["fpr"])
    axes[1].set_title(T["f1b_title"])
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(fig_dir(), "fig_baseline_comparison.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"图1 -> {out}")


def fig2_adfa(agg):
    variants = [("adfa_skip0", "bigram"), ("adfa_skip1", "skip-1 bigram"),
                ("adfa_skip2", "skip-2 bigram")]
    variants = [(k, l) for k, l in variants if k in agg]
    if not variants:
        print("图2: 无 ADFA 数据，跳过")
        return
    classes = sorted(agg[variants[0][0]].get("per_class_agg", {}).keys())
    if not classes:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(classes))
    w = 0.25
    for i, (key, label) in enumerate(variants):
        pc = agg[key]["per_class_agg"]
        means = [pc.get(c, {}).get("mean", 0) * 100 for c in classes]
        cis = [pc.get(c, {}).get("ci95", 0) * 100 for c in classes]
        ax.bar(x + (i - 1) * w, means, w, yerr=cis, capsize=3, label=label, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") if LANG == "zh" else c.replace("_", "\n")
                        for c in classes], fontsize=9)
    ax.set_ylabel(T["det_pct"])
    ax.set_ylim(0, 105)
    ax.set_title(T["adfa_title"])
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    fpr_note = T["adfa_fpr_note"] + " / ".join(
        f"{agg[key].get('fpr_f', {}).get('mean', 0)*100:.1f}%" for key, _ in variants)
    # 宏平均虚线 + 标注（纵向充分错开：上方 +2 / 下方 -6 / 下方 -14）
    y_off = [2.0, -7.0, -16.0]
    for i, (key, label) in enumerate(variants):
        m = agg[key].get("macro_f", {}).get("mean", 0) * 100
        ci = agg[key].get("macro_f", {}).get("ci95", 0) * 100
        ax.axhline(m, linestyle="--", alpha=0.4,
                   color=sns.color_palette('colorblind')[i])
        ax.text(len(classes) - 0.5, m + y_off[i],
                T["adfa_macro"].format(label=label, m=m, ci=ci),
                fontsize=7, ha="right", va="center",
                color=sns.color_palette('colorblind')[i])
    ax.text(0.02, 0.02, fpr_note, transform=ax.transAxes, fontsize=8,
            color="gray", va="bottom")

    fig.tight_layout()
    out = os.path.join(fig_dir(), "fig_adfa_perclass.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"图2 -> {out}")


def fig3_score_distribution():
    labels = {"lstm_keys": T["arch_keys"], "lstm_full": T["arch_full"],
              "tinygpt": T["arch_tiny"]}
    paths = {k: os.path.join(DET, "results", f"scores_{k}.npz")
             for k in ["lstm_keys", "lstm_full", "tinygpt"]}
    available = {k: p for k, p in paths.items() if os.path.exists(p)}
    if not available:
        print("图3: 无分数数据，跳过")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(4.2 * len(available), 3.6))
    if len(available) == 1:
        axes = [axes]
    for ax, (arch, path) in zip(axes, available.items()):
        d = np.load(path)
        bs, rs, tau = d["benign_scores"], d["rove_scores"], float(d["tau"])
        cap = max(np.percentile(np.concatenate([bs, rs]), 99.5), tau * 1.5)
        ax.hist(bs[bs < cap], bins=80, density=True, alpha=0.55, label=T["benign"])
        ax.hist(rs[rs < cap], bins=80, density=True, alpha=0.55, label=T["aaa_rove"])
        ax.axvline(tau, color="red", linestyle="--", linewidth=1.5,
                   label=f"τ={tau:.2f}")
        ax.set_xlabel(T["score_x"])
        ax.set_ylabel(T["density"])
        ax.set_title(labels[arch], fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(T["fig3_title"], fontsize=11)
    fig.tight_layout()
    out = os.path.join(fig_dir(), "fig_score_distribution.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"图3 -> {out}")


def fig4_aaa_mimicry():
    path = os.path.join(DET, "..", "reports", "aaa_calibration.json")
    if not os.path.exists(path):
        print("图4: 无 AAA 校准数据，跳过")
        return
    with open(path) as f:
        rep = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    dt_order = [f"DT{i}" for i in range(7)]
    x = np.arange(7)
    w = 0.27
    for i, (key, label) in enumerate([("benign", T["benign_host"]),
                                      ("rove", T["aaa_rove_l"]),
                                      ("bee", T["aaa_bee_l"])]):
        probs = [rep[key]["dt_probs"].get(d, 0) * 100 for d in dt_order]
        axes[0].bar(x + (i - 1) * w, probs, w, label=label, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dt_order)
    axes[0].set_ylabel(T["share_pct"])
    axes[0].set_xlabel(T["dt_x"])
    axes[0].set_title(T["fig4a_title"])
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    kl = rep["kl_divergence"]
    fields = ["ET", "PROC", "ARGV", "DT"]
    x2 = np.arange(len(fields))
    w2 = 0.35
    for i, tag in enumerate(["rove", "bee"]):
        vals = [kl.get(f"{f}_{tag}_vs_benign", 0) for f in fields]
        axes[1].bar(x2 + (i - 0.5) * w2, vals, w2,
                    label=T["aaa_rove_l"] if tag == "rove" else T["aaa_bee_l"],
                    alpha=0.85)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(fields)
    axes[1].set_ylabel(T["kl_y"])
    axes[1].set_title(T["fig4b_title"])
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(fig_dir(), "fig_aaa_mimicry.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"图4 -> {out}")


def main():
    argv = sys.argv[1:]
    lang = "zh"
    if "--lang" in argv:
        i = argv.index("--lang")
        lang = argv[i + 1] if i + 1 < len(argv) else "zh"
        del argv[i:i + 2]
    setup_lang(lang)
    results_dir = argv[0] if argv else os.path.join(DET, "results")
    if os.path.exists(os.path.join(results_dir, "aggregate_summary.json")):
        agg = load_agg(results_dir)
        fig1_baseline_comparison(agg)
        fig2_adfa(agg)
    else:
        print(f"未找到 aggregate_summary.json，跳过图1/图2")
    fig3_score_distribution()
    fig4_aaa_mimicry()
    print(f"完成（lang={lang}）")


if __name__ == "__main__":
    main()
