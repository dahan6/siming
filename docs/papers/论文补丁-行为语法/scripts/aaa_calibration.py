#!/usr/bin/env python3
"""AAA 对抗强度量化特征提取

目的：为论文提供"AAA 与文献自适应攻击者的对比"的量化证据。
从轨迹数据提取可度量特征：

  1. 时序特征：事件间隔分布、CV（变异系数）、频率跟随能力
  2. 行为词汇：进程名多样性、ET 分布、ARGV 骨架多样性
  3. 拟态特征：与良性分布的 KL 散度（逐字段）
  4. 自适应标志：rebirth/变形/自删除类事件占比

对比对象（文献锚点，由论文文字部分引用）：
  - gym-malware (Anderson 2018)：仅变异 API 调用序列，无时机适应
  - MERLIN：RL 静态逃逸，无分布拟合
  - AAA：时序拟态 + 宿主频率跟随 + 渐稳减速 + rebirth 自变异

用法: aaa_calibration.py
输出: ../reports/aaa_calibration.json + 终端打印
"""
import json
import os
import sys
from collections import Counter

import numpy as np

DET = os.path.dirname(os.path.abspath(__file__))
BENIGN = os.path.expanduser("~/defense-lab/data/host_tokens_clean.jsonl")
ROVE = os.path.join(DET, "data", "rove_attacks.jsonl")
BEE = os.path.join(DET, "data", "bee_active.jsonl")

DT_ORDER = [f"DT{i}" for i in range(7)]


def load_tokens(path, limit=None):
    seqs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = e.get("tokens")
            if t:
                seqs.append(t)
    return seqs


def field_dist(seqs, idx, prefix):
    """某字段的分布"""
    c = Counter()
    for t in seqs:
        if idx < len(t) and t[idx].startswith(prefix):
            c[t[idx]] += 1
    return c


def kl_div(p, q, keys):
    """两个 Counter 分布的 KL 散度（带平滑）"""
    pk = np.array([p.get(k, 0) for k in keys], dtype=float)
    qk = np.array([q.get(k, 0) for k in keys], dtype=float)
    pk = (pk + 1) / (pk.sum() + len(keys))
    qk = (qk + 1) / (qk.sum() + len(keys))
    return float(np.sum(pk * np.log(pk / qk)))


def analyze(name, seqs):
    dt_c = field_dist(seqs, 7, "DT")
    n = max(1, sum(dt_c.values()))
    dt_probs = [dt_c.get(d, 0) / n for d in DT_ORDER]

    proc_c = field_dist(seqs, 1, "PROC:")
    et_c = field_dist(seqs, 0, "ET:")
    argv_c = field_dist(seqs, 2, "ARGV")

    # CV 估计：DT 桶中点近似（桶边界: 1,10,100,1000,10000,60000 ms）
    midpoints = np.array([0.5, 5, 55, 550, 5500, 35000, 120000])
    samples = np.concatenate([
        np.full(dt_c.get(d, 0), midpoints[i]) for i, d in enumerate(DT_ORDER)])
    cv = float(np.std(samples) / np.mean(samples)) if len(samples) and samples.mean() > 0 else 0.0

    return {
        "name": name,
        "n_events": len(seqs),
        "dt_probs": {d: round(dt_c.get(d, 0) / n, 4) for d in DT_ORDER},
        "dt_cv_estimate": round(cv, 3),
        "proc_vocab": len(proc_c),
        "et_vocab": len(et_c),
        "argv_vocab": len(argv_c),
        "proc_top10": proc_c.most_common(10),
        "et_dist": dict(et_c),
        "proc_counter": proc_c,
        "et_counter": et_c,
        "argv_counter": argv_c,
    }


def main():
    benign = load_tokens(BENIGN, limit=100000)
    rove = load_tokens(ROVE)
    bee = load_tokens(BEE)

    a_ben = analyze("benign_host", benign)
    a_rove = analyze("AAA_rove", rove)
    a_bee = analyze("AAA_bee_active", bee)

    # 逐字段 KL 散度（AAA vs 良性）——拟态强度度量
    kl_report = {}
    counters = {"ET": "et_counter", "PROC": "proc_counter",
                "ARGV": "argv_counter", "DT": None}
    for field in ["ET", "PROC", "ARGV", "DT"]:
        for tag, a in [("rove", a_rove), ("bee", a_bee)]:
            if field == "DT":
                # DT 用 dt_probs 重建 counter
                ck_a = Counter({d: int(p * a["n_events"]) for d, p in a["dt_probs"].items()})
                ck_b = Counter({d: int(p * a_ben["n_events"]) for d, p in a_ben["dt_probs"].items()})
                keys = set(ck_a) | set(ck_b)
            else:
                ck_a = a[counters[field]]
                ck_b = a_ben[counters[field]]
                keys = set(ck_a) | set(ck_b)
            kl = kl_div(ck_a, ck_b, keys)
            kl_report[f"{field}_{tag}_vs_benign"] = round(kl, 3)

    # 自适应标志统计
    adaptive_markers = Counter()
    for line in open(ROVE):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        adaptive_markers[e.get("label", "?")] += 1

    report = {
        "benign": {k: v for k, v in a_ben.items() if "counter" not in k},
        "rove": {k: v for k, v in a_rove.items() if "counter" not in k},
        "bee": {k: v for k, v in a_bee.items() if "counter" not in k},
        "kl_divergence": kl_report,
        "aaa_label_dist": dict(adaptive_markers),
    }

    out = os.path.join(DET, "..", "reports", "aaa_calibration.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 终端摘要
    print("=" * 60)
    print("  AAA 对抗强度量化特征")
    print("=" * 60)
    for a in [a_ben, a_rove, a_bee]:
        print(f"\n[{a['name']}] n={a['n_events']}")
        print(f"  DT CV≈{a['dt_cv_estimate']}  进程词汇 {a['proc_vocab']}  "
              f"ET词汇 {a['et_vocab']}  ARGV词汇 {a['argv_vocab']}")
        print(f"  DT 分布: {a['dt_probs']}")
    print(f"\n逐字段 KL 散度（越小越像良性）: {kl_report}")
    print(f"\nAAA 标签分布: {dict(adaptive_markers)}")
    print(f"\n报告 -> {out}")


if __name__ == "__main__":
    main()
