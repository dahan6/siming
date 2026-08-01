#!/usr/bin/env python3
"""deploy_scorer v3 端到端验证：在adaptive agent遥测 vs 良性数据上跑完整五网管道

验证目标：
1. adaptive agent活跃期：告警覆盖率（各 P 层的命中分布）
2. 良性夜间：误报率（FPR）
3. 各模块贡献：时序 vs 自适应 vs 模式 vs 上下文 vs 稀有度
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

from train_prior import TinyGPT, CTX
from pattern_db import PatternDB
from temporal_analyzer import TemporalAnalyzer
from adaptive_detector import AdaptiveDetector

DET = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 0.3
CTX_SLOTS = {"PARENT", "DST", "DT"}
RARE_SLOTS = {"PROC", "ARGV"}


def slot_of(tok):
    if ":" in tok:
        return tok.split(":")[0]
    if tok.startswith("ARGV"):
        return "ARGV"
    if tok.startswith("DT"):
        return "DT"
    return tok


def convert_7to8(tokens_7):
    """7-token → 8-token（在 DST 后插入 PC:NONE）"""
    out = list(tokens_7)
    has_pc = any(t.startswith("PC:") for t in out)
    if not has_pc:
        dst_idx = next((i for i, t in enumerate(out) if t.startswith("DST:")), 5)
        out.insert(dst_idx + 1, "PC:NONE")
    return out


def run_pipeline(model_dir, events_jsonl, hour_filter=None, n_max=50000, label="", use_vm_tau=False):
    """跑完整五网管道"""
    # 加载模型
    ckpt = torch.load(os.path.join(model_dir, "prior.pt"),
                      map_location=DEVICE, weights_only=False)
    stoi = ckpt["stoi"]
    model = TinyGPT(len(stoi)).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    # τ 优先级：slot_tau_local > slot_tau_vm > slot_tau
    slot_tau = {}
    for tau_name in ("slot_tau_local.json", "slot_tau_vm.json", "slot_tau.json"):
        tau_path = os.path.join(model_dir, tau_name)
        if os.path.exists(tau_path):
            slot_tau = json.load(open(tau_path))["slot_tau"]
            break
    db = PatternDB(os.path.join(DET, "patterns.jsonl"))
    temporal = TemporalAnalyzer(min_samples=20)
    adaptive = AdaptiveDetector(window_size=200, cooldown=50)

    window = []
    ewma = 0.0
    n_scored = 0
    n_alert = Counter()
    alert_details = Counter()  # by type
    temporal_alerts = []
    adaptive_alerts = []
    pattern_alerts = []

    for line in open(events_jsonl):
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue

        if hour_filter and e["ts"][11:13] not in hour_filter:
            continue
        if n_scored >= n_max:
            break

        tokens_7 = e["tokens"]
        tokens = convert_7to8(tokens_7)
        ts_str = e.get("ts", "")

        # 从时间戳重建 ms 时间戳
        from datetime import datetime
        try:
            ts = datetime.fromisoformat(ts_str).timestamp() * 1000
        except Exception:
            ts = n_scored * 1000

        n_scored += 1

        # === 网络1: 稀有度（prior model） ===
        ids = [stoi.get(t, 0) for t in tokens]
        window = (window + ids)[-CTX:]
        n, L = len(ids), len(window)
        start = max(L - n, 1)
        with torch.no_grad():
            x = torch.tensor(window, device=DEVICE).unsqueeze(0)
            lp = torch.log_softmax(model(x), dim=-1)[0]
            tgt = torch.tensor(window[start:L], device=DEVICE)
            nll = -lp[start - 1:L - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        s_ev = nll.max().item()
        ewma = ALPHA * s_ev + (1 - ALPHA) * ewma

        fired = []
        for t, v in zip(tokens, nll.tolist()):
            s = slot_of(t)
            if v > slot_tau.get(s, 1.0):
                fired.append((s, t, round(v, 2)))
        n_unk = sum(1 for t in tokens if t not in stoi)
        ctx_fired = [x for x in fired if x[0] in CTX_SLOTS]
        rare_fired = [x for x in fired if x[0] in RARE_SLOTS]

        # === 网络2: 模式库 ===
        hits = db.match(tokens)
        patterns = [{"id": h["id"], "technique": h["technique"],
                     "severity": h["severity"]} for h in hits]

        # === 网络3: 自适应检测 ===
        adapt_alerts = adaptive.update(tokens, ts=ts_str)
        adapt_p5 = [a for a in adapt_alerts if a["severity"] >= 5]
        adapt_low = [a for a in adapt_alerts if a["severity"] < 5]

        # === 网络4: 时序分析 ===
        proc_name = "?"
        uid_val = None
        for t in tokens:
            if t.startswith("PROC:"):
                proc_name = t.split(":", 1)[1]
            elif t.startswith("UID:"):
                uid_val = t.split(":", 1)[1] if ":" in t else None
        temporal_result = temporal.update(proc_name, ts, ewma, uid=uid_val)

        # === 融合判定 ===
        # 按严重度拆分模式命中
        strong_patterns = [p for p in patterns if p["severity"] >= 4]
        weak_patterns = [p for p in patterns if p["severity"] < 4]

        if adapt_p5:
            prio = "P0"
            for a in adapt_p5:
                adaptive_alerts.append(a["type"])
                alert_details[a["type"]] += 1
        elif strong_patterns:
            prio = "P1"
            for p in strong_patterns:
                pattern_alerts.append(p["id"])
                alert_details[f"pattern:{p['id']}"] += 1
        elif ctx_fired:
            prio = "P2"
            alert_details["ctx_anomaly"] += 1
        elif rare_fired or n_unk > 0:
            prio = "P3"
            alert_details["rarity"] += 1
        elif temporal_result and temporal_result["verdict"] == "anomalous":
            prio = "P4"
            temporal_alerts.append(proc_name)
            alert_details[f"temporal:{proc_name}"] += 1
        elif adapt_low or weak_patterns:
            prio = "P5"
            for a in adapt_low:
                adaptive_alerts.append(a["type"])
                alert_details[a["type"]] += 1
            for p in weak_patterns:
                pattern_alerts.append(p["id"])
                alert_details[f"weak:{p['id']}"] += 1
        else:
            continue

        n_alert[prio] += 1

    # 报告
    total_alerts = sum(n_alert.values())
    fpr = total_alerts / max(n_scored, 1) * 100

    print(f"\n{'='*60}")
    print(f"=== {label} ===")
    print(f"{'='*60}")
    print(f"事件数: {n_scored}")
    print(f"总告警: {total_alerts} ({fpr:.2f}%)")
    print(f"\n各层告警分布:")
    for p in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        if n_alert[p] > 0:
            pct = n_alert[p] / max(total_alerts, 1) * 100
            print(f"  {p}: {n_alert[p]:>5} ({pct:.1f}%)")

    if pattern_alerts:
        print(f"\n模式命中 (top 10):")
        for pid, c in Counter(pattern_alerts).most_common(10):
            print(f"  {pid}: {c}")

    if temporal_alerts:
        print(f"\n时序异常进程 (top 10):")
        for proc, c in Counter(temporal_alerts).most_common(10):
            print(f"  {proc}: {c}")

    if adaptive_alerts:
        print(f"\n自适应告警类型:")
        for atype, c in Counter(adaptive_alerts).most_common():
            print(f"  {atype}: {c}")

    print(f"\n详细告警分类 (top 15):")
    for detail, c in alert_details.most_common(15):
        print(f"  {detail}: {c}")

    return {"n_scored": n_scored, "n_alert": dict(n_alert),
            "fpr": fpr, "details": dict(alert_details)}


def main():
    # 优先使用 VM 通用模型
    model_dir = os.path.join(DET, "model-vm-universal")
    if not os.path.exists(os.path.join(model_dir, "prior.pt")):
        model_dir = os.path.join(DET, "model-current")
    clone_path = os.path.expanduser(
        "~/data/telemetry/clone_events.jsonl")
    regime_path = os.path.expanduser(
        "~/data/telemetry/regime_events.jsonl")

    # 测试 1: adaptive agent活跃期（13:00-14:00）
    agent_result = run_pipeline(
        model_dir, regime_path, hour_filter={"13"}, n_max=50000,
        label="adaptive agent活跃期 (regime 13:00) [VM通用模型]", use_vm_tau=True)

    # 测试 2: 良性夜间（02:00-05:00）
    benign_result = run_pipeline(
        model_dir, clone_path, hour_filter={"02", "03", "04", "05"}, n_max=20000,
        label="良性夜间 (clone 02-05) [VM通用模型]", use_vm_tau=True)

    # 汇总
    print(f"\n{'='*60}")
    print(f"=== 汇总对比 ===")
    print(f"{'='*60}")
    print(f"{'指标':<25}{'adaptive agent':>12}{'良性':>12}")
    print(f"{'-'*49}")
    print(f"{'事件数':<25}{agent_result['n_scored']:>12}{benign_result['n_scored']:>12}")
    print(f"{'总告警率':<25}{agent_result['fpr']:>11.2f}%{benign_result['fpr']:>11.2f}%")

    agent_p0 = agent_result["n_alert"].get("P0", 0)
    agent_p1 = agent_result["n_alert"].get("P1", 0)
    agent_p4 = agent_result["n_alert"].get("P4", 0)
    agent_p5 = agent_result["n_alert"].get("P5", 0)
    benign_p0 = benign_result["n_alert"].get("P0", 0)
    benign_p4 = benign_result["n_alert"].get("P4", 0)
    benign_p5 = benign_result["n_alert"].get("P5", 0)

    print(f"{'P0 自适应高危':<25}{agent_p0:>12}{benign_p0:>12}")
    print(f"{'P1 模式命中':<25}{agent_p1:>12}{benign_result['n_alert'].get('P1', 0):>12}")
    print(f"{'P4 时序异常':<25}{agent_p4:>12}{benign_p4:>12}")
    print(f"{'P5 自适应低危':<25}{agent_p5:>12}{benign_p5:>12}")

    # 新增模块（P0+P4+P5）的增益
    new_agent = agent_p0 + agent_p4 + agent_p5
    new_benign = benign_p0 + benign_p4 + benign_p5
    print(f"\n新模块增益 (P0+P4+P5):")
    print(f"  adaptive agent: {new_agent} 条 ({new_agent/max(agent_result['n_scored'],1)*100:.1f}%)")
    print(f"  良性: {new_benign} 条 ({new_benign/max(benign_result['n_scored'],1)*100:.1f}%)")


if __name__ == "__main__":
    main()
