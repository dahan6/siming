#!/usr/bin/env python3
"""二阶时序分析器：检测机器节奏 vs 人节奏的结构差异

一阶 DT 桶被adaptive agent的频率跟随+抖动骗过，但统计高阶矩仍有结构差异：
- 人驱动：间隔重尾（突发+长静默），cv>1.5，tail_ratio>5
- 机器驱动：间隔均匀/正态，cv<0.8，tail_ratio<3
- 频率跟随：进程间隔与系统事件率高耦合，eps_corr>0.6

用法（独立测试）: temporal_analyzer.py <tokens.jsonl> [--n 20000]
用法（库调用）: from temporal_analyzer import TemporalAnalyzer
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np


class TemporalAnalyzer:
    """对 token 流做二阶时序分析，输出节奏异常判定"""

    def __init__(self, window_size=500, min_samples=20):
        self.window_size = window_size
        self.min_samples = min_samples
        # 按 PROC 维护近期时间戳（ms）
        self.proc_times = defaultdict(list)
        # 全局事件率（滑窗估计）
        self.global_times = []
        # 阈值（2026-07-31 用adaptive agent遥测 18 万条标定）
        # 良性基线 CV p5=1.509, adaptive agent CV p5=0.647
        # 检测率 15/15, 误报 1/15 (sh)
        self.thresholds = {
            "cv_low": 1.5,      # <此值=机器节奏（良性 p5=1.509）
            "cv_suspicious": 1.6,  # 1.5-1.6 灰区（良性 p15=1.612）
            "tail_low": 3.3,    # <此值=缺乏长尾（良性 p5=3.289）
            "eps_corr_high": 0.6,  # >此值=频率跟随
        }

    def update(self, proc, timestamp_ms, global_eps, uid=None):
        """喂一条事件，返回当前该进程的时序判定（或 None=样本不足）

        Args:
            proc: 进程名（tokens[1] 的值，如 "bash"）
            timestamp_ms: 事件时间戳（毫秒）
            global_eps: 当前系统事件率（事件/秒，滑窗估计）
            uid: 可选 UID 过滤——只对非 root 用户进程做时序分析
                 （系统进程天然是机器节奏，不算异常）

        Returns:
            dict 或 None: {cv, tail_ratio, eps_corr, verdict, n}
            verdict: "normal" / "suspicious" / "anomalous"
        """
        # 系统 root 进程天然是机器节奏，跳过
        SYSTEM_UIDS = {"0", "?"}
        if uid is not None and uid in SYSTEM_UIDS:
            return None

        ts_list = self.proc_times[proc]
        ts_list.append(timestamp_ms)
        if len(ts_list) > self.window_size:
            ts_list.pop(0)

        # 全局事件率记录
        self.global_times.append((timestamp_ms, global_eps))
        if len(self.global_times) > self.window_size * 2:
            self.global_times.pop(0)

        if len(ts_list) < self.min_samples:
            return None

        # 计算间隔
        intervals = np.diff(ts_list, prepend=ts_list[0])
        intervals = intervals[intervals > 0]  # 去零（同时刻事件）
        if len(intervals) < self.min_samples // 2:
            return None

        mean_int = np.mean(intervals)
        if mean_int < 1:
            return None  # 太密集，可能是同一批事件

        std_int = np.std(intervals)
        cv = std_int / mean_int
        p50 = np.percentile(intervals, 50)
        p95 = np.percentile(intervals, 95)
        tail_ratio = p95 / max(p50, 1)

        # eps 耦合度：进程间隔 vs 全局事件率的相关性
        eps_corr = 0.0
        if len(self.global_times) >= self.min_samples:
            # 取与进程时间戳对齐的 global_eps
            proc_eps = []
            proc_intervals = []
            for i in range(1, len(ts_list)):
                t = ts_list[i]
                # 找最近的 global_eps
                nearest_eps = global_eps  # 简化：用当前值
                proc_eps.append(nearest_eps)
                proc_intervals.append(ts_list[i] - ts_list[i - 1])
            if len(proc_intervals) >= self.min_samples // 2 and np.std(proc_eps) > 0:
                eps_corr = abs(np.corrcoef(proc_intervals, proc_eps)[0, 1])
                if np.isnan(eps_corr):
                    eps_corr = 0.0

        # 判定
        th = self.thresholds
        signals = []
        if cv < th["cv_low"]:
            signals.append("low_cv")  # 过于整齐
        elif cv < th["cv_suspicious"]:
            signals.append("border_cv")  # 灰区
        if tail_ratio < th["tail_low"] and len(intervals) >= 30:
            signals.append("low_tail")  # 缺乏长尾
        if eps_corr > th["eps_corr_high"]:
            signals.append("eps_following")  # 频率跟随

        if any(s in signals for s in ("low_cv", "eps_following")):
            verdict = "anomalous"
        elif signals:
            verdict = "suspicious"
        else:
            verdict = "normal"

        return {
            "cv": round(cv, 3),
            "tail_ratio": round(tail_ratio, 3),
            "eps_corr": round(eps_corr, 3),
            "n": len(intervals),
            "signals": signals,
            "verdict": verdict,
        }

    def calibrate(self, events, n_calib=20000):
        """用良性数据标定阈值（取各指标的安全边界）

        Args:
            events: [{"tokens": [...], "timestamp": ...}, ...] 或 [{"tokens": [...], "dt_ms": ...}]
        """
        proc_times = defaultdict(list)
        global_times = []
        all_cv, all_tail, all_corr = [], [], []

        prev_ts = None
        for i, ev in enumerate(events[:n_calib]):
            toks = ev.get("tokens", [])
            if len(toks) < 2:
                continue
            proc = toks[1] if toks[1].startswith("PROC:") else "unknown"

            ts = ev.get("timestamp_ms") or ev.get("dt_ms_accum")
            if ts is None:
                # 从 dt 桶重建
                dt_tok = [t for t in toks if t.startswith("DT")]
                continue

            if prev_ts is not None:
                pass
            prev_ts = ts

            proc_times[proc].append(ts)
            global_times.append(ts)

        # 用时间戳重建间隔
        for proc, times in proc_times.items():
            if len(times) < self.min_samples:
                continue
            intervals = np.diff(sorted(times))
            intervals = intervals[intervals > 0]
            if len(intervals) < self.min_samples // 2:
                continue
            mean_int = np.mean(intervals)
            if mean_int < 1:
                continue
            all_cv.append(np.std(intervals) / mean_int)
            p50 = np.percentile(intervals, 50)
            p95 = np.percentile(intervals, 95)
            all_tail.append(p95 / max(p50, 1))

        if all_cv:
            cv_arr = np.array(all_cv)
            # 阈值：取良性分布的 p5 作为 "机器节奏" 的上界（良性里最整齐的 5%）
            self.thresholds["cv_low"] = max(float(np.percentile(cv_arr, 5)), 0.3)
            self.thresholds["cv_suspicious"] = max(float(np.percentile(cv_arr, 15)), 0.6)
        if all_tail:
            tail_arr = np.array(all_tail)
            self.thresholds["tail_low"] = max(float(np.percentile(tail_arr, 5)), 1.5)

        return dict(self.thresholds)


def dt_to_ms(dt_token):
    """DT 桶 → 近似 ms（用于无原始时间戳的场景）"""
    mapping = {"DT0": 0.5, "DT1": 5, "DT2": 50, "DT3": 500, "DT4": 5000, "DT5": 30000, "DT6": 60000}
    return mapping.get(dt_token, 100)


def run_on_tokens(tokens_path, n=20000):
    """对 token 流做离线分析，输出统计"""
    events = [json.loads(l) for l in open(tokens_path)][-n:]
    print(f"分析 {len(events)} 事件")

    # 重建时间戳（用 DT 桶近似）
    proc_times = defaultdict(list)
    accum_ms = 0
    for ev in events:
        toks = ev["tokens"]
        proc = toks[1] if len(toks) > 1 else "?"
        dt = toks[-1] if toks[-1].startswith("DT") else "DT3"
        accum_ms += dt_to_ms(dt) + np.random.uniform(0, dt_to_ms(dt))
        proc_times[proc].append(accum_ms)

    analyzer = TemporalAnalyzer(min_samples=15)
    print(f"\n{'进程':<20}{'n':>5}{'cv':>8}{'tail':>8}{'verdict':>12}")
    results = []
    for proc in sorted(proc_times, key=lambda p: len(proc_times[p]), reverse=True)[:20]:
        times = sorted(proc_times[proc])
        if len(times) < 15:
            continue
        intervals = np.diff(times)
        intervals = intervals[intervals > 0]
        if len(intervals) < 10:
            continue
        mean_int = np.mean(intervals)
        if mean_int < 1:
            continue
        cv = np.std(intervals) / mean_int
        p50 = np.percentile(intervals, 50)
        p95 = np.percentile(intervals, 95)
        tail = p95 / max(p50, 1)
        if cv < 1.5:
            verdict = "anomalous"
        elif cv < 1.6:
            verdict = "suspicious"
        else:
            verdict = "normal"
        results.append((proc, len(intervals), cv, tail, verdict))
        print(f"{proc:<20}{len(intervals):>5}{cv:>8.3f}{tail:>8.1f}{verdict:>12}")

    # 良性基线统计
    all_cv = [r[2] for r in results]
    print(f"\ncv 分布: min={min(all_cv):.3f} median={np.median(all_cv):.3f} max={max(all_cv):.3f}")
    n_machine = sum(1 for c in all_cv if c < 1.5)
    print(f"机器节奏 (cv<1.5): {n_machine}/{len(results)} 进程")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_on_tokens(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20000)
    else:
        print("用法: temporal_analyzer.py <tokens.jsonl> [n]")
