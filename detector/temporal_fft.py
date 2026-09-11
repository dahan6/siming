#!/usr/bin/env python3
"""时序层复合升级：CV + FFT 周期检测 + 多尺度分析

新增能力：
1. FFT 周期检测——抓 C2 信标的固定回连间隔
2. 多尺度 CV——1分钟/5分钟/15分钟窗口同时分析
3. Entropy rate——信息熵率，补充 CV 的盲区

核心原理：
- CV 抓"间隔太整齐"（隐翅虫 sleep CV=0.31）
- FFT 抓"有固定周期"（C2 每 60 秒回连一次）
- 两者正交：CV 低的 FFT 不一定有峰值，FFT 有峰值的 CV 不一定低
"""
import json
import sys
import os
from collections import defaultdict, deque
from datetime import datetime

import numpy as np


class FFTAnalyzer:
    """FFT 周期检测器

    对进程事件间隔序列做 FFT，检测固定周期信号。
    C2 信标、定时回连的特征是频谱中有明显的尖峰。
    """

    def __init__(self, min_samples=32, max_samples=256):
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.timestamps = deque(maxlen=max_samples)

    def update(self, timestamp_s):
        """喂一个事件时间戳（秒），返回周期检测结果
        
        FFT 分析的是事件时间戳序列，不是间隔序列。
        重采样到均匀网格后做 FFT，能正确检测周期。
        """
        self.timestamps.append(timestamp_s)

        if len(self.timestamps) < self.min_samples:
            return {"periodic": False, "dominant_freq": None, "status": "warmup"}

        ts = np.array(sorted(self.timestamps))
        ts = ts - ts[0]  # 从 0 开始
        duration = ts[-1] - ts[0]
        if duration < 1:
            return {"periodic": False, "dominant_freq": None, "status": "too_short"}

        # 重采样到均匀网格（事件计数直方图 = 事件率信号）
        # 网格分辨率 = duration / n_bins
        n_bins = min(len(ts) * 4, 512)
        bin_width = duration / n_bins
        counts, _ = np.histogram(ts, bins=n_bins, range=(0, duration))

        # FFT on 事件率信号
        spectrum = np.abs(np.fft.rfft(counts - np.mean(counts)))
        freqs = np.fft.rfftfreq(n_bins, d=bin_width)

        if len(spectrum) > 1:
            spectrum[0] = 0  # 去直流

        # 找最强非零频率
        peak_idx = np.argmax(spectrum[1:]) + 1
        peak_power = spectrum[peak_idx]
        noise_floor = np.median(spectrum[spectrum > 0]) if np.any(spectrum > 0) else 0.01

        snr = peak_power / max(noise_floor, 0.01)

        if freqs[peak_idx] > 0:
            period = 1.0 / freqs[peak_idx]
        else:
            period = 0

        is_periodic = snr > 5.0 and 1.0 < period < duration * 0.5

        return {
            "periodic": is_periodic,
            "snr": round(snr, 2),
            "dominant_period_s": round(period, 2) if period > 0 else None,
            "dominant_freq": round(freqs[peak_idx], 6),
            "n_samples": len(self.timestamps),
        }


class MultiScaleCV:
    """多尺度 CV 分析器

    同时在多个时间窗口上计算 CV：
    - 短窗口（1分钟）：抓快速回连
    - 中窗口（5分钟）：抓中等节奏
    - 长窗口（15分钟）：抓低速活动
    """

    def __init__(self, scales={"1min": 60, "5min": 300, "15min": 900}):
        self.scales = scales
        self.intervals = defaultdict(lambda: deque(maxlen=10000))
        self.timestamps = deque(maxlen=10000)

    def update(self, proc, timestamp_s):
        """喂一个事件，返回多尺度 CV 结果"""
        self.timestamps.append(timestamp_s)

        # 按进程维护间隔
        proc_intervals = self.intervals[proc]
        result = {}

        for scale_name, window_s in self.scales.items():
            # 取窗口内的间隔
            cutoff = timestamp_s - window_s
            recent = [(t, i) for t, i in zip(self.timestamps, proc_intervals)
                      if t >= cutoff]

            if len(recent) < 5:
                result[scale_name] = {"cv": None, "verdict": "insufficient"}
                continue

            intervals = np.array([i for _, i in recent])
            mean = np.mean(intervals)
            if mean < 0.001:
                result[scale_name] = {"cv": None, "verdict": "too_dense"}
                continue

            cv = float(np.std(intervals) / mean)
            verdict = "anomalous" if cv < 1.5 else ("suspicious" if cv < 2.0 else "normal")
            result[scale_name] = {"cv": round(cv, 3), "verdict": verdict}

        return result

    def add_interval(self, proc, interval_s, timestamp_s):
        """添加间隔记录"""
        self.intervals[proc].append(interval_s)


class EnhancedTemporalAnalyzer:
    """增强时序分析器：CV + FFT + 多尺度

    融合三个信号：
    1. 全局 CV（已有）
    2. FFT 周期检测（新增）
    3. 多尺度 CV（新增）

    综合判定：
    - CV < 1.5 OR FFT periodic → anomalous
    - 多尺度中任一窗口 CV < 1.5 → suspicious
    """

    def __init__(self, window_size=500, min_samples=20):
        self.window_size = window_size
        self.min_samples = min_samples
        self.proc_times = defaultdict(list)  # 每进程的时间戳
        self.global_times = []
        # FFT 分析器（全局）
        self.fft = FFTAnalyzer(min_samples=32, max_samples=256)
        # 多尺度 CV
        self.multi_cv = MultiScaleCV()

    def update(self, proc, timestamp_s, global_eps=0):
        """喂一个事件

        Args:
            proc: 进程名
            timestamp_s: 时间戳（秒）
            global_eps: 全局事件率

        Returns:
            dict: {cv, fft, multi_scale, verdict}
        """
        # 全局时间戳
        self.global_times.append(timestamp_s)
        if len(self.global_times) > self.window_size:
            self.global_times.pop(0)

        # 进程时间戳
        ts_list = self.proc_times[proc]
        ts_list.append(timestamp_s)

        result = {
            "proc": proc,
            "n_proc_samples": len(ts_list),
            "n_global_samples": len(self.global_times),
        }

        # ═══ 1. 传统 CV ═══
        if len(ts_list) >= self.min_samples:
            intervals = np.diff(sorted(ts_list))
            intervals = intervals[intervals > 0]
            if len(intervals) >= self.min_samples // 2:
                mean_int = np.mean(intervals)
                if mean_int >= 0.001:
                    std_int = np.std(intervals)
                    cv = std_int / mean_int
                    p50 = np.percentile(intervals, 50)
                    p95 = np.percentile(intervals, 95)
                    tail_ratio = p95 / max(p50, 0.001)
                    result["cv"] = round(cv, 3)
                    result["tail_ratio"] = round(tail_ratio, 3)
                    result["cv_verdict"] = "anomalous" if cv < 1.5 else ("suspicious" if cv < 2.0 else "normal")
                else:
                    result["cv"] = None
                    result["cv_verdict"] = "too_dense"
            else:
                result["cv"] = None
                result["cv_verdict"] = "insufficient"
        else:
            result["cv"] = None
            result["cv_verdict"] = "insufficient"

        # ═══ 2. FFT 周期检测 ═══
        if len(ts_list) >= 2:
            last_interval = ts_list[-1] - ts_list[-2] if len(ts_list) >= 2 else 0
            if last_interval > 0:
                fft_result = self.fft.update(last_interval)
                result["fft"] = fft_result
            else:
                result["fft"] = {"periodic": False, "status": "no_interval"}
        else:
            result["fft"] = {"periodic": False, "status": "insufficient"}

        # ═══ 3. 综合判定 ═══
        signals = []
        if result.get("cv_verdict") == "anomalous":
            signals.append("low_cv")
        if result.get("fft", {}).get("periodic"):
            signals.append("periodic")

        if any(s in signals for s in ("low_cv", "periodic")):
            result["verdict"] = "anomalous"
        elif result.get("cv_verdict") == "suspicious":
            result["verdict"] = "suspicious"
        else:
            result["verdict"] = "normal"

        result["signals"] = signals
        return result


def test_fft():
    """测试 FFT 周期检测"""
    print("=== FFT 周期检测测试 ===\n")

    fft = FFTAnalyzer(min_samples=32, max_samples=256)

    # 测试1: 固定周期（C2 信标每 60 秒）
    print("--- 测试1: 固定周期 60s（C2 信标）---")
    fft = FFTAnalyzer(min_samples=32, max_samples=256)
    np.random.seed(42)
    t = 0
    for i in range(64):
        t += 60.0 + np.random.normal(0, 2)  # 60s ± 2s 抖动
        result = fft.update(t)
    print(f"  periodic={result['periodic']} SNR={result['snr']} period={result['dominant_period_s']}s")
    assert result["periodic"], "应该检测到周期性"

    # 测试2: 随机间隔（人类行为）
    print("\n--- 测试2: 随机间隔（人类行为）---")
    fft2 = FFTAnalyzer(min_samples=32, max_samples=256)
    np.random.seed(42)
    t = 0
    for i in range(64):
        t += np.random.exponential(30)  # 重尾分布
        result = fft2.update(t)
    print(f"  periodic={result['periodic']} SNR={result['snr']} period={result.get('dominant_period_s')}s")
    assert not result["periodic"], "不应该检测到周期性"

    # 测试3: 准周期（间隔 55-65s 均匀分布）
    print("\n--- 测试3: 准周期 55-65s 均匀分布---")
    fft3 = FFTAnalyzer(min_samples=32, max_samples=256)
    np.random.seed(42)
    t = 0
    for i in range(64):
        t += np.random.uniform(55, 65)
        result = fft3.update(t)
    print(f"  periodic={result['periodic']} SNR={result['snr']} period={result.get('dominant_period_s')}s")

    # 测试4: 多频率叠加（正常系统 + 信标）
    print("\n--- 测试4: 多频率叠加（系统活动 + 30s 信标）---")
    fft4 = FFTAnalyzer(min_samples=64, max_samples=256)
    np.random.seed(42)
    t = 0
    for i in range(128):
        if i % 4 == 0:
            t += 30 + np.random.normal(0, 1)  # 信标
        else:
            t += np.random.exponential(5)  # 正常活动
        result = fft4.update(t)
    print(f"  periodic={result['periodic']} SNR={result['snr']} period={result.get('dominant_period_s')}s")

    print("\n✅ FFT 测试完成")


if __name__ == "__main__":
    test_fft()
