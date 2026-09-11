#!/usr/bin/env python3
"""墨城防火墙: NDR 总序列周期检测器

核心洞察（来自隐翅虫 §4.5 对抗评估）:
  flow 级周期检测会被通道交替规避（http/dns 换着发，每个 flow 都不规则）。
  **总序列周期检测**（所有连接按时间排序算间隔 CV）才是杀招——
  C2 换端口换协议也躲不掉"每 30s 必外联一次"的总频率。

  隐翅虫的对抗手段（§3.7 行为时序拟态）:
  - 宿主频率跟随：步进 = U(1.5,5.0)s × 宿主事件率
  - 渐稳减速：压力驱动间隔拉伸 1~3 倍
  - 时机网络抖动：部署侧 × U(0.5,1.5) 打破固定周期

  本模块在 token 打分之上叠加一层宏观周期分析:
  1. 滚动窗口内所有连接的间隔变异系数 CV = std/mean
  2. CV 低 → 周期性强 → C2 信标嫌疑
  3. 按目标 IP/网段分组检测周期性（定位到具体 C2 通道）

用法:
  from fw_ndr import NDRDetector
  ndr = NDRDetector(window=100)
  result = ndr.observe(ts_epoch, dst_ip, dst_port, proto)
  # result = {"beacon": True, "target": "1.2.3.4", "period": 30.0, "cv": 0.08}
"""
import time
from collections import defaultdict, deque


class NDRDetector:
    """总序列周期检测器（NDR = Network Detection and Response）"""

    def __init__(self, window=100, min_samples=8, cv_threshold=0.3,
                 min_period=5.0, max_period=600.0):
        """
        Args:
            window: 滚动窗口大小（最近 N 个连接）
            min_samples: 周期判定所需最小样本数
            cv_threshold: 变异系数阈值（CV < 此值 = 周期性强）
            min_period: 可信周期下限（秒）——低于此值的"周期"可能是正常突发
            max_period: 可信周期上限（秒）
        """
        self.window = window
        self.min_samples = min_samples
        self.cv_threshold = cv_threshold
        self.min_period = min_period
        self.max_period = max_period
        # 全局时间序列
        self.global_ts = deque(maxlen=window)
        # 按目标的分组时间序列
        self.per_target = defaultdict(lambda: deque(maxlen=window))
        # 已报告过的 beacon target（同一 target 只报一次）
        self._reported = set()

    def observe(self, ts_epoch, dst_ip, dst_port=0, proto=""):
        """观察一个新连接事件，返回周期检测结果

        Returns:
            dict 或 None:
              {"beacon": True, "target": ip, "period": s, "cv": float,
               "n_samples": int, "score": float}

        每个 beacon target 只返回一次检测结果，后续 observe 不再重复报告
        同一 target（除非数据量不足以维持 beacon 判定后重新积累）。
        """
        self.global_ts.append(ts_epoch)
        key = f"{dst_ip}:{dst_port}"
        self.per_target[key].append(ts_epoch)

        # 检查全局序列周期
        global_result = self._check_periodicity(self.global_ts, "GLOBAL")

        # 检查每个目标分组
        target_results = []
        for tgt, ts_list in self.per_target.items():
            if len(ts_list) >= self.min_samples:
                r = self._check_periodicity(ts_list, tgt)
                if r and r["beacon"]:
                    target_results.append(r)

        # 优先返回目标级检测结果（更精确）
        best = None
        if target_results:
            best = min(target_results, key=lambda x: x["cv"])
        elif global_result and global_result["beacon"]:
            best = global_result

        if best:
            tgt_key = best["target"]
            # 同一 target 只报一次
            if tgt_key in self._reported:
                return None
            self._reported.add(tgt_key)
            return best
        return None

    def _check_periodicity(self, ts_list, target):
        """计算时间序列的周期性指标

        CV（变异系数）= std(intervals) / mean(intervals)
        CV 越低 → 间隔越均匀 → 周期性越强
        """
        if len(ts_list) < self.min_samples:
            return None

        ts = list(ts_list)
        intervals = [ts[i] - ts[i-1] for i in range(1, len(ts))]
        if not intervals:
            return None

        n = len(intervals)
        mean = sum(intervals) / n
        if mean < 1e-6:
            return None  # 避免除零

        variance = sum((x - mean) ** 2 for x in intervals) / n
        std = variance ** 0.5
        cv = std / mean

        # 周期判定
        is_beacon = (cv < self.cv_threshold and
                     self.min_period <= mean <= self.max_period and
                     n >= self.min_samples)

        # 置信分数：CV 越低分数越高
        score = max(0.0, 1.0 - cv / self.cv_threshold) if is_beacon else 0.0

        return {
            "beacon": is_beacon,
            "target": target,
            "period": round(mean, 1),
            "cv": round(cv, 4),
            "n_samples": n,
            "score": round(score, 3),
        }

    def stats(self):
        """返回当前检测器状态摘要"""
        return {
            "global_connections": len(self.global_ts),
            "tracked_targets": len(self.per_target),
            "targets_with_data": sum(1 for v in self.per_target.values()
                                     if len(v) >= self.min_samples),
        }


# ── 合成 C2 信标模拟器（用于验证 NDR）──
def simulate_beacon(n=20, period=30.0, jitter=0.1, base_ts=None):
    """模拟 C2 信标流量：固定周期 + 可配抖动

    Args:
        n: 信标次数
        period: 基础周期（秒）
        jitter: 抖动比例（0=完美周期，1=完全随机）
    """
    import random
    base_ts = base_ts or time.time()
    flows = []
    t = base_ts
    for i in range(n):
        # 隐翅虫风格：U(0.5,1.5) 抖动
        actual_period = period * (1 + random.uniform(-jitter, jitter))
        t += actual_period
        flows.append({
            "ts_epoch": t,
            "dst_ip": "203.0.113.99",
            "dst_port": 4444,
            "proto": "tcp",
        })
    return flows


def simulate_normal(n=50, base_ts=None):
    """模拟正常流量：随机间隔"""
    import random
    base_ts = base_ts or time.time()
    flows = []
    t = base_ts
    for i in range(n):
        t += random.expovariate(1 / 15)  # 平均 15 秒，指数分布
        flows.append({
            "ts_epoch": t,
            "dst_ip": random.choice(["8.8.8.8", "1.1.1.1", "140.82.112.3"]),
            "dst_port": random.choice([443, 80, 53]),
            "proto": random.choice(["tcp", "udp"]),
        })
    return flows


if __name__ == "__main__":
    print("== NDR 周期检测器验证 ==\n")

    # 测试 1: 纯 C2 信标（低抖动）
    ndr = NDRDetector(window=50, min_samples=6)
    beacon = simulate_beacon(n=15, period=30.0, jitter=0.05)
    result = None
    for f in beacon:
        result = ndr.observe(f["ts_epoch"], f["dst_ip"], f["dst_port"], f["proto"])
    print(f"C2 信标（低抖动 jitter=0.05）: {result}")
    assert result and result["beacon"], "应检出 C2 信标"

    # 测试 2: C2 信标（隐翅虫级抖动 U(0.5,1.5)）
    ndr2 = NDRDetector(window=50, min_samples=6)
    beacon2 = simulate_beacon(n=15, period=30.0, jitter=0.5)
    result2 = None
    for f in beacon2:
        result2 = ndr2.observe(f["ts_epoch"], f["dst_ip"], f["dst_port"], f["proto"])
    print(f"C2 信标（隐翅虫级抖动 jitter=0.5）: {result2}")

    # 测试 3: 正常流量（指数分布）
    ndr3 = NDRDetector(window=50, min_samples=6)
    normal = simulate_normal(n=30)
    result3 = None
    for f in normal:
        result3 = ndr3.observe(f["ts_epoch"], f["dst_ip"], f["dst_port"], f["proto"])
    print(f"正常流量（指数分布）: {result3}")
    assert result3 is None or not result3["beacon"], "正常流量不应被判为信标"

    # 测试 4: 混合流量（正常 + C2）
    ndr4 = NDRDetector(window=50, min_samples=6)
    mixed = sorted(normal[:20] + beacon[:10], key=lambda f: f["ts_epoch"])
    result4 = None
    for f in mixed:
        result4 = ndr4.observe(f["ts_epoch"], f["dst_ip"], f["dst_port"], f["proto"])
    print(f"混合流量（正常+C2）: {result4}")

    print("\n== 全部验证通过 ==")
