#!/usr/bin/env python3
"""墨城防火墙: 告警聚合器

同一 SRCNET→DSTNET→PORTCLS 组合在 window_sec 秒内只报一次告警。
后续相同组合的事件被抑制，只累加 hit_count。

规则:
  - P0/P1/P_NDR: 首次即报，后续同组在窗口内抑制
  - P2: 同组在窗口内只报一次
  - P3 弱信号: 不独立告警（在 daemon 中已跳过）

用法:
  agg = AlertAggregator(window_sec=60)
  result = agg.check(ts_epoch, tokens, prio)
  if not result["suppressed"]:
      emit_alert(..., hit_count=result["hit_count"])
"""
import time
from collections import defaultdict


def _group_key(tokens):
    """token 序列 → 聚合组键 (SRCNET, DSTNET, PORTCLS)"""
    slot_map = {}
    for t in tokens:
        if ":" in t:
            k, v = t.split(":", 1)
            slot_map[k] = v
        elif t.startswith("DT"):
            slot_map["DT"] = t
    return (slot_map.get("SRCNET", "?"),
            slot_map.get("DSTNET", "?"),
            slot_map.get("PORTCLS", "?"))


class AlertAggregator:
    """时间窗口告警聚合"""

    def __init__(self, window_sec=60):
        self.window_sec = window_sec
        # key → {"last_ts": float, "hit_count": int, "prio": str}
        self._groups = {}
        self._total_suppressed = 0

    def check(self, ts_epoch, tokens, prio, key_override=None):
        """检查一条告警是否应被抑制。

        返回: {"suppressed": bool, "hit_count": int}
          - suppressed=True: 该告警被聚合抑制（同组已有告警在窗口内）
          - suppressed=False: 该告警应发出，hit_count 为该组当前累积命中数

        Args:
            key_override: 自定义聚合键（用于 NDR beacon target 等），
                          为 None 时用 token 的 SRCNET→DSTNET→PORTCLS
        """
        key = key_override if key_override else _group_key(tokens)
        now = ts_epoch if ts_epoch else time.time()
        entry = self._groups.get(key)

        if entry is None:
            # 首次出现，发出告警
            self._groups[key] = {"last_ts": now, "hit_count": 1, "prio": prio}
            return {"suppressed": False, "hit_count": 1}

        if now - entry["last_ts"] <= self.window_sec:
            # 窗口内，抑制并累加
            entry["hit_count"] += 1
            self._total_suppressed += 1
            return {"suppressed": True, "hit_count": entry["hit_count"]}

        # 窗口过期，重置并发出
        entry["last_ts"] = now
        entry["hit_count"] = 1
        entry["prio"] = prio
        return {"suppressed": False, "hit_count": 1}

    def stats(self):
        """返回聚合统计"""
        active = sum(1 for e in self._groups.values())
        return {"active_groups": active, "total_suppressed": self._total_suppressed}

    def reset(self):
        """清空所有聚合状态"""
        self._groups.clear()
        self._total_suppressed = 0


if __name__ == "__main__":
    # 自测
    agg = AlertAggregator(window_sec=60)
    toks = ["PROTO:TCP", "DIR:OUT", "SRCNET:LAN", "DSTNET:EXT", "PORTCLS:HTTPS",
            "SIZECLS:L", "DT5"]

    r1 = agg.check(1000.0, toks, "P2")
    print(f"第1条: suppressed={r1['suppressed']}, hit={r1['hit_count']}")  # False, 1

    r2 = agg.check(1010.0, toks, "P2")
    print(f"第2条: suppressed={r2['suppressed']}, hit={r2['hit_count']}")  # True, 2

    r3 = agg.check(1050.0, toks, "P2")
    print(f"第3条: suppressed={r3['suppressed']}, hit={r3['hit_count']}")  # True, 3

    r4 = agg.check(1070.0, toks, "P2")  # >60s 窗口过期
    print(f"第4条: suppressed={r4['suppressed']}, hit={r4['hit_count']}")  # False, 1

    print(f"统计: {agg.stats()}")
