#!/usr/bin/env python3
"""墨城防火墙: 连接状态机

从 conntrack state 字段提取连接生命周期状态，
为检测引擎提供上下文：NEW 连续出现 = 扫描，ESTABLISHED = 正常通信。

状态分类:
  NEW          - 新连接（SYN_SENT, 第一包）
  ESTABLISHED  - 已建立（正常通信）
  RELATED      - 关联连接（FTP 数据通道等）
  INVALID      - 无效连接（防火墙应直接 DROP）
  CLOSING      - 正在关闭（TIME_WAIT, CLOSE, FIN_WAIT）

在 token 层面新增一个维度: CONNSTATE
"""
import re

# conntrack state → 状态分类映射
STATE_MAP = {
    "SYN_SENT": "NEW",
    "SYN_RECV": "NEW",
    "NEW": "NEW",
    "ESTABLISHED": "ESTABLISHED",
    "TIME_WAIT": "CLOSING",
    "CLOSE": "CLOSING",
    "CLOSE_WAIT": "CLOSING",
    "FIN_WAIT": "CLOSING",
    "FIN_WAIT1": "CLOSING",
    "FIN_WAIT2": "CLOSING",
    "LAST_ACK": "CLOSING",
    "CLOSED": "CLOSING",
}

# conntrack 事件类型 → 状态
EVENT_STATE = {
    "NEW": "NEW",
    "UPDATE": "ESTABLISHED",
    "DESTROY": "CLOSING",
}


def classify_state(ct_state="", event_type=""):
    """conntrack state 或事件类型 → 标准状态分类

    Returns: "NEW" / "ESTABLISHED" / "RELATED" / "INVALID" / "CLOSING" / "UNKNOWN"
    """
    if event_type and event_type in EVENT_STATE:
        return EVENT_STATE[event_type]
    if ct_state in STATE_MAP:
        return STATE_MAP[ct_state]
    if ct_state == "RELATED":
        return "RELATED"
    if ct_state == "INVALID":
        return "INVALID"
    return "UNKNOWN"


class ConnectionStateTracker:
    """跟踪连接五元组的状态转移

    用途:
      - NEW 连续出现（同一目标不同端口）= 端口扫描
      - 大量 ESTABLISHED 到同一目标 = 数据传输中
      - INVALID 连接 = 直接 DROP 候选
    """

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.conntrack = {}  # (proto, src, sport, dst, dport) → state
        self.new_counts = {}  # dst_ip → NEW 连接计数（滑动窗口）

    def update(self, proto, src_ip, src_port, dst_ip, dst_port, ct_state="", event_type=""):
        """更新连接状态，返回当前状态分类 + 上下文信息

        Returns:
            dict: {"state": str, "dst_new_count": int, "is_scan_pattern": bool}
        """
        key = (proto, src_ip, src_port, dst_ip, dst_port)
        state = classify_state(ct_state, event_type)

        # 更新连接表
        self.conntrack[key] = state

        # LRU 淘汰
        if len(self.conntrack) > self.max_size:
            oldest = list(self.conntrack.keys())[:self.max_size // 2]
            for k in oldest:
                del self.conntrack[k]

        # NEW 计数：同一目标 IP 的 NEW 连接数
        if state == "NEW":
            self.new_counts[dst_ip] = self.new_counts.get(dst_ip, 0) + 1
        elif state in ("ESTABLISHED", "CLOSING"):
            self.new_counts.pop(dst_ip, None)

        dst_new = self.new_counts.get(dst_ip, 0)
        # 同一目标 5+ 个 NEW 连接 = 扫描模式
        is_scan = dst_new >= 5

        return {
            "state": state,
            "dst_new_count": dst_new,
            "is_scan_pattern": is_scan,
        }

    def stats(self):
        from collections import Counter
        states = Counter(self.conntrack.values())
        return {
            "total_tracked": len(self.conntrack),
            "by_state": dict(states),
            "scan_targets": sum(1 for c in self.new_counts.values() if c >= 5),
        }


if __name__ == "__main__":
    tracker = ConnectionStateTracker()

    # 模拟正常连接
    r = tracker.update("tcp", "198.51.100.1", 50000, "8.8.8.8", 443, "SYN_SENT")
    print(f"正常 SYN: {r}")
    r = tracker.update("tcp", "198.51.100.1", 50000, "8.8.8.8", 443, "ESTABLISHED")
    print(f"正常 EST: {r}")

    # 模拟端口扫描
    for port in [21, 22, 23, 80, 443, 3306, 8080]:
        tracker.update("tcp", "198.51.100.1", 50000 + port, "10.0.0.1", port, "SYN_SENT")
    print(f"扫描后: {tracker.stats()}")

    # INVALID
    r = tracker.update("tcp", "evil", 1234, "198.51.100.1", 22, "INVALID")
    print(f"INVALID: {r}")
    assert r["state"] == "INVALID"
    print("\n状态机测试通过")
