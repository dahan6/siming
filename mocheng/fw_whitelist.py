#!/usr/bin/env python3
"""墨城防火墙: 持久白名单匹配

加载 fw_whitelist.jsonl，对 IP/CIDR 做快速匹配。
匹配白名单的事件跳过 P1-P3（只保留 P0 强模式命中）。

白名单格式（每行一条）:
  {"ip": "10.0.0.0/8", "reason": "内网 LAN"}
  {"ip": "198.51.100.1", "reason": "本地 DNS/网关"}
  以 # 开头的行为注释，跳过。

用法:
  from fw_whitelist import Whitelist
  wl = Whitelist()          # 加载默认 fw_whitelist.jsonl
  wl.load("other.jsonl")    # 或指定路径
  if wl.matches("10.0.1.5"):
      ...
"""
import ipaddress
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fw_paths import BASE_DIR

DEFAULT_PATH = os.path.join(BASE_DIR, "fw_whitelist.jsonl")


class Whitelist:
    """IP/CIDR 白名单匹配器"""

    def __init__(self, path=None):
        self.path = path or DEFAULT_PATH
        self._networks = []   # list[ipaddress.IPv4Network | IPv6Network]
        self._reasons = []
        self.load(self.path)

    def load(self, path):
        """加载白名单文件（JSONL，支持注释行）"""
        self._networks = []
        self._reasons = []
        if not path or not os.path.exists(path):
            return
        for line in open(path, errors="replace"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # 纯 IP 行
                entry = {"ip": line}
            ip = entry.get("ip", "").strip()
            if not ip:
                continue
            try:
                # 单 IP → 按版本确定前缀（IPv4=/32, IPv6=/128）
                if "/" not in ip:
                    addr = ipaddress.ip_address(ip)
                    prefix = "32" if addr.version == 4 else "128"
                    net = ipaddress.ip_network(f"{ip}/{prefix}", strict=False)
                else:
                    net = ipaddress.ip_network(ip, strict=False)
                self._networks.append(net)
                self._reasons.append(entry.get("reason", ""))
            except ValueError:
                pass  # 无效 IP，跳过

    def matches(self, ip):
        """IP 是否在白名单中（返回 True/False）"""
        if not ip or not self._networks:
            return False
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return False
        for net in self._networks:
            # 只比较同版本（IPv4 vs IPv6）
            if net.version == addr.version and addr in net:
                return True
        return False

    def matches_any(self, *ips):
        """多个 IP 中任一匹配则返回 True"""
        for ip in ips:
            if self.matches(ip):
                return True
        return False

    def stats(self):
        return {"entries": len(self._networks), "path": self.path}


if __name__ == "__main__":
    wl = Whitelist()
    print(f"白名单: {wl.stats()}")
    for net, reason in zip(wl._networks, wl._reasons):
        print(f"  {net}  {reason}")
    # 测试
    for test_ip in ["127.0.0.1", "10.0.1.5", "198.51.100.100", "8.8.8.8"]:
        print(f"  {test_ip}: {'白名单' if wl.matches(test_ip) else '不在白名单'}")
