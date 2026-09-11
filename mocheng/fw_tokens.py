#!/usr/bin/env python3
"""墨城防火墙 M1: 网络流事件 → 7-token 离散化

每事件 7 个 token: [PROTO][DIR][SRCNET][DSTNET][PORTCLS][SIZECLS][DT]

支持 IPv4/IPv6、Docker/K8s 网络段、CGNAT、link-local。
"""
import json
import re
import sys
import os
import ipaddress

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fw_paths import DATA_DIR, FLOWS_FILE, TOKENS_FILE

HAS_IP = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


def net_class(ip):
    """IP → 网络分类 token 值（支持 IPv4 + IPv6）"""
    if not ip or ip == "?" or ip == "*":
        return "UNKNOWN"
    # ── IPv6 ──
    if ":" in ip:
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return "UNKNOWN"
        if addr.is_loopback:
            return "LOOPBACK"
        if addr.is_link_local:
            return "LINKLOCAL"
        if addr.is_private:  # ULA fc00::/7, 等
            return "LAN"
        if addr.is_multicast:
            return "MULTICAST"
        return "EXT"
    # ── IPv4 ──
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return "UNKNOWN"
    if addr.is_loopback:
        return "LOOPBACK"
    if addr.is_link_local:  # 169.254.0.0/16（AWS metadata 等）
        return "LINKLOCAL"
    if addr.is_multicast:  # 224.0.0.0/4
        return "MULTICAST"
    # CGNAT 100.64.0.0/10
    if ip.startswith("100.64.") or ip.startswith("100.65.") or \
       ip.startswith("100.66.") or ip.startswith("100.67.") or \
       ip.startswith("100.68.") or ip.startswith("100.69.") or \
       ip.startswith("100.7") or ip.startswith("100.8") or \
       ip.startswith("100.9") or ip.startswith("100.10") or \
       ip.startswith("100.11") or ip.startswith("100.12"):
        return "LAN"
    # 配置化的 VPN 检测
    for prefix in VPN_PREFIXES:
        if ip.startswith(prefix):
            return "VPN"
    if addr.is_private:
        return "LAN"
    return "EXT"


# VPN 网段（可被 fw_config.toml 覆盖）
VPN_PREFIXES = ["10.100.", "10.200."]


def set_vpn_prefixes(prefixes):
    """从配置文件更新 VPN 网段"""
    global VPN_PREFIXES
    VPN_PREFIXES = prefixes


# ── 端口/服务分类 ──
PORT_MAP = {
    20: "FTP", 21: "FTP", 22: "SSH", 23: "TELNET",
    25: "SMTP", 53: "DNS", 67: "DHCP", 68: "DHCP",
    80: "HTTP", 110: "POP3", 123: "NTP", 135: "RPC",
    139: "SMB", 143: "IMAP", 161: "SNMP", 389: "LDAP",
    443: "HTTPS", 445: "SMB", 465: "SMTPS", 587: "SMTP",
    636: "LDAPS", 993: "IMAPS", 995: "POP3S",
    1433: "MSSQL", 1521: "ORACLE", 2049: "NFS",
    2375: "DOCKER", 2376: "DOCKER",  # Docker API
    3000: "WEBAPP", 3306: "MYSQL", 3389: "RDP", 5432: "POSTGRES",
    5601: "KIBANA", 5900: "VNC", 6379: "REDIS",
    6443: "K8S",  # Kubernetes API
    8080: "HTTPALT", 8443: "HTTPSALT", 9000: "WEBAPP",
    9090: "PROMETHEUS", 9200: "ES", 9300: "ES",
    11211: "MEMCACHE", 27017: "MONGO",
}


def port_class(port):
    """端口 → 服务分类 token 值"""
    port = int(port) if port else 0
    if port in PORT_MAP:
        return PORT_MAP[port]
    if port == 0:
        return "NONE"
    if port < 1024:
        return "WELLKNOWN"
    if port < 49152:
        return "HIGHPORT"
    return "EPHEMERAL"


# ── 流量大小分类 ──
SIZE_BUCKETS = [1, 200, 5_000, 100_000]  # ZERO=0B, S<200, M<5K, L<100K, XL>=100K


def size_class(n_bytes):
    """字节 → 大小分类"""
    n = int(n_bytes) if n_bytes else 0
    if n == 0:
        return "ZERO"  # SYN-only / 空探测
    for i, b in enumerate(SIZE_BUCKETS[1:]):  # 跳过第一个(1)
        if n < b:
            return ["S", "M", "L"][i]
    return "XL"


# ── 时间间隔桶 ──
DT_BUCKETS_MS = [1, 10, 100, 1000, 10_000, 60_000]


def dt_bucket(delta_ms):
    """毫秒间隔 → DT0~DT6 桶"""
    d = int(delta_ms) if delta_ms else 0
    for i, b in enumerate(DT_BUCKETS_MS):
        if d < b:
            return f"DT{i}"
    return "DT6"


def flow_to_tokens(flow, delta_ms):
    """一条网络流事件 → 7 个 token"""
    proto = str(flow.get("proto", "?")).upper()
    if proto not in ("TCP", "UDP", "ICMP"):
        proto = "OTHER"
    direction = str(flow.get("dir", "?")).upper()
    if direction not in ("IN", "OUT", "FWD", "LOOPBACK"):
        direction = "UNKNOWN"
    src = net_class(str(flow.get("src_ip", "?")))
    dst = net_class(str(flow.get("dst_ip", "?")))
    port = flow.get("dst_port", flow.get("src_port", 0))
    svc = port_class(port)
    sz = size_class(flow.get("bytes", 0))
    dt = dt_bucket(delta_ms)
    return [
        f"PROTO:{proto}",
        f"DIR:{direction}",
        f"SRCNET:{src}",
        f"DSTNET:{dst}",
        f"PORTCLS:{svc}",
        f"SIZECLS:{sz}",
        dt,
    ]


def parse_flow_line(line):
    """从 JSON 行解析网络流事件"""
    line = line.strip()
    if not line or not line.startswith("{"):
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else FLOWS_FILE
    out_path = sys.argv[2] if len(sys.argv) > 2 else TOKENS_FILE

    records = []
    for line in open(in_path, errors="replace"):
        flow = parse_flow_line(line)
        if flow:
            ts_sort = flow.get("ts_sort", flow.get("ts_epoch", flow.get("ts", "")))
            records.append((ts_sort, flow))
    records.sort(key=lambda r: r[0])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    prev_ts_epoch = None
    with open(out_path, "w") as out:
        for ts_sort, flow in records:
            ts_epoch = flow.get("ts_epoch", 0)
            if prev_ts_epoch and ts_epoch:
                delta = max(0, int((ts_epoch - prev_ts_epoch) * 1000))
            else:
                delta = flow.get("delta_ms", 0)
            prev_ts_epoch = ts_epoch or None
            tokens = flow_to_tokens(flow, delta)
            out.write(json.dumps({
                "ts": flow.get("ts", ""),
                "host": flow.get("host", "fw"),
                "tokens": tokens,
                "raw": {k: flow.get(k) for k in
                        ("src_ip", "dst_ip", "dst_port", "proto", "bytes", "dir")},
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"解析 {n} 个网络流事件 -> {out_path}")


if __name__ == "__main__":
    main()
