#!/usr/bin/env python3
"""墨城防火墙: 网络流采集器

多数据源自动降级采集，自动推断流量方向。

数据源优先级:
  1. conntrack -E -o timestamp -e NEW（实时事件流，最佳）
  2. conntrack -L（快照，次选）
  3. /proc/net/nf_conntrack（无需 root）
  4. ss -tnu（无 conntrack 时的兜底）

方向推断:
  src 是本机 IP → OUT，dst 是本机 IP → IN，都不是 → FWD

用法:
  fw_collect.py [--event] [--interval 2] [--out PATH] [--once] [--duration 300]
"""
import json
import os
import re
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fw_paths import DATA_DIR, LIVE_FLOWS_FILE, EVENT_FLOWS_FILE

# ── 本机 IP 自动检测 ──

def detect_local_ips():
    """获取本机所有 IP 地址（IPv4 + IPv6）"""
    ips = set()
    # 方法 1: socket（基本）
    try:
        hostname = socket.gethostname()
        ips.add(socket.gethostbyname(hostname))
    except (socket.gaierror, OSError):
        pass
    # 方法 2: UDP 连接 trick（不走真实流量，拿出口 IP）
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    # 方法 3: 遍历所有网卡
    try:
        import netifaces
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            for af in (netifaces.AF_INET, netifaces.AF_INET6):
                if af in addrs:
                    for a in addrs[af]:
                        ip = a.get("addr", "").split("%")[0]  # 去掉 scope
                        ips.add(ip)
    except ImportError:
        pass
    # 方法 4: ip addr 命令
    try:
        result = subprocess.run(["ip", "-o", "addr"],
                                capture_output=True, text=True, timeout=5)
        for line in result.stdout.split("\n"):
            parts = line.split()
            if len(parts) >= 4 and parts[2] in ("inet", "inet6"):
                ip = parts[3].split("/")[0]
                ips.add(ip)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    ips.discard("127.0.0.1")
    ips.discard("::1")
    return ips


# ── 方向推断 ──

def infer_direction(src_ip, dst_ip, local_ips):
    """根据本机 IP 列表推断流量方向"""
    src_is_local = src_ip in local_ips
    dst_is_local = dst_ip in local_ips
    if src_is_local and not dst_is_local:
        return "out"
    if dst_is_local and not src_is_local:
        return "in"
    if src_is_local and dst_is_local:
        return "loopback"
    return "fwd"  # 转发流量


# ── 正则 ──

# conntrack -L 文本: proto num timeout [state] src=... dst=... sport=... dport=...
CT_LIST_RE = re.compile(
    r"(?P<proto>\w+)\s+\d+\s+\d+\s+(?P<state>\w+)?\s*"
    r"src=(?P<src_ip>\S+)\s+dst=(?P<dst_ip>\S+)\s+"
    r"sport=(?P<src_port>\d+)\s+dport=(?P<dst_port>\d+)"
)
# /proc/net/nf_conntrack: family num proto num timeout [state] src=... dst=...
CT_PROC_RE = re.compile(
    r"\w+\s+\d+\s+(?P<proto>\w+)\s+\d+\s+\d+\s+(?P<state>\w+)?\s*"
    r"src=(?P<src_ip>\S+)\s+dst=(?P<dst_ip>\S+)\s+"
    r"sport=(?P<src_port>\d+)\s+dport=(?P<dst_port>\d+)"
)
# conntrack -E 事件: [ts] [event] proto num timeout state src=... dst=...
CT_EVENT_RE = re.compile(
    r"\[(?P<ts>[\d.]+)\]\s+\[?(?P<event>\w+)\]?\s+"
    r"(?P<proto>\w+)\s+\d+\s+\d+\s+(?P<state>\w+)?\s*"
    r"src=(?P<src_ip>\S+)\s+dst=(?P<dst_ip>\S+)\s+"
    r"sport=(?P<src_port>\d+)\s+dport=(?P<dst_port>\d+)"
)
CT_BYTES_RE = re.compile(r"bytes=(?P<bytes>\d+)")

# ss -tnu 输出: State Recv-Q Send-Q Local Address:Port Peer Address:Port
SS_RE = re.compile(
    r"(?P<state>\w+)\s+\d+\s+\d+\s+"
    r"\[?(?P<local_ip>[\d.:a-fA-F]+)\]?: (?P<local_port>\d+)\s+"
    r"\[?(?P<peer_ip>[\d.:a-fA-F]+)\]?: (?P<peer_port>\d+)"
)


def make_flow(ts_epoch, proto, src_ip, dst_ip, src_port, dst_port,
              n_bytes=0, state="", local_ips=None):
    """构造一条 flow 事件 dict"""
    direction = infer_direction(src_ip, dst_ip, local_ips or set())
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts_epoch)),
        "ts_epoch": ts_epoch,
        "ts_sort": ts_epoch,
        "proto": proto,
        "dir": direction,
        "src_ip": src_ip,
        "src_port": int(src_port),
        "dst_ip": dst_ip,
        "dst_port": int(dst_port),
        "bytes": int(n_bytes),
        "action": "accept",
        "host": socket.gethostname(),
        "state": state,
    }


# ── 数据源: conntrack -L（快照）──

def read_conntrack_L(local_ips):
    """conntrack -L 快照模式"""
    try:
        result = subprocess.run(["conntrack", "-L"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return [], "conntrack -L 失败: " + result.stderr.strip()[:100]
    except FileNotFoundError:
        return [], "conntrack 命令不存在"
    except (subprocess.TimeoutExpired, PermissionError) as e:
        return [], f"conntrack -L 权限/超时: {e}"
    flows = []
    for line in result.stdout.split("\n"):
        m = CT_LIST_RE.search(line)
        if not m:
            continue
        bm = CT_BYTES_RE.search(line)
        flows.append(make_flow(
            time.time(), m.group("proto"),
            m.group("src_ip"), m.group("dst_ip"),
            m.group("src_port"), m.group("dst_port"),
            bm.group("bytes") if bm else 0,
            m.group("state") or "", local_ips))
    return flows, f"conntrack -L ({len(flows)} 条)"


# ── 数据源: /proc/net/nf_conntrack ──

def read_proc_conntrack(local_ips):
    """/proc/net/nf_conntrack（无需 root）"""
    try:
        with open("/proc/net/nf_conntrack") as f:
            lines = f.readlines()
    except (FileNotFoundError, PermissionError) as e:
        return [], f"/proc/net/nf_conntrack 不可用: {e}"
    flows = []
    for line in lines:
        m = CT_PROC_RE.search(line)
        if not m:
            continue
        bm = CT_BYTES_RE.search(line)
        flows.append(make_flow(
            time.time(), m.group("proto"),
            m.group("src_ip"), m.group("dst_ip"),
            m.group("src_port"), m.group("dst_port"),
            bm.group("bytes") if bm else 0,
            m.group("state") or "", local_ips))
    return flows, f"/proc/net/nf_conntrack ({len(flows)} 条)"


# ── 数据源: ss（兜底，无 conntrack 时）──

def read_ss(local_ips):
    """ss -tnu（TCP/UDP 连接表，无需额外安装）"""
    try:
        result = subprocess.run(["ss", "-tnu"],
                                capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return [], f"ss 不可用: {e}"
    flows = []
    for line in result.stdout.split("\n")[1:]:  # 跳过表头
        m = SS_RE.search(line)
        if not m:
            continue
        local_ip = m.group("local_ip").strip("[]")
        peer_ip = m.group("peer_ip").strip("[]")
        proto = "tcp" if "tcp" in line.lower() else "udp"
        flows.append(make_flow(
            time.time(), proto,
            local_ip, peer_ip,
            m.group("local_port"), m.group("peer_port"),
            0, m.group("state"), local_ips))
    return flows, f"ss -tnu ({len(flows)} 条)"


# ── 数据源选择 ──

def collect_snapshot(local_ips):
    """尝试所有快照数据源，返回 (flows, source_info)"""
    for reader in [read_conntrack_L, read_proc_conntrack, read_ss]:
        flows, info = reader(local_ips)
        if flows:
            return flows, info
        print(f"  [skip] {info}")
    return [], "所有数据源均不可用"


def collect_events(duration, out_path, local_ips):
    """conntrack -E 实时事件流采集"""
    try:
        proc = subprocess.Popen(
            ["conntrack", "-E", "-o", "timestamp", "-e", "NEW"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1)
    except FileNotFoundError:
        print("[WARN] conntrack 命令不存在，回退到快照模式")
        return collect_snapshot(local_ips)

    print(f"conntrack -E 事件流采集 {duration}s ...")
    n = 0
    out_path = out_path or EVENT_FLOWS_FILE
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    start = time.time()
    with open(out_path, "w") as f:
        for line in proc.stdout:
            if time.time() - start > duration:
                break
            m = CT_EVENT_RE.search(line)
            if not m:
                continue
            ts = float(m.group("ts"))
            bm = CT_BYTES_RE.search(line)
            flow = make_flow(
                ts, m.group("proto"),
                m.group("src_ip"), m.group("dst_ip"),
                m.group("src_port"), m.group("dst_port"),
                bm.group("bytes") if bm else 0,
                m.group("state") or "", local_ips)
            f.write(json.dumps(flow, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 200 == 0:
                print(f"  [{time.time()-start:.0f}s] {n} NEW 事件")
    proc.kill()
    print(f"采集完成: {n} 条 -> {out_path}")
    return n


# ── 主函数 ──

def main():
    import argparse
    ap = argparse.ArgumentParser(description="墨城防火墙流量采集器")
    ap.add_argument("--out", default=None, help="输出文件路径")
    ap.add_argument("--once", action="store_true", help="单次快照后退出")
    ap.add_argument("--event", action="store_true", help="使用 conntrack -E 事件流模式")
    ap.add_argument("--duration", type=int, default=300, help="事件流采集时长（秒）")
    ap.add_argument("--interval", type=float, default=5.0, help="快照采样间隔")
    ap.add_argument("--max-flows", type=int, default=100000, help="最大采集条数")
    args = ap.parse_args()

    out_path = args.out or LIVE_FLOWS_FILE
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 检测本机 IP
    local_ips = detect_local_ips()
    print(f"本机 IP: {', '.join(sorted(local_ips)) if local_ips else '(未检测到)'}")

    # 事件流模式
    if args.event:
        n = collect_events(args.duration, out_path, local_ips)
        print(f"事件流采集完成: {n} 条 -> {out_path}")
        return

    # 快照模式
    total = 0
    seen_keys = set()
    while True:
        flows, info = collect_snapshot(local_ips)
        if not flows:
            print(f"[WARN] {info}")
            print("尝试: sudo modprobe nf_conntrack && sudo apt install conntrack")
            if args.once:
                return
            time.sleep(args.interval)
            continue

        n_new = 0
        with open(out_path, "a") as f:
            for fl in flows:
                key = (fl["proto"], fl["src_ip"], fl["src_port"],
                       fl["dst_ip"], fl["dst_port"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                f.write(json.dumps(fl, ensure_ascii=False) + "\n")
                n_new += 1
                total += 1

        ts = time.strftime("%F %T")
        print(f"[{ts}] [{info}] 新增 {n_new}, 累计 {total}")
        if args.once or total >= args.max_flows:
            break
        time.sleep(args.interval)

    print(f"采集完成: {total} 条 -> {out_path}")


if __name__ == "__main__":
    main()
