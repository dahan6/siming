#!/usr/bin/env python3
"""Siming M1: 遥测解析器
把 rsyslog 汇聚的 tracee 事件解析为离散 token 流（JSONL）。

每事件 7 个 token: [ET][PROC][ARGV_SKEL][PARENT][UID][DST][DT]
设计依据: docs/检测模型架构设计.md 第 1 节（字段级离散化，不上 VQ-VAE）
"""
import json
import re
import sys
import glob
import os
import ipaddress

BASE64ISH = re.compile(r"^[A-Za-z0-9+/=]{20,}$")
HAS_URL = re.compile(r"https?://")
HAS_IP = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

# LAN classification: RFC1918/link-local addresses count as LAN. Override with
# SIMING_LAN_PREFIX (e.g. "10.0.0.") to restrict LAN to a specific subnet.
LAN_PREFIX = os.environ.get("SIMING_LAN_PREFIX")


def _is_lan(ip):
    if LAN_PREFIX:
        return ip.startswith(LAN_PREFIX)
    try:
        addr = ipaddress.ip_address(ip)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except ValueError:
        return False

# 敏感路径分类：ARGV 骨架(N1P)会抹平"写哪个文件"，PATHCLASS 补回这个维度
# 动机：原型学习中 tee→rc.local / tee→~/.bashrc / tee→ld.so.preload 三个技术不可分
PATH_CLASSES = [
    ("ETC_SYSTEMD", ("/etc/systemd/",)),
    ("ETC_LD", ("/etc/ld.so",)),
    ("ETC_CRON", ("/etc/cron",)),
    ("HOME_RC", (".bashrc", ".bash_profile", ".profile", ".zshrc")),
    ("SSH_KEYS", (".ssh/",)),
    ("TMP", ("/tmp/", "/dev/shm/")),
    ("ETC_PASSWD", ("/etc/passwd", "/etc/shadow", "/etc/sudoers")),
]


def pathclass_token(argv):
    if not isinstance(argv, list):
        return "PC:NONE"
    for a in argv[1:]:
        if not isinstance(a, str):
            continue
        for cls, pats in PATH_CLASSES:
            if any(p in a for p in pats):
                return f"PC:{cls}"
    return "PC:OTHER" if any(isinstance(a, str) and a.startswith("/") for a in argv[1:]) else "PC:NONE"


DT_BUCKETS_MS = [1, 10, 100, 1000, 10_000, 60_000]


def args_to_dict(event):
    return {a["name"]: a.get("value") for a in event.get("args", [])}


def dt_bucket(delta_ms):
    for i, b in enumerate(DT_BUCKETS_MS):
        if delta_ms < b:
            return f"DT{i}"
    return "DT6"


def argv_skeleton(argv):
    if not isinstance(argv, list):
        return "ARGV0"
    n = len(argv) - 1  # 去掉 argv[0]
    nb = "N0" if n <= 0 else "N1" if n == 1 else "N2" if n <= 4 else "N3"
    flags = ""
    rest = argv[1:]
    if any(HAS_URL.search(a) for a in rest):
        flags += "U"
    if any(HAS_IP.search(a) for a in rest):
        flags += "I"
    if any(a.startswith("/") for a in rest):
        flags += "P"
    if any(BASE64ISH.match(a) for a in rest):
        flags += "B"
    return f"ARGV:{nb}{flags or '-'}"


def dst_token(event_name, args):
    if event_name != "security_socket_connect":
        return "DST:NONE"
    remote = str(args.get("remote_addr", ""))
    m = HAS_IP.search(remote)
    if not m:
        return "DST:OTHER"
    ip = m.group(0)
    # 端口分级：知名服务端口 vs 高端口（高端口外联是经典异常信号）
    port = 0
    pm = re.search(r":(\d+)$", remote)
    if pm:
        port = int(pm.group(1))
    pc = "WELL" if port in (22, 53, 80, 443, 514, 123) else "HIGH"
    if _is_lan(ip):
        return f"DST:LAN:{pc}"
    return f"DST:EXT:{pc}"  # external connection from an isolated host = strong signal


def parse_line(line):
    """从 rsyslog 行提取 tracee JSON。返回 (ts_iso, host, event_dict) 或 None。"""
    if "tracee:" not in line:
        return None
    try:
        head, payload = line.split("tracee:", 1)
        host = head.split()[1]
        event = json.loads(payload.strip())
        ts_iso = head.split()[0]
        return ts_iso, host, event
    except (ValueError, IndexError, json.JSONDecodeError):
        return None


def event_to_tokens(event, delta_ms):
    name = event.get("eventName", "unknown")
    args = args_to_dict(event)
    proc = event.get("processName", "?")
    parent = str(args.get("prev_comm") or "?")
    uid = event.get("userId", "?")
    et = {"sched_process_exec": "EXEC", "security_socket_connect": "CONN"}.get(name, name)
    return [
        f"ET:{et}",
        f"PROC:{proc}",
        argv_skeleton(args.get("argv")),
        f"PARENT:{parent}",
        f"UID:{uid}",
        dst_token(name, args),
        dt_bucket(delta_ms),
    ]


def main():
    in_glob = sys.argv[1] if len(sys.argv) > 1 else "/var/log/siming/*/*.log"
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
        "~/siming/detector/data/tokens.jsonl")

    # 收集 (epoch_ns, ts_iso, host, event) 并全局排序，保证 ΔT 正确
    records = []
    for path in sorted(glob.glob(in_glob)):
        with open(path, errors="replace") as f:
            for line in f:
                r = parse_line(line)
                if r:
                    ts_iso, host, ev = r
                    records.append((ev.get("timestamp", 0), ts_iso, host, ev))
    records.sort(key=lambda r: r[0])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, "w") as out:
        prev_ts = None
        for epoch_ns, ts_iso, host, ev in records:
            delta_ms = 0 if prev_ts is None else max(0, (epoch_ns - prev_ts) // 1_000_000)
            prev_ts = epoch_ns
            out.write(json.dumps({
                "ts": ts_iso, "host": host,
                "tokens": event_to_tokens(ev, delta_ms),
            }) + "\n")
            n += 1
    print(f"解析 {n} 个事件 -> {out_path}")


if __name__ == "__main__":
    main()
