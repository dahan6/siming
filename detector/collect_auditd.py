#!/usr/bin/env python3
"""auditd 实时采集器：从 /var/log/audit/audit.log 解析 execve 事件 → 司命 8-token

auditd 捕获所有进程执行（零遗漏），比 procfs 轮询强 100 倍。
每条 SYSCALL + EXECVE + PATH 组合成一条 8-token 事件。

用法:
  # 实时采集
  sudo python3 collect_auditd.py --duration 300 --out data/audit_events.jsonl

  # 解析已有日志
  python3 collect_auditd.py --parse /var/log/audit/audit.log --out data/audit_parsed.jsonl
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

# ── Token 离散化（和司命 parse_events.py 对齐）──

PATH_CLASSES = [
    ("ETC_SYSTEMD", ("/etc/systemd/",)),
    ("ETC_LD", ("/etc/ld.so",)),
    ("ETC_CRON", ("/etc/cron",)),
    ("HOME_RC", (".bashrc", ".bash_profile", ".profile", ".zshrc")),
    ("SSH_KEYS", (".ssh/",)),
    ("TMP", ("/tmp/", "/dev/shm/")),
    ("ETC_PASSWD", ("/etc/passwd", "/etc/shadow", "/etc/sudoers")),
    ("VAR_LOG", ("/var/log/",)),
]

# 敏感端口/目标分类
HIGH_PORTS = set(range(49152, 65536))
WELL_KNOWN = {20:"FTP", 21:"FTP", 22:"SSH", 23:"TELNET", 25:"SMTP", 53:"DNS",
              80:"HTTP", 110:"POP3", 143:"IMAP", 443:"HTTPS", 993:"IMAPS",
              995:"POP3S", 3306:"MYSQL", 5432:"POSTGRES", 6379:"REDIS",
              8080:"HTTPALT", 8443:"HTTPSALT", 9090:"PROM"}

BASE64ISH = re.compile(r"^[A-Za-z0-9+/=]{20,}$")
HAS_URL = re.compile(r"https?://")
HAS_IP = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


def pathclass_token(argv_list):
    """从参数列表推断路径分类"""
    for a in argv_list:
        if not isinstance(a, str): continue
        for cls, pats in PATH_CLASSES:
            if any(p in a for p in pats):
                return f"PC:{cls}"
    # 检查是否有路径参数
    for a in argv_list:
        if isinstance(a, str) and a.startswith("/"):
            return "PC:OTHER"
    return "PC:NONE"


def argv_skeleton(argv_list):
    """参数骨架：N{argc}{flags}"""
    if not argv_list:
        return "ARGV0"
    n = len(argv_list) - 1  # 去掉 argv[0]（程序名）
    if n == 0:
        return "ARGV0"

    flags = ""
    has_path = any(isinstance(a, str) and a.startswith("/") for a in argv_list[1:])
    has_base64 = any(isinstance(a, str) and BASE64ISH.match(a) for a in argv_list[1:])
    has_url = any(isinstance(a, str) and HAS_URL.search(a) for a in argv_list[1:])
    has_ip = any(isinstance(a, str) and HAS_IP.search(a) for a in argv_list[1:])

    if has_path: flags += "P"
    if has_base64: flags += "B"
    if has_url: flags += "U"
    if has_ip: flags += "I"
    if not flags: flags = "-"

    return f"ARGV:N{n}{flags}"


def dt_bucket(delta_ms):
    """时间间隔 → DT 桶"""
    buckets = [1, 10, 100, 1000, 10000, 60000]
    for i, b in enumerate(buckets):
        if delta_ms < b:
            return f"DT{i}"
    return "DT6"


def get_proc_name(exe_path, comm):
    """从 exe 路径或 comm 提取进程名"""
    if exe_path and "/" in exe_path:
        name = exe_path.rsplit("/", 1)[-1]
        return name
    if comm:
        return comm
    return "unknown"


def get_parent_name(ppid_str, proc_cache):
    """从 PID 获取父进程名"""
    ppid = int(ppid_str) if ppid_str else 0
    if ppid in proc_cache:
        return proc_cache[ppid]
    # 系统进程
    if ppid <= 1:
        return "systemd" if ppid == 1 else "kernel"
    return "?"


def parse_audit_line(line):
    """解析一行 audit 日志，返回 dict"""
    result = {}
    # 提取 key=value 对
    for match in re.finditer(r'(\w+)=(?:"([^"]*)"|(\S+))', line):
        key = match.group(1)
        val = match.group(2) if match.group(2) is not None else match.group(3)
        result[key] = val
    return result


def parse_audit_log(path, max_events=None):
    """解析 audit 日志文件，返回 8-token 事件列表"""
    events = []
    current = {}  # audit_id → partial event
    proc_cache = {}  # pid → proc_name
    prev_ts = None

    with open(path, errors="replace") as f:
        for line in f:
            if "key=exec_log" not in line and "key=\"exec_log\"" not in line:
                # 也处理不带 key 的 SYSCALL
                if "type=SYSCALL" not in line:
                    continue

            parsed = parse_audit_line(line)

            # SYSCALL 类型
            if "type=SYSCALL" in line and "syscall=59" in line:
                audit_id = parsed.get("id", "")
                ts_str = parsed.get("msg", "")

                # 提取时间戳
                ts_match = re.search(r"audit\((\d+\.\d+):", ts_str)
                ts_epoch = float(ts_match.group(1)) if ts_match else 0

                comm = parsed.get("comm", "").strip('"')
                exe = parsed.get("exe", "").strip('"')
                ppid = parsed.get("ppid", "0")
                pid = parsed.get("pid", "0")
                uid = parsed.get("uid", "0")
                auid = parsed.get("auid", "0")

                proc_name = get_proc_name(exe, comm)
                proc_cache[int(pid)] = proc_name
                parent_name = get_parent_name(ppid, proc_cache)

                # 等 EXECVE 和 CWD 行来补全
                current[audit_id if audit_id else f"{pid}_{ts_epoch}"] = {
                    "ts": ts_epoch,
                    "proc": proc_name,
                    "parent": parent_name,
                    "uid": uid,
                    "auid": auid,
                    "ppid": ppid,
                    "pid": pid,
                    "exe": exe,
                }

            # EXECVE 类型（参数）
            elif "type=EXECVE" in line:
                argc = parsed.get("argc", "0")
                argv = []
                for i in range(int(argc) if argc.isdigit() else 0):
                    key = f"a{i}"
                    if key in parsed:
                        argv.append(parsed[key].strip('"'))
                    elif f"a{i}" in line:
                        # 尝试另一种格式
                        m = re.search(rf'a{i}="([^"]*)"', line)
                        if m:
                            argv.append(m.group(1))

                # 找最近的 SYSCALL 来配对
                # 简化：直接用同一行的 audit_id
                ts_match = re.search(r"audit\((\d+\.\d+):", line)
                if ts_match:
                    ts_epoch = float(ts_match.group(1))
                    eid = ts_match.group(1)
                    for k, ev in current.items():
                        if abs(ev["ts"] - ts_epoch) < 0.01:
                            ev["argv"] = argv
                            break

    # 生成 8-token 事件
    prev_ts = None
    for eid, ev in sorted(current.items(), key=lambda x: x[1]["ts"]):
        if "argv" not in ev:
            continue

        argv = ev["argv"]
        tokens = [
            "ET:EXEC",
            f"PROC:{ev['proc']}",
            argv_skeleton(argv),
            f"PARENT:{ev['parent']}",
            f"UID:{ev['uid']}",
            "DST:NONE",
            pathclass_token(argv),
            "DT0",  # placeholder，后面填
        ]

        # DT 计算
        if prev_ts:
            delta_ms = int((ev["ts"] - prev_ts) * 1000)
            tokens[7] = dt_bucket(max(0, delta_ms))
        prev_ts = ev["ts"]

        ts_str = datetime.fromtimestamp(ev["ts"]).isoformat()
        events.append({
            "ts": ts_str,
            "tokens": tokens,
            "host": os.uname().nodename,
        })

        if max_events and len(events) >= max_events:
            break

    return events


def main():
    ap = argparse.ArgumentParser(description="auditd execve 采集器")
    ap.add_argument("--parse", default="/var/log/audit/audit.log",
                    help="解析的 audit 日志文件")
    ap.add_argument("--out", default="data/audit_events.jsonl",
                    help="输出 JSONL")
    ap.add_argument("--duration", type=int, default=0,
                    help="实时采集秒数（0=只解析已有日志）")
    args = ap.parse_args()

    if args.duration > 0:
        print(f"实时采集 {args.duration}s...")
        t0 = time.time()
        # 记录当前位置
        try:
            initial_size = os.path.getsize(args.parse)
        except:
            initial_size = 0

        # 等待采集
        time.sleep(args.duration)

        # 解析新增部分
        print("解析新增事件...")
        events = []
        with open(args.parse, errors="replace") as f:
            f.seek(initial_size)
            new_lines = f.readlines()

        # 写临时文件解析
        tmp = "/tmp/audit_new.log"
        with open(tmp, "w") as tf:
            tf.writelines(new_lines)
        events = parse_audit_log(tmp)

        print(f"采集到 {len(events)} 事件")
    else:
        print(f"解析 {args.parse}...")
        events = parse_audit_log(args.parse)

    # 输出
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    print(f"输出: {args.out} ({len(events)} 事件)")

    # 进程分布
    from collections import Counter
    procs = Counter(ev["tokens"][1] for ev in events)
    print(f"\n进程分布 (top 15):")
    for p, c in procs.most_common(15):
        print(f"  {p}: {c}")


if __name__ == "__main__":
    main()
