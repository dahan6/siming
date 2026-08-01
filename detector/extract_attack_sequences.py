#!/usr/bin/env python3
"""从遥测数据中提取攻击行为序列（通用版）

给定一个 token 事件流（JSONL），按行为模式分离候选攻击序列。
可用于：
  - 从实验遥测中提取已知攻击的行为序列
  - 从 tracee 采集数据中分离异常子序列
  - 为原型训练提供标注样本

分离策略（参数可调）：
  1. 时间窗：事件密度异常的时段
  2. 进程特征：PARENT:sh/bash + 非系统子进程
  3. 命令集：按已知攻击原语分类（recon/probe/persist/exfil/privesc）
  4. 序列切分：按间隔窗口（默认 30s）分组，≥3 个动作 = 一个序列

用法:
  # 从任意 token JSONL 提取
  python3 extract_attack_sequences.py input.jsonl --window 30 --min-actions 3

  # 从多个文件合并
  python3 extract_attack_sequences.py file1.jsonl file2.jsonl --output sequences.jsonl
"""
import json
import os
import sys
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# 动作原语映射（可扩展）
ACTION_PRIMITIVES = {
    "probe":      {"cat", "head", "tail", "less", "more", "readlink"},
    "recon":      {"ss", "ip", "ifconfig", "netstat", "ps", "ls", "find",
                   "grep", "env", "id", "whoami", "uname", "hostname",
                   "uptime", "free", "df", "wc", "du", "sort", "sed",
                   "journalctl", "pgrep", "date"},
    "persist":    {"crontab", "at", "systemctl", "setsid", "nohup"},
    "communicate":{"curl", "wget", "nc", "ncat", "python3", "python", "logger"},
    "propagate":  {"ping", "nmap", "ssh", "scp", "sftp"},
    "exfil":      {"tar", "gzip", "zip", "base64"},
    "privesc":    {"sudo", "su", "pkexec"},
    "evasion":    {"rm", "mv", "chmod", "chown"},
    "impact":     {"kill", "pkill"},
}

ALL_ACTION_CMDS = set()
for cmds in ACTION_PRIMITIVES.values():
    ALL_ACTION_CMDS.update(cmds)

# 默认排除的系统背景进程
SYSTEM_PROCS = {
    "systemd", "snapd", "sleep", "dbus-daemon", "rsyslogd", "cron", "atd",
    "sshd", "agetty", "login", "systemd-journal", "systemd-udevd",
    "systemd-resolved", "systemd-networkd", "systemd-timesyncd",
    "systemd-logind", "polkitd", "accounts-daemon", "networkd-dispat",
    "unattended-upgr", "apt", "dpkg", "snap", "fwupd", "ModemManager",
    "avahi-daemon", "cupsd", "wpa_supplicant", "irqbalance", "multipathd",
    "packagekitd", "udisks2", "colord", "update-motd-fsc",
    "anacron", "logrotate", "run-parts", "cloud-id", "apt-helper",
    "debian-sa1", "sftp-server",
}

SYSTEM_PARENTS = {
    "systemd", "cron", "atd", "rsyslogd", "snapd", "unattended-upgr",
    "apt", "dpkg", "run-parts", "anacron", "logrotate", "update-motd-fsc",
    "cloud-id", "apt-helper", "sudo",
}


def parse_tokens(line):
    """解析 JSONL 行（兼容 7-token 和 8-token）"""
    e = json.loads(line)
    toks = e["tokens"]
    d = {"ts": e["ts"], "host": e.get("host", "?")}
    for t in toks:
        if t.startswith("DT") and ":" not in t:
            d["DT"] = t[2:]
        elif ":" in t:
            slot, val = t.split(":", 1)
            d[slot] = val
    return d


def classify_action(proc):
    """将命令映射到动作原语"""
    for primitive, cmds in ACTION_PRIMITIVES.items():
        if proc in cmds:
            return primitive
    return "unknown"


def infer_pc(ev):
    """从 ARGV 推断 PC 槽"""
    proc = ev.get("PROC", "")
    argv = ev.get("ARGV", "")
    if proc in ("crontab", "at"): return "PC:ETC_CRON"
    if proc == "systemctl": return "PC:ETC_SYSTEMD"
    if proc in ("cat", "head", "tail") and "passwd" in argv: return "PC:ETC_PASSWD"
    if proc in ("ssh", "scp", "sftp"): return "PC:SSH_KEYS"
    if proc in ("tar", "gzip", "zip"): return "PC:TMP"
    return "PC:NONE"


def to_8token(ev):
    """转为标准 8-token 格式"""
    pc = infer_pc(ev)
    return [
        f"ET:{ev.get('ET','EXEC')}",
        f"PROC:{ev.get('PROC','?')}",
        f"ARGV:{ev.get('ARGV','0')}",
        f"PARENT:{ev.get('PARENT','?')}",
        f"UID:{ev.get('UID','0')}",
        f"DST:{ev.get('DST','NONE')}",
        pc,
        f"DT{ev.get('DT','0')}",
    ]


def extract_sequences(events, window_s=30, min_actions=3):
    """从事件流中提取行为序列"""
    candidates = []
    for ev in events:
        parent = ev.get("PARENT", "")
        uid = ev.get("UID", "")
        proc = ev.get("PROC", "")
        if parent in ("sh", "bash", "(sh)", "(bash)") and proc not in SYSTEM_PROCS:
            ev["_primitive"] = classify_action(proc)
            candidates.append(ev)

    if not candidates:
        return []

    sequences = []
    current_seq = [candidates[0]]
    for i in range(1, len(candidates)):
        t1 = datetime.fromisoformat(candidates[i-1]["ts"])
        t2 = datetime.fromisoformat(candidates[i]["ts"])
        gap = (t2 - t1).total_seconds()
        if gap > window_s:
            if len(current_seq) >= min_actions:
                sequences.append(current_seq)
            current_seq = [candidates[i]]
        else:
            current_seq.append(candidates[i])
    if len(current_seq) >= min_actions:
        sequences.append(current_seq)

    return sequences


def main():
    ap = argparse.ArgumentParser(description="攻击行为序列提取器（通用版）")
    ap.add_argument("inputs", nargs="+", help="输入 JSONL 文件")
    ap.add_argument("--window", type=int, default=30, help="序列切分窗口（秒）")
    ap.add_argument("--min-actions", type=int, default=3, help="最少动作数")
    ap.add_argument("--output", default="attack_sequences.jsonl")
    ap.add_argument("--exclude-hours", default=None,
                    help="排除的小时（逗号分隔，如 13,14,15）")
    a = ap.parse_args()

    exclude = set(a.exclude_hours.split(",")) if a.exclude_hours else set()

    all_events = []
    for path in a.inputs:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                ev = parse_tokens(line)
                if exclude and ev["ts"][11:13] in exclude:
                    continue
                all_events.append(ev)
            except Exception:
                continue

    all_events.sort(key=lambda e: e["ts"])
    print(f"总事件: {len(all_events)}")

    sequences = extract_sequences(all_events, a.window, a.min_actions)
    print(f"提取序列: {len(sequences)}")

    n_written = 0
    with open(a.output, "w") as f:
        for seq in sequences:
            tokens_8 = [to_8token(ev) for ev in seq]
            prims = [ev.get("_primitive", "unknown") for ev in seq]
            label = Counter(prims).most_common(1)[0][0]
            entry = {
                "type": "sequence",
                "label": label,
                "host": seq[0].get("host", "?"),
                "ts_start": seq[0]["ts"],
                "ts_end": seq[-1]["ts"],
                "n_events": len(seq),
                "tokens": tokens_8,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"输出: {a.output} ({n_written} 序列)")


if __name__ == "__main__":
    main()
