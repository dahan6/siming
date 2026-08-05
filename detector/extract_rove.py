#!/usr/bin/env python3
"""从隐翅虫遥测提取训练数据。

数据源（只读）：
  ~/adaptive-agent-sim/checkpoints/clone_events.jsonl  (101K, 7-token)
  ~/adaptive-agent-sim/checkpoints/regime_events.jsonl  (79K, 7-token)

输出：
  rove_sequences.jsonl  — 隐翅虫行为序列（8-token，补 PC 槽）
  rove_stats.json       — 提取统计

分离策略：
  隐翅虫用 sh -c 间接执行，PARENT 是 sh/bash，UID:1000（range 用户）。
  但正常管理操作也用 bash。分离依据：
  1. 时间窗：实验高峰期（13:00-14:30）事件密度异常
  2. 节奏：bee 步进 lognormal(μ=0.9,σ=0.7)，CV≈0.5-0.7
  3. 动作组合：bee 的 12 原语映射到特定命令集
  4. 序列模式：recon→persist→communicate 的任务链
"""
import json
import sys
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# 隐翅虫的动作原语映射（从 v4 文档 §3 提取）
# 12 原语 → 命令集
ROVE_PRIMITIVES = {
    "probe":      {"cat", "head", "tail", "less", "more", "readlink"},
    "recon":      {"ss", "ip", "ifconfig", "netstat", "ps", "ls", "find",
                   "grep", "env", "id", "whoami", "uname", "hostname",
                   "uptime", "free", "df", "wc", "du", "sort", "sed",
                   "journalctl", "pgrep", "date", "timedatectl", "loginctl"},
    "persist":    {"crontab", "at", "systemctl", "setsid", "nohup"},
    "communicate":{"curl", "wget", "nc", "ncat", "python3", "python", "logger"},
    "propagate":  {"ping", "nmap", "ssh", "scp", "sftp"},
    "exfil":      {"tar", "gzip", "zip", "base64"},
    "privesc":    {"sudo", "su", "pkexec"},
    "evasion":    {"rm", "mv", "chmod", "chown"},
    "lateral":    {"ssh", "scp", "sftp", "rsync"},
    "discovery":  {"find", "locate", "which", "whereis"},
    "collection": {"cat", "cp", "tar", "zip"},
    "impact":     {"kill", "pkill", "rm"},
    "timing":     {"seq", "sleep"},
}

# 所有 bee 可能使用的命令
ROVE_CMDS = set()
for cmds in ROVE_PRIMITIVES.values():
    ROVE_CMDS.update(cmds)

# 系统背景进程（排除）
SYSTEM_PROCS = {
    "systemd", "snapd", "sleep", "dbus-daemon", "rsyslogd", "cron", "atd",
    "sshd", "agetty", "login", "systemd-journal", "systemd-udevd",
    "systemd-resolved", "systemd-networkd", "systemd-timesyncd",
    "systemd-logind", "polkitd", "accounts-daemon", "networkd-dispat",
    "unattended-upgr", "apt", "dpkg", "snap", "fwupd", "ModemManager",
    "avahi-daemon", "cupsd", "wpa_supplicant", "irqbalance", "multipathd",
    "packagekitd", "udisks2", "colord", "switcheroo-cont",
    "power-profiles-da", "thermald", "bolt", "nvme", "update-motd-fsc",
    "anacron", "logrotate", "certwatch", "man-db", "mlocate", "updatedb",
    "aptitude", "popularity-contest", "ubuntu-advant", "landscape",
    "pollinate", "entropy", "timedate", "cloud-id", "apt-helper",
    "debian-sa1", "run-parts", "sftp-server",
}

# 系统父进程（排除）
SYSTEM_PARENTS = {
    "systemd", "cron", "atd", "rsyslogd", "snapd", "unattended-upgr",
    "apt", "dpkg", "run-parts", "anacron", "logrotate", "update-motd-fsc",
    "cloud-id", "apt-helper", "sudo",
}


def parse_7token(line):
    """解析 7-token JSONL 行。
    注意：DT 槽格式是 DT{N}（无冒号），其他槽是 SLOT:VAL。
    """
    e = json.loads(line)
    toks = e["tokens"]
    d = {"ts": e["ts"], "host": e.get("host", "?")}
    for t in toks:
        if t.startswith("DT") and ":" not in t:
            # DT4 → slot=DT, val=4
            d["DT"] = t[2:]
        elif ":" in t:
            slot, val = t.split(":", 1)
            d[slot] = val
    return d


def infer_pc(ev):
    """从 ARGV 推断 PC 槽（7→8 token 补全）"""
    argv = ev.get("ARGV", "")
    proc = ev.get("PROC", "")
    # 简化推断：基于命令类别
    if proc in ("crontab", "at"):
        return "PC:ETC_CRON"
    if proc == "systemctl":
        return "PC:ETC_SYSTEMD"
    if proc in ("cat", "head", "tail") and "passwd" in argv:
        return "PC:ETC_PASSWD"
    if proc in ("ssh", "scp", "sftp"):
        return "PC:SSH_KEYS"
    if proc in ("tar", "gzip", "zip"):
        return "PC:TMP"
    return "PC:NONE"


def to_8token(ev):
    """7-token → 8-token（补 PC 槽）"""
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


def classify_action(proc):
    """将命令映射到动作原语"""
    for primitive, cmds in ROVE_PRIMITIVES.items():
        if proc in cmds:
            return primitive
    return "unknown"


def extract_sequences(events, window_s=30, min_actions=3):
    """从事件流中提取行为序列。

    策略：找 PARENT:sh/bash + UID:1000 的事件簇，
    按时间窗分组，每窗内 ≥min_actions 个 bee 命令 = 一个序列。
    """
    # 过滤候选事件
    candidates = []
    for ev in events:
        parent = ev.get("PARENT", "")
        uid = ev.get("UID", "")
        proc = ev.get("PROC", "")
        # bee 特征：sh/bash 父进程 + 非系统子进程
        if parent in ("sh", "bash", "(sh)", "(bash)") and proc not in SYSTEM_PROCS:
            ev["_primitive"] = classify_action(proc)
            candidates.append(ev)

    if not candidates:
        return []

    # 按时间窗分组
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


def analyze_temporal(sequences):
    """分析序列的时序特征"""
    stats = {
        "n_sequences": len(sequences),
        "total_events": sum(len(s) for s in sequences),
        "seq_lengths": [],
        "intervals": [],
        "primitive_dist": Counter(),
        "proc_dist": Counter(),
        "uid_dist": Counter(),
        "dst_dist": Counter(),
    }

    for seq in sequences:
        stats["seq_lengths"].append(len(seq))
        for ev in seq:
            stats["primitive_dist"][ev.get("_primitive", "?")] += 1
            stats["proc_dist"][ev.get("PROC", "?")] += 1
            stats["uid_dist"][ev.get("UID", "?")] += 1
            stats["dst_dist"][ev.get("DST", "NONE")] += 1

        # 间隔
        for i in range(1, len(seq)):
            t1 = datetime.fromisoformat(seq[i-1]["ts"])
            t2 = datetime.fromisoformat(seq[i]["ts"])
            dt = (t2 - t1).total_seconds()
            if 0 < dt < 300:
                stats["intervals"].append(dt)

    return stats


def main():
    src_dir = Path(os.path.expanduser("~/adaptive-agent-sim/checkpoints"))
    out_dir = Path(__file__).parent

    all_events = []
    for fname in ["clone_events.jsonl", "regime_events.jsonl"]:
        path = src_dir / fname
        if not path.exists():
            print(f"[WARN] {path} 不存在，跳过")
            continue
        events = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(parse_7token(line))
                except Exception:
                    continue
        print(f"[INFO] {fname}: {len(events)} 事件")
        all_events.extend(events)

    # 按时间排序
    all_events.sort(key=lambda e: e["ts"])
    print(f"[INFO] 合并: {len(all_events)} 事件")

    # 提取序列
    sequences = extract_sequences(all_events, window_s=30, min_actions=3)
    print(f"[INFO] 提取序列: {len(sequences)}")

    # 分析
    stats = analyze_temporal(sequences)
    print(f"\n=== 提取统计 ===")
    print(f"序列数: {stats['n_sequences']}")
    print(f"总事件: {stats['total_events']}")
    if stats["seq_lengths"]:
        import statistics
        print(f"序列长度: 中位={statistics.median(stats['seq_lengths']):.0f} "
              f"均值={statistics.mean(stats['seq_lengths']):.1f} "
              f"最大={max(stats['seq_lengths'])}")
    if stats["intervals"]:
        import statistics
        iv = stats["intervals"]
        print(f"间隔: 中位={statistics.median(iv):.2f}s "
              f"均值={statistics.mean(iv):.2f}s "
              f"CV={statistics.stdev(iv)/statistics.mean(iv):.3f}")

    print(f"\n原语分布:")
    for p, c in stats["primitive_dist"].most_common():
        print(f"  {p}: {c}")

    print(f"\n进程分布 (top 15):")
    for p, c in stats["proc_dist"].most_common(15):
        print(f"  {p}: {c}")

    print(f"\nUID 分布:")
    for u, c in stats["uid_dist"].most_common():
        print(f"  {u}: {c}")

    print(f"\nDST 分布:")
    for d, c in stats["dst_dist"].most_common():
        print(f"  {d}: {c}")

    # 输出序列（8-token JSONL）
    out_path = out_dir / "rove_sequences.jsonl"
    n_written = 0
    with open(out_path, "w") as f:
        for seq in sequences:
            tokens_8 = [to_8token(ev) for ev in seq]
            # 找序列中最常见的原语作为标签
            prims = [ev.get("_primitive", "unknown") for ev in seq]
            label = Counter(prims).most_common(1)[0][0]
            entry = {
                "type": "sequence",
                "genus": "rove_beetle",
                "label": label,
                "host": seq[0].get("host", "?"),
                "ts_start": seq[0]["ts"],
                "ts_end": seq[-1]["ts"],
                "n_events": len(seq),
                "tokens": tokens_8,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"\n[OUT] {out_path}: {n_written} 序列")

    # 输出统计
    stats_out = out_dir / "rove_stats.json"
    # Counter 不能直接 JSON 序列化
    stats_json = {
        "n_sequences": stats["n_sequences"],
        "total_events": stats["total_events"],
        "primitive_dist": dict(stats["primitive_dist"].most_common()),
        "proc_dist": dict(stats["proc_dist"].most_common(30)),
        "uid_dist": dict(stats["uid_dist"].most_common()),
        "dst_dist": dict(stats["dst_dist"].most_common()),
    }
    if stats["seq_lengths"]:
        import statistics
        stats_json["seq_len_median"] = statistics.median(stats["seq_lengths"])
        stats_json["seq_len_mean"] = statistics.mean(stats["seq_lengths"])
        stats_json["seq_len_max"] = max(stats["seq_lengths"])
    if stats["intervals"]:
        import statistics
        iv = stats["intervals"]
        stats_json["interval_median_s"] = statistics.median(iv)
        stats_json["interval_mean_s"] = statistics.mean(iv)
        stats_json["interval_cv"] = statistics.stdev(iv) / statistics.mean(iv)

    with open(stats_out, "w") as f:
        json.dump(stats_json, f, indent=2, ensure_ascii=False)
    print(f"[OUT] {stats_out}")


if __name__ == "__main__":
    main()
