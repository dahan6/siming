#!/usr/bin/env python3
"""自动弱标注器：在 token 序列上用规则启发式打行为标签

标签体系（6 类）：
  benign     - 正常运维/开发行为
  recon      - 侦察行为（端口扫描、进程枚举、文件搜索）
  persist    - 持久化（crontab/at/systemd/rc 文件）
  exfil      - 数据窃取（读敏感文件+网络外联）
  privesc    - 提权（SUID 查找/setuid/sudo 异常链）
  lateral    - 横向移动（SSH/SCP/SMB 内网探测）

标注策略：滑动窗口（W 事件），检查窗口内的行为模式。
每个窗口输出一个标签 + 置信度。

用法:
  # 标注 audit 数据
  python3 auto_labeler.py data/audit_all.jsonl --out data/audit_labeled.jsonl

  # 标注 VM 遥测
  python3 auto_labeler.py data/vm_multi_train.jsonl --out data/vm_labeled.jsonl
"""
import json
import os
import re
import sys
import argparse
from collections import Counter, defaultdict

# ═══ 规则定义 ═══

# 侦察命令集（收紧：只留主动侦察工具，排除系统后台常用命令）
# 关键区分：系统后台高频用的（readlink/uname/whoami/id/uptime/free/df/env）不算侦察
# 攻击者主动侦察的核心命令：网络枚举 + 进程枚举 + 文件搜索
RECON_PROCS = {
    # 网络侦察（核心）
    "ss", "netstat", "lsof", "ip", "ifconfig", "arp", "nmap",
    # 进程侦察（核心）
    "ps", "top", "htop", "pgrep", "pidof",
    # 文件搜索（核心——攻击者用它找目标）
    "find", "locate",
    # 安全日志侦察
    "journalctl", "dmesg",
}
# 注意：uname/hostname/whoami/id/who/w/uptime/free/df/du/vmstat/iostat/
#       env/printenv/set/readlink/realpath/which/whereis/last
#       这些是系统后台高频命令，不算侦察

# 侦察的高危路径参数
RECON_PATHS = ["/etc/shadow", "/etc/sudoers"]  # passwd 不算高危（太常见）

# 持久化命令
PERSIST_PROCS = {"crontab", "at", "atq", "atrm", "batch"}
PERSIST_PATHS = ["/etc/systemd/", "/etc/cron", ".bashrc", ".bash_profile",
                 ".profile", ".zshrc", "/etc/init.d/", "/etc/rc"]

# 提权信号（收紧：sudo/su 是正常操作，只有异常路径才算）
# sudo 本身不算提权——它是合法的权限切换工具
# 真正的提权信号：
#   1. SUID 查找（find perm 4000）—— 在找可利用的 SUID 二进制
#   2. 非 sudo 链的 UID:0 —— 可能利用了 SUID 漏洞
#   3. capability 操作（capsh/getcap/setcap）
PRIVESC_PROCS = {"pkexec", "doas"}  # 排除 sudo/su（太常见）
PRIVESC_SUID_FIND = re.compile(r"perm.*4000|perm.*u\+s|SUID", re.I)
PRIVESC_CAP_PROCS = {"capsh", "getcap", "setcap"}

# 窃取信号
EXFIL_READ_PROCS = {"cat", "head", "tail", "less", "more", "dd", "cp", "scp"}
EXFIL_PATHS = ["/etc/passwd", "/etc/shadow", "/etc/sudoers", ".ssh/", ".ssh/id_"]
EXFIL_NET_PROCS = {"curl", "wget", "nc", "ncat", "python3", "python", "perl"}

# 横向移动
LATERAL_PROCS = {"ssh", "scp", "sftp", "rsync"}


def extract_proc(tokens):
    """从 8-token 提取 PROC 值"""
    for t in tokens:
        if t.startswith("PROC:"):
            return t.split(":", 1)[1]
    return ""


def extract_argv(tokens):
    for t in tokens:
        if t.startswith("ARGV:"):
            return t
    return ""


def extract_parent(tokens):
    for t in tokens:
        if t.startswith("PARENT:"):
            return t.split(":", 1)[1]
    return ""


def extract_uid(tokens):
    for t in tokens:
        if t.startswith("UID:"):
            return t.split(":", 1)[1]
    return ""


def extract_pc(tokens):
    for t in tokens:
        if t.startswith("PC:"):
            return t.split(":", 1)[1]
    return ""


def extract_dst(tokens):
    for t in tokens:
        if t.startswith("DST:"):
            return t.split(":", 1)[1]
    return "NONE"


def label_window(events, start, end):
    """对一个窗口 [start, end) 的事件打标签
    
    返回: (label, confidence, evidence)
    """
    window = events[start:end]
    n = len(window)
    
    # 统计窗口内行为
    procs = [extract_proc(e["tokens"]) for e in window]
    proc_set = set(procs)
    
    # 计数
    n_recon = sum(1 for p in procs if p in RECON_PROCS)
    n_persist = sum(1 for p in procs if p in PERSIST_PROCS)
    n_privesc = sum(1 for p in procs if p in PRIVESC_PROCS)
    n_lateral = sum(1 for p in procs if p in LATERAL_PROCS)
    
    # 路径检查
    has_passwd_read = False
    has_ssh_key = False
    has_systemd_write = False
    has_cron_write = False
    has_rc_write = False
    has_suid_find = False
    has_ext_conn = False
    
    for e in window:
        tokens = e["tokens"]
        argv = extract_argv(tokens)
        pc = extract_pc(tokens)
        dst = extract_dst(tokens)
        proc = extract_proc(tokens)
        
        # 敏感文件读取
        if any(p in pc for p in ["ETC_PASSWD"]):
            has_passwd_read = True
        if "SSH_KEYS" in pc:
            has_ssh_key = True
        
        # 持久化路径
        if "ETC_SYSTEMD" in pc:
            has_systemd_write = True
        if "ETC_CRON" in pc:
            has_cron_write = True
        if "HOME_RC" in pc:
            has_rc_write = True
        
        # SUID 查找
        if proc == "find":
            if any(k in argv.lower() for k in ["4000", "perm", "suid"]):
                has_suid_find = True
        
        # 外联
        if "EXT" in dst:
            has_ext_conn = True
    
    # 判定规则（按优先级）
    evidence = []
    
    # ── PRIVESC（收紧：只认真实提权信号）──
    # SUID 查找（攻击者找可利用的 SUID 二进制）
    if has_suid_find:
        evidence.append(f"SUID 查找 (find perm 4000)")
        return "privesc", 0.85, evidence
    
    # capability 操作
    if any(p in PRIVESC_CAP_PROCS for p in procs):
        evidence.append(f"capability 操作 ({[p for p in procs if p in PRIVESC_CAP_PROCS]})")
        return "privesc", 0.8, evidence
    
    # python3/python UID:0 但父进程是 bash（不是 sudo/su）—— SUID 利用
    for e in window:
        proc = extract_proc(e["tokens"])
        uid = extract_uid(e["tokens"])
        parent = extract_parent(e["tokens"])
        if proc in ("python3", "python") and uid == "0" and parent in ("bash", "sh"):
            # 确认窗口内没有 sudo（排除合法 sudo python3）
            has_sudo_in_window = any(extract_proc(e2["tokens"]) == "sudo" for e2 in window)
            if not has_sudo_in_window:
                evidence.append(f"SUID python3 UID:0 parent={parent} (无 sudo 链)")
                return "privesc", 0.9, evidence
    
    # ── EXFIL（收紧：要求读取类进程 + 网络外联同时出现）──
    # 读敏感文件 + 外联
    EXFIL_READERS = {"cat", "head", "tail", "less", "more", "dd"}
    has_passwd_by_reader = any(
        "ETC_PASSWD" in extract_pc(e["tokens"]) and
        extract_proc(e["tokens"]) in EXFIL_READERS
        for e in window
    )
    if has_passwd_by_reader and has_ext_conn:
        evidence.append("敏感文件读取 + 外联")
        return "exfil", 0.8, evidence
    
    # cat passwd 后跟 python3/curl 外联
    passwd_idx = None
    for i, e in enumerate(window):
        if "ETC_PASSWD" in extract_pc(e["tokens"]):
            passwd_idx = i
    if passwd_idx is not None:
        for e in window[passwd_idx:]:
            if extract_proc(e["tokens"]) in EXFIL_NET_PROCS and "EXT" in extract_dst(e["tokens"]):
                evidence.append("cat passwd → 网络工具外联")
                return "exfil", 0.85, evidence
    
    # SSH 密钥读取（收紧：只认 cat/less/tail 读 .ssh，不认 grep/readlink/find）
    ssh_key_readers = {"cat", "less", "more", "tail", "head", "dd", "cp"}
    if has_ssh_key:
        # 检查是读取类进程还是搜索/处理类
        ssh_key_procs = [extract_proc(e["tokens"]) for e in window
                         if "SSH_KEYS" in extract_pc(e["tokens"])]
        if any(p in ssh_key_readers for p in ssh_key_procs):
            evidence.append("SSH 密钥读取 (cat/less)")
            return "exfil", 0.75, evidence
    
    # ── PERSIST ──
    if has_cron_write or has_systemd_write:
        evidence.append(f"持久化路径写入 (cron={has_cron_write}, systemd={has_systemd_write})")
        return "persist", 0.85, evidence
    
    if has_rc_write and any(p in PERSIST_PROCS for p in procs):
        evidence.append("RC 文件 + 持久化命令")
        return "persist", 0.75, evidence
    
    if n_persist >= 2:
        evidence.append(f"多个持久化命令 ({n_persist})")
        return "persist", 0.7, evidence
    
    # ── LATERAL ──
    if n_lateral >= 2:
        evidence.append(f"多个横向移动工具 ({n_lateral})")
        return "lateral", 0.7, evidence
    
    # ── RECON（收紧：要求密集连续侦察）──
    # 连续 5+ 侦察命令（且不含睡眠/系统进程打断）
    max_recon_streak = 0
    cur_streak = 0
    for p in procs:
        if p in RECON_PROCS:
            cur_streak += 1
            max_recon_streak = max(max_recon_streak, cur_streak)
        elif p in ("sleep", "dash", "sh", "systemd-executor", "snap", "snapctl",
                    "snap-confine", "snap-seccomp", "snap-exec", "getent",
                    "unix_chkpwd", "dirname", "md5sum", "chmod"):
            # 系统后台命令不打断连续性判断但也不计入侦察
            pass
        else:
            cur_streak = 0
    
    if max_recon_streak >= 6:
        evidence.append(f"连续侦察 ({max_recon_streak} 个)")
        return "recon", 0.8, evidence
    
    # 高密度侦察（>70% 且绝对数量 >=8）
    if n_recon >= 8 and n_recon >= n * 0.7:
        evidence.append(f"侦察密集 ({n_recon}/{n} = {n_recon/n*100:.0f}%)")
        return "recon", 0.7, evidence
    
    # ── BENIGN ──
    return "benign", 0.9, ["default benign"]


def label_file(input_path, output_path, window=16, stride=4):
    """对整个文件做滑动窗口标注"""
    events = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                e = json.loads(line)
                if "tokens" in e and len(e["tokens"]) >= 4:
                    events.append(e)
            except:
                continue
    
    if not events:
        print(f"无有效事件: {input_path}")
        return
    
    # 滑动窗口标注
    labels = []
    for i in range(0, len(events) - window + 1, stride):
        label, conf, evidence = label_window(events, i, i + window)
        labels.append({
            "start_idx": i,
            "end_idx": i + window,
            "ts_start": events[i].get("ts", ""),
            "ts_end": events[min(i+window-1, len(events)-1)].get("ts", ""),
            "label": label,
            "confidence": conf,
            "evidence": evidence,
            "n_events": window,
            "tokens_sample": [extract_proc(e["tokens"]) for e in events[i:i+min(5,window)]],
        })
    
    # 输出
    with open(output_path, "w") as f:
        for l in labels:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    
    # 统计
    label_dist = Counter(l["label"] for l in labels)
    print(f"标注: {len(labels)} 窗口 ({input_path})")
    print(f"  分布: {dict(label_dist.most_common())}")
    for label in ["benign", "recon", "persist", "exfil", "privesc", "lateral"]:
        items = [l for l in labels if l["label"] == label]
        if items:
            avg_conf = sum(l["confidence"] for l in items) / len(items)
            print(f"  {label:10s}: {len(items):>5} (conf={avg_conf:.2f})")
    
    return label_dist


def main():
    ap = argparse.ArgumentParser(description="弱标注器")
    ap.add_argument("input", help="输入 JSONL")
    ap.add_argument("--out", required=True, help="输出 JSONL")
    ap.add_argument("--window", type=int, default=16, help="窗口大小")
    ap.add_argument("--stride", type=int, default=4, help="步幅")
    args = ap.parse_args()
    
    label_file(args.input, args.out, args.window, args.stride)


if __name__ == "__main__":
    main()
