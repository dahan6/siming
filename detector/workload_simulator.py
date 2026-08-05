#!/usr/bin/env python3
"""宿主机多样化工作负载模拟器

模拟不同用户角色的工作行为，产生丰富的正常进程事件，
让司命 prior 模型见到更广泛的正常分布。

角色：
1. 运维管理员：systemctl/journalctl/apt/docker/ufw
2. 开发者：python/gcc/cargo/npm/git/vim/make
3. 分析师：ss/ps/find/grep/cat/head/tail/wc/sort/awk
4. 普通用户：ls/cd/cat/echo/date/whoami/uptime

每个角色执行 5-10 分钟的随机命令序列，
procfs 采集器同时捕获行为事件。
"""
import os
import random
import subprocess
import time
import sys

def run(cmd, timeout=10):
    """安全执行命令，不产生实际副作用"""
    try:
        subprocess.run(cmd, shell=True, timeout=timeout,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

ROLES = {
    "ops": {
        "weight": 0.3,
        "commands": [
            "systemctl status sshd",
            "systemctl list-units --type=service --state=running",
            "journalctl -n 20 --no-pager",
            "df -h",
            "free -m",
            "uptime",
            "docker ps -a 2>/dev/null || true",
            "docker images 2>/dev/null || true",
            "ufw status 2>/dev/null || true",
            "apt list --installed 2>/dev/null | head -20",
            "systemctl is-active cron",
            "systemctl is-active ssh",
            "lsblk",
            "ip addr show",
            "ip route show",
        ],
    },
    "developer": {
        "weight": 0.3,
        "commands": [
            "python3 -c 'print(1+1)'",
            "python3 --version",
            "pip3 list 2>/dev/null | head -10",
            "which gcc && gcc --version | head -1",
            "which cargo && cargo --version",
            "which node && node --version",
            "which npm && npm --version",
            "git --version",
            "git -C /home/adazz18 status 2>/dev/null || true",
            "git -C /home/adazz18 log --oneline -3 2>/dev/null || true",
            "ls -la /home/adazz18/",
            "find /home/adazz18 -name '*.py' -maxdepth 2 2>/dev/null | head -10",
            "cat /home/adazz18/.bashrc | head -5",
            "wc -l /home/adazz18/.bashrc",
        ],
    },
    "analyst": {
        "weight": 0.2,
        "commands": [
            "ss -tlnp 2>/dev/null || ss -tln",
            "ss -tnp 2>/dev/null | head -20",
            "ps aux --sort=-%cpu | head -15",
            "ps aux --sort=-%mem | head -10",
            "find /var/log -name '*.log' -mtime -1 2>/dev/null | head -10",
            "grep -c 'error' /var/log/syslog 2>/dev/null || true",
            "head -50 /var/log/syslog 2>/dev/null || true",
            "tail -20 /var/log/auth.log 2>/dev/null || true",
            "awk '{print $1}' /var/log/syslog 2>/dev/null | sort | uniq -c | sort -rn | head -5 || true",
            "wc -l /var/log/syslog 2>/dev/null || true",
            "du -sh /var/log/ 2>/dev/null",
            "du -sh /tmp/ 2>/dev/null",
            "stat /etc/passwd",
            "stat /etc/shadow",
        ],
    },
    "user": {
        "weight": 0.2,
        "commands": [
            "ls -la /home/adazz18/",
            "ls -la /tmp/",
            "cat /etc/hostname",
            "date",
            "whoami",
            "id",
            "uname -a",
            "hostname",
            "env | head -10",
            "echo hello",
            "pwd",
            "which bash",
            "cal 2>/dev/null || true",
            "ls -la /home/adazz18/Downloads/ 2>/dev/null || true",
        ],
    },
}


def simulate(duration_min=5):
    """模拟指定分钟的工作负载"""
    duration_sec = duration_min * 60
    t0 = time.time()
    n = 0

    # 构建加权命令池
    pool = []
    for role, info in ROLES.items():
        for cmd in info["commands"]:
            pool.append((role, cmd, info["weight"]))

    while time.time() - t0 < duration_sec:
        # 加权随机选择
        weights = [p[2] for p in pool]
        choice = random.choices(pool, weights=weights)[0]
        role, cmd, _ = choice

        run(cmd)
        n += 1

        # 随机间隔（模拟人节奏）
        delay = random.uniform(0.5, 3.0)
        time.sleep(delay)

        if n % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{elapsed:.0f}s] {n} 命令执行", flush=True)

    return n


if __name__ == "__main__":
    mins = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"工作负载模拟 {mins} 分钟...")
    n = simulate(mins)
    print(f"完成: {n} 条命令")
