#!/usr/bin/env bash
# Siming：良性工作负载生成器（在 VM 内运行，模拟普通运维/用户活动）
# 用法: bash benign_workload.sh [轮数]
N="${1:-80}"
ACTIONS=(
  "ls -la /var/log | head -20"
  "ps aux --sort=-%mem | head -10"
  "systemctl status cron --no-pager"
  "df -h"
  "free -m"
  "ss -tlnp"
  "journalctl -n 15 --no-pager"
  "cat /proc/loadavg"
  "find /etc -maxdepth 1 -name '*.conf' | head"
  "grep -r listen /etc/ssh/sshd_config 2>/dev/null"
  "tar tf /var/log/apt/history.log 2>/dev/null || cat /var/log/apt/history.log | tail -5"
  "ls /home/range"
  "cat /etc/os-release"
  "uptime"
  "systemctl list-units --type=service --state=running --no-pager | head -15"
  "echo test-\$RANDOM > /tmp/wl.tmp && cat /tmp/wl.tmp && rm /tmp/wl.tmp"
  "head -5 /etc/passwd"
  "ip addr show"
)
for i in $(seq 1 "$N"); do
  eval "${ACTIONS[$((RANDOM % ${#ACTIONS[@]}))]}" > /dev/null 2>&1
  sleep $((RANDOM % 12 + 3))
done
echo "workload done: $N actions"
