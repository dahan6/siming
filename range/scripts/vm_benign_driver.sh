#!/usr/bin/env bash
# 良性活动驱动器：在目标主机上制造"日常"行为分布（供静默学习窗采集）
# 只跑普通管理/用户动作，绝不碰攻击性命令。在 VM 内运行。
# 用法: vm_benign_driver.sh [持续秒数，默认 1800]
DUR=${1:-1800}
END=$((SECONDS + DUR))
ACTIONS=(
  "ls -la /etc /var/log /tmp >/dev/null"
  "cat /var/log/syslog 2>/dev/null | tail -20 >/dev/null"
  "systemctl status cron --no-pager >/dev/null 2>&1"
  "systemctl list-units --type=service --no-pager >/dev/null 2>&1"
  "ps aux >/dev/null"
  "df -h >/dev/null"
  "free -m >/dev/null"
  "uptime"
  "dpkg -l 2>/dev/null | tail -10 >/dev/null"
  "apt-get update -o Debug::NoLocking=1 >/dev/null 2>&1 || true"
  "apt list --upgradable 2>/dev/null | head -5 >/dev/null"
  "journalctl -n 30 --no-pager >/dev/null 2>&1"
  "crontab -l 2>/dev/null || true"
  "ss -tlnp 2>/dev/null >/dev/null"
  "cat /etc/passwd >/dev/null"
  "find /var/log -name '*.log' -mmin -60 2>/dev/null | head -5 >/dev/null"
  "hostnamectl status >/dev/null"
  "lsmod | head -10 >/dev/null"
  "echo test > /tmp/.drv && cat /tmp/.drv && rm /tmp/.drv"
  "tar czf /tmp/.drv.tgz /etc/hostname /etc/hosts 2>/dev/null && rm /tmp/.drv.tgz"
)
while [ $SECONDS -lt $END ]; do
  for a in "${ACTIONS[@]}"; do
    eval "$a"
    sleep $((RANDOM % 8 + 2))
    [ $SECONDS -ge $END ] && break
  done
done
echo "driver done"
