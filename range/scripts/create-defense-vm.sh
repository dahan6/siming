#!/usr/bin/env bash
# 创建防御侧采集 VM（与红方 lado-range 完全隔离的独立资产）
# 用法: create-defense-vm.sh [vm名，默认 defense-atomic-0]
set -euo pipefail
VM="${1:-defense-atomic-0}"
DL=~/defense-lab/range
IMG=$DL/images/ubuntu-22.04-cloudimg.img
DISK=$DL/images/${VM}.qcow2
SEED=$DL/images/${VM}-seed.iso

[ -f "$DISK" ] && { echo "已存在: $DISK"; exit 1; }

# cloud-init：设置登录密码 + 装基础工具 + 关自动更新（保持环境可复现）
cat > /tmp/${VM}-user-data <<EOF
#cloud-config
hostname: ${VM}
ssh_pwauth: true
chpasswd:
  list: |
    ubuntu:defense
  expire: false
runcmd:
  - apt-get update
  - apt-get install -y curl git cron at rsyslog jq
  - systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer || true
EOF
cloud-localds "$SEED" /tmp/${VM}-user-data

qemu-img create -f qcow2 -b "$IMG" -F qcow2 "$DISK" 20G
virt-install --name "$VM" --memory 2048 --vcpus 2 \
  --disk "$DISK",format=qcow2 --disk "$SEED",device=cdrom \
  --os-variant ubuntu22.04 --network network=default \
  --graphics none --console pty,target_type=serial --noautoconsole --import

echo "VM $VM 已创建。登录: ssh ubuntu@<ip>（密码 defense）"
