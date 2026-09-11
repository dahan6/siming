#!/bin/bash
# VM 激活脚本：让 VM 产生丰富的正常行为
# 在 VM 内执行，模拟多角色工作负载 + 采集

VM_IP=$1
VM_PASS=${2:?"usage: $0 <VM_IP> <VM_PASS> [WINDOW]"}
DURATION=${3:-600}  # 默认 10 分钟

if [ -z "$VM_IP" ]; then
    echo "用法: $0 <vm_ip> [password] [duration_sec]"
    echo "示例: $0 198.51.100.88 <vm-password> 600"
    exit 1
fi

echo "=== VM 激活: $VM_IP ($DURATION 秒) ==="

# 在 VM 上执行的工作负载（通过 SSH）
sshpass -p "$VM_PASS" ssh -o StrictHostKeyChecking=no root@$VM_IP << VM_SCRIPT
#!/bin/bash
DURATION=$DURATION
END_TIME=\$(($(date +%s) + DURATION))

# 角色模拟函数
simulate_ops() {
    # 运维管理员
    systemctl status sshd 2>/dev/null
    systemctl list-units --type=service --state=running 2>/dev/null | head -10
    journalctl -n 20 --no-pager 2>/dev/null
    df -h
    free -m
    uptime
    docker ps -a 2>/dev/null || true
    apt list --installed 2>/dev/null | head -10
    lsblk
    ip addr show | head -20
}

simulate_dev() {
    python3 -c 'print(1+1)' 2>/dev/null
    python3 --version 2>/dev/null
    which gcc && gcc --version | head -1 2>/dev/null
    git --version 2>/dev/null
    find /var/log -name '*.log' -maxdepth 2 2>/dev/null | head -5
    ls -la /root/ /tmp/
    cat /etc/hostname
    uname -a
}

simulate_analyst() {
    ps aux --sort=-%cpu | head -10
    ss -tlnp 2>/dev/null || ss -tln
    ss -tnp 2>/dev/null | head -10
    netstat -tlnp 2>/dev/null | head -10
    cat /etc/passwd | head -10
    grep root /etc/passwd
    find / -perm -4000 -type f 2>/dev/null | head -5
    crontab -l 2>/dev/null || echo "no crontab"
    cat /proc/cpuinfo | head -5
    env | head -10
}

simulate_user() {
    ls -la /root/
    ls -la /tmp/
    date
    whoami
    id
    hostname
    pwd
    which bash python3 git
    echo "hello world"
    history 2>/dev/null | tail -5 || true
}

echo "开始模拟 (\$DURATION 秒)..."
COUNT=0
while [ \$(date +%s) -lt \$END_TIME ]; do
    ROLE=\$((RANDOM % 4))
    case \$ROLE in
        0) simulate_ops ;;
        1) simulate_dev ;;
        2) simulate_analyst ;;
        3) simulate_user ;;
    esac
    COUNT=\$((COUNT + 1))
    sleep \$((RANDOM % 3 + 1))
done

echo "完成: \$COUNT 轮模拟"
VM_SCRIPT

echo "=== VM 激活完成 ==="
