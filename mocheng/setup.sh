#!/usr/bin/env bash
# 墨城防火墙一键安装 + 自动训练
# 用法: bash setup.sh [--duration 300] [--install-service]
set -euo pipefail

FW="$(cd "$(dirname "$0")" && pwd)"
PY="${MOCHENG_PY:-python3}"
DURATION="${1:-300}"
INSTALL_SERVICE="${2:-}"

echo "================================================"
echo "  墨城防火墙 · 一键安装"
echo "  项目路径: $FW"
echo "  Python:   $PY ($($PY --version 2>&1))"
echo "================================================"

# ── 1. 检测发行版 + 安装系统依赖 ──
echo ""
echo "[1/6] 检测系统依赖..."
NEED_INSTALL=()
command -v conntrack >/dev/null 2>&1 || NEED_INSTALL+=("conntrack")
command -v iptables >/dev/null 2>&1 || NEED_INSTALL+=("iptables")
if [ ${#NEED_INSTALL[@]} -gt 0 ]; then
    echo "  缺少: ${NEED_INSTALL[*]}"
    if command -v apt-get >/dev/null 2>&1; then
        echo "  安装中 (apt)..."
        sudo apt-get update -qq && sudo apt-get install -y -qq "${NEED_INSTALL[@]}"
    elif command -v yum >/dev/null 2>&1; then
        echo "  安装中 (yum)..."
        sudo yum install -y -q "${NEED_INSTALL[@]}"
    elif command -v dnf >/dev/null 2>&1; then
        echo "  安装中 (dnf)..."
        sudo dnf install -y -q "${NEED_INSTALL[@]}"
    else
        echo "  [WARN] 无法自动安装，请手动安装: ${NEED_INSTALL[*]}"
    fi
else
    echo "  conntrack + iptables 已就绪"
fi

# ── 2. Python 依赖 ──
echo ""
echo "[2/6] 安装 Python 依赖..."
$PY -m pip install -q -r "$FW/requirements.txt" 2>&1 | tail -3
echo "  Python 依赖已安装"

# ── 3. 加载内核模块 ──
echo ""
echo "[3/6] 检查 conntrack 内核模块..."
sudo modprobe nf_conntrack 2>/dev/null && echo "  nf_conntrack 已加载" || echo "  [WARN] modprobe 失败（可能已在内核中）"

# ── 4. 采集真实流量 ──
echo ""
echo "[4/6] 采集真实流量 (${DURATION}s)..."
echo "  请在此期间正常使用网络（浏览网页、SSH、DNS 等）"
# 先试 conntrack -E 事件流，失败则用快照
if sudo conntrack -E -o timestamp -e NEW >/dev/null 2>&1 &
   KILL_PID=$!; sleep 1; kill $KILL_PID 2>/dev/null; wait $KILL_PID 2>/dev/null; then
    sudo $PY "$FW/fw_collect.py" --event --duration "$DURATION" --out "$FW/data/event_flows.jsonl"
    RAW="$FW/data/event_flows.jsonl"
else
    echo "  conntrack -E 不可用，使用快照模式..."
    sudo $PY "$FW/fw_collect.py" --interval 5 --once --out "$FW/data/live_flows.jsonl"
    RAW="$FW/data/live_flows.jsonl"
fi

N_FLOWS=$(wc -l < "$RAW" 2>/dev/null || echo 0)
echo "  采集到 $N_FLOWS 条流量"
[ "$N_FLOWS" -lt 500 ] && echo "  [WARN] 流量不足 500 条，模型质量可能较差"

# ── 5. 训练 + 校准 ──
echo ""
echo "[5/6] 训练模型 + 校准..."
TOK="${RAW%.jsonl}_tokens.jsonl"
$PY "$FW/fw_tokens.py" "$RAW" "$TOK"
nice -n 19 $PY "$FW/train_prior.py" "$TOK" "$FW/model" 2>&1 | grep -E "词表|参数|epoch|基线|已保存"
$PY "$FW/fw_calibrate.py" "$FW/model" "$TOK" 2>&1 | grep -E "槽位|已保存"

# ── 6. 验证 ──
echo ""
echo "[6/6] 模型验证..."
VAL=$($PY "$FW/fw_validate.py" "$FW/model" "$TOK" 500 2>/dev/null || echo "")
if echo "$VAL" | grep -q "PASS"; then
    FPR=$(echo "$VAL" | grep -oP 'FPR\(EWMA\)=\K[\d.]+%')
    echo "  ✅ 验收通过！FPR=$FPR"
else
    echo "  ⚠️ 验收未完全通过（数据量可能不足），但防火墙仍可运行"
    echo "$VAL" | grep -E "τ\(|分离|达标" || true
fi

# ── 安装 systemd 服务 ──
if [ "$INSTALL_SERVICE" = "--install-service" ]; then
    echo ""
    echo "[额外] 安装 systemd 服务..."
    # 生成动态路径的 service 文件
    sed "s|/home/lab/mocheng-firewall|$FW|g; s|/home/lab/miniconda3/envs/ai/bin/python|$(which $PY)|g" \
        "$FW/mocheng-firewall.service" | sudo tee /etc/systemd/system/mocheng-firewall.service >/dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable mocheng-firewall
    echo "  服务已安装（sudo systemctl start mocheng-firewall 启动）"
fi

echo ""
echo "================================================"
echo "  安装完成！"
echo "  测试: $PY $FW/fw_daemon.py model --once --no-enforce"
echo "  启动: $PY $FW/fw_daemon.py model --loop --interval 10"
echo "  体检: $PY $FW/fw_ctl.py status"
echo "================================================"
