#!/usr/bin/env bash
# 墨城防火墙: 新主机自适应上线管线（onboarding）
# 流程：部署采集器 → 静默学习窗 → 拉取解析 → 训练 → 校准 → 原型 → 自动验收
#       不达标 → 延长学习窗重试（最多 MAX_TRY 次）
#
# 用法: onboard_fw.sh <目标ssh(如 ubuntu@198.51.100.88)> <模型名> [学习窗秒数=1800]
set -euo pipefail
TARGET="${1:?用法: onboard_fw.sh <user@ip> <模型名> [窗秒]}"
NAME="${2:?缺模型名}"
WIN="${3:-1800}"
MAX_TRY=3
FW=~/mocheng-firewall
PY=~/miniconda3/envs/ai/bin/python
HOST_PART="sshpass -p <vm-password> ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=8 $TARGET"
SCP="sshpass -p <vm-password> scp -o StrictHostKeyChecking=no"
MODEL_DIR="$FW/model-$NAME"
RAW="$FW/data/onboard_${NAME}_flows.jsonl"
TOK="$FW/data/onboard_${NAME}_tokens.jsonl"
REPORT="$FW/reports/onboard_${NAME}.log"
mkdir -p "$FW/reports" "$MODEL_DIR" "$FW/data"

echo "== 墨城 onboarding $NAME @ $TARGET ==" | tee "$REPORT"

# 部署采集器
$SCP "$FW/fw_collect.py" "$TARGET:/tmp/fw_collect.py" >/dev/null 2>&1
$SCP "$FW/fw_tokens.py" "$TARGET:/tmp/fw_tokens.py" >/dev/null 2>&1
$HOST_PART "chmod +x /tmp/fw_collect.py /tmp/fw_tokens.py"

for try in $(seq 1 $MAX_TRY); do
  echo "-- 第 $try 轮学习窗 ${WIN}s --" | tee -a "$REPORT"

  # 在目标机启动采集（conntrack 快照循环）
  $HOST_PART "nohup python3 /tmp/fw_collect.py --interval 2 --max-flows 100000 \
    --out /tmp/fw_flows.jsonl > /dev/null 2>&1 &"
  sleep "$WIN"
  $HOST_PART "pkill -f fw_collect.py" 2>/dev/null || true

  # 拉取数据
  $SCP "$TARGET:/tmp/fw_flows.jsonl" "$RAW" >/dev/null 2>&1
  N=$(wc -l < "$RAW" 2>/dev/null || echo 0)
  echo "  累计事件 $N" | tee -a "$REPORT"
  [ "$N" -lt 2000 ] && { echo "  事件不足 2000，延长学习窗" | tee -a "$REPORT"; continue; }

  # 解析 → 训练 → 校准 → 原型
  $PY "$FW/fw_tokens.py" "$RAW" "$TOK" | tee -a "$REPORT"
  nice -n 19 $PY "$FW/train_prior.py" "$TOK" "$MODEL_DIR" 2>&1 \
    | grep -E "token 总数|参数量|epoch |基线 NLL|已保存" | tee -a "$REPORT"
  nice -n 19 $PY "$FW/fw_calibrate.py" "$MODEL_DIR" "$TOK" 20000 2>&1 \
    | grep -E "槽位|已保存" | tee -a "$REPORT"
  nice -n 19 $PY "$FW/fw_proto.py" "$MODEL_DIR" --gen-samples --k 5 2>&1 \
    | grep -E "样本|接住|误报|码本" | tee -a "$REPORT"

  # 自动验收：留出 FPR + 合成异常
  VAL=$(nice -n 19 $PY "$FW/fw_validate.py" "$MODEL_DIR" "$TOK" 20000 2>/dev/null)
  echo "$VAL" | grep -E "τ\(|用例|分离|达标" | tee -a "$REPORT"
  FPR=$(echo "$VAL" | grep -oP 'FPR\(EWMA\)=\K[\d.]+%' | tr -d '%')
  DETECT=$(echo "$VAL" | grep -oP '全部检出=\K\w+')
  if [ "$DETECT" = "True" ] && $PY -c "exit(0 if float('${FPR:-99}') <= 2.0 else 1)"; then
    echo "== 验收通过：FPR(EWMA)=${FPR}% 检出全过 ==" | tee -a "$REPORT"
    echo "ONBOARD_OK $NAME FPR=$FPR" | tee -a "$REPORT"
    # 切换符号链接
    ln -sfn "$MODEL_DIR" "$FW/model-current"
    echo "模型已切换 -> model-current" | tee -a "$REPORT"
    exit 0
  fi
  echo "  验收未过（FPR=${FPR}% 检出=$DETECT），延长学习窗重训" | tee -a "$REPORT"
done
echo "ONBOARD_FAIL $NAME" | tee -a "$REPORT"
exit 1
