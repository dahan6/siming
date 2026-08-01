#!/usr/bin/env bash
# 新主机自适应上线管线（onboarding）
# 流程：静默学习窗（良性活动+tracee）→ 拉取解析 → 训练 → 校准 → 自动验收
#       不达标 → 延长学习窗重试（最多 MAX_TRY 次）
# 用法: onboard.sh <目标ssh(如 ubuntu@<defense-vm-ip>)> <模型名> [学习窗秒数=1800]
set -euo pipefail
TARGET="${1:?用法: onboard.sh <user@ip> <模型名> [窗秒]}"
NAME="${2:?缺模型名}"
WIN="${3:-1800}"
MAX_TRY=3
DL=~/siming
PY=~/miniconda3/envs/ai/bin/python
HOST_PART="sshpass -p defense ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=8 $TARGET"
SCP="sshpass -p defense scp -o StrictHostKeyChecking=no"
MODEL_DIR="$DL/detector/model-$NAME"
RAW="$DL/data/onboard_${NAME}.jsonl"
TOK="$DL/data/onboard_${NAME}_tokens.jsonl"
REPORT="$DL/reports/onboard_${NAME}.log"
mkdir -p "$DL/reports" "$MODEL_DIR"

echo "== onboarding $NAME @ $TARGET ==" | tee "$REPORT"

# 部署采集与驱动
$SCP /usr/local/bin/tracee-defense "$TARGET:/tmp/tracee-bin" >/dev/null 2>&1
$SCP ~/siming/range/scripts/vm_benign_driver.sh "$TARGET:/tmp/driver.sh" >/dev/null 2>&1
$HOST_PART "chmod +x /tmp/tracee-bin /tmp/driver.sh"

for try in $(seq 1 $MAX_TRY); do
  echo "-- 第 $try 轮学习窗 ${WIN}s --" | tee -a "$REPORT"
  $HOST_PART "echo defense | sudo -S bash -c 'nohup /tmp/tracee-bin --events sched_process_exec,security_socket_connect --output json > /tmp/cap.jsonl 2>/dev/null & nohup /tmp/driver.sh $WIN > /dev/null 2>&1 &'"
  sleep "$WIN"
  $HOST_PART "echo defense | sudo -S pkill tracee-bin; echo defense | sudo -S pkill -f driver.sh" 2>/dev/null || true
  $HOST_PART "echo defense | sudo -S cat /tmp/cap.jsonl" >> "$RAW"
  N=$(wc -l < "$RAW")
  echo "  累计事件 $N" | tee -a "$REPORT"
  [ "$N" -lt 5000 ] && { echo "  事件不足 5000，延长学习窗" | tee -a "$REPORT"; continue; }

  $PY "$DL/detector/parse_raw_tracee.py" "$RAW" "$TOK" | tee -a "$REPORT"
  nice -n 19 $PY "$DL/detector/train_prior.py" "$TOK" "$MODEL_DIR" 2>&1 \
    | grep -E "token 总数|参数量|epoch |基线 NLL|已保存" | tee -a "$REPORT"
  nice -n 19 $PY "$DL/detector/calibrate_slot_tau.py" "$MODEL_DIR" "$TOK" 20000 2>&1 \
    | grep -E "槽位|已保存" | tee -a "$REPORT"

  # 自动验收：留出 FPR + 合成异常
  VAL=$(nice -n 19 $PY "$DL/detector/validate_host.py" "$MODEL_DIR" "$TOK" 20000 2>/dev/null)
  echo "$VAL" | grep -E "τ\(|用例|分离|达标" | tee -a "$REPORT"
  FPR=$(echo "$VAL" | grep -oP 'FPR\(EWMA\)=\K[\d.]+%' | tr -d '%')
  DETECT=$(echo "$VAL" | grep -oP '全部检出=\K\w+')
  if [ "$DETECT" = "True" ] && $PY -c "exit(0 if float('$FPR') <= 1.5 else 1)"; then
    echo "== 验收通过：FPR(EWMA)=$FPR% 检出全过 ==" | tee -a "$REPORT"
    echo "ONBOARD_OK $NAME FPR=$FPR" | tee -a "$REPORT"
    exit 0
  fi
  echo "  验收未过（FPR=$FPR% 检出=$DETECT），延长学习窗重训" | tee -a "$REPORT"
done
echo "ONBOARD_FAIL $NAME" | tee -a "$REPORT"
exit 1
