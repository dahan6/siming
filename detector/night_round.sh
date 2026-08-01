#!/usr/bin/env bash
# 夜间迭代一轮：解析 → 训练 → 验证 → 记日志
# 用法: night_round.sh <轮次号>
set -euo pipefail
ROUND="${1:?用法: night_round.sh <轮次号>}"
PY=~/miniconda3/envs/ai/bin/python
DL=~/siming
LOG=$DL/reports/night_iterations.md

mkdir -p "$DL/reports"
cd "$DL/detector"

echo "== [$(date '+%F %T')] 第 ${ROUND} 轮 ==" | tee -a "$LOG"

# 1. 解析（保留全量，训练用 train_prior 内部 85/15 切分）
nice -n 19 $PY parse_raw_tracee.py "$DL/data/host_tracee.jsonl" "$DL/data/host_tokens.jsonl" | tee -a "$LOG"
wc -l "$DL/data/host_tokens.jsonl" | tee -a "$LOG"

# 2. 训练（输出到独立目录，不覆盖 VM 版模型）
MODEL_DIR="$DL/detector/model-host-r${ROUND}"
nice -n 19 $PY train_prior.py "$DL/data/host_tokens.jsonl" "$MODEL_DIR" 2>&1 \
  | grep --line-buffered -E "token 总数|参数量|epoch |基线 NLL|已保存" | tee -a "$LOG"

# 3. 验证
nice -n 19 $PY validate_host.py "$MODEL_DIR" "$DL/data/host_tokens.jsonl" 2>&1 \
  | grep -v "warning\|_fwd" | tee -a "$LOG"

echo "" | tee -a "$LOG"
