#!/usr/bin/env bash
# ADFA-LD 多 seed 矩阵（云端双卡并行）：3 变体 × 5 seed
# 需 ADFA-LD 数据集解包在当前目录（Training_Data_Master/ 等）
# 用法: bash run_adfa_matrix_cloud.sh
set -u
cd "$(dirname "$0")/.."
PY=${MOCHENG_PY:-python3}
mkdir -p results logs

run_one() {
  local skip=$1 seed=$2 gpu=$3
  local out="results/adfa_skip${skip}_seed${seed}.json"
  [ -f "$out" ] && { echo "[skip] skip=$skip seed=$seed"; return; }
  echo "[$(date +%T)] GPU$gpu ADFA skip=$skip seed=$seed"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/adfa_bigram.py ADFA-LD --epochs 3 --valsplit 3500 \
    --skip $skip --seed $seed --out "$out" \
    > "logs/adfa_skip${skip}_seed${seed}.log" 2>&1
  echo "[$(date +%T)] GPU$gpu done skip=$skip seed=$seed (exit $?)"
}

for skip in 0 1 2; do
  for seed in 0 1 2 3 4; do echo "$skip $seed"; done
done | {
  i=0
  while read -r skip seed; do
    gpu=$((i % 2))
    run_one "$skip" "$seed" "$gpu" &
    i=$((i+1))
    while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 5; done
  done
  wait
}
echo "== ADFA 矩阵完成 =="
