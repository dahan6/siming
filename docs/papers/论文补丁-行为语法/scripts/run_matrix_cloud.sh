#!/usr/bin/env bash
# 双卡并行实验编排（DeepLog 对比矩阵：3 架构 × 5 seed = 15 轮）
# 在实验目录下运行，需 data/host_tokens_clean.jsonl + rove_attacks.jsonl + bee_active.jsonl
# 用法: bash run_matrix_cloud.sh
set -u
cd "$(dirname "$0")/.."
PY=${MOCHENG_PY:-python3}
mkdir -p results logs

run_one() {
  local arch=$1 seed=$2 gpu=$3
  local out="results/deeplog_${arch}_seed${seed}.json"
  [ -f "$out" ] && { echo "[skip] $arch seed$seed"; return; }
  echo "[$(date +%T)] GPU$gpu 开始 $arch seed$seed"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/deeplog_baseline.py \
    --arch "$arch" --seed "$seed" --out "$out" \
    > "logs/${arch}_seed${seed}.log" 2>&1
  echo "[$(date +%T)] GPU$gpu 完成 $arch seed$seed (exit $?)"
}

for arch in lstm_keys lstm_full tinygpt; do
  for seed in 0 1 2 3 4; do echo "$arch $seed"; done
done | {
  i=0
  while read -r arch seed; do
    gpu=$((i % 2))
    run_one "$arch" "$seed" "$gpu" &
    i=$((i+1))
    while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 5; done
  done
  wait
}
echo "== 全部 15 轮完成 =="
