#!/usr/bin/env bash
# 图3 分布数据：三架构各训一个子采样模型，导出分数数组
set -u
cd ~/defense-lab/detector
PY=~/miniconda3/envs/ai/bin/python
export BENIGN_PATH=~/defense-lab/data/host_tokens_clean.jsonl
mkdir -p results logs
for arch in lstm_keys lstm_full tinygpt; do
  out="results/scores_${arch}.npz"
  [ -f "$out" ] && { echo "[skip] $arch"; continue; }
  echo "[$(date +%T)] 训练+打分 $arch"
  nice -n 19 $PY deeplog_baseline.py --arch $arch --seed 0 \
    --benign-max 150000 --benign-holdout 6000 --attack-max 20000 \
    --dump-scores "$out" \
    --out "results/dist_${arch}_seed0.json" \
    > "logs/dist_${arch}.log" 2>&1
  echo "[$(date +%T)] done $arch (exit $?)"
done
echo "== 分布数据完成 =="
