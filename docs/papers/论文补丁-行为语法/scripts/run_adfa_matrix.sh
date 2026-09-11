#!/usr/bin/env bash
# ADFA-LD 多 seed 矩阵：3 变体 × 5 seed，本机后台串行（nice 19）
set -u
cd ~/defense-lab/detector
PY=~/miniconda3/envs/ai/bin/python
ADFA=~/defense-lab/data/adfa/raw/ADFA-LD
mkdir -p results logs

for skip in 0 1 2; do
  for seed in 0 1 2 3 4; do
    out="results/adfa_skip${skip}_seed${seed}.json"
    [ -f "$out" ] && { echo "[skip] skip=$skip seed=$seed"; continue; }
    echo "[$(date +%T)] ADFA skip=$skip seed=$seed"
    nice -n 19 $PY adfa_bigram.py "$ADFA" --epochs 3 --valsplit 3500 \
      --skip $skip --seed $seed --out "$out" \
      > "logs/adfa_skip${skip}_seed${seed}.log" 2>&1
    echo "[$(date +%T)] done (exit $?)"
  done
done
echo "== ADFA 矩阵完成 =="
ls results/adfa_*.json | wc -l
