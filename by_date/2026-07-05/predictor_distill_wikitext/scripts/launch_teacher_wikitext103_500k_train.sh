#!/usr/bin/env bash
set -euo pipefail

REPO=/raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
PAIRS=$REPO/MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/data/wikitext103_pairs_train_500k.jsonl
OUT=$REPO/MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/teacher_cache_wikitext103_train_500k
LOG=$REPO/MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/logs/teacher_wikitext103_train_500k

mkdir -p "$OUT" "$LOG"

for gpu in 0 1 2 3 4 5 6 7; do
  start=$((gpu * 62500))
  end=$(((gpu + 1) * 62500))
  CUDA_VISIBLE_DEVICES=$gpu "$PY" \
    "$REPO/MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/scripts/generate_teacher_scores.py" \
    --pairs "$PAIRS" \
    --out-dir "$OUT" \
    --start "$start" \
    --end "$end" \
    --limit 0 \
    --device cuda:0 \
    > "$LOG/shard_${start}_${end}.log" 2>&1 &
  echo $! > "$LOG/shard_${start}_${end}.pid"
  echo "launched train gpu=$gpu start=$start end=$end pid=$(cat "$LOG/shard_${start}_${end}.pid")"
done

wait
