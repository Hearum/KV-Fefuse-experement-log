#!/usr/bin/env bash
set -euo pipefail
REPO=/raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
PAIRS=MOTIVATION_EXPERIMENTS/native_initialized_layer4_distill/data/musique_pairs.jsonl
OUT=MOTIVATION_EXPERIMENTS/native_initialized_layer4_distill/teacher_cache_musique_val
mkdir -p "$REPO/$OUT" "$REPO/MOTIVATION_EXPERIMENTS/native_initialized_layer4_distill/logs"
cd "$REPO"
N=$(wc -l < "$PAIRS")
SHARD=$(( (N + 7) / 8 ))
for gpu in 0 1 2 3 4 5 6 7; do
  start=$((gpu * SHARD)); end=$((start + SHARD)); if [ "$end" -gt "$N" ]; then end=$N; fi
  [ "$start" -ge "$N" ] && continue
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/scripts/generate_teacher_scores.py \
    --pairs "$PAIRS" --out-dir "$OUT" --start "$start" --end "$end" --limit 0 --device cuda:0 \
    > "MOTIVATION_EXPERIMENTS/native_initialized_layer4_distill/logs/musique_teacher_${start}_${end}.log" 2>&1 &
  echo $! > "MOTIVATION_EXPERIMENTS/native_initialized_layer4_distill/logs/musique_teacher_${start}_${end}.pid"
  echo "launched gpu=$gpu start=$start end=$end pid=$(cat MOTIVATION_EXPERIMENTS/native_initialized_layer4_distill/logs/musique_teacher_${start}_${end}.pid)"
done
