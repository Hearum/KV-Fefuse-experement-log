#!/usr/bin/env bash
set -euo pipefail
REPO=/raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
OUT=$REPO/MOTIVATION_EXPERIMENTS/online_ttft_profile_draft_select_sparse_update_rrf18_pptrue_sweep_fastkv
LOG=$OUT/logs/run.log
mkdir -p "$OUT/logs"
cd "$REPO"
RATES=(0.00 0.05 0.15 0.30 0.50 0.70 0.99 1.00)
echo "[$(date +%F %T)] start draft-select sparse-update profile sweep" | tee -a "$LOG"
for rate in "${RATES[@]}"; do
  echo "[$(date +%F %T)] start rate=$rate" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" tools_profile_draft_select_sparse_update_rrf18_pptrue.py \
    --model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct \
    --draft_model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct \
    --model_name Qwen2.5-7B-Instruct \
    --data_path ./data/result_reflect.json \
    --dataset_name musique \
    --cache_path /raid/home/hming/fusionrag-reflect-full-cache \
    --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3 \
    --output_dir "$OUT" \
    --device cuda:0 \
    --topk 10 \
    --rate "$rate" \
    --max_main_questions 80 \
    --max_sub_questions 80 \
    --warmup_sub_questions 8 \
    --rrf_k 18 \
    --threshold_factor 0.5 \
    > "$OUT/logs/rate_${rate}.log" 2>&1
  echo "[$(date +%F %T)] done rate=$rate" | tee -a "$LOG"
done
echo "[$(date +%F %T)] finished sweep" | tee -a "$LOG"
