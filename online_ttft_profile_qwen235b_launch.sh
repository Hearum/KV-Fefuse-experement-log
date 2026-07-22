#!/usr/bin/env bash
set -euo pipefail

REPO=/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
MODEL=/home/hming/models/Qwen3-235B-A22B
DRAFT=/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
CACHE_QK=/home/hming/fusionrag-qwen3-235b-three-groups-unified-prompt-cache/online_qk_rate015
CACHE_DRAFT=/home/hming/fusionrag-qwen3-235b-three-groups-unified-prompt-cache/online_draft_rate015
EXP=$REPO/MOTIVATION_EXPERIMENTS/online_ttft_profile_qwen235b
LOG=$EXP/logs/qwen235b_profile_sweep.log

RATES=(0.00 0.05 0.15 0.30 0.50 0.70 0.99 1.00)
MAX_MAIN=50
MAX_SUB=60
WARMUP=3

mkdir -p "$EXP/logs" "$EXP/qk_rate_sweep" "$EXP/draft_rate_sweep"

run_qk() {
  local rate="$1"
  echo "[$(date '+%F %T')] START qk rate=$rate host=$(hostname)" | tee -a "$LOG"
  cd "$REPO"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 "$PY" tools_profile_clean_rate_sweep_qwen235b.py \
    --model_type qwen3_moe \
    --model_path "$MODEL" \
    --model_name Qwen3-235B-A22B \
    --data_path ./data/result_reflect.json \
    --dataset_name musique \
    --cache_path "$CACHE_QK" \
    --bge_model_path "$BGE" \
    --output_dir "$EXP/qk_rate_sweep" \
    --device cuda:0 \
    --use_multi_gpu true \
    --topk 10 \
    --rate "$rate" \
    --max_main_questions "$MAX_MAIN" \
    --max_sub_questions "$MAX_SUB" \
    --warmup_sub_questions "$WARMUP" \
    > "$EXP/logs/qk_rate_${rate}.log" 2>&1
  echo "[$(date '+%F %T')] DONE qk rate=$rate" | tee -a "$LOG"
}

run_draft() {
  local rate="$1"
  echo "[$(date '+%F %T')] START draft rate=$rate host=$(hostname)" | tee -a "$LOG"
  cd "$REPO"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 "$PY" tools_profile_draft_dispatch_rrf18_pptrue_qwen235b.py \
    --model_type qwen3_moe \
    --model_path "$MODEL" \
    --draft_model_path "$DRAFT" \
    --model_name Qwen3-235B-A22B \
    --data_path ./data/result_reflect.json \
    --dataset_name musique \
    --cache_path "$CACHE_DRAFT" \
    --bge_model_path "$BGE" \
    --output_dir "$EXP/draft_rate_sweep" \
    --device cuda:0 \
    --use_multi_gpu true \
    --topk 10 \
    --rate "$rate" \
    --max_main_questions "$MAX_MAIN" \
    --max_sub_questions "$MAX_SUB" \
    --warmup_sub_questions "$WARMUP" \
    > "$EXP/logs/draft_rate_${rate}.log" 2>&1
  echo "[$(date '+%F %T')] DONE draft rate=$rate" | tee -a "$LOG"
}

echo "[$(date '+%F %T')] BEGIN Qwen3-235B profile sweep" | tee -a "$LOG"
for rate in "${RATES[@]}"; do
  run_qk "$rate"
done
for rate in "${RATES[@]}"; do
  run_draft "$rate"
done

"$PY" "$EXP/summarize_qwen235b_profile.py" || true
echo "[$(date '+%F %T')] FINISHED Qwen3-235B profile sweep" | tee -a "$LOG"
