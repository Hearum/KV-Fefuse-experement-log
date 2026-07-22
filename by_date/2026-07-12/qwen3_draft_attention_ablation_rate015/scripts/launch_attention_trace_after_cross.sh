#!/usr/bin/env bash
set -euo pipefail

REPO=/raid/home/hming/FusionRAG-pca-analysis
EXP=$REPO/MOTIVATION_EXPERIMENTS/qwen3_draft_attention_ablation_rate015
OUT=$EXP/attention_trace_uniform_alpha0p25_0_10
TRACE=$OUT/attention_trace.jsonl
LOG=$OUT/run.log

mkdir -p "$OUT"

echo "[$(date '+%F %T')] waiting for qjy000 cross-dataset fixed workers to finish" | tee -a "$LOG"
while pgrep -af fixed_gpu_cross_dataset_workers.py >/dev/null; do
  sleep 300
done

echo "[$(date '+%F %T')] waiting for gpu0 to be mostly free" | tee -a "$LOG"
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -n 1 | tr -d ' ')
  if [[ "$used" -lt 10000 ]]; then
    break
  fi
  sleep 120
done

cd "$REPO"
rm -f "$TRACE"

CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
FUSIONRAG_REPROCESS_ATTENTION_ABLATION=uniform \
FUSIONRAG_REPROCESS_ATTENTION_ABLATION_ALPHA=0.25 \
FUSIONRAG_REPROCESS_ATTENTION_TRACE_JSONL="$TRACE" \
FUSIONRAG_REPROCESS_ATTENTION_TRACE_LAYERS=0,8,16,24,32,40,48,56,63 \
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python test_fusionrag_reflect_preprocess_exp.py \
  --model_type qwen3 \
  --model_path /mnt/qjhs-sh-lab-01/models/Qwen3-32B \
  --model_name Qwen3-32B \
  --data_path ./data/result_reflect.json \
  --dataset_name musique \
  --cache_path /raid/home/hming/fusionrag-qwen3-attn-trace-cache/musique \
  --result_path "$OUT" \
  --start_sample 0 \
  --end_sample 10 \
  --rate 0.15 \
  --topk 10 \
  --preprocess true \
  --recall_method bge \
  --reprocess_method DraftModel \
  --draft_model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct \
  --revert_rope true \
  --preprocess_scope global \
  --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3 \
  --device cuda:0 \
  --use_multi_gpu false \
  --openai_base_url http://36.150.226.221:32355/v1 \
  --openai_api_key api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS \
  --openai_model GLM-5.2 \
  >> "$LOG" 2>&1

python3 "$EXP/scripts/summarize_attention_trace.py" \
  "$TRACE" \
  "$OUT/attention_trace_by_layer.csv" \
  >> "$LOG" 2>&1

echo "[$(date '+%F %T')] done" | tee -a "$LOG"
