#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
MODEL=/mnt/qjhs-sh-lab-01/models/Qwen3-32B
DRAFT=/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
if [[ -d /raid/home/hming/fusionrag-reflect-qwen3-full-cache ]]; then
  CACHE=/raid/home/hming/fusionrag-reflect-qwen3-full-cache
else
  CACHE=/home/hming/fusionrag-reflect-qwen3-full-cache
fi
OUT_ROOT="$REPO/MOTIVATION_EXPERIMENTS/qwen3_draft_attention_ablation_rate015"
API_URL=http://36.150.226.221:32355/v1
API_KEY=api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS

run_cfg() {
  local mode="$1"
  local gpu="$2"
  local start="$3"
  local end="$4"
  local alpha="${5:-1.0}"
  local alpha_label=${alpha//./p}
  local label="${mode}_draft_rate015"
  if [[ "$alpha" != "1.0" && "$alpha" != "1" ]]; then
    label="${mode}_alpha${alpha_label}_draft_rate015"
  fi
  local out_dir="$OUT_ROOT/$label/seg_${start}_${end}"
  mkdir -p "$out_dir" "$OUT_ROOT/logs"

  echo "[$(date '+%F %T')] start mode=$mode alpha=$alpha gpu=$gpu segment=$start-$end" | tee -a "$OUT_ROOT/logs/${label}_seg_${start}_${end}.log"
  cd "$REPO"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONUNBUFFERED=1 \
  FUSIONRAG_REPROCESS_ATTENTION_ABLATION="$mode" \
  FUSIONRAG_REPROCESS_ATTENTION_ABLATION_ALPHA="$alpha" \
  "$PY" test_fusionrag_reflect_preprocess_exp.py \
    --model_type qwen3 \
    --model_path "$MODEL" \
    --model_name Qwen3-32B \
    --data_path ./data/result_reflect.json \
    --dataset_name musique \
    --cache_path "$CACHE" \
    --result_path "$out_dir" \
    --start_sample "$start" \
    --end_sample "$end" \
    --rate 0.15 \
    --topk 10 \
    --preprocess true \
    --recall_method bge \
    --reprocess_method DraftModel \
    --draft_model_path "$DRAFT" \
    --revert_rope true \
    --preprocess_scope global \
    --bge_model_path "$BGE" \
    --device cuda:0 \
    --use_multi_gpu false \
    --openai_base_url "$API_URL" \
    --openai_api_key "$API_KEY" \
    --openai_model GLM-5.2 \
    > "$out_dir/run.log" 2>&1
  echo "[$(date '+%F %T')] done mode=$mode alpha=$alpha gpu=$gpu segment=$start-$end" | tee -a "$OUT_ROOT/logs/${label}_seg_${start}_${end}.log"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  run_cfg "$@"
  exit 0
fi

mode="${1:-uniform}"
alpha="${2:-1.0}"
if [[ "$mode" != "uniform" && "$mode" != "random" ]]; then
  echo "usage: $0 {uniform|random} [alpha]" >&2
  exit 2
fi

segments=("0 25" "25 50" "50 75" "75 100" "100 125" "125 150" "150 175" "175 200")
mkdir -p "$OUT_ROOT/logs"
for gpu in 0 1 2 3 4 5 6 7; do
  read -r start end <<< "${segments[$gpu]}"
  alpha_label=${alpha//./p}
  run_label="$mode"
  if [[ "$alpha" != "1.0" && "$alpha" != "1" ]]; then
    run_label="${mode}_alpha${alpha_label}"
  fi
  nohup "$0" --worker "$mode" "$gpu" "$start" "$end" "$alpha" \
    > "$OUT_ROOT/logs/${run_label}_gpu${gpu}_seg_${start}_${end}.outer.log" 2>&1 < /dev/null &
  echo $! > "$OUT_ROOT/logs/${run_label}_gpu${gpu}_seg_${start}_${end}.pid"
  echo "launched mode=$mode alpha=$alpha gpu=$gpu segment=$start-$end pid=$(cat "$OUT_ROOT/logs/${run_label}_gpu${gpu}_seg_${start}_${end}.pid")"
done
