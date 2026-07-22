#!/usr/bin/env bash
set -euo pipefail

REPO=/raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
MODEL=/mnt/qjhs-sh-lab-01/models/Qwen3-32B
DRAFT=/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
CACHE=/raid/home/hming/fusionrag-reflect-qwen3-full-cache
EXP="$REPO/MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration"
OUT_ROOT="$EXP/qwen3_docgen_fair_budget"
FIXED="$EXP/hybrid_old70_docgen30_rate0p1/chunk_fixed_sets_npz"
API_URL=http://36.150.226.221:32355/v1
API_KEY=api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS
METHOD=hybrid_old70_docgen30_frequency_global
RUN=offline10_hybrid_old70_docgen30_draft005

run_segment() {
  local gpu="$1" start="$2" end="$3"
  local out_dir="$OUT_ROOT/$RUN/seg_${start}_${end}"
  mkdir -p "$out_dir" "$OUT_ROOT/logs"
  cd "$REPO"
  echo "START $RUN gpu=$gpu seg=$start-$end $(date)"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
  FUSIONRAG_RESIDUAL_ONLINE_RATE=0.05 FUSIONRAG_RESIDUAL_ONLINE_METHOD=DraftModel \
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
    --offline_fixed_set_dir "$FIXED" \
    --offline_fixed_set_method "$METHOD" \
    --offline_fixed_set_rate 0.10 \
    > "$out_dir/run.log" 2>&1
  echo "DONE $RUN gpu=$gpu seg=$start-$end $(date)"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  run_segment "$@"
  exit 0
fi

mkdir -p "$OUT_ROOT/logs"
segments=("0 25" "25 50" "50 75" "75 100" "100 125" "125 150" "150 175" "175 200")
for gpu in 0 1 2 3 4 5 6 7; do
  read -r start end <<< "${segments[$gpu]}"
  nohup "$0" --worker "$gpu" "$start" "$end" > "$OUT_ROOT/logs/${RUN}_seg_${start}_${end}.outer.log" 2>&1 < /dev/null &
  echo $! > "$OUT_ROOT/logs/${RUN}_seg_${start}_${end}.pid"
  echo "launched gpu=$gpu seg=$start-$end pid=$(cat "$OUT_ROOT/logs/${RUN}_seg_${start}_${end}.pid")"
done
