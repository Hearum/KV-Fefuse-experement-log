#!/usr/bin/env bash
set -euo pipefail

REPO=/raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
MODEL=/mnt/qjhs-sh-lab-01/models/Qwen3-32B
DRAFT=/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
CACHE=/raid/home/hming/fusionrag-reflect-qwen3-full-cache
OUT_ROOT="$REPO/MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline"
FIXED="$REPO/MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_hybrid_qk_draft_rate015_full/chunk_fixed_sets_npz"
API_URL=http://36.150.226.221:32355/v1
API_KEY=api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS

run_cfg() {
  local gpu="$1"
  local start="$2"
  local end="$3"
  local cfg="$4"
  local out_dir="$OUT_ROOT/$cfg/seg_${start}_${end}"
  mkdir -p "$out_dir" "$OUT_ROOT/logs"

  local extra=()
  local method="FusionRAG"
  if [[ "$cfg" == "online_qk_rate015" ]]; then
    :
  elif [[ "$cfg" == "online_draft_rate015" ]]; then
    method="DraftModel"
    extra+=(--draft_model_path "$DRAFT")
  elif [[ "$cfg" == "offline_hybrid70_rate015" ]]; then
    extra+=(--offline_fixed_set_dir "$FIXED" --offline_fixed_set_method hybrid_draft70_qk30_score_per_chunk --offline_fixed_set_rate 0.15)
  else
    echo "unknown cfg: $cfg" >&2
    return 2
  fi

  echo "[$(date '+%F %T')] start cfg=$cfg gpu=$gpu segment=$start-$end" | tee -a "$OUT_ROOT/logs/seg_${start}_${end}.log"
  cd "$REPO"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" test_fusionrag_reflect_preprocess_exp.py \
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
    --reprocess_method "$method" \
    --revert_rope true \
    --preprocess_scope global \
    --bge_model_path "$BGE" \
    --device cuda:0 \
    --use_multi_gpu false \
    --openai_base_url "$API_URL" \
    --openai_api_key "$API_KEY" \
    --openai_model GLM-5.2 \
    "${extra[@]}" \
    > "$out_dir/run.log" 2>&1
  echo "[$(date '+%F %T')] done cfg=$cfg gpu=$gpu segment=$start-$end" | tee -a "$OUT_ROOT/logs/seg_${start}_${end}.log"
}

run_segment() {
  local gpu="$1"
  local start="$2"
  local end="$3"
  run_cfg "$gpu" "$start" "$end" online_qk_rate015
  run_cfg "$gpu" "$start" "$end" online_draft_rate015
  run_cfg "$gpu" "$start" "$end" offline_hybrid70_rate015
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  run_segment "$@"
  exit 0
fi

cd "$REPO"
mkdir -p "$OUT_ROOT/logs"
segments=("0 25" "25 50" "50 75" "75 100" "100 125" "125 150" "150 175" "175 200")
for gpu in 0 1 2 3 4 5 6 7; do
  read -r start end <<< "${segments[$gpu]}"
  nohup "$0" --worker "$gpu" "$start" "$end" \
    > "$OUT_ROOT/logs/gpu${gpu}_seg_${start}_${end}.outer.log" 2>&1 < /dev/null &
  echo $! > "$OUT_ROOT/logs/gpu${gpu}_seg_${start}_${end}.pid"
  echo "launched gpu=$gpu segment=$start-$end pid=$(cat "$OUT_ROOT/logs/gpu${gpu}_seg_${start}_${end}.pid")"
done
