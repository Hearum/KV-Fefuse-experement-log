#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REPO=$(cd "$EXP_DIR/../.." && pwd)
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
MODEL=/mnt/qjhs-sh-lab-01/models/Qwen3-32B
DRAFT=/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
CACHE_ROOT=/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-kv-blend-beta-cache
API_URL=http://36.150.226.221:32355/v1
API_KEY=api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/results"

run_one() {
  local beta="$1"
  local gpu="$2"
  local mode="${3:-kv}"
  local tag="beta${beta//./p}_${mode}_0_25"
  local out_dir="$EXP_DIR/results/$tag"
  local cache_dir="$CACHE_ROOT/shared_musique_0_25"
  mkdir -p "$out_dir" "$cache_dir"
  cd "$REPO"
  echo "[$(date '+%F %T')] start beta=$beta mode=$mode gpu=$gpu" | tee "$EXP_DIR/logs/$tag.launch.log"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  FUSIONRAG_REPROCESS_KV_BLEND_BETA="$beta" \
  FUSIONRAG_REPROCESS_KV_BLEND_MODE="$mode" \
  "$PY" test_fusionrag_reflect_preprocess_exp.py \
    --model_type qwen3 \
    --model_path "$MODEL" \
    --model_name Qwen3-32B \
    --data_path ./data/result_reflect.json \
    --dataset_name musique \
    --cache_path "$cache_dir" \
    --result_path "$out_dir" \
    --start_sample 0 \
    --end_sample 25 \
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
  echo "[$(date '+%F %T')] done beta=$beta mode=$mode gpu=$gpu" | tee -a "$EXP_DIR/logs/$tag.launch.log"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  run_one "$@"
  exit 0
fi

betas=(0 0.25 0.5 0.75 1.0)
gpus=(${FUSIONRAG_KV_BLEND_GPUS:-0 1 2 3 4})
for i in "${!betas[@]}"; do
  beta="${betas[$i]}"
  gpu="${gpus[$i]}"
  tag="beta${beta//./p}_kv_0_25"
  nohup "$0" --worker "$beta" "$gpu" kv > "$EXP_DIR/logs/$tag.outer.log" 2>&1 < /dev/null &
  echo $! > "$EXP_DIR/logs/$tag.pid"
  echo "launched beta=$beta gpu=$gpu pid=$(cat "$EXP_DIR/logs/$tag.pid")"
done
