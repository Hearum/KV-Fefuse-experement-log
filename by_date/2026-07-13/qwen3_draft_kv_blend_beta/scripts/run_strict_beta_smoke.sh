#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/raid/home/hming/FusionRAG-pca-analysis}
PY=${PY:-/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python}
MODEL=${MODEL:-/mnt/qjhs-sh-lab-01/models/Qwen3-32B}
DRAFT=${DRAFT:-/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct}
BGE=${BGE:-/mnt/qjhs-sh-lab-01/models/bge-m3}
CACHE=${CACHE:-/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-attn-alpha-cross-cache-v2/worker_gpu2/hotpotqa}
OUT_ROOT=${OUT_ROOT:-/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_strict_smoke}
API_URL=${API_URL:-http://36.150.226.221:32355/v1}
API_KEY=${API_KEY:-api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS}
API_MODEL=${API_MODEL:-GLM-5.2}
GPU=${GPU:-0}
START=${START:-0}
END=${END:-1}

run_case() {
  local name="$1"
  local rate="$2"
  local beta="$3"
  local strict="$4"
  local out="$OUT_ROOT/${name}_hotpot_${START}_${END}"
  mkdir -p "$out"
  cd "$REPO"
  local envs=(CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1)
  if [[ "$strict" == "1" ]]; then
    envs+=(FUSIONRAG_STRICT_BETA_QUERY_PREFILL=1 FUSIONRAG_REPROCESS_KV_BLEND_BETA="$beta" FUSIONRAG_REPROCESS_KV_BLEND_MODE=kv)
  fi
  env "${envs[@]}" "$PY" test_fusionrag_reflect_preprocess_exp.py \
    --model_type qwen3 \
    --model_path "$MODEL" \
    --model_name Qwen3-32B \
    --data_path MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/hotpotqa_reflect.json \
    --dataset_name hotpotqa \
    --cache_path "$CACHE" \
    --result_path "$out" \
    --start_sample "$START" \
    --end_sample "$END" \
    --rate "$rate" \
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
    --openai_model "$API_MODEL" \
    > "$out/run.log" 2>&1
}

run_case strict_beta0 0.15 0 1
run_case strict_beta1 0.15 1 1
run_case true_rate0 0 0 0
