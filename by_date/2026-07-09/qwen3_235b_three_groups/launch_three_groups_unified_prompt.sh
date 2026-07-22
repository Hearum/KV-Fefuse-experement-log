#!/usr/bin/env bash
set -euo pipefail

REPO=/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
MODEL=/home/hming/models/Qwen3-235B-A22B
DRAFT=/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
OUT_ROOT="$REPO/MOTIVATION_EXPERIMENTS/qwen3_235b_three_groups_unified_prompt"
CACHE=/home/hming/fusionrag-qwen3-235b-three-groups-unified-prompt-cache
API_URL=http://36.150.226.221:32355/v1
API_KEY=api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS

mkdir -p "$OUT_ROOT/logs" "$CACHE"

run_cfg() {
  local cfg="$1"
  local rate="0.15"
  local method="FusionRAG"
  local out_dir="$OUT_ROOT/$cfg/full_0_200"
  local extra=()

  if [[ "$cfg" == "full_rate1" ]]; then
    rate="1.0"
  elif [[ "$cfg" == "online_qk_rate015" ]]; then
    rate="0.15"
  elif [[ "$cfg" == "online_draft_rate015" ]]; then
    rate="0.15"
    method="DraftModel"
    extra+=(--draft_model_path "$DRAFT")
  else
    echo "unknown cfg: $cfg" >&2
    return 2
  fi

  mkdir -p "$out_dir"
  echo "[$(date "+%F %T")] start cfg=$cfg rate=$rate method=$method" | tee -a "$OUT_ROOT/logs/three_groups.log"
  cd "$REPO"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 "$PY" test_fusionrag_reflect_preprocess_exp.py \
    --model_type qwen3_moe \
    --model_path "$MODEL" \
    --model_name Qwen3-235B-A22B \
    --data_path ./data/result_reflect.json \
    --dataset_name musique \
    --cache_path "$CACHE/$cfg" \
    --result_path "$out_dir" \
    --start_sample 0 \
    --end_sample 200 \
    --rate "$rate" \
    --topk 10 \
    --preprocess true \
    --recall_method bge \
    --reprocess_method "$method" \
    --revert_rope true \
    --preprocess_scope global \
    --bge_model_path "$BGE" \
    --device cuda:0 \
    --use_multi_gpu true \
    --openai_base_url "$API_URL" \
    --openai_api_key "$API_KEY" \
    --openai_model GLM-5.2 \
    "${extra[@]}" \
    > "$out_dir/run.log" 2>&1
  echo "[$(date "+%F %T")] done cfg=$cfg" | tee -a "$OUT_ROOT/logs/three_groups.log"
}

run_cfg full_rate1
run_cfg online_qk_rate015
run_cfg online_draft_rate015
