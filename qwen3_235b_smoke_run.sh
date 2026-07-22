#!/usr/bin/env bash
set -euo pipefail

REPO=/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
MODEL=/home/hming/models/Qwen3-235B-A22B
DRAFT=/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
EXP="$REPO/MOTIVATION_EXPERIMENTS/qwen3_235b_moe_adapter_smoke_v5"
CACHE=/home/hming/fusionrag-qwen3-235b-moe-adapter-smoke-v4-cache
API_URL=http://36.150.226.221:32355/v1
API_KEY_FILE="$REPO/MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/launch_qwen3_rate015_segments.sh"
API_KEY="$(grep -m1 '^API_KEY=' "$API_KEY_FILE" | cut -d= -f2-)"

mkdir -p "$EXP/logs" "$CACHE"

run_cfg() {
  local cfg="$1"
  local method="FusionRAG"
  local out_dir="$EXP/$cfg/seg_0_1"
  local extra=()

  if [[ "$cfg" == "online_draft_rate015" ]]; then
    method="DraftModel"
    extra+=(--draft_model_path "$DRAFT")
  elif [[ "$cfg" != "online_qk_rate015" ]]; then
    echo "unknown cfg: $cfg" >&2
    return 2
  fi

  mkdir -p "$out_dir"
  echo "[$(date '+%F %T')] start cfg=$cfg" | tee -a "$EXP/logs/smoke.log"
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
    --end_sample 1 \
    --rate 0.15 \
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
  echo "[$(date '+%F %T')] done cfg=$cfg" | tee -a "$EXP/logs/smoke.log"
}

run_cfg online_qk_rate015
run_cfg online_draft_rate015
