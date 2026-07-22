#!/usr/bin/env bash
set -euo pipefail
ROOT=/raid/home/hming/FusionRAG-pca-analysis
cd "$ROOT"
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
EXP=$ROOT/MOTIVATION_EXPERIMENTS/offline_raw_qk_selector_rate015
FIXED=$EXP/combined/chunk_fixed_sets_npz
CACHE=/raid/home/hming/fusionrag-reflect-full-cache
MODEL=/mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
DATA=$ROOT/data/result_reflect.json
mkdir -p "$EXP/generation_logs"
common=(
  --model_type qwen2
  --model_path "$MODEL"
  --model_name Qwen2.5-7B-Instruct
  --bge_model_path "$BGE"
  --data_path "$DATA"
  --dataset_name musique
  --cache_path "$CACHE"
  --topk 10
  --preprocess_scope global
  --recall_method bge
  --revert_rope true
  --use_multi_gpu false
  --openai_base_url http://36.150.226.221:32355/v1
  --openai_api_key api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS
  --openai_model GLM-5.2
  --reprocess_method FusionRAG
  --rate 0.15
  --offline_fixed_set_dir "$FIXED"
  --offline_fixed_set_rate 0.15
)
run_one() {
  local gpu=$1; shift
  local label=$1; shift
  local preprocess=$1; shift
  local method=$1; shift
  local out=$EXP/$label
  mkdir -p "$out"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    export PYTHONUNBUFFERED=1
    "$PY" test_fusionrag_reflect_preprocess_exp.py \
      "${common[@]}" \
      --result_path "$out" \
      --device cuda:0 \
      --preprocess "$preprocess" \
      --offline_fixed_set_method "$method"
  ) > "$EXP/generation_logs/$label.log" 2>&1 &
  echo $! > "$EXP/generation_logs/$label.pid"
}
run_one 0 rawqk_freq_preprocess_runtime true qk_frequency_per_chunk
run_one 1 rawqk_mean_preprocess_runtime true qk_mean_score_per_chunk
run_one 2 rawqk_freq_raw_runtime false qk_frequency_per_chunk
run_one 3 rawqk_mean_raw_runtime false qk_mean_score_per_chunk
wait
