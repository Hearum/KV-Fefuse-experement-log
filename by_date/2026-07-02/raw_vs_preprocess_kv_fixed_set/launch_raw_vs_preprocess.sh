#!/usr/bin/env bash
set -euo pipefail

ROOT=/raid/home/hming/FusionRAG-pca-analysis
cd "$ROOT"

PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
EXP=$ROOT/MOTIVATION_EXPERIMENTS/raw_vs_preprocess_kv_fixed_set
CACHE=/raid/home/hming/fusionrag-reflect-full-cache
MODEL=/mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct
DRAFT=/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
DATA=$ROOT/data/result_reflect.json
QK_DIR=$ROOT/MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_qk_rate015_full/combined/chunk_fixed_sets_npz
DRAFT_DIR=$ROOT/MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_draft_rate015_full/combined/chunk_fixed_sets_npz
HYBRID_DIR=$ROOT/MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_hybrid_qk_draft_rate015_full/chunk_fixed_sets_npz

common_args=(
  --model_type qwen2
  --model_path "$MODEL"
  --model_name Qwen2.5-7B-Instruct
  --bge_model_path "$BGE"
  --data_path "$DATA"
  --dataset_name musique
  --cache_path "$CACHE"
  --topk 10
  --preprocess false
  --preprocess_scope global
  --recall_method bge
  --revert_rope true
  --use_multi_gpu false
  --openai_base_url http://36.150.226.221:32355/v1
  --openai_api_key api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS
  --openai_model GLM-5.2
)

run_one() {
  local gpu=$1; shift
  local label=$1; shift
  local out=$EXP/$label
  mkdir -p "$out"
  echo "[$(date +%F %T)] START $label on GPU $gpu" | tee "$EXP/$label.log"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    export PYTHONUNBUFFERED=1
    "$PY" test_fusionrag_reflect_preprocess_exp.py \
      "${common_args[@]}" \
      --result_path "$out" \
      --device cuda:0 \
      "$@"
  ) >> "$EXP/$label.log" 2>&1 &
  echo $! > "$EXP/$label.pid"
}

run_one 0 raw_rate0_no_doc_recompute \
  --reprocess_method FusionRAG --rate 0.0

run_one 1 raw_online_qk_rate015 \
  --reprocess_method FusionRAG --rate 0.15

run_one 2 raw_offline_qk_frequency \
  --reprocess_method FusionRAG --rate 0.15 \
  --offline_fixed_set_dir "$QK_DIR" \
  --offline_fixed_set_method qk_frequency_per_chunk \
  --offline_fixed_set_rate 0.15

run_one 3 raw_offline_qk_mean \
  --reprocess_method FusionRAG --rate 0.15 \
  --offline_fixed_set_dir "$QK_DIR" \
  --offline_fixed_set_method qk_mean_score_per_chunk \
  --offline_fixed_set_rate 0.15

run_one 4 raw_offline_draft_frequency \
  --reprocess_method FusionRAG --rate 0.15 \
  --offline_fixed_set_dir "$DRAFT_DIR" \
  --offline_fixed_set_method draft_frequency_per_chunk \
  --offline_fixed_set_rate 0.15

run_one 5 raw_offline_draft_mean \
  --reprocess_method FusionRAG --rate 0.15 \
  --offline_fixed_set_dir "$DRAFT_DIR" \
  --offline_fixed_set_method draft_mean_score_per_chunk \
  --offline_fixed_set_rate 0.15

run_one 6 raw_hybrid_draft70_qk30 \
  --reprocess_method FusionRAG --rate 0.15 \
  --offline_fixed_set_dir "$HYBRID_DIR" \
  --offline_fixed_set_method hybrid_draft70_qk30_score_per_chunk \
  --offline_fixed_set_rate 0.15

(
  export FUSIONRAG_FORCE_SPARSE_ATTENTION=1
  export FUSIONRAG_DRAFT_SELECTION_VARIANT=profile
  run_one 7 raw_online_draft_profile_sparse \
    --reprocess_method DraftModel --rate 0.15 \
    --draft_model_path "$DRAFT"
)

wait

echo "[$(date +%F %T)] all raw-vs-preprocess jobs finished" | tee -a "$EXP/launch.log"
