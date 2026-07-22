#!/usr/bin/env bash
set -euo pipefail

REPO=/raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
MODEL=/home/hming/models/Qwen3-235B-A22B
DRAFT=/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
EXP=$REPO/MOTIVATION_EXPERIMENTS/qwen3_235b_param_scaling_cross_dataset
OUT_ROOT=$EXP/results
CACHE_ROOT=/home/hming/fusionrag-qwen3-235b-param-scaling-cache
API_URL=http://36.150.226.221:32355/v1
API_KEY=api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS

mkdir -p "$EXP/logs" "$OUT_ROOT" "$CACHE_ROOT"

run_one() {
  local dataset="$1"
  local data_path="$2"
  local end_sample="$3"
  local method_label="$4"
  local rate="0.15"
  local reprocess="FusionRAG"
  local extra=()

  case "$method_label" in
    full_rate1)
      rate="1.0"
      ;;
    online_qk_rate015)
      rate="0.15"
      ;;
    online_draft_rate015)
      rate="0.15"
      reprocess="DraftModel"
      extra+=(--draft_model_path "$DRAFT")
      ;;
    offline3b_mean_rate015)
      rate="0.15"
      extra+=(--offline_fixed_set_dir "$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/fixed_sets_${dataset}_3b/chunk_fixed_sets_npz" --offline_fixed_set_method offline3b_mean_score_global --offline_fixed_set_rate 0.15)
      ;;
    offline3b_freq_boundary2_rate015)
      rate="0.15"
      extra+=(--offline_fixed_set_dir "$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/fixed_sets_${dataset}_3b/chunk_fixed_sets_npz" --offline_fixed_set_method offline3b_freq_boundary0p02_global --offline_fixed_set_rate 0.15)
      ;;
    offline32b_top2_rate015)
      rate="0.15"
      extra+=(--offline_fixed_set_dir "$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/fixed_sets_${dataset}_32b/chunk_fixed_sets_npz" --offline_fixed_set_method offline32b_top2_mean_global --offline_fixed_set_rate 0.15)
      ;;
    *)
      echo "unknown method_label=$method_label" >&2
      return 2
      ;;
  esac

  local out_dir="$OUT_ROOT/$dataset/$method_label/full_0_${end_sample}"
  local cache_dir="$CACHE_ROOT/$dataset/$method_label"
  mkdir -p "$out_dir" "$cache_dir"

  echo "[$(date '+%F %T')] START host=$(hostname) dataset=$dataset method=$method_label rate=$rate" | tee -a "$EXP/logs/qjy000_queue.log"
  cd "$REPO"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 "$PY" test_fusionrag_reflect_preprocess_exp.py \
    --model_type qwen3_moe \
    --model_path "$MODEL" \
    --model_name Qwen3-235B-A22B \
    --data_path "$data_path" \
    --dataset_name "$dataset" \
    --cache_path "$cache_dir" \
    --result_path "$out_dir" \
    --start_sample 0 \
    --end_sample "$end_sample" \
    --rate "$rate" \
    --topk 10 \
    --preprocess true \
    --recall_method bge \
    --reprocess_method "$reprocess" \
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
  echo "[$(date '+%F %T')] DONE host=$(hostname) dataset=$dataset method=$method_label" | tee -a "$EXP/logs/qjy000_queue.log"
}

run_dataset() {
  local dataset="$1"
  local data_path="$2"
  local end_sample="$3"
  for method in \
    full_rate1 \
    online_qk_rate015 \
    online_draft_rate015 \
    offline3b_mean_rate015 \
    offline3b_freq_boundary2_rate015 \
    offline32b_top2_rate015
  do
    run_one "$dataset" "$data_path" "$end_sample" "$method"
  done
}

run_dataset 2wikimqa "$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/2wikimqa_reflect.json" 200
run_dataset triviaqa "$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/triviaqa_reflect.json" 270
