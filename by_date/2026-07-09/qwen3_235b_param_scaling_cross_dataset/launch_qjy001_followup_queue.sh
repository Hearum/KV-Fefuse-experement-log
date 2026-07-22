#!/usr/bin/env bash
set -euo pipefail

REPO=/home/hming/FusionRAG-pca-analysis
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

wait_for_existing_musique() {
  echo "[$(date '+%F %T')] waiting qwen3_235b_three_groups_unified_prompt to finish" | tee -a "$EXP/logs/qjy001_followup_queue.log"
  while tmux has-session -t qwen3_235b_three_groups_unified_prompt 2>/dev/null; do
    sleep 300
  done
  echo "[$(date '+%F %T')] existing musique three-group session finished" | tee -a "$EXP/logs/qjy001_followup_queue.log"
}

run_one_cross() {
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
      echo "unknown cross method_label=$method_label" >&2
      return 2
      ;;
  esac

  local out_dir="$OUT_ROOT/$dataset/$method_label/full_0_${end_sample}"
  local cache_dir="$CACHE_ROOT/$dataset/$method_label"
  mkdir -p "$out_dir" "$cache_dir"
  echo "[$(date '+%F %T')] START host=$(hostname) dataset=$dataset method=$method_label rate=$rate" | tee -a "$EXP/logs/qjy001_followup_queue.log"
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
  echo "[$(date '+%F %T')] DONE host=$(hostname) dataset=$dataset method=$method_label" | tee -a "$EXP/logs/qjy001_followup_queue.log"
}

run_musique_offline() {
  local dataset=musique
  local data_path="$REPO/data/result_reflect.json"
  local end_sample=200
  local rate=0.15

  declare -a labels=(offline3b_mean_rate015 offline3b_freq_boundary2_rate015 offline32b_top2_rate015)
  declare -a dirs=(
    "$REPO/MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_draft_smart_global_rate015_full/chunk_fixed_sets_npz"
    "$REPO/MOTIVATION_EXPERIMENTS/offline_boundary_mix_rate015/chunk_fixed_sets_npz"
    "$REPO/MOTIVATION_EXPERIMENTS/offline_draft32b_teacher_rate015/fixed_sets_control_qwen3_32b/chunk_fixed_sets_npz"
  )
  declare -a methods=(draft_smart_mean_score_global draft_smart_freq_boundary0p02_global draft32b_smart_top2_mean_global)

  for i in 0 1 2; do
    local method_label="${labels[$i]}"
    local fixed_dir="${dirs[$i]}"
    local fixed_method="${methods[$i]}"
    local out_dir="$OUT_ROOT/$dataset/$method_label/full_0_${end_sample}"
    local cache_dir="$CACHE_ROOT/$dataset/$method_label"
    mkdir -p "$out_dir" "$cache_dir"
    echo "[$(date '+%F %T')] START host=$(hostname) dataset=$dataset method=$method_label fixed=$fixed_method" | tee -a "$EXP/logs/qjy001_followup_queue.log"
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
      --reprocess_method FusionRAG \
      --revert_rope true \
      --preprocess_scope global \
      --bge_model_path "$BGE" \
      --device cuda:0 \
      --use_multi_gpu true \
      --offline_fixed_set_dir "$fixed_dir" \
      --offline_fixed_set_method "$fixed_method" \
      --offline_fixed_set_rate 0.15 \
      --openai_base_url "$API_URL" \
      --openai_api_key "$API_KEY" \
      --openai_model GLM-5.2 \
      > "$out_dir/run.log" 2>&1
    echo "[$(date '+%F %T')] DONE host=$(hostname) dataset=$dataset method=$method_label" | tee -a "$EXP/logs/qjy001_followup_queue.log"
  done
}

run_hotpotqa_all() {
  for method in \
    full_rate1 \
    online_qk_rate015 \
    online_draft_rate015 \
    offline3b_mean_rate015 \
    offline3b_freq_boundary2_rate015 \
    offline32b_top2_rate015
  do
    run_one_cross hotpotqa "$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/hotpotqa_reflect.json" 260 "$method"
  done
}

wait_for_existing_musique
run_musique_offline
run_hotpotqa_all
