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
LOG=$EXP/logs/qjy001_resume_remaining.log

mkdir -p "$EXP/logs" "$OUT_ROOT" "$CACHE_ROOT"

rate_tag() {
  if [[ "$1" == "1.0" ]]; then
    echo "rate_1.0"
  else
    echo "rate_0.15"
  fi
}

unique_rows_for_csv() {
  local pattern="$1"
  "$PY" - "$pattern" <<'PY'
import csv, glob, sys
paths=glob.glob(sys.argv[1], recursive=True)
if not paths:
    print(0)
    raise SystemExit
best=0
for path in paths:
    try:
        with open(path, encoding="utf-8", newline="") as f:
            rows=list(csv.DictReader(f))
        keys={(r.get("Main Question",""), r.get("Sub Question","")) for r in rows}
        best=max(best, len(keys))
    except Exception:
        pass
print(best)
PY
}

maybe_skip() {
  local dataset="$1"
  local method_label="$2"
  local end_sample="$3"
  local rate="$4"
  local out_dir="$OUT_ROOT/$dataset/$method_label/full_0_${end_sample}"
  local tag
  tag=$(rate_tag "$rate")
  local pattern="$out_dir/**/*.csv"
  local rows
  rows=$(unique_rows_for_csv "$pattern")
  if [[ "$rows" -ge "$end_sample" ]]; then
    echo "[$(date '+%F %T')] SKIP dataset=$dataset method=$method_label rows=$rows expected=$end_sample" | tee -a "$LOG"
    return 0
  fi
  echo "[$(date '+%F %T')] NEED_RUN dataset=$dataset method=$method_label rows=$rows expected=$end_sample rate_tag=$tag" | tee -a "$LOG"
  return 1
}

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
      if [[ "$dataset" == "musique" ]]; then
        extra+=(--offline_fixed_set_dir "$REPO/MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_draft_smart_global_rate015_full/chunk_fixed_sets_npz" --offline_fixed_set_method draft_smart_mean_score_global --offline_fixed_set_rate 0.15)
      else
        extra+=(--offline_fixed_set_dir "$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/fixed_sets_${dataset}_3b/chunk_fixed_sets_npz" --offline_fixed_set_method offline3b_mean_score_global --offline_fixed_set_rate 0.15)
      fi
      ;;
    offline3b_freq_boundary2_rate015)
      rate="0.15"
      if [[ "$dataset" == "musique" ]]; then
        extra+=(--offline_fixed_set_dir "$REPO/MOTIVATION_EXPERIMENTS/offline_boundary_mix_rate015/chunk_fixed_sets_npz" --offline_fixed_set_method draft_smart_freq_boundary0p02_global --offline_fixed_set_rate 0.15)
      else
        extra+=(--offline_fixed_set_dir "$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/fixed_sets_${dataset}_3b/chunk_fixed_sets_npz" --offline_fixed_set_method offline3b_freq_boundary0p02_global --offline_fixed_set_rate 0.15)
      fi
      ;;
    offline32b_top2_rate015)
      rate="0.15"
      if [[ "$dataset" == "musique" ]]; then
        extra+=(--offline_fixed_set_dir "$REPO/MOTIVATION_EXPERIMENTS/offline_draft32b_teacher_rate015/fixed_sets_control_qwen3_32b/chunk_fixed_sets_npz" --offline_fixed_set_method draft32b_smart_top2_mean_global --offline_fixed_set_rate 0.15)
      else
        extra+=(--offline_fixed_set_dir "$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/fixed_sets_${dataset}_32b/chunk_fixed_sets_npz" --offline_fixed_set_method offline32b_top2_mean_global --offline_fixed_set_rate 0.15)
      fi
      ;;
    *)
      echo "unknown method_label=$method_label" >&2
      return 2
      ;;
  esac

  if maybe_skip "$dataset" "$method_label" "$end_sample" "$rate"; then
    return 0
  fi

  local out_dir="$OUT_ROOT/$dataset/$method_label/full_0_${end_sample}"
  local cache_dir="$CACHE_ROOT/$dataset/$method_label"
  mkdir -p "$out_dir" "$cache_dir"

  echo "[$(date '+%F %T')] START host=$(hostname) dataset=$dataset method=$method_label rate=$rate" | tee -a "$LOG"
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
  echo "[$(date '+%F %T')] DONE host=$(hostname) dataset=$dataset method=$method_label" | tee -a "$LOG"
}

run_methods() {
  local dataset="$1"
  local data_path="$2"
  local end_sample="$3"
  shift 3
  for method in "$@"; do
    run_one "$dataset" "$data_path" "$end_sample" "$method"
  done
}

echo "[$(date '+%F %T')] RESUME remaining qwen3-235b param-scaling queue on $(hostname)" | tee -a "$LOG"

run_methods musique "$REPO/data/result_reflect.json" 200 \
  offline32b_top2_rate015

run_methods hotpotqa "$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/hotpotqa_reflect.json" 260 \
  full_rate1 \
  online_qk_rate015 \
  online_draft_rate015 \
  offline3b_mean_rate015 \
  offline3b_freq_boundary2_rate015 \
  offline32b_top2_rate015

run_methods 2wikimqa "$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/2wikimqa_reflect.json" 200 \
  offline32b_top2_rate015

run_methods triviaqa "$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/triviaqa_reflect.json" 270 \
  full_rate1 \
  online_qk_rate015 \
  online_draft_rate015 \
  offline3b_mean_rate015 \
  offline3b_freq_boundary2_rate015 \
  offline32b_top2_rate015

cd "$EXP"
"$PY" summarize_results.py || true
echo "[$(date '+%F %T')] FINISHED remaining queue" | tee -a "$LOG"
