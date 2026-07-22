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
LOG=$EXP/logs/qjy001_recovery_remaining.log

mkdir -p "$EXP/logs" "$OUT_ROOT" "$CACHE_ROOT"

unique_rows_for_dir() {
  local dir="$1"
  "$PY" - "$dir" <<'PY'
import csv, glob, os, sys
paths=glob.glob(os.path.join(sys.argv[1], "**", "*.csv"), recursive=True)
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

run_one() {
  local dataset="$1"
  local end_sample="$2"
  local method_label="$3"
  local data_path="$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/${dataset}_reflect.json"
  local rate="0.15"
  local reprocess="FusionRAG"
  local extra=()

  case "$method_label" in
    online_draft_rate015)
      reprocess="DraftModel"
      extra+=(--draft_model_path "$DRAFT")
      ;;
    offline3b_mean_rate015)
      extra+=(--offline_fixed_set_dir "$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/fixed_sets_${dataset}_3b/chunk_fixed_sets_npz" --offline_fixed_set_method offline3b_mean_score_global --offline_fixed_set_rate 0.15)
      ;;
    offline3b_freq_boundary2_rate015)
      extra+=(--offline_fixed_set_dir "$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/fixed_sets_${dataset}_3b/chunk_fixed_sets_npz" --offline_fixed_set_method offline3b_freq_boundary0p02_global --offline_fixed_set_rate 0.15)
      ;;
    offline32b_top2_rate015)
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

  local rows
  rows=$(unique_rows_for_dir "$out_dir")
  if [[ "$rows" -ge "$end_sample" ]]; then
    echo "[$(date '+%F %T')] SKIP dataset=$dataset method=$method_label rows=$rows expected=$end_sample" | tee -a "$LOG"
    return 0
  fi

  echo "[$(date '+%F %T')] START host=$(hostname) dataset=$dataset method=$method_label rows=$rows expected=$end_sample rate=$rate" | tee -a "$LOG"
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

echo "[$(date '+%F %T')] START qjy001 recovery remaining 235B queue" | tee -a "$LOG"

run_one 2wikimqa 200 offline32b_top2_rate015
run_one hotpotqa 260 online_draft_rate015
run_one hotpotqa 260 offline3b_mean_rate015
run_one hotpotqa 260 offline3b_freq_boundary2_rate015
run_one hotpotqa 260 offline32b_top2_rate015
run_one triviaqa 270 online_draft_rate015
run_one triviaqa 270 offline3b_mean_rate015
run_one triviaqa 270 offline3b_freq_boundary2_rate015
run_one triviaqa 270 offline32b_top2_rate015

cd "$EXP"
"$PY" summarize_results.py || true
echo "[$(date '+%F %T')] FINISHED qjy001 recovery remaining 235B queue" | tee -a "$LOG"
