#!/usr/bin/env bash
set -euo pipefail

REPO=/raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
MODEL=/mnt/qjhs-sh-lab-01/models/Qwen3-32B
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
CACHE=/raid/home/hming/fusionrag-reflect-qwen3-full-cache
EXP="$REPO/MOTIVATION_EXPERIMENTS/offline_draft32b_teacher_rate015"
OUT_ROOT="$EXP/accuracy_runs_control_qwen3_32b_boundary_mix"
FIXED="$EXP/fixed_sets_control_qwen3_32b_boundary_mix/chunk_fixed_sets_npz"
API_URL=http://36.150.226.221:32355/v1
API_KEY=api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS
METHODS=(
  draft32b_smart_freq_boundary0p02_global
  draft32b_smart_mean_boundary0p02_global
  draft32b_smart_top2_boundary0p02_global
  draft32b_smart_freq_boundary0p03_global
  draft32b_smart_mean_boundary0p03_global
  draft32b_smart_top2_boundary0p03_global
  draft32b_smart_freq_boundary0p05_global
  draft32b_smart_mean_boundary0p05_global
  draft32b_smart_top2_boundary0p05_global
)

run_cfg() {
  local gpu="$1" start="$2" end="$3" method="$4"
  local out_dir="$OUT_ROOT/$method/seg_${start}_${end}"
  mkdir -p "$out_dir" "$EXP/logs"
  cd "$REPO"
  if [[ -f "$out_dir/run.log" ]] && grep -q "FINAL RESULTS" "$out_dir/run.log"; then
    echo "[$(date '+%F %T')] skip complete boundary method=$method segment=$start-$end" | tee -a "$EXP/logs/boundary_accuracy_segments.log"
    return 0
  fi
  echo "[$(date '+%F %T')] start boundary accuracy method=$method gpu=$gpu segment=$start-$end" | tee -a "$EXP/logs/boundary_accuracy_segments.log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" test_fusionrag_reflect_preprocess_exp.py \
    --model_type qwen3 \
    --model_path "$MODEL" \
    --model_name Qwen3-32B \
    --data_path ./data/result_reflect.json \
    --dataset_name musique \
    --cache_path "$CACHE" \
    --result_path "$out_dir" \
    --start_sample "$start" \
    --end_sample "$end" \
    --rate 0.15 \
    --topk 10 \
    --preprocess true \
    --recall_method bge \
    --reprocess_method FusionRAG \
    --revert_rope true \
    --preprocess_scope global \
    --bge_model_path "$BGE" \
    --device cuda:0 \
    --use_multi_gpu false \
    --openai_base_url "$API_URL" \
    --openai_api_key "$API_KEY" \
    --openai_model GLM-5.2 \
    --offline_fixed_set_dir "$FIXED" \
    --offline_fixed_set_method "$method" \
    --offline_fixed_set_rate 0.15 \
    > "$out_dir/run.log" 2>&1
  echo "[$(date '+%F %T')] done boundary accuracy method=$method gpu=$gpu segment=$start-$end" | tee -a "$EXP/logs/boundary_accuracy_segments.log"
}

run_segment() {
  local gpu="$1" start="$2" end="$3"
  for method in "${METHODS[@]}"; do
    run_cfg "$gpu" "$start" "$end" "$method"
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  run_segment "$@"
  exit 0
fi

mkdir -p "$EXP/logs"
segments=("0 25" "25 50" "50 75" "75 100" "100 125" "125 150" "150 175" "175 200")
for gpu in 0 1 2 3 4 5 6 7; do
  read -r start end <<< "${segments[$gpu]}"
  nohup "$0" --worker "$gpu" "$start" "$end" > "$EXP/logs/boundary_accuracy_gpu${gpu}_${start}_${end}.outer.log" 2>&1 < /dev/null &
  echo $! > "$EXP/logs/boundary_accuracy_gpu${gpu}_${start}_${end}.pid"
  echo "launched boundary accuracy gpu=$gpu segment=$start-$end pid=$(cat "$EXP/logs/boundary_accuracy_gpu${gpu}_${start}_${end}.pid")"
done
