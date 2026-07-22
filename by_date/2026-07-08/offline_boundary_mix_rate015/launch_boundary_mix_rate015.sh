#!/usr/bin/env bash
set -euo pipefail

REPO=/raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
MODEL=/mnt/qjhs-sh-lab-01/models/Qwen3-32B
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
CACHE=/raid/home/hming/fusionrag-reflect-qwen3-full-cache
OUT_ROOT="$REPO/MOTIVATION_EXPERIMENTS/offline_boundary_mix_rate015/accuracy_runs"
FIXED="$REPO/MOTIVATION_EXPERIMENTS/offline_boundary_mix_rate015/chunk_fixed_sets_npz"
API_URL=http://36.150.226.221:32355/v1
API_KEY=api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS

METHODS=(
  draft_smart_freq_boundary0p02_global
  draft_smart_mean_boundary0p02_global
  draft_smart_freq_boundary0p03_global
  draft_smart_mean_boundary0p03_global
  draft_smart_freq_boundary0p05_global
  draft_smart_mean_boundary0p05_global
)

run_cfg() {
  local gpu="$1" start="$2" end="$3" method="$4"
  local out_dir="$OUT_ROOT/$method/seg_${start}_${end}"
  mkdir -p "$out_dir" "$OUT_ROOT/logs"
  echo "[$(date '+%F %T')] start method=$method gpu=$gpu segment=$start-$end" | tee -a "$OUT_ROOT/logs/seg_${start}_${end}.log"
  cd "$REPO"
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
  echo "[$(date '+%F %T')] done method=$method gpu=$gpu segment=$start-$end" | tee -a "$OUT_ROOT/logs/seg_${start}_${end}.log"
}

run_worker() {
  local gpu="$1"
  shift
  while [[ $# -gt 0 ]]; do
    local start="$1" end="$2"
    shift 2
    for method in "${METHODS[@]}"; do
      run_cfg "$gpu" "$start" "$end" "$method"
    done
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  run_worker "$@"
  exit 0
fi

mkdir -p "$OUT_ROOT/logs"
cd "$REPO"

# Use currently free GPUs. Each worker processes two 25-example segments sequentially.
workers=(
  "2 0 25 100 125"
  "5 25 50 125 150"
  "6 50 75 150 175"
  "7 75 100 175 200"
)

for spec in "${workers[@]}"; do
  read -r gpu s1 e1 s2 e2 <<< "$spec"
  nohup "$0" --worker "$gpu" "$s1" "$e1" "$s2" "$e2" > "$OUT_ROOT/logs/gpu${gpu}.outer.log" 2>&1 < /dev/null &
  echo $! > "$OUT_ROOT/logs/gpu${gpu}.pid"
  echo "launched gpu=$gpu segments=${s1}-${e1},${s2}-${e2} pid=$(cat "$OUT_ROOT/logs/gpu${gpu}.pid")"
done
