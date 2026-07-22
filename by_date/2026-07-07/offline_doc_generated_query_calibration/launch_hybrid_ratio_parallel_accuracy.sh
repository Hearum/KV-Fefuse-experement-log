#!/usr/bin/env bash
set -euo pipefail
REPO=/raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
MODEL=/mnt/qjhs-sh-lab-01/models/Qwen3-32B
DRAFT=/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
CACHE=/raid/home/hming/fusionrag-reflect-qwen3-full-cache
EXP="$REPO/MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration"
OUT_ROOT="$EXP/qwen3_docgen_fair_budget"
API_URL=http://36.150.226.221:32355/v1
API_KEY=api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS
run_segment() {
  local gpu="$1" label="$2" method="$3" fixed="$4" start="$5" end="$6"
  local run="offline10_hybrid_${label}_draft005"
  local out_dir="$OUT_ROOT/$run/seg_${start}_${end}"
  mkdir -p "$out_dir" "$OUT_ROOT/logs"
  cd "$REPO"
  echo "START $run gpu=$gpu seg=$start-$end $(date)"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
  FUSIONRAG_RESIDUAL_ONLINE_RATE=0.05 FUSIONRAG_RESIDUAL_ONLINE_METHOD=DraftModel \
  "$PY" test_fusionrag_reflect_preprocess_exp.py \
    --model_type qwen3 --model_path "$MODEL" --model_name Qwen3-32B \
    --data_path ./data/result_reflect.json --dataset_name musique --cache_path "$CACHE" \
    --result_path "$out_dir" --start_sample "$start" --end_sample "$end" \
    --rate 0.15 --topk 10 --preprocess true --recall_method bge \
    --reprocess_method DraftModel --draft_model_path "$DRAFT" --revert_rope true \
    --preprocess_scope global --bge_model_path "$BGE" --device cuda:0 --use_multi_gpu false \
    --openai_base_url "$API_URL" --openai_api_key "$API_KEY" --openai_model GLM-5.2 \
    --offline_fixed_set_dir "$fixed" --offline_fixed_set_method "$method" --offline_fixed_set_rate 0.10 \
    > "$out_dir/run.log" 2>&1
  echo "DONE $run gpu=$gpu seg=$start-$end $(date)"
}
if [[ "${1:-}" == "--worker" ]]; then shift; run_segment "$@"; exit 0; fi
mkdir -p "$OUT_ROOT/logs"
# old90 on GPUs 0-3, old50 on GPUs 4-7
segs=("0 50" "50 100" "100 150" "150 200")
for i in 0 1 2 3; do
  read -r s e <<< "${segs[$i]}"
  nohup "$0" --worker "$i" old90_docgen10 hybrid_old90_docgen10_frequency_global "$EXP/hybrid_old90_docgen10_rate0p1/chunk_fixed_sets_npz" "$s" "$e" > "$OUT_ROOT/logs/offline10_hybrid_old90_docgen10_draft005_seg_${s}_${e}.outer.log" 2>&1 < /dev/null &
  echo $! > "$OUT_ROOT/logs/offline10_hybrid_old90_docgen10_draft005_seg_${s}_${e}.pid"
  gpu=$((i+4))
  nohup "$0" --worker "$gpu" old50_docgen50 hybrid_old50_docgen50_frequency_global "$EXP/hybrid_old50_docgen50_rate0p1/chunk_fixed_sets_npz" "$s" "$e" > "$OUT_ROOT/logs/offline10_hybrid_old50_docgen50_draft005_seg_${s}_${e}.outer.log" 2>&1 < /dev/null &
  echo $! > "$OUT_ROOT/logs/offline10_hybrid_old50_docgen50_draft005_seg_${s}_${e}.pid"
done
