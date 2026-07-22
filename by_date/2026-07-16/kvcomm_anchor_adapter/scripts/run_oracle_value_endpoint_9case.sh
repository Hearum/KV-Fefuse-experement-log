#!/usr/bin/env bash
set -euo pipefail

cd /raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
EXP=MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter
ADAPTER_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2
TAG=oracle_value_mixed_bge3_random9_a12_top1_seed20260716
LOG_ROOT="$EXP/results/oracle_compatibility/logs"

declare -A EXPECTED=([38]=24 [43]=19 [24]=22 [54]=25 [14]=22 [68]=21 [66]=20 [13]=24 [63]=23)
for ex in 38 43 24 54 14 68 66 13 63; do
  key_dir="$ADAPTER_ROOT/${TAG}_e${ex}/key"
  while [[ $(find "$key_dir" -type f 2>/dev/null | wc -l) -lt ${EXPECTED[$ex]} ]]; do
    sleep 20
  done
done

run_endpoint() {
  local gpu="$1"
  local ex="$2"
  local start=$((ex - 1))
  local adapter="$ADAPTER_ROOT/${TAG}_e${ex}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" \
    MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py \
    --dataset musique-v2 \
    --method position_adapter_preprocess_rate0 \
    --rate 0.0 \
    --start "$start" \
    --end "$ex" \
    --gpu 0 \
    --result-root "$EXP/results/oracle_compatibility/e${ex}_${TAG}" \
    --static-key-bias-path "$adapter/key" \
    --static-value-bias-path "$adapter/value" \
    --static-key-bias-require-all \
    --static-value-bias-require-all \
    >"$LOG_ROOT/e${ex}.endpoint.log" 2>&1
}

(run_endpoint 0 38; run_endpoint 0 63) &
run_endpoint 1 43 &
run_endpoint 2 24 &
run_endpoint 3 54 &
run_endpoint 4 14 &
run_endpoint 5 68 &
run_endpoint 6 66 &
run_endpoint 7 13 &
wait
