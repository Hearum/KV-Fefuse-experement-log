#!/usr/bin/env bash
set -euo pipefail

cd /raid/home/hming/FusionRAG-pca-analysis

PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
EXP=MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter
ADAPTER_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2
LOG_ROOT="$EXP/results/mixed_anchor_pool/logs"
mkdir -p "$LOG_ROOT"

run_case() {
  local gpu="$1"
  local ex="$2"
  local start=$((ex - 1))
  local tag="mixed_bge3_random9_a12_top4_pdocs3_cachedmatch_seed20260716"
  local adapter="$ADAPTER_ROOT/${tag}_e${ex}"
  local result="$EXP/results/mixed_anchor_pool/e${ex}_${tag}"

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" \
    "$EXP/scripts/build_setup_v2_offline_prefix_bank_biases.py" \
    --source preprocess \
    --example "$ex" \
    --prefix-source mixed \
    --bge-anchors 3 \
    --random-anchors 9 \
    --topk-anchors 4 \
    --prefix-docs 3 \
    --matcher hybrid \
    --temperature 0.07 \
    --random-seed 20260716 \
    --output-dir "$adapter" \
    --device cuda:0 \
    >"$LOG_ROOT/e${ex}.build.log" 2>&1

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" \
    MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py \
    --dataset musique-v2 \
    --method position_adapter_preprocess_rate0 \
    --rate 0.0 \
    --start "$start" \
    --end "$ex" \
    --gpu 0 \
    --result-root "$result" \
    --static-key-bias-path "$adapter/key" \
    --static-value-bias-path "$adapter/value" \
    --static-key-bias-require-all \
    --static-value-bias-require-all \
    >"$LOG_ROOT/e${ex}.endpoint.log" 2>&1
}

(run_case 0 38; run_case 0 63) &
run_case 1 43 &
run_case 2 24 &
run_case 3 54 &
run_case 4 14 &
run_case 5 68 &
run_case 6 66 &
run_case 7 13 &
wait
