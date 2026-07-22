#!/usr/bin/env bash
set -euo pipefail

ROOT=/raid/home/hming/FusionRAG-pca-analysis
cd "$ROOT"

EXP=MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter
OUT=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/oracle_kv_lstsq_retrieval_trace_a32_fullprefix_ridge1e-4_seed20260716_e24_probe
LOG_ROOT="$EXP/results/oracle_span_trace_a32_e24/logs"
mkdir -p "$LOG_ROOT"

while [[ $(find "$OUT/key" -maxdepth 1 -type f -name '24_*_key_bias.pt' | wc -l) -lt 22 ]]; do
  sleep 20
done

CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py \
  --dataset musique-v2 --method position_adapter_preprocess_rate0 --rate 0.0 \
  --start 23 --end 24 --gpu 0 \
  --result-root "$EXP/results/oracle_span_trace_a32_e24/endpoint" \
  --static-key-bias-path "$OUT/key" --static-value-bias-path "$OUT/value" \
  --static-key-bias-require-all --static-value-bias-require-all \
  >"$LOG_ROOT/e24.endpoint.log" 2>&1
