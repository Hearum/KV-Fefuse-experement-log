#!/usr/bin/env bash
set -euo pipefail
cd /raid/home/hming/FusionRAG-pca-analysis

PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
EXP=MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter
ADAPTER=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/oracle_kv_lstsq_retrieval_trace_a12_fullprefix_ridge1e-4_seed20260716_e24
LOG_ROOT="$EXP/results/oracle_span_trace_fullprefix_e24/logs"

while [[ $(find "$ADAPTER/key" -type f 2>/dev/null | wc -l) -lt 22 ]]; do sleep 20; done
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py \
  --dataset musique-v2 --method position_adapter_preprocess_rate0 --rate 0.0 \
  --start 23 --end 24 --gpu 0 \
  --result-root "$EXP/results/oracle_span_trace_fullprefix_e24/endpoint" \
  --static-key-bias-path "$ADAPTER/key" --static-value-bias-path "$ADAPTER/value" \
  --static-key-bias-require-all --static-value-bias-require-all \
  >"$LOG_ROOT/e24.endpoint.log" 2>&1
