#!/usr/bin/env bash
set -euo pipefail

if [[ -d /raid/home/hming/FusionRAG-pca-analysis ]]; then
  ROOT=/raid/home/hming/FusionRAG-pca-analysis
else
  ROOT=/home/hming/FusionRAG-pca-analysis
fi
cd "$ROOT"

PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
EXP=MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter
ADAPTER_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2
TAG=oracle_kv_lstsq_retrieval_trace_a12_fullprefix_ridge1e-4_seed20260716_e24
HOST=$(hostname)
LOG_ROOT="$EXP/results/oracle_span_trace_fullprefix_e24/layer_diagnostics"
mkdir -p "$LOG_ROOT"

run_chunk() {
  local gpu="$1"
  local chunk="$2"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" \
    "$EXP/scripts/build_setup_v2_offline_prefix_bank_biases.py" \
    --source preprocess --example 24 --prefix-source retrieval_traces \
    --trace-anchors 12 --prefix-docs 0 --topk 10 --retrieval-topk 50 \
    --matcher hybrid --selection-mode oracle_kv_lstsq --oracle-ridge 1e-4 \
    --random-seed 20260716 --chunk-ids "$chunk" \
    --metadata-name "metadata_layerdiag_${HOST}_gpu${gpu}_chunk${chunk}.json" \
    --output-dir "$ADAPTER_ROOT/$TAG" --device cuda:0 \
    >"$LOG_ROOT/${HOST}_gpu${gpu}_chunk${chunk}.log" 2>&1
}

run_chunk 0 2 &
run_chunk 1 7 &
run_chunk 2 12 &
run_chunk 3 17 &
run_chunk 4 20 &
run_chunk 5 21 &
wait
