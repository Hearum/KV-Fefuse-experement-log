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
TAG=oracle_kv_lstsq_retrieval_trace_a12_fullprefix_ridge1e-4_seed20260716
HOST=$(hostname)
LOG_ROOT="$EXP/results/oracle_span_trace_fullprefix_e24/logs"
mkdir -p "$LOG_ROOT"

run_shard() {
  local gpu="$1"
  local chunks="$2"
  local metadata_chunks="${chunks//,/_}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" \
    "$EXP/scripts/build_setup_v2_offline_prefix_bank_biases.py" \
    --source preprocess --example 24 --prefix-source retrieval_traces \
    --trace-anchors 12 --prefix-docs 0 --topk 10 --retrieval-topk 50 \
    --matcher hybrid --selection-mode oracle_kv_lstsq --oracle-ridge 1e-4 \
    --random-seed 20260716 --chunk-ids "$chunks" \
    --metadata-name "metadata_${HOST}_gpu${gpu}_chunks${metadata_chunks}_e24_complete.json" \
    --output-dir "$ADAPTER_ROOT/${TAG}_e24" --device cuda:0 \
    >"$LOG_ROOT/${HOST}_gpu${gpu}.build.log" 2>&1
}

case "$HOST" in
  qjhs-sh-lab-01)
    (run_shard 0 2; run_shard 0 21) &
    run_shard 1 3 &
    run_shard 2 4 &
    run_shard 3 5 &
    run_shard 4 7 &
    run_shard 5 8 &
    run_shard 6 9 &
    run_shard 7 10 &
    ;;
  qjhs-sh-lab-04)
    run_shard 0 12 &
    run_shard 1 13 &
    run_shard 2 14 &
    run_shard 3 15 &
    run_shard 4 16 &
    run_shard 5 18 &
    run_shard 6 19 &
    run_shard 7 20 &
    ;;
  *) echo "Unsupported host: $HOST" >&2; exit 2 ;;
esac
wait
