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
OUT=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/oracle_kv_lstsq_layerwise_retrieval_trace_a12_fullprefix_ridge1e-4_seed20260716_e24_probe
LOG_ROOT="$EXP/results/oracle_span_trace_layerwise_probe/logs"
HOST=$(hostname)
mkdir -p "$LOG_ROOT"

run_chunk() {
  local gpu="$1"
  local chunk="$2"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" \
    "$EXP/scripts/build_setup_v2_offline_prefix_bank_biases.py" \
    --source preprocess --example 24 --prefix-source retrieval_traces \
    --trace-anchors 12 --prefix-docs 0 --topk 10 --retrieval-topk 50 \
    --matcher hybrid --selection-mode oracle_kv_lstsq_layerwise --oracle-ridge 1e-4 \
    --random-seed 20260716 --chunk-ids "$chunk" \
    --metadata-name "metadata_${HOST}_gpu${gpu}_chunk${chunk}.json" \
    --output-dir "$OUT" --device cuda:0 \
    >"$LOG_ROOT/${HOST}_gpu${gpu}_chunk${chunk}.log" 2>&1
}

for pair in 0:2 1:7 2:12 3:17 4:20 5:21; do
  run_chunk "${pair%%:*}" "${pair##*:}" &
done
wait
