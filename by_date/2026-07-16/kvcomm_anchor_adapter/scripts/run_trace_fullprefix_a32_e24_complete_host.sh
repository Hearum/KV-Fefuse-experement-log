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
OUT=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/oracle_kv_lstsq_retrieval_trace_a32_fullprefix_ridge1e-4_seed20260716_e24_probe
LOG_ROOT="$EXP/results/oracle_span_trace_a32_e24/logs"
HOST=$(hostname)
mkdir -p "$LOG_ROOT"

run_chunk() {
  local gpu="$1"
  local chunk="$2"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" \
    "$EXP/scripts/build_setup_v2_offline_prefix_bank_biases.py" \
    --source preprocess --example 24 --prefix-source retrieval_traces \
    --trace-anchors 32 --prefix-docs 0 --topk 10 --retrieval-topk 50 \
    --matcher hybrid --selection-mode oracle_kv_lstsq --oracle-ridge 1e-4 \
    --random-seed 20260716 --chunk-ids "$chunk" \
    --metadata-name "metadata_${HOST}_gpu${gpu}_chunk${chunk}.json" \
    --output-dir "$OUT" --device cuda:0 \
    >"$LOG_ROOT/${HOST}_gpu${gpu}_chunk${chunk}.log" 2>&1
}

case "$HOST" in
  qjhs-sh-lab-01) pairs=(0:1 1:3 2:5 3:6 4:8 5:9 6:11) ;;
  qjhs-sh-lab-04) pairs=(0:13 1:14 2:15 3:16 4:18 5:19 6:22) ;;
  *) echo "Unsupported host: $HOST" >&2; exit 2 ;;
esac

for pair in "${pairs[@]}"; do
  run_chunk "${pair%%:*}" "${pair##*:}" &
done
wait
