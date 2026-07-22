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
TAG=oracle_kv_lstsq_mixed_bge3_random9_a12_ridge1e-4_seed20260716
HOST=$(hostname)
LOG_ROOT="$EXP/results/oracle_span/logs"
mkdir -p "$LOG_ROOT"

run_shard() {
  local gpu="$1" ex="$2" chunks="$3"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" \
    "$EXP/scripts/build_setup_v2_offline_prefix_bank_biases.py" \
    --source preprocess --example "$ex" --prefix-source mixed \
    --bge-anchors 3 --random-anchors 9 --prefix-docs 3 \
    --matcher hybrid --selection-mode oracle_kv_lstsq --oracle-ridge 1e-4 \
    --random-seed 20260716 --chunk-ids "$chunks" \
    --metadata-name "metadata_${HOST}_gpu${gpu}_e${ex}.json" \
    --output-dir "$ADAPTER_ROOT/${TAG}_e${ex}" --device cuda:0 \
    >"$LOG_ROOT/${HOST}_gpu${gpu}_e${ex}.build.log" 2>&1
}

case "$HOST" in
  qjhs-sh-lab-01)
    run_shard 0 38 2,3,4,5,6 &
    run_shard 1 38 7,8,9,10,11,12 &
    run_shard 2 43 1,2,3,4,5 &
    run_shard 3 43 6,7,8,9,10 &
    run_shard 4 24 1,2,3,4,5,6 &
    run_shard 5 24 7,8,9,10,11 &
    run_shard 6 54 1,2,3,4,5,6,7 &
    run_shard 7 54 8,9,10,11,12,13 &
    ;;
  qjhs-sh-lab-04)
    run_shard 0 38 13,14,15,16,17,18 &
    run_shard 1 38 19,20,21,22,23,24 &
    run_shard 2 43 11,12,13,14,15 &
    run_shard 3 43 16,17,18,19 &
    run_shard 4 24 12,13,14,15,16,17 &
    run_shard 5 24 18,19,20,21,22 &
    run_shard 6 54 14,15,16,17,18,19 &
    run_shard 7 54 20,21,22,23,24,25 &
    ;;
  *) echo "Unsupported host: $HOST" >&2; exit 2 ;;
esac
wait
