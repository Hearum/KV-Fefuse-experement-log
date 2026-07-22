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
TAG=oracle_value_mixed_bge3_random9_a12_top1_seed20260716
HOST=$(hostname)
LOG_ROOT="$EXP/results/oracle_compatibility/logs"
mkdir -p "$LOG_ROOT"

run_shard() {
  local gpu="$1"
  local ex="$2"
  local chunks="$3"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" \
    "$EXP/scripts/build_setup_v2_offline_prefix_bank_biases.py" \
    --source preprocess \
    --example "$ex" \
    --prefix-source mixed \
    --bge-anchors 3 \
    --random-anchors 9 \
    --topk-anchors 1 \
    --prefix-docs 3 \
    --matcher hybrid \
    --selection-mode oracle_value_l2 \
    --temperature 0.07 \
    --random-seed 20260716 \
    --chunk-ids "$chunks" \
    --metadata-name "metadata_${HOST}_gpu${gpu}_e${ex}.json" \
    --output-dir "$ADAPTER_ROOT/${TAG}_e${ex}" \
    --device cuda:0 \
    >"$LOG_ROOT/${HOST}_gpu${gpu}_e${ex}.build.log" 2>&1
}

case "$HOST" in
  qjhs-sh-lab-01)
    (run_shard 0 38 2,3,4,5,6,7,8,9,10,11,12; run_shard 0 63 1,2,3,4,5,6,7,8,9,10,11,12) &
    run_shard 1 43 1,2,3,4,5,6,7,8,9,10 &
    run_shard 2 24 1,2,3,4,5,6,7,8,9,10,11 &
    run_shard 3 54 1,2,3,4,5,6,7,8,9,10,11,12,13 &
    run_shard 4 14 1,2,3,4,5,6,7,8,9,10,11 &
    run_shard 5 68 1,2,3,4,5,6,7,8,9,10,11 &
    run_shard 6 66 1,2,3,4,5,6,7,8,9,10 &
    run_shard 7 13 1,2,3,4,5,6,7,8,9,10,11,12 &
    ;;
  qjhs-sh-lab-04)
    (run_shard 0 38 13,14,15,16,17,18,19,20,21,22,23,24; run_shard 0 63 13,14,15,16,17,18,19,20,21,22,23) &
    run_shard 1 43 11,12,13,14,15,16,17,18,19 &
    run_shard 2 24 12,13,14,15,16,17,18,19,20,21,22 &
    run_shard 3 54 14,15,16,17,18,19,20,21,22,23,24,25 &
    run_shard 4 14 12,13,14,15,16,17,18,19,20,21,22 &
    run_shard 5 68 12,13,14,15,16,17,18,19,20,21 &
    run_shard 6 66 11,12,13,14,15,16,17,18,19,20 &
    run_shard 7 13 13,14,15,16,17,18,19,20,21,22,23,24 &
    ;;
  *)
    echo "Unsupported host: $HOST" >&2
    exit 2
    ;;
esac
wait
