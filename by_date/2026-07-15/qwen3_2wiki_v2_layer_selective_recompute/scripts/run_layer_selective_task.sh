#!/usr/bin/env bash
set -euo pipefail
COND="$1"
START="$2"
END="$3"
GPU="$4"
RATE="${5:-0.15}"
ROOT=/raid/home/hming/FusionRAG-pca-analysis
EXP=$ROOT/MOTIVATION_EXPERIMENTS/by_date/2026-07-15/qwen3_2wiki_v2_layer_selective_recompute
case "$COND" in
  all_layers)
    unset FUSIONRAG_REPROCESS_KEEP_KEY_LAYERS || true
    unset FUSIONRAG_REPROCESS_KEEP_VALUE_LAYERS || true
    ;;
  v_late_59_63)
    export FUSIONRAG_REPROCESS_KEEP_KEY_LAYERS=none
    export FUSIONRAG_REPROCESS_KEEP_VALUE_LAYERS=59-63
    ;;
  v_late_56_63)
    export FUSIONRAG_REPROCESS_KEEP_KEY_LAYERS=none
    export FUSIONRAG_REPROCESS_KEEP_VALUE_LAYERS=56-63
    ;;
  k_mid_45_52)
    export FUSIONRAG_REPROCESS_KEEP_KEY_LAYERS=45-52
    export FUSIONRAG_REPROCESS_KEEP_VALUE_LAYERS=none
    ;;
  kv_gap_core)
    export FUSIONRAG_REPROCESS_KEEP_KEY_LAYERS=45-52
    export FUSIONRAG_REPROCESS_KEEP_VALUE_LAYERS=59-63
    ;;
  kv_gap_wide)
    export FUSIONRAG_REPROCESS_KEEP_KEY_LAYERS=45-52
    export FUSIONRAG_REPROCESS_KEEP_VALUE_LAYERS=56-63
    ;;
  *)
    echo "unknown condition: $COND" >&2
    exit 2
    ;;
esac
mkdir -p "$EXP/results/$COND"
cd "$ROOT"
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py \
  --dataset 2wikimqa-v2 \
  --method online_qk \
  --rate "$RATE" \
  --start "$START" \
  --end "$END" \
  --gpu "$GPU" \
  --result-root "$EXP/results/$COND"
