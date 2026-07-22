#!/usr/bin/env bash
set -euo pipefail

ROOT=/raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
EXP_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_fresh_preprocess_qjy000_lab03_v2_20260714
CACHE_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-qjy000-lab03-v2-20260714
LAUNCHER=MOTIVATION_EXPERIMENTS/qwen3_baseline_provenance_audit/scripts/run_strict_fresh_cache_qk_draft_rate015.py

mkdir -p "$EXP_ROOT/logs" "$CACHE_ROOT"
cd "$ROOT"

FUSIONRAG_STRICT_FRESH_DATASETS=musique \
FUSIONRAG_STRICT_FRESH_METHODS=online_qk_rate015,online_draft_rate015 \
FUSIONRAG_STRICT_FRESH_GPUS=0,1,2,3,4,5,6,7 \
FUSIONRAG_STRICT_FRESH_EXP_ROOT="$EXP_ROOT" \
FUSIONRAG_STRICT_FRESH_CACHE_ROOT="$CACHE_ROOT" \
nohup "$PY" "$LAUNCHER" \
  > "$EXP_ROOT/logs/launcher_qjy000.log" 2>&1 < /dev/null &

echo $! > "$EXP_ROOT/logs/launcher_qjy000.pid"
echo "launched pid=$(cat "$EXP_ROOT/logs/launcher_qjy000.pid") exp_root=$EXP_ROOT cache_root=$CACHE_ROOT"
