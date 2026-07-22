#!/usr/bin/env bash
set -euo pipefail

ROOT=/raid/home/hming/FusionRAG-pca-analysis
EXP=$ROOT/MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
QUERY_JSONL=$EXP/generated_queries/docgen_queries.jsonl
OUT=$EXP/docgen_chunk_draft_smart_full
LOG=$EXP/logs

mkdir -p "$OUT" "$LOG"

for gpu in 0 1 2 3 4 5 6 7; do
  start=$((gpu * 25))
  end=$((start + 25))
  (
    cd "$ROOT"
    echo "START docgen chunk fixed sets gpu=$gpu examples=$start-$end $(date)"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" "$EXP/build_docgen_chunk_fixed_sets.py" \
      --docgen-query-jsonl "$QUERY_JSONL" \
      --out-dir "$OUT" \
      --example-start "$start" \
      --example-end "$end" \
      --rates 0.10,0.15,0.20 \
      --method docgen_draft_smart_frequency_global \
      --device cuda:0
    echo "DONE docgen chunk fixed sets gpu=$gpu examples=$start-$end $(date)"
  ) > "$LOG/docgen_chunk_fixed_sets_${start}_${end}.log" 2>&1 &
done

wait
echo "ALL DONE $(date)"
