#!/usr/bin/env bash
set -euo pipefail
ROOT=/raid/home/hming/FusionRAG-pca-analysis
cd "$ROOT"
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
EXP=$ROOT/MOTIVATION_EXPERIMENTS/offline_raw_qk_selector_rate015
RAW_CACHE=/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache
mkdir -p "$EXP/logs"
for shard in 0 1 2 3 4 5 6 7; do
  start=$((shard * 25))
  out="$EXP/shard_${start}_25"
  mkdir -p "$out"
  (
    export CUDA_VISIBLE_DEVICES=$shard
    export PYTHONUNBUFFERED=1
    "$PY" tools_reflect_offline_qk_fixed_sets.py \
      --model-path /mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct \
      --data-path ./data/result_reflect.json \
      --bge-model-path /mnt/qjhs-sh-lab-01/models/bge-m3 \
      --preprocess-cache-dir "$RAW_CACHE" \
      --out-dir "$out" \
      --example-start "$start" \
      --num-examples 25 \
      --control-count 8 \
      --rate 0.15 \
      --device cuda:0
  ) > "$EXP/logs/shard_${start}_25.log" 2>&1 &
  echo $! > "$EXP/logs/shard_${start}_25.pid"
done
wait
mkdir -p "$EXP/combined/chunk_fixed_sets_npz" "$EXP/combined/score_cache_npz"
find "$EXP" -path "*/chunk_fixed_sets_npz/*.npz" -exec cp -f {} "$EXP/combined/chunk_fixed_sets_npz/" \;
find "$EXP" -path "*/score_cache_npz/*.npz" -exec cp -f {} "$EXP/combined/score_cache_npz/" \;
cat > "$EXP/combined/README.md" <<README
# Combined Offline Raw-QK Fixed Sets

- dataset: data/result_reflect.json
- rate: 0.15
- calibration queries per example: 8, from other examples only
- selector KV: raw document KV cache
- raw KV cache: $RAW_CACHE
- methods: qk_frequency_per_chunk, qk_mean_score_per_chunk
- generated from 8 shards under ../shard_*
README
