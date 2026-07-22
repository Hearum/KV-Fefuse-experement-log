#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_dataset_balanced_v2_20260714
CACHE=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-cross-dataset-rate-sweep-dataset-balanced-v2-cache-20260714
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
SCRIPT=MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/scripts/run_cross_dataset_qk_draft_rate_sweep.py
mkdir -p "$ROOT/logs"

launch_one() {
  local host="$1"
  local workdir="$2"
  local dataset="$3"
  local gpu="$4"
  local shard="$5"
  local log="$ROOT/logs/launcher_${dataset}_${host}_gpu${gpu}_shard${shard}.log"
  if [[ "$host" == "qjy000" ]]; then
    cd "$workdir"
    FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_ROOT="$ROOT" \
    FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_ROOT="$CACHE" \
    FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_DATASETS="$dataset" \
    FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_GPUS="$gpu" \
    FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_SHARD_COUNT=7 \
    FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_SHARD_INDEX="$shard" \
    nohup "$PY" "$SCRIPT" > "$log" 2>&1 < /dev/null &
    echo "${dataset}_${host}_gpu${gpu}_shard${shard}=$!"
  else
    ssh "$host" "cd '$workdir' && FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_ROOT='$ROOT' FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_ROOT='$CACHE' FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_DATASETS='$dataset' FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_GPUS='$gpu' FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_SHARD_COUNT=7 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_SHARD_INDEX='$shard' nohup '$PY' '$SCRIPT' > '$log' 2>&1 < /dev/null & echo ${dataset}_${host}_gpu${gpu}_shard${shard}=\$!"
  fi
}

# 2WikiMQA: 7 shards / 7 GPUs
launch_one qjy000 /raid/home/hming/FusionRAG-pca-analysis 2wikimqa 3 0
launch_one qjy000 /raid/home/hming/FusionRAG-pca-analysis 2wikimqa 4 1
launch_one qjy000 /raid/home/hming/FusionRAG-pca-analysis 2wikimqa 5 2
launch_one qjy000 /raid/home/hming/FusionRAG-pca-analysis 2wikimqa 6 3
launch_one qjy000 /raid/home/hming/FusionRAG-pca-analysis 2wikimqa 7 4
launch_one qjy001 /home/hming/FusionRAG-pca-analysis 2wikimqa 0 5
launch_one qjy001 /home/hming/FusionRAG-pca-analysis 2wikimqa 1 6

# HotpotQA: 7 shards / 7 GPUs
launch_one qjy001 /home/hming/FusionRAG-pca-analysis hotpotqa 2 0
launch_one qjy001 /home/hming/FusionRAG-pca-analysis hotpotqa 3 1
launch_one qjy001 /home/hming/FusionRAG-pca-analysis hotpotqa 4 2
launch_one qjy001 /home/hming/FusionRAG-pca-analysis hotpotqa 5 3
launch_one qjy001 /home/hming/FusionRAG-pca-analysis hotpotqa 6 4
launch_one qjy001 /home/hming/FusionRAG-pca-analysis hotpotqa 7 5
launch_one qjy003 /home/hming/FusionRAG-pca-analysis hotpotqa 0 6

# TriviaQA: 7 shards / 7 GPUs
launch_one qjy003 /home/hming/FusionRAG-pca-analysis triviaqa 1 0
launch_one qjy003 /home/hming/FusionRAG-pca-analysis triviaqa 2 1
launch_one qjy003 /home/hming/FusionRAG-pca-analysis triviaqa 3 2
launch_one qjy003 /home/hming/FusionRAG-pca-analysis triviaqa 4 3
launch_one qjy003 /home/hming/FusionRAG-pca-analysis triviaqa 5 4
launch_one qjy003 /home/hming/FusionRAG-pca-analysis triviaqa 6 5
launch_one qjy003 /home/hming/FusionRAG-pca-analysis triviaqa 7 6
