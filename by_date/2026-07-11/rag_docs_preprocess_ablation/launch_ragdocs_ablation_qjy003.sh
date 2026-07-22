#!/usr/bin/env bash
set -euo pipefail

REPO=/home/hming/FusionRAG-pca-analysis
EXP="$REPO/MOTIVATION_EXPERIMENTS/rag_docs_preprocess_ablation"
BASE_EXP="$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization"
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
MODEL=/mnt/qjhs-sh-lab-01/models/Qwen3-32B
DRAFT=/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
CACHE=${CACHE:-/dev/shm/hming/fusionrag-ragdocs-preprocess-ablation-cache-v3}
GPU_LIST=${GPU_LIST:-"0 1 2 3"}
read -r -a GPUS <<< "$GPU_LIST"
WORKER_COUNT=${#GPUS[@]}
API_URL=${API_URL:-http://36.150.226.221:32355/v1}
API_KEY=${API_KEY:-$(grep -m1 '^API_KEY=' "$REPO/MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/launch_qwen3_rate015_segments.sh" | cut -d= -f2-)}

mkdir -p "$EXP/logs" "$EXP/results" "$CACHE"
LOG="$EXP/logs/supervisor.log"
TASKS="$EXP/logs/tasks.tsv"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

make_tasks() {
  : > "$TASKS"
  local idx=0
  for dspec in "2wikimqa 200" "hotpotqa 260" "triviaqa 270"; do
    read -r dataset n <<< "$dspec"
    for cfg in ragdocs_online_qk_rate015 ragdocs_online_draft_rate015; do
      local s=0
      while [[ "$s" -lt "$n" ]]; do
        local e=$((s + 25))
        if [[ "$e" -gt "$n" ]]; then
          e="$n"
        fi
        printf "%s\t%s\t%s\t%s\t%s\n" "$idx" "$dataset" "$cfg" "$s" "$e" >> "$TASKS"
        idx=$((idx + 1))
        s="$e"
      done
    done
  done
  log "task list written: $TASKS ($(wc -l < "$TASKS") tasks)"
}

run_one_task() {
  local gpu="$1"
  local dataset="$2"
  local cfg="$3"
  local start="$4"
  local end="$5"

  local out_dir="$EXP/results/$cfg/$dataset/seg_${start}_${end}"
  mkdir -p "$out_dir"
  local run_log="$out_dir/run.log"
  if [[ -f "$run_log" ]] && grep -q "FINAL RESULTS" "$run_log"; then
    log "skip finished cfg=$cfg dataset=$dataset segment=$start-$end gpu=$gpu"
    return 0
  fi

  local method="FusionRAG"
  local extra=()
  if [[ "$cfg" == "ragdocs_online_draft_rate015" ]]; then
    method="DraftModel"
    extra+=(--draft_model_path "$DRAFT")
  elif [[ "$cfg" != "ragdocs_online_qk_rate015" ]]; then
    log "unknown cfg=$cfg"
    return 2
  fi

  log "start cfg=$cfg dataset=$dataset segment=$start-$end gpu=$gpu"
  cd "$REPO"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" test_fusionrag_reflect_preprocess_exp.py \
    --model_type qwen3 \
    --model_path "$MODEL" \
    --model_name Qwen3-32B \
    --data_path "$BASE_EXP/data/${dataset}_reflect.json" \
    --dataset_name "$dataset" \
    --cache_path "$CACHE/worker_gpu${gpu}/$dataset/$cfg" \
    --result_path "$out_dir" \
    --start_sample "$start" \
    --end_sample "$end" \
    --rate 0.15 \
    --topk 10 \
    --preprocess true \
    --recall_method rag_docs \
    --reprocess_method "$method" \
    --revert_rope true \
    --preprocess_scope global \
    --bge_model_path "$BGE" \
    --device cuda:0 \
    --use_multi_gpu false \
    --openai_base_url "$API_URL" \
    --openai_api_key "$API_KEY" \
    --openai_model GLM-5.2 \
    "${extra[@]}" \
    > "$run_log" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    log "fail cfg=$cfg dataset=$dataset segment=$start-$end gpu=$gpu rc=$rc"
    return "$rc"
  fi
  log "done cfg=$cfg dataset=$dataset segment=$start-$end gpu=$gpu"
}

worker() {
  local rank="$1"
  local gpu="$2"
  while IFS=$'\t' read -r idx dataset cfg start end; do
    if [[ $((idx % WORKER_COUNT)) -ne "$rank" ]]; then
      continue
    fi
    run_one_task "$gpu" "$dataset" "$cfg" "$start" "$end" || true
  done < "$TASKS"
  log "worker gpu=$gpu finished"
}

run_workers() {
  make_tasks
  log "launching $WORKER_COUNT workers on qjy003 gpu_list=$GPU_LIST cache=$CACHE"
  local rank=0
  for gpu in "${GPUS[@]}"; do
    nohup "$0" --worker "$rank" "$gpu" > "$EXP/logs/worker_gpu${gpu}.outer.log" 2>&1 < /dev/null &
    echo $! > "$EXP/logs/worker_gpu${gpu}.pid"
    log "launched worker rank=$rank gpu=$gpu pid=$(cat "$EXP/logs/worker_gpu${gpu}.pid")"
    rank=$((rank + 1))
  done

  while true; do
    local running
    running=$(ps -ef | { grep "launch_ragdocs_ablation_qjy003.sh --worker" || true; } | { grep -v grep || true; } | wc -l)
    "$PY" "$EXP/summarize_ragdocs_ablation.py" > "$EXP/logs/latest_summary.log" 2>&1 || true
    log "workers running=$running"
    if [[ "$running" -eq 0 ]]; then
      break
    fi
    sleep 300
  done
  "$PY" "$EXP/summarize_ragdocs_ablation.py" > "$EXP/logs/final_summary.log" 2>&1 || true
  log "supervisor done"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  worker "$@"
  exit 0
fi

log "supervisor start host=$(hostname)"
run_workers
