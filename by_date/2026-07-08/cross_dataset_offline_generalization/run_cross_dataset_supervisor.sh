#!/usr/bin/env bash
set -euo pipefail

REPO=/raid/home/hming/FusionRAG-pca-analysis
EXP="$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization"
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
MODEL=/mnt/qjhs-sh-lab-01/models/Qwen3-32B
DRAFT=/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
CACHE=/raid/home/hming/fusionrag-crossdataset-qwen3-cache
API_URL=${API_URL:-http://36.150.226.221:32355/v1}
API_KEY=${API_KEY:-$(grep -m1 '^API_KEY=' "$REPO/MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/launch_qwen3_rate015_segments.sh" | cut -d= -f2-)}

mkdir -p "$EXP/logs" "$EXP/results"
LOG="$EXP/logs/supervisor.log"
TASKS="$EXP/logs/accuracy_tasks.tsv"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

count_cache() {
  local dataset="$1"
  local teacher="$2"
  find "$EXP/score_cache_${dataset}_${teacher}/score_cache_npz" -name 'reflect_draft_example*_scores.npz' 2>/dev/null | wc -l
}

wait_score_cache() {
  log "等待 3B/32B score cache 完成并等待写入进程退出"
  while true; do
    local ok=1
    local msg=""
    for spec in "2wikimqa 200" "hotpotqa 260" "triviaqa 270"; do
      read -r d n <<< "$spec"
      for t in 3b 32b; do
        local c
        c=$(count_cache "$d" "$t")
        msg+="$d-$t=$c/$n "
        if [[ "$c" -lt "$n" ]]; then
          ok=0
        fi
      done
    done
    local running
    running=$(ps -ef | { grep tmp_build_control_draft_score_cache.py || true; } | { grep cross_dataset || true; } | { grep -v grep || true; } | wc -l)
    msg+="running_score_process=$running"
    log "$msg"
    if [[ "$ok" -eq 1 && "$running" -eq 0 ]]; then
      break
    fi
    sleep 300
  done
  log "score cache 已完成"
}

derive_fixed_sets() {
  log "开始派生 offline fixed sets"
  for d in 2wikimqa hotpotqa triviaqa; do
    for t in 3b 32b; do
      local prefix="offline${t}"
      local out="$EXP/fixed_sets_${d}_${t}"
      local marker="$out/.done"
      if [[ -f "$marker" ]]; then
        log "skip derive: $d $t 已存在"
        continue
      fi
      rm -rf "$out"
      "$PY" "$EXP/scripts/tmp_derive_generic_smart_sets.py" \
        --score-cache-dir "$EXP/score_cache_${d}_${t}/score_cache_npz" \
        --out-dir "$out" \
        --prefix "$prefix" \
        --rate 0.15 \
        --boundary-rate 0.02 \
        > "$EXP/logs/derive_${d}_${t}.log" 2>&1
      touch "$marker"
      log "derive done: $d $t -> $out"
    done
  done
}

make_tasks() {
  : > "$TASKS"
  local idx=0
  for dspec in "2wikimqa 200" "hotpotqa 260" "triviaqa 270"; do
    read -r d n <<< "$dspec"
    for cfg in full_rate1 online_qk_rate015 online_draft_rate015 offline3b_mean offline3b_freq_boundary2 offline32b_top2; do
      local s=0
      while [[ "$s" -lt "$n" ]]; do
        local e=$((s + 25))
        if [[ "$e" -gt "$n" ]]; then
          e="$n"
        fi
        printf "%s\t%s\t%s\t%s\t%s\n" "$idx" "$d" "$cfg" "$s" "$e" >> "$TASKS"
        idx=$((idx + 1))
        s="$e"
      done
    done
  done
  log "accuracy task list written: $TASKS ($(wc -l < "$TASKS") tasks)"
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

  local rate="0.15"
  local method="FusionRAG"
  local extra=()
  if [[ "$cfg" == "full_rate1" ]]; then
    rate="1.0"
  elif [[ "$cfg" == "online_qk_rate015" ]]; then
    :
  elif [[ "$cfg" == "online_draft_rate015" ]]; then
    method="DraftModel"
    extra+=(--draft_model_path "$DRAFT")
  elif [[ "$cfg" == "offline3b_mean" ]]; then
    extra+=(--offline_fixed_set_dir "$EXP/fixed_sets_${dataset}_3b/chunk_fixed_sets_npz" --offline_fixed_set_method offline3b_mean_score_global --offline_fixed_set_rate 0.15)
  elif [[ "$cfg" == "offline3b_freq_boundary2" ]]; then
    extra+=(--offline_fixed_set_dir "$EXP/fixed_sets_${dataset}_3b/chunk_fixed_sets_npz" --offline_fixed_set_method offline3b_freq_boundary0p02_global --offline_fixed_set_rate 0.15)
  elif [[ "$cfg" == "offline32b_top2" ]]; then
    extra+=(--offline_fixed_set_dir "$EXP/fixed_sets_${dataset}_32b/chunk_fixed_sets_npz" --offline_fixed_set_method offline32b_top2_mean_global --offline_fixed_set_rate 0.15)
  else
    log "unknown cfg=$cfg"
    return 2
  fi

  log "start accuracy cfg=$cfg dataset=$dataset segment=$start-$end gpu=$gpu"
  cd "$REPO"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" test_fusionrag_reflect_preprocess_exp.py \
    --model_type qwen3 \
    --model_path "$MODEL" \
    --model_name Qwen3-32B \
    --data_path "$EXP/data/${dataset}_reflect.json" \
    --dataset_name "$dataset" \
    --cache_path "$CACHE/worker_gpu${gpu}/$dataset" \
    --result_path "$out_dir" \
    --start_sample "$start" \
    --end_sample "$end" \
    --rate "$rate" \
    --topk 10 \
    --preprocess true \
    --recall_method bge \
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
  log "done accuracy cfg=$cfg dataset=$dataset segment=$start-$end gpu=$gpu"
}

worker() {
  local gpu="$1"
  while IFS=$'\t' read -r idx dataset cfg start end; do
    if [[ $((idx % 8)) -ne "$gpu" ]]; then
      continue
    fi
    run_one_task "$gpu" "$dataset" "$cfg" "$start" "$end"
  done < "$TASKS"
  log "worker gpu=$gpu finished"
}

run_accuracy() {
  make_tasks
  log "启动 8 个 accuracy worker"
  for gpu in 0 1 2 3 4 5 6 7; do
    nohup "$0" --worker "$gpu" > "$EXP/logs/accuracy_worker_gpu${gpu}.outer.log" 2>&1 < /dev/null &
    echo $! > "$EXP/logs/accuracy_worker_gpu${gpu}.pid"
    log "launched accuracy worker gpu=$gpu pid=$(cat "$EXP/logs/accuracy_worker_gpu${gpu}.pid")"
  done

  while true; do
    local running
    running=$(ps -ef | { grep "run_cross_dataset_supervisor.sh --worker" || true; } | { grep -v grep || true; } | wc -l)
    "$PY" "$EXP/summarize_cross_dataset.py" > "$EXP/logs/latest_summary.log" 2>&1 || true
    log "accuracy workers running=$running"
    if [[ "$running" -eq 0 ]]; then
      break
    fi
    sleep 300
  done
  "$PY" "$EXP/summarize_cross_dataset.py" > "$EXP/logs/final_summary.log" 2>&1
  log "accuracy 全部完成，结果已汇总"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  worker "$@"
  exit 0
fi

log "supervisor start"
wait_score_cache
derive_fixed_sets
run_accuracy
log "supervisor done"
