#!/usr/bin/env bash
set -euo pipefail

REPO=/raid/home/hming/FusionRAG-pca-analysis
EXP="$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization"
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
MODEL=/mnt/qjhs-sh-lab-01/models/Qwen3-32B
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
CACHE=/raid/home/hming/fusionrag-crossdataset-rerun-cache
API_URL=${API_URL:-http://36.150.226.221:32355/v1}
API_KEY=${API_KEY:-$(grep -m1 '^API_KEY=' "$REPO/MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/launch_qwen3_rate015_segments.sh" | cut -d= -f2-)}

mkdir -p "$EXP/logs/rerun_failed_trivia"
LOG="$EXP/logs/rerun_failed_trivia/rerun.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

run_one() {
  local gpu="$1"
  local cfg="$2"
  local start="$3"
  local end="$4"
  local out_dir="$EXP/results/$cfg/triviaqa/seg_${start}_${end}"
  local cache_dir="$CACHE/gpu${gpu}/${cfg}/seg_${start}_${end}"
  local method="FusionRAG"
  local extra=()

  if [[ "$cfg" == "offline3b_freq_boundary2" ]]; then
    extra+=(--offline_fixed_set_dir "$EXP/fixed_sets_triviaqa_3b/chunk_fixed_sets_npz" --offline_fixed_set_method offline3b_freq_boundary0p02_global --offline_fixed_set_rate 0.15)
  elif [[ "$cfg" == "offline32b_top2" ]]; then
    extra+=(--offline_fixed_set_dir "$EXP/fixed_sets_triviaqa_32b/chunk_fixed_sets_npz" --offline_fixed_set_method offline32b_top2_mean_global --offline_fixed_set_rate 0.15)
  else
    echo "unknown cfg=$cfg" >&2
    exit 2
  fi

  rm -rf "$out_dir" "$cache_dir"
  mkdir -p "$out_dir" "$cache_dir"
  log "start cfg=$cfg seg=$start-$end gpu=$gpu"
  cd "$REPO"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" test_fusionrag_reflect_preprocess_exp.py \
    --model_type qwen3 \
    --model_path "$MODEL" \
    --model_name Qwen3-32B \
    --data_path "$EXP/data/triviaqa_reflect.json" \
    --dataset_name triviaqa \
    --cache_path "$cache_dir" \
    --result_path "$out_dir" \
    --start_sample "$start" \
    --end_sample "$end" \
    --rate 0.15 \
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
    > "$out_dir/run.log" 2>&1
  rm -rf "$cache_dir"
  log "done cfg=$cfg seg=$start-$end gpu=$gpu"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  gpu="$1"
  shift
  while [[ "$#" -gt 0 ]]; do
    cfg="$1"; start="$2"; end="$3"
    shift 3
    run_one "$gpu" "$cfg" "$start" "$end"
  done
  exit 0
fi

nohup "$0" --worker 0 offline3b_freq_boundary2 150 175 offline32b_top2 175 200 > "$EXP/logs/rerun_failed_trivia/gpu0.outer.log" 2>&1 < /dev/null & echo $! > "$EXP/logs/rerun_failed_trivia/gpu0.pid"; log "launched gpu0 pid=$(cat "$EXP/logs/rerun_failed_trivia/gpu0.pid")"
nohup "$0" --worker 1 offline3b_freq_boundary2 175 200 offline32b_top2 200 225 > "$EXP/logs/rerun_failed_trivia/gpu1.outer.log" 2>&1 < /dev/null & echo $! > "$EXP/logs/rerun_failed_trivia/gpu1.pid"; log "launched gpu1 pid=$(cat "$EXP/logs/rerun_failed_trivia/gpu1.pid")"
nohup "$0" --worker 2 offline3b_freq_boundary2 200 225 offline32b_top2 225 250 > "$EXP/logs/rerun_failed_trivia/gpu2.outer.log" 2>&1 < /dev/null & echo $! > "$EXP/logs/rerun_failed_trivia/gpu2.pid"; log "launched gpu2 pid=$(cat "$EXP/logs/rerun_failed_trivia/gpu2.pid")"
nohup "$0" --worker 3 offline3b_freq_boundary2 225 250 offline32b_top2 250 270 > "$EXP/logs/rerun_failed_trivia/gpu3.outer.log" 2>&1 < /dev/null & echo $! > "$EXP/logs/rerun_failed_trivia/gpu3.pid"; log "launched gpu3 pid=$(cat "$EXP/logs/rerun_failed_trivia/gpu3.pid")"
nohup "$0" --worker 4 offline32b_top2 75 100 > "$EXP/logs/rerun_failed_trivia/gpu4.outer.log" 2>&1 < /dev/null & echo $! > "$EXP/logs/rerun_failed_trivia/gpu4.pid"; log "launched gpu4 pid=$(cat "$EXP/logs/rerun_failed_trivia/gpu4.pid")"
nohup "$0" --worker 5 offline32b_top2 100 125 > "$EXP/logs/rerun_failed_trivia/gpu5.outer.log" 2>&1 < /dev/null & echo $! > "$EXP/logs/rerun_failed_trivia/gpu5.pid"; log "launched gpu5 pid=$(cat "$EXP/logs/rerun_failed_trivia/gpu5.pid")"
nohup "$0" --worker 6 offline32b_top2 125 150 > "$EXP/logs/rerun_failed_trivia/gpu6.outer.log" 2>&1 < /dev/null & echo $! > "$EXP/logs/rerun_failed_trivia/gpu6.pid"; log "launched gpu6 pid=$(cat "$EXP/logs/rerun_failed_trivia/gpu6.pid")"
nohup "$0" --worker 7 offline32b_top2 150 175 > "$EXP/logs/rerun_failed_trivia/gpu7.outer.log" 2>&1 < /dev/null & echo $! > "$EXP/logs/rerun_failed_trivia/gpu7.pid"; log "launched gpu7 pid=$(cat "$EXP/logs/rerun_failed_trivia/gpu7.pid")"

wait
"$PY" "$EXP/summarize_cross_dataset.py" > "$EXP/logs/rerun_failed_trivia/final_summary_after_rerun.log" 2>&1 || true
log "rerun script done"
