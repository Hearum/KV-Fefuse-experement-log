#!/usr/bin/env bash
set -euo pipefail

REPO=/raid/home/hming/FusionRAG-pca-analysis
EXP="$REPO/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization"
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
MODEL=/mnt/qjhs-sh-lab-01/models/Qwen3-32B
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
CACHE_ROOT=${CACHE_ROOT:-/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-crossdataset-offline32b-trivia-missing-cache-20260714}
API_URL=${API_URL:-http://36.150.226.221:32355/v1}
API_KEY=${API_KEY:-$(grep -m1 ^API_KEY= "$REPO/MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/launch_qwen3_rate015_segments.sh" | cut -d= -f2-)}

LOG_DIR="$EXP/logs/rerun_offline32b_top2_trivia_missing_20260714"
mkdir -p "$LOG_DIR" "$CACHE_ROOT"
LOG="$LOG_DIR/launcher.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

run_one() {
  local gpu="$1"
  local start="$2"
  local end="$3"
  local out_dir="$EXP/results/offline32b_top2/triviaqa/seg_${start}_${end}"
  local run_log="$out_dir/run.log"
  local cache_dir="$CACHE_ROOT/gpu${gpu}/seg_${start}_${end}"

  mkdir -p "$out_dir" "$cache_dir"
  if [[ -f "$run_log" ]] && grep -q "FINAL RESULTS" "$run_log"; then
    log "skip already finished seg=$start-$end gpu=$gpu"
    return 0
  fi
  if [[ -f "$run_log" ]]; then
    cp "$run_log" "$out_dir/run.log.failed_$(date '+%Y%m%d_%H%M%S')"
  fi

  log "start offline32b_top2 triviaqa seg=$start-$end gpu=$gpu"
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
    --reprocess_method FusionRAG \
    --revert_rope true \
    --preprocess_scope global \
    --bge_model_path "$BGE" \
    --device cuda:0 \
    --use_multi_gpu false \
    --openai_base_url "$API_URL" \
    --openai_api_key "$API_KEY" \
    --openai_model GLM-5.2 \
    --offline_fixed_set_dir "$EXP/fixed_sets_triviaqa_32b/chunk_fixed_sets_npz" \
    --offline_fixed_set_method offline32b_top2_mean_global \
    --offline_fixed_set_rate 0.15 \
    > "$run_log" 2>&1
  log "done offline32b_top2 triviaqa seg=$start-$end gpu=$gpu"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  run_one "$@"
  exit 0
fi

log "launch missing offline32b_top2 triviaqa segments"
nohup "$0" --worker 0 175 200 > "$LOG_DIR/gpu0_seg175_200.outer.log" 2>&1 < /dev/null & echo $! > "$LOG_DIR/gpu0_seg175_200.pid"; log "launched seg=175-200 pid=$(cat "$LOG_DIR/gpu0_seg175_200.pid") gpu=0"
nohup "$0" --worker 1 225 250 > "$LOG_DIR/gpu1_seg225_250.outer.log" 2>&1 < /dev/null & echo $! > "$LOG_DIR/gpu1_seg225_250.pid"; log "launched seg=225-250 pid=$(cat "$LOG_DIR/gpu1_seg225_250.pid") gpu=1"
nohup "$0" --worker 2 250 270 > "$LOG_DIR/gpu2_seg250_270.outer.log" 2>&1 < /dev/null & echo $! > "$LOG_DIR/gpu2_seg250_270.pid"; log "launched seg=250-270 pid=$(cat "$LOG_DIR/gpu2_seg250_270.pid") gpu=2"

log "launcher done"
