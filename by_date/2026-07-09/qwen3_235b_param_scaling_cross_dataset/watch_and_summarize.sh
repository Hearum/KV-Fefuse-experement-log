#!/usr/bin/env bash
set -u

REPO=/raid/home/hming/FusionRAG-pca-analysis
EXP=$REPO/MOTIVATION_EXPERIMENTS/qwen3_235b_param_scaling_cross_dataset
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
LOG_LOCAL=/tmp/qwen3_235b_param_scaling_watcher.log

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_LOCAL"
}

run_summary() {
  ssh qjy000 "cd '$EXP' && '$PY' summarize_results.py >/tmp/qwen3_235b_summary_last.log 2>&1 && tail -1 /tmp/qwen3_235b_summary_last.log" | tee -a "$LOG_LOCAL" || true
}

session_alive() {
  local host="$1"
  local session="$2"
  ssh "$host" "tmux has-session -t '$session' 2>/dev/null" >/dev/null 2>&1
}

log "watcher started"
run_summary

while true; do
  alive=0
  if session_alive qjy000 qwen3_235b_param_qjy000_resume; then
    alive=1
  fi
  if session_alive qjy001 qwen3_235b_param_qjy001_followup; then
    alive=1
  fi

  if [[ "$alive" -eq 0 ]]; then
    log "both experiment queues finished; generating final summary"
    run_summary
    ssh qjy000 "cat >> '$EXP/RUN_STATUS.md' <<'EOF'

## Watcher 结束记录

- 结束时间：$(date '+%F %T')
- 已自动重扫结果并写入：\`MOTIVATION_EXPERIMENTS/qwen3_235b_param_scaling_cross_dataset/RESULTS_SUMMARY.md\`
EOF"
    log "watcher finished"
    exit 0
  fi

  log "queues still running; refreshing summary"
  run_summary
  sleep 600 || true
done
