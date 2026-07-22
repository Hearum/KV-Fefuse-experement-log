#!/usr/bin/env bash
set -u

REPO=/home/hming/FusionRAG-pca-analysis
EXP=$REPO/MOTIVATION_EXPERIMENTS/qwen3_235b_param_scaling_cross_dataset
SCRIPT=$EXP/launch_qjy001_resume_remaining.sh
LOG=$EXP/logs/qjy003_wait_then_resume.log
SESSION=qwen3_235b_param_qjy003_resume_remaining

mkdir -p "$EXP/logs"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

all_gpus_idle() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{ if ($1 >= 1024) busy++ } END { print busy+0 }')
  [[ "$used" -eq 0 ]]
}

log "qjy003 watcher started; will launch only when all GPUs are idle"

while true; do
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    log "target session already running: $SESSION"
    sleep 600
    continue
  fi

  if all_gpus_idle; then
    log "all GPUs idle; starting remaining queue"
    cd "$REPO" || exit 1
    tmux new-session -d -s "$SESSION" "$SCRIPT"
    tmux ls | grep "$SESSION" | tee -a "$LOG" || true
    exit 0
  fi

  log "GPUs still busy; waiting"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tee -a "$LOG" || true
  sleep 600
done
