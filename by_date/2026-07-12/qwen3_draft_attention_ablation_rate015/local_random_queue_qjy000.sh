#!/usr/bin/env bash
set -euo pipefail
ROOT=/raid/home/hming/FusionRAG-pca-analysis
EXP=MOTIVATION_EXPERIMENTS/qwen3_draft_attention_ablation_rate015
cd "$ROOT"
TASKS=(
  "random 0.1 75 100"
  "random 0.1 100 125"
  "random 0.1 125 150"
  "random 0.1 150 175"
  "random 0.1 175 200"
  "random 0.5 0 25"
  "random 0.5 25 50"
  "random 0.5 50 75"
  "random 0.5 75 100"
)
label_of() { local mode=$1 alpha=$2; local al=${alpha//./p}; echo "${mode}_alpha${al}_draft_rate015"; }
is_done() { local mode=$1 alpha=$2 s=$3 e=$4; local label=$(label_of "$mode" "$alpha"); local dir="$EXP/$label/seg_${s}_${e}"; grep -q 'FINAL RESULTS' "$dir/run.log" 2>/dev/null && test -f "$dir/Qwen3-32B/musique/DraftModel_global_topk10_bge/rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv"; }
is_running() { local mode=$1 alpha=$2 s=$3 e=$4; local label=$(label_of "$mode" "$alpha"); pgrep -af "$EXP/$label/seg_${s}_${e}" >/dev/null; }
free_gpus() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$2+0 < 10000 {gsub(/ /,"",$1); print $1}'; }
launch_task() { local gpu=$1 mode=$2 alpha=$3 s=$4 e=$5; local al=${alpha//./p}; local lp="queue_${mode}_alpha${al}_qjy000_gpu${gpu}_seg_${s}_${e}"; nohup "$EXP/launch_attention_ablation.sh" --worker "$mode" "$gpu" "$s" "$e" "$alpha" > "$EXP/logs/${lp}.outer.log" 2>&1 < /dev/null & echo $! > "$EXP/logs/${lp}.pid"; echo "[$(date '+%F %T')] launched $mode alpha=$alpha seg=$s-$e gpu=$gpu pid=$(cat "$EXP/logs/${lp}.pid")"; }
while true; do
  pending=0; launched=0; used=" "
  for t in "${TASKS[@]}"; do
    read -r mode alpha s e <<< "$t"
    if is_done "$mode" "$alpha" "$s" "$e" || is_running "$mode" "$alpha" "$s" "$e"; then continue; fi
    pending=$((pending+1))
    for gpu in $(free_gpus); do
      case "$used" in *" $gpu "*) continue;; esac
      launch_task "$gpu" "$mode" "$alpha" "$s" "$e"
      used="$used$gpu "
      launched=$((launched+1))
      break
    done
  done
  echo "[$(date '+%F %T')] qjy000 queue pending=$pending launched=$launched"
  if [ "$pending" -eq 0 ]; then break; fi
  sleep 60
done
