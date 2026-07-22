#!/usr/bin/env bash
set -euo pipefail
ROOT=/raid/home/hming/FusionRAG-pca-analysis
EXP=MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
MODEL=/mnt/qjhs-sh-lab-01/models/Qwen3-32B
DRAFT=/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3
API_URL=http://36.150.226.221:32355/v1
API_KEY=api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS
CACHE=/raid/home/hming/fusionrag-qwen3-attn-alpha-cross-cache
cd "$ROOT"
mkdir -p "$EXP/logs" "$CACHE"
# qjy000 owns 2wikimqa and first half of hotpotqa.
TASKS=(
  "2wikimqa data/2wikimqa-200.jsonl uniform 0.1 0 25"
  "2wikimqa data/2wikimqa-200.jsonl uniform 0.1 25 50"
  "2wikimqa data/2wikimqa-200.jsonl uniform 0.1 50 75"
  "2wikimqa data/2wikimqa-200.jsonl uniform 0.1 75 100"
  "2wikimqa data/2wikimqa-200.jsonl uniform 0.1 100 125"
  "2wikimqa data/2wikimqa-200.jsonl uniform 0.1 125 150"
  "2wikimqa data/2wikimqa-200.jsonl uniform 0.1 150 175"
  "2wikimqa data/2wikimqa-200.jsonl uniform 0.1 175 200"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl uniform 0.1 0 25"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl uniform 0.1 25 50"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl uniform 0.1 50 75"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl uniform 0.1 75 100"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl uniform 0.1 100 125"
  "2wikimqa data/2wikimqa-200.jsonl uniform 0.25 0 25"
  "2wikimqa data/2wikimqa-200.jsonl uniform 0.25 25 50"
  "2wikimqa data/2wikimqa-200.jsonl uniform 0.25 50 75"
  "2wikimqa data/2wikimqa-200.jsonl uniform 0.25 75 100"
  "2wikimqa data/2wikimqa-200.jsonl uniform 0.25 100 125"
  "2wikimqa data/2wikimqa-200.jsonl uniform 0.25 125 150"
  "2wikimqa data/2wikimqa-200.jsonl uniform 0.25 150 175"
  "2wikimqa data/2wikimqa-200.jsonl uniform 0.25 175 200"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl uniform 0.25 0 25"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl uniform 0.25 25 50"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl uniform 0.25 50 75"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl uniform 0.25 75 100"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl uniform 0.25 100 125"
  "2wikimqa data/2wikimqa-200.jsonl random 0.05 0 25"
  "2wikimqa data/2wikimqa-200.jsonl random 0.05 25 50"
  "2wikimqa data/2wikimqa-200.jsonl random 0.05 50 75"
  "2wikimqa data/2wikimqa-200.jsonl random 0.05 75 100"
  "2wikimqa data/2wikimqa-200.jsonl random 0.05 100 125"
  "2wikimqa data/2wikimqa-200.jsonl random 0.05 125 150"
  "2wikimqa data/2wikimqa-200.jsonl random 0.05 150 175"
  "2wikimqa data/2wikimqa-200.jsonl random 0.05 175 200"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl random 0.05 0 25"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl random 0.05 25 50"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl random 0.05 50 75"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl random 0.05 75 100"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl random 0.05 100 125"
  "2wikimqa data/2wikimqa-200.jsonl random 0.1 0 25"
  "2wikimqa data/2wikimqa-200.jsonl random 0.1 25 50"
  "2wikimqa data/2wikimqa-200.jsonl random 0.1 50 75"
  "2wikimqa data/2wikimqa-200.jsonl random 0.1 75 100"
  "2wikimqa data/2wikimqa-200.jsonl random 0.1 100 125"
  "2wikimqa data/2wikimqa-200.jsonl random 0.1 125 150"
  "2wikimqa data/2wikimqa-200.jsonl random 0.1 150 175"
  "2wikimqa data/2wikimqa-200.jsonl random 0.1 175 200"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl random 0.1 0 25"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl random 0.1 25 50"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl random 0.1 50 75"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl random 0.1 75 100"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl random 0.1 100 125"
  "2wikimqa data/2wikimqa-200.jsonl random 0.25 0 25"
  "2wikimqa data/2wikimqa-200.jsonl random 0.25 25 50"
  "2wikimqa data/2wikimqa-200.jsonl random 0.25 50 75"
  "2wikimqa data/2wikimqa-200.jsonl random 0.25 75 100"
  "2wikimqa data/2wikimqa-200.jsonl random 0.25 100 125"
  "2wikimqa data/2wikimqa-200.jsonl random 0.25 125 150"
  "2wikimqa data/2wikimqa-200.jsonl random 0.25 150 175"
  "2wikimqa data/2wikimqa-200.jsonl random 0.25 175 200"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl random 0.25 0 25"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl random 0.25 25 50"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl random 0.25 50 75"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl random 0.25 75 100"
  "hotpotqa data/hotpotqa-260-100-10-doc.jsonl random 0.25 100 125"
)
label_of() { local mode=$1 alpha=$2; local al=${alpha//./p}; echo "${mode}_alpha${al}"; }
run_dir() { local ds=$1 mode=$2 alpha=$3 s=$4 e=$5; echo "$EXP/$(label_of "$mode" "$alpha")/$ds/seg_${s}_${e}"; }
is_done() { local ds=$1 path=$2 mode=$3 alpha=$4 s=$5 e=$6; local d=$(run_dir "$ds" "$mode" "$alpha" "$s" "$e"); grep -q 'FINAL RESULTS' "$d/run.log" 2>/dev/null && find "$d" -name 'rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv' | grep -q .; }
is_running() { local ds=$1 path=$2 mode=$3 alpha=$4 s=$5 e=$6; local d=$(run_dir "$ds" "$mode" "$alpha" "$s" "$e"); pgrep -af "$d" >/dev/null; }
free_gpus() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$2+0 < 10000 {gsub(/ /,"",$1); print $1}'; }
launch_task() {
  local gpu=$1 ds=$2 data_path=$3 mode=$4 alpha=$5 s=$6 e=$7
  local out_dir=$(run_dir "$ds" "$mode" "$alpha" "$s" "$e")
  local lp="queue_${ds}_$(label_of "$mode" "$alpha")_qjy000_gpu${gpu}_seg_${s}_${e}"
  mkdir -p "$out_dir"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 FUSIONRAG_REPROCESS_ATTENTION_ABLATION="$mode" FUSIONRAG_REPROCESS_ATTENTION_ABLATION_ALPHA="$alpha" \
    "$PY" test_fusionrag_reflect_preprocess_exp.py \
      --model_type qwen3 --model_path "$MODEL" --model_name Qwen3-32B \
      --data_path "$data_path" --dataset_name "$ds" --cache_path "$CACHE/$ds" --result_path "$out_dir" \
      --start_sample "$s" --end_sample "$e" --rate 0.15 --topk 10 --preprocess true --recall_method bge \
      --reprocess_method DraftModel --draft_model_path "$DRAFT" --revert_rope true --preprocess_scope global \
      --bge_model_path "$BGE" --device cuda:0 --use_multi_gpu false \
      --openai_base_url "$API_URL" --openai_api_key "$API_KEY" --openai_model GLM-5.2 \
      > "$out_dir/run.log" 2>&1 < /dev/null &
  echo $! > "$EXP/logs/${lp}.pid"
  echo "[$(date '+%F %T')] launched ds=$ds mode=$mode alpha=$alpha seg=$s-$e gpu=$gpu pid=$(cat "$EXP/logs/${lp}.pid")"
}
while true; do
  pending=0; launched=0; used=" "
  for t in "${TASKS[@]}"; do
    read -r ds path mode alpha s e <<< "$t"
    if is_done "$ds" "$path" "$mode" "$alpha" "$s" "$e" || is_running "$ds" "$path" "$mode" "$alpha" "$s" "$e"; then continue; fi
    pending=$((pending+1))
    for gpu in $(free_gpus); do
      case "$used" in *" $gpu "*) continue;; esac
      launch_task "$gpu" "$ds" "$path" "$mode" "$alpha" "$s" "$e"
      used="$used$gpu "; launched=$((launched+1)); break
    done
  done
  echo "[$(date '+%F %T')] qjy000 cross queue pending=$pending launched=$launched"
  if [ "$pending" -eq 0 ]; then break; fi
  sleep 90
done
