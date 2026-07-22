#!/usr/bin/env bash
set -euo pipefail
mode="$1"
rate="$2"
gpu="$3"
cd /raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
OUT_ROOT=MOTIVATION_EXPERIMENTS/kv_update_rate_sweep_nojudge
tag="${mode}_rate${rate//./p}"
out="${OUT_ROOT}/${tag}"
log="${OUT_ROOT}/${tag}.parallel.gpu${gpu}.log"
csv="${out}/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_${rate}_revert_rope.csv"
if [[ -f "$csv" ]] && grep -q "FINAL RESULTS" "$log" 2>/dev/null; then
  echo "SKIP completed ${tag}"
  exit 0
fi
echo "START ${tag} gpu=${gpu} $(date)" | tee "$log"
FUSIONRAG_SKIP_LLM_JUDGE=1 \
FUSIONRAG_STRICT_REPROCESS_ABLATION=1 \
FUSIONRAG_CLEAN_STRICT_ABLATION=1 \
FUSIONRAG_REPROCESS_UPDATE_MODE="$mode" \
CUDA_VISIBLE_DEVICES="$gpu" \
TMPDIR=/raid/home/hming/tmp \
"$PY" test_fusionrag_reflect_preprocess_exp.py \
  --model_type qwen \
  --model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct \
  --model_name Qwen2.5-7B-Instruct \
  --data_path ./data/result_reflect.json \
  --dataset_name musique \
  --cache_path /raid/home/hming/fusionrag-reflect-full-cache/ \
  --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3 \
  --topk 10 \
  --preprocess True \
  --recall_method bge \
  --random_seed 42 \
  --fixed_doc_idx 0 \
  --reprocess_method FusionRAG \
  --revert_rope True \
  --use_multi_gpu False \
  --preprocess_scope global \
  --use_entropy_selection False \
  --entropy_top_k 4 \
  --draft_layer_selection entropy \
  --vattention_topk_ratio 0.5 \
  --epsilon 0.1 \
  --delta 0.05 \
  --min_rate 0.05 \
  --max_rate 0.5 \
  --long_decode False \
  --long_decode_max_tokens 1000 \
  --result_path "$out" \
  --rate "$rate" >> "$log" 2>&1
echo "END ${tag} gpu=${gpu} $(date)" | tee -a "$log"
tail -n 8 "$log"
