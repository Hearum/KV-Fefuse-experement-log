#!/usr/bin/env bash
set -euo pipefail
cd /raid/home/hming/FusionRAG-pca-analysis

PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
OUT_ROOT=MOTIVATION_EXPERIMENTS/kv_update_rate_sweep_nojudge
GPU=${CUDA_VISIBLE_DEVICES:-6}
COMMON_ARGS=(
  --model_type qwen
  --model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct
  --model_name Qwen2.5-7B-Instruct
  --data_path ./data/result_reflect.json
  --dataset_name musique
  --cache_path /raid/home/hming/fusionrag-reflect-full-cache/
  --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3
  --topk 10
  --preprocess True
  --recall_method bge
  --random_seed 42
  --fixed_doc_idx 0
  --reprocess_method FusionRAG
  --revert_rope True
  --use_multi_gpu False
  --preprocess_scope global
  --use_entropy_selection False
  --entropy_top_k 4
  --draft_layer_selection entropy
  --vattention_topk_ratio 0.5
  --epsilon 0.1
  --delta 0.05
  --min_rate 0.05
  --max_rate 0.5
  --long_decode False
  --long_decode_max_tokens 1000
)

run_one() {
  local mode="$1"
  local rate="$2"
  local tag="${mode}_rate${rate//./p}"
  local out="${OUT_ROOT}/${tag}"
  local log="${OUT_ROOT}/${tag}.log"
  if [[ -f "${out}/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_${rate}_revert_rope.csv" ]]; then
    echo "SKIP existing ${tag}" | tee -a "${OUT_ROOT}/run_sweep.log"
    return 0
  fi
  echo "START ${tag} $(date)" | tee -a "${OUT_ROOT}/run_sweep.log" "$log"
  FUSIONRAG_SKIP_LLM_JUDGE=1 \
  FUSIONRAG_STRICT_REPROCESS_ABLATION=1 \
  FUSIONRAG_CLEAN_STRICT_ABLATION=1 \
  FUSIONRAG_REPROCESS_UPDATE_MODE="$mode" \
  CUDA_VISIBLE_DEVICES="$GPU" \
  TMPDIR=/raid/home/hming/tmp \
  "$PY" test_fusionrag_reflect_preprocess_exp.py \
    "${COMMON_ARGS[@]}" \
    --result_path "$out" \
    --rate "$rate" >> "$log" 2>&1
  echo "END ${tag} $(date)" | tee -a "${OUT_ROOT}/run_sweep.log" "$log"
  tail -n 8 "$log" | tee -a "${OUT_ROOT}/run_sweep.log"
}

# rate=0 and rate=1 are shared baselines, but rate=1 still needs one full-recompute run.
run_one kv 1.0
for rate in 0.3 0.5 0.8; do
  for mode in kv v_only k_only; do
    run_one "$mode" "$rate"
  done
done
