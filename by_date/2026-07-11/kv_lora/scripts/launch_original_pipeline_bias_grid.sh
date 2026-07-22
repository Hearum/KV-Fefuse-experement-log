#!/usr/bin/env bash
set -euo pipefail
cd /home/hming/FusionRAG-pca-analysis
bias_root=MOTIVATION_EXPERIMENTS/kv_lora/results/static_bias_grid_50
out=MOTIVATION_EXPERIMENTS/kv_lora/results/original_pipeline_bias_grid
pp_bias_root=MOTIVATION_EXPERIMENTS/kv_lora/results/static_bias_grid_50_preprocess
mkdir -p "$out/logs"
configs=(m0n2 m0n4 m0n8 m1n16 m3n2 m3n16 m5n16 m10n16)
for cfg in "${configs[@]}"; do
  while [[ $(find "$bias_root/$cfg" -name '*_value_bias.pt' 2>/dev/null | wc -l) -lt 550 ]]; do sleep 30; done
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/kv_lora/scripts/convert_raw_bias_to_preprocess_bias.py \
    --raw-bias-dir "$bias_root/$cfg" --output-dir "$pp_bias_root/$cfg" \
    --raw-cache /home/hming/fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique/kv_cache \
    --preprocess-cache /home/hming/fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique/preprocess_kv_cache_global_topk10_bge
done
common=(
  --model_type qwen3 --model_path /mnt/qjhs-sh-lab-01/models/Qwen3-32B --model_name Qwen3-32B
  --data_path ./data/result_reflect.json --dataset_name musique
  --cache_path /home/hming/fusionrag-reflect-qwen3-full-cache
  --result_path "/home/hming/FusionRAG-pca-analysis/$out"
  --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3
  --topk 10 --recall_method bge --revert_rope true --max_samples 50 --max_cache_len 8192
  --reprocess_method FusionRAG --rate 0 --static_value_bias_require_all true
)
run_triple() {
  local gpu=$1 cfg=$2 bias="/home/hming/FusionRAG-pca-analysis/$bias_root/$2" pp_bias="/home/hming/FusionRAG-pca-analysis/$pp_bias_root/$2"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
    /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python test_fusionrag_reflect_preprocess_exp.py \
    "${common[@]}" --preprocess false --static_value_bias_path "$bias" \
    > "$out/logs/${cfg}_raw.log" 2>&1
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
    /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python test_fusionrag_reflect_preprocess_exp.py \
    "${common[@]}" --preprocess true --static_value_bias_path "$bias" \
    > "$out/logs/${cfg}_preprocess_cross_rawbias.log" 2>&1
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$gpu" \
    /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python test_fusionrag_reflect_preprocess_exp.py \
    "${common[@]}" --preprocess true --static_value_bias_path "$pp_bias" \
    > "$out/logs/${cfg}_preprocess_source_bias.log" 2>&1
}
for i in 0 1 2 3 4 5 6 7; do
  run_triple "$i" "${configs[$i]}" &
  echo $! > "$out/logs/${configs[$i]}.worker.pid"
done
wait
