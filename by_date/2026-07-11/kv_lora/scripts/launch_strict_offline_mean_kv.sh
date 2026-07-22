#!/usr/bin/env bash
set -euo pipefail
cd /raid/home/hming/FusionRAG-pca-analysis
out=MOTIVATION_EXPERIMENTS/kv_lora/results/original_pipeline_strict_offline_mean_kv
mkdir -p "$out/logs"
common=(--model_type qwen3 --model_path /mnt/qjhs-sh-lab-01/models/Qwen3-32B --model_name Qwen3-32B --data_path ./data/result_reflect.json --dataset_name musique --cache_path /raid/home/hming/fusionrag-reflect-qwen3-full-cache --result_path "/raid/home/hming/FusionRAG-pca-analysis/$out" --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3 --topk 10 --recall_method bge --revert_rope true --max_samples 5 --max_cache_len 8192 --reprocess_method FusionRAG --rate 0 --static_key_bias_require_all true --static_value_bias_require_all true)
run() {
 local gpu=$1 source=$2 preprocess=$3 bias="/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora/results/strict_offline_mean_kv_${2}_ex0to4_m6"
 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python test_fusionrag_reflect_preprocess_exp.py "${common[@]}" --preprocess "$preprocess" --static_key_bias_path "$bias" --static_value_bias_path "$bias" > "$out/logs/${source}_mean_kv.log" 2>&1 < /dev/null &
 echo $! > "$out/logs/${source}_mean_kv.pid"
}
run 0 raw false
run 1 preprocess true
