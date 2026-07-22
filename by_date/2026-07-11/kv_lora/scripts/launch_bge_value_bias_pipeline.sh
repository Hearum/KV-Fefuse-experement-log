#!/usr/bin/env bash
set -euo pipefail
cd /raid/home/hming/FusionRAG-pca-analysis
out=MOTIVATION_EXPERIMENTS/kv_lora/results/original_pipeline_bge_value_bias
bias=/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora/results/static_bias_bge_preprocess_minus_raw_50
mkdir -p "$out/logs"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python test_fusionrag_reflect_preprocess_exp.py \
 --model_type qwen3 --model_path /mnt/qjhs-sh-lab-01/models/Qwen3-32B --model_name Qwen3-32B \
 --data_path ./data/result_reflect.json --dataset_name musique --cache_path /raid/home/hming/fusionrag-reflect-qwen3-full-cache \
 --result_path /raid/home/hming/FusionRAG-pca-analysis/$out --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3 \
 --topk 10 --recall_method bge --revert_rope true --max_samples 50 --max_cache_len 8192 --reprocess_method FusionRAG \
 --rate 0 --preprocess false --static_value_bias_path "$bias" --static_value_bias_require_all true \
 > "$out/logs/rawK_bgeV_rate0.log" 2>&1 < /dev/null &
echo $! > "$out/logs/rawK_bgeV_rate0.pid"
