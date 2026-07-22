#!/usr/bin/env bash
set -euo pipefail
cd /raid/home/hming/FusionRAG-pca-analysis
out=MOTIVATION_EXPERIMENTS/kv_lora/results/original_pipeline_shared_adapter_second_holdout
adapter=/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora/results/shared_preprocess_residual_adapter.pt
mkdir -p "$out/logs"
common=(--model_type qwen3 --model_path /mnt/qjhs-sh-lab-01/models/Qwen3-32B --model_name Qwen3-32B --data_path ./data/result_reflect.json --dataset_name musique --cache_path /raid/home/hming/fusionrag-reflect-qwen3-full-cache --result_path "/raid/home/hming/FusionRAG-pca-analysis/$out" --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3 --topk 10 --recall_method bge --revert_rope true --max_samples 100 --start_sample 50 --end_sample 100 --max_cache_len 8192 --reprocess_method FusionRAG)
run(){ local gpu=$1 name=$2;shift 2;OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python test_fusionrag_reflect_preprocess_exp.py "${common[@]}" "$@" > "$out/logs/$name.log" 2>&1 < /dev/null & echo $! > "$out/logs/$name.pid";}
run 0 full --rate 1 --preprocess false
run 1 preprocess --rate 0 --preprocess true
run 2 adapter_full --rate 0 --preprocess true --static_kv_linear_adapter_path "$adapter" --static_kv_linear_adapter_mode full --static_kv_linear_adapter_scale 0.5
run 3 adapter_rank64 --rate 0 --preprocess true --static_kv_linear_adapter_path "$adapter" --static_kv_linear_adapter_mode rank64 --static_kv_linear_adapter_scale 0.5
