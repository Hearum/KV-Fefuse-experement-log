#!/usr/bin/env bash
set -euo pipefail
cd /raid/home/hming/FusionRAG-pca-analysis
out=MOTIVATION_EXPERIMENTS/kv_lora/results/original_pipeline_refit_adapter_second_holdout
adapter=/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora/results/shared_preprocess_residual_adapter_refit50.pt
mkdir -p "$out/logs"
common=(--model_type qwen3 --model_path /mnt/qjhs-sh-lab-01/models/Qwen3-32B --model_name Qwen3-32B --data_path ./data/result_reflect.json --dataset_name musique --cache_path /raid/home/hming/fusionrag-reflect-qwen3-full-cache --result_path "/raid/home/hming/FusionRAG-pca-analysis/$out" --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3 --topk 10 --recall_method bge --revert_rope true --max_samples 100 --start_sample 50 --end_sample 100 --max_cache_len 8192 --reprocess_method FusionRAG --rate 0 --preprocess true --static_kv_linear_adapter_path "$adapter" --static_kv_linear_adapter_scale 0.5)
for spec in 0:full 1:rank64;do gpu=${spec%%:*};mode=${spec##*:};OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$gpu nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python test_fusionrag_reflect_preprocess_exp.py "${common[@]}" --static_kv_linear_adapter_mode "$mode" > "$out/logs/$mode.log" 2>&1 < /dev/null & echo $! > "$out/logs/$mode.pid";done
