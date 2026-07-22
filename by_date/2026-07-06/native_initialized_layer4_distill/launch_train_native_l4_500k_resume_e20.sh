#!/usr/bin/env bash
set -euo pipefail
REPO=/raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
cd "$REPO"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} "$PY" -m accelerate.commands.launch \
  --num_processes 8 \
  --mixed_precision bf16 \
  MOTIVATION_EXPERIMENTS/native_initialized_layer4_distill/scripts/train_native_layer4_selector.py \
  --cache-dir MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/teacher_cache_wikitext103_train_500k \
  --val-cache-dir MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/teacher_cache_wikitext103_val_50k \
  --musique-val-cache-dir MOTIVATION_EXPERIMENTS/native_initialized_layer4_distill/teacher_cache_musique_val \
  --out-dir MOTIVATION_EXPERIMENTS/native_initialized_layer4_distill/checkpoints/native_l4_wikitext500k_e20_resume \
  --resume-from MOTIVATION_EXPERIMENTS/native_initialized_layer4_distill/checkpoints/native_l4_wikitext500k_e1/native_layer4_selector.pt \
  --train-limit 0 \
  --val-limit 5000 \
  --batch-size 2 \
  --epochs 20 \
  --lr 1e-6 \
  --temperature 2.0 \
  --log-every 25 \
  --save-every-epoch
