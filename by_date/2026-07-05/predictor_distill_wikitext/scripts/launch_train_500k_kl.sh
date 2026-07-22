#!/usr/bin/env bash
set -euo pipefail

REPO=/raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python

cd "$REPO"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} "$PY" -m accelerate.commands.launch \
  --num_processes 8 \
  --mixed_precision fp16 \
  MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/scripts/predictor_distill_accelerate.py \
  --cache-dir MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/teacher_cache_500k \
  --out-dir MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/checkpoints/kl500k_h512_l4 \
  --hidden 512 \
  --layers 4 \
  --heads 8 \
  --batch-size 64 \
  --epochs 20 \
  --lr 3e-4 \
  --temperature 2.0 \
  --bce-weight 0.0 \
  --eval-ratios 0.05,0.10,0.15,0.30
