#!/usr/bin/env bash
set -euo pipefail

REPO=/raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python

cd "$REPO"

OUT_DIR=MOTIVATION_EXPERIMENTS/selector_aware_draft_model/checkpoints/first12_attnprob_kl_100k_e3

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} "$PY" -m accelerate.commands.launch \
  --num_processes 4 \
  --mixed_precision bf16 \
  MOTIVATION_EXPERIMENTS/native_initialized_layer4_distill/scripts/train_native_layer4_selector.py \
  --cache-dir MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/teacher_cache_wikitext103_train_500k \
  --val-cache-dir MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/teacher_cache_wikitext103_val_50k \
  --musique-val-cache-dir MOTIVATION_EXPERIMENTS/native_initialized_layer4_distill/teacher_cache_musique_val \
  --out-dir "$OUT_DIR" \
  --layers 12 \
  --score-mode attn_prob \
  --train-limit 100000 \
  --val-limit 5000 \
  --batch-size 1 \
  --epochs 3 \
  --lr 1e-6 \
  --temperature 2.0 \
  --eval-ratios 0.05,0.10,0.15,0.30 \
  --log-every 25 \
  --save-every-epoch
