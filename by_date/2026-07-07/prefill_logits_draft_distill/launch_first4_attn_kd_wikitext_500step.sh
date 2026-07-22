#!/usr/bin/env bash
set -euo pipefail
REPO=/raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
cd "$REPO"
OUT=MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/outputs/first4_wikitext_attn_kd_s256_step500
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} "$PY" -m torch.distributed.run --nproc_per_node=4 \
  MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/scripts/train_small_draft_lm_attn_kd.py \
  --out-dir "$OUT" \
  --cache-path MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/data/wikitext_blocks_s256_n20000.pt \
  --student-layers 4 \
  --seq-len 256 \
  --max-sequences 20000 \
  --max-steps 500 \
  --per-device-batch-size 1 \
  --grad-accum 4 \
  --lr 5e-6 \
  --temperature 2.0 \
  --kl-weight 1.0 \
  --ce-weight 0.1 \
  --attn-weight 1.0 \
  --teacher-attn-mode lastn \
  --teacher-attn-last-n 18 \
  --student-attn-mode last \
  --log-every 10 \
  --save-every 100
