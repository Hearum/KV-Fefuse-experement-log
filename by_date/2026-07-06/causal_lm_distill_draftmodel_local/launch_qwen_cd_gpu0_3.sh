#!/usr/bin/env bash
set -euo pipefail
REPO=/raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
EXP=$REPO/MOTIVATION_EXPERIMENTS/causal_lm_distill_draftmodel_local
OUT=$EXP/outputs/qwen_cd_wikitext103_4layer_gpu0_3_s256_n20000_step2000
LOG=$EXP/logs/qwen_cd_wikitext103_4layer_gpu0_3_s256_n20000_step2000.log
mkdir -p "$OUT" "$EXP/logs" "$EXP/data"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export MASTER_PORT=${MASTER_PORT:-29631}
nohup "$PY" -m torch.distributed.run --nproc_per_node=4 --master_port "$MASTER_PORT" \
  MOTIVATION_EXPERIMENTS/causal_lm_distill_draftmodel_local/scripts/train_qwen_causal_distill.py \
  --teacher-model /mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct \
  --student-init-model /mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct \
  --text-path MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/data/wikitext103_train.txt \
  --out-dir "$OUT" \
  --cache-path MOTIVATION_EXPERIMENTS/causal_lm_distill_draftmodel_local/data/wikitext103_qwen_blocks_s256_n20000.pt \
  --student-layers 4 \
  --seq-len 256 \
  --max-sequences 20000 \
  --epochs 5 \
  --max-steps 2000 \
  --per-device-batch-size 1 \
  --grad-accum 8 \
  --lr 1e-5 \
  --temperature 2.0 \
  --kl-weight 1.0 \
  --ce-weight 0.1 \
  --log-every 10 \
  --save-every 200 \
  > "$LOG" 2>&1 &
echo $! > "$EXP/logs/qwen_cd_wikitext103_4layer_gpu0_3_s256_n20000_step2000.pid"
echo "launched pid=$(cat "$EXP/logs/qwen_cd_wikitext103_4layer_gpu0_3_s256_n20000_step2000.pid") log=$LOG out=$OUT"
