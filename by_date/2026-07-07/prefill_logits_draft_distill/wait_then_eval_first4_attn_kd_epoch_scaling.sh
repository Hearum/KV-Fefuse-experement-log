#!/usr/bin/env bash
set -euo pipefail
REPO=/raid/home/hming/FusionRAG-pca-analysis
cd "$REPO"
PID=$(cat MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/first4_wikitext_attn_kd_s256_10epoch_resume.pid)
echo "waiting training pid=$PID"
while kill -0 "$PID" 2>/dev/null; do
  sleep 60
done
echo "training finished; start epoch scaling eval"
bash MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/eval_first4_attn_kd_epoch_scaling.sh
cat >> MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/README.md <<"MD"

## Epoch scaling eval 完成

已自动完成每个 epoch checkpoint 的 selector-overlap 评估，汇总见：

- `MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/epoch_scaling_eval_first4_attn_kd/summary.csv`
- `MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/epoch_scaling_eval_first4_attn_kd/README.md`
MD
