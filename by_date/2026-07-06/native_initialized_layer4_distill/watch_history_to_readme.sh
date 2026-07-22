#!/usr/bin/env bash
set -u
cd /raid/home/hming/FusionRAG-pca-analysis
EXP=MOTIVATION_EXPERIMENTS/native_initialized_layer4_distill
PYTHON=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
while true; do
  "$PYTHON" "$EXP/update_readme_with_history.py" || true
  sleep 60
done
