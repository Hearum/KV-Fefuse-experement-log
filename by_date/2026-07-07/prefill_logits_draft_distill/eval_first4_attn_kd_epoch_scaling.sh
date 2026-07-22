#!/usr/bin/env bash
set -euo pipefail
REPO=/raid/home/hming/FusionRAG-pca-analysis
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
cd "$REPO"
BASE=MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/outputs/first4_wikitext_attn_kd_s256_step500
OUT_ROOT=MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/epoch_scaling_eval_first4_attn_kd
mkdir -p "$OUT_ROOT"
STEPS=(500 1250 2500 3750 5000 6250 7500 8750 10000 11250 12500)
for STEP in "${STEPS[@]}"; do
  if [[ "$STEP" == "12500" ]]; then
    CKPT="$BASE/training_state_final_step012500.pt"
    [[ -f "$CKPT" ]] || CKPT="$BASE/training_state_step012500.pt"
  elif [[ "$STEP" == "500" ]]; then
    CKPT="$BASE/training_state_final_step000500.pt"
  else
    CKPT=$(printf "%s/training_state_step%06d.pt" "$BASE" "$STEP")
  fi
  if [[ ! -f "$CKPT" ]]; then
    echo "[skip] missing checkpoint step=$STEP path=$CKPT"
    continue
  fi
  OUT=$(printf "%s/step%06d" "$OUT_ROOT" "$STEP")
  if [[ -f "$OUT/metrics.json" ]]; then
    echo "[skip] already evaluated step=$STEP"
    continue
  fi
  echo "[eval] step=$STEP ckpt=$CKPT"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PY" \
    MOTIVATION_EXPERIMENTS/causal_lm_distill_draftmodel_local/scripts/eval_cd_layer4_against_full_draft.py \
    --cd-checkpoint "$CKPT" \
    --out-dir "$OUT" \
    --layers 4 \
    --score-mode attn_prob \
    --wiki-limit 5000 \
    --musique-limit 0 \
    --batch-size 1 \
    --eval-ratios 0.05,0.10,0.15,0.30
 done

"$PY" - <<"PY"
import csv, json
from pathlib import Path
root=Path("MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/epoch_scaling_eval_first4_attn_kd")
rows=[]
for d in sorted(root.glob("step*")):
    p=d/"metrics.json"
    if not p.exists():
        continue
    step=int(d.name.replace("step", ""))
    data=json.loads(p.read_text())
    for split in ["wiki", "musique"]:
        m=data.get(split, {})
        rows.append({
            "step": step,
            "epoch_equiv": step/1250,
            "split": split,
            "kl": m.get("val_kl"),
            "r05": m.get("val_recall_r0p05"),
            "j05": m.get("val_jaccard_r0p05"),
            "r10": m.get("val_recall_r0p1"),
            "j10": m.get("val_jaccard_r0p1"),
            "r15": m.get("val_recall_r0p15"),
            "j15": m.get("val_jaccard_r0p15"),
            "r30": m.get("val_recall_r0p3"),
            "j30": m.get("val_jaccard_r0p3"),
        })
out=root/"summary.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=["step","epoch_equiv","split","kl","r05","j05","r10","j10","r15","j15","r30","j30"])
    w.writeheader(); w.writerows(rows)
lines=["# First4 Attention-Aware KD Epoch Scaling\n\n", "| step | epoch | split | KL | R@15 | J@15 | R@30 | J@30 |\n", "|---:|---:|---|---:|---:|---:|---:|---:|\n"]
for r in rows:
    lines.append(f"| {r[step]} | {r[epoch_equiv]:.2f} | {r[split]} | {r[kl]:.4f} | {r[r15]:.4f} | {r[j15]:.4f} | {r[r30]:.4f} | {r[j30]:.4f} |\n")
(root/"README.md").write_text("".join(lines), encoding="utf-8")
print("wrote", out)
print("".join(lines))
PY
