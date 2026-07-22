#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path("/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/structured_kv_adapter")
data = json.loads((root / "results/shared_mlp_predictor_balanced.json").read_text())
epochs = [row["epoch"] for row in data["history"]]
train = [row["train_loss"] for row in data["history"]]
validation = [row["validation_loss"] for row in data["history"]]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(epochs, train, marker="o", label="Train")
ax.plot(epochs, validation, marker="o", label="Validation")
ax.axvline(data["best_epoch"], color="#555555", linestyle="--", label="Selected epoch")
ax.set(xlabel="Epoch", ylabel="Normalized coefficient MSE", title="Balanced shared MLP training curve")
ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
fig.savefig(root / "figures/shared_mlp_balanced_training_curve.png", dpi=180)
