#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXP = Path("/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter")
data = pd.read_csv(EXP / "results/strict_anchor_transfer/summary.csv")
methods = ["no_update", "random_anchor", "mean_anchor", "position_top8", "oracle_anchor"]
labels = ["No update", "Random", "Mean", "Position top-8", "Oracle"]
colors = ["#9CA3AF", "#E45756", "#72B7B2", "#4C78A8", "#F2CF5B"]
panels = [
    ("raw", "k", "Raw / RoPE-aligned K"),
    ("raw", "v", "Raw / V"),
    ("preprocess", "k", "Preprocess / RoPE-aligned K"),
    ("preprocess", "v", "Preprocess / V"),
]

fig, axes = plt.subplots(1, 4, figsize=(14.2, 4.1), sharey=True)
for ax, (source, kind, title) in zip(axes, panels):
    subset = data[(data.source == source) & (data.kind == kind)].set_index("method")
    values = [float(subset.loc[method, "recovery"]) for method in methods]
    bars = ax.bar(range(len(methods)), values, color=colors, width=0.72)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(methods)), labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 1.08)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
axes[0].set_ylabel("Remaining Delta L2 (lower is better)")
fig.tight_layout()
out = EXP / "figures/strict_anchor_recovery.png"
fig.savefig(out, dpi=200)
print(out)
