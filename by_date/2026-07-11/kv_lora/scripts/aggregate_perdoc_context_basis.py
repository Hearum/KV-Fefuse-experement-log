#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

root = Path("/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora")
files = sorted((root / "results").glob("perdoc_context_basis_t?_40train10test.csv"))
data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
keys = ["part", "target", "source", "kind", "context_index", "rank"]
sums = data.groupby(keys, as_index=False)[
    ["delta_norm2", "base_norm2", "error_norm2", "prediction_norm2", "target_prediction_dot"]
].sum()
sums["original_gap"] = (sums.delta_norm2 / sums.base_norm2).pow(.5)
sums["delta_recovery_error"] = (sums.error_norm2 / sums.delta_norm2).pow(.5)
sums["final_kv_error"] = (sums.error_norm2 / sums.base_norm2).pow(.5)
sums["explained_delta_energy"] = 1 - sums.error_norm2 / sums.delta_norm2
sums["cosine"] = sums.target_prediction_dot / (sums.delta_norm2 * sums.prediction_norm2).pow(.5)
sums.to_csv(root / "results/perdoc_context_basis_5targets_50contexts_40train10test_context.csv", index=False)

metrics = ["original_gap", "delta_recovery_error", "final_kv_error", "explained_delta_energy", "cosine"]
summary = []
for (source, kind, rank), group in sums.groupby(["source", "kind", "rank"]):
    item = {"source": source, "kind": kind, "rank": int(rank), "n_heldout_contexts": len(group), "n_targets": group.target.nunique()}
    for metric in metrics:
        values = group[metric]
        item.update({
            f"{metric}_mean": float(values.mean()), f"{metric}_std": float(values.std()),
            f"{metric}_median": float(values.median()), f"{metric}_p10": float(values.quantile(.1)),
            f"{metric}_p90": float(values.quantile(.9)),
        })
    summary.append(item)
(root / "results/perdoc_context_basis_5targets_50contexts_40train10test_summary.json").write_text(
    json.dumps({"files": [str(x) for x in files], "aggregation": "sum norm squares over 64 layers per heldout context, then distribution over 5*10 contexts", "summary": summary}, indent=2) + "\n"
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, source in zip(axes, ["raw", "preprocess"]):
    for kind, color in [("k", "#2471A3"), ("v", "#C0392B")]:
        part = sums[(sums.source == source) & (sums.kind == kind)]
        agg = part.groupby("rank").delta_recovery_error.agg(["mean", lambda x: x.quantile(.1), lambda x: x.quantile(.9)])
        ax.plot(agg.index, agg.iloc[:, 0], marker="o", label=kind.upper(), color=color)
        ax.fill_between(agg.index, agg.iloc[:, 1], agg.iloc[:, 2], color=color, alpha=.16)
    ax.set_title(source); ax.set_xlabel("Context-template rank"); ax.grid(alpha=.25); ax.legend()
axes[0].set_ylabel("Held-out Delta relative L2 error")
fig.tight_layout(); fig.savefig(root / "figures/perdoc_context_basis_5targets_heldout.png", dpi=180)
print(json.dumps(summary, indent=2))
