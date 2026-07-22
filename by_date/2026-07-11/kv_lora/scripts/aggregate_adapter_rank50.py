#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
BASE = ROOT / "MOTIVATION_EXPERIMENTS/kv_lora"
PARTS = BASE / "results/adapter_rank50_parts"
OUT = BASE / "results"
FIG = BASE / "figures"
N = 50
Q = [0.1, 0.5, 0.9]


def stats(frame, columns):
    groups = []
    for keys, group in frame.groupby(columns, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {
            column: (value.item() if hasattr(value, "item") else value)
            for column, value in zip(columns, keys)
        }
        values = group["value"]
        row.update(
            mean=float(values.mean()),
            std=float(values.std(ddof=1)),
            median=float(values.median()),
            p10=float(values.quantile(Q[0])),
            p90=float(values.quantile(Q[2])),
            n=int(values.count()),
        )
        groups.append(row)
    return groups


global_df = pd.concat(
    [pd.read_csv(path) for path in sorted(PARTS.glob("p*_global.csv"))],
    ignore_index=True,
)
layer_df = pd.concat(
    [pd.read_csv(path) for path in sorted(PARTS.glob("p*_layers.csv"))],
    ignore_index=True,
)
required = {("raw", "k"), ("raw", "v"), ("preprocess", "k"), ("preprocess", "v")}
valid = []
for example, group in global_df.groupby("example"):
    if len(group) == 4 and set(zip(group.source, group.kind)) == required:
        valid.append(int(example))
selected = sorted(valid)[:N]
if len(selected) < N:
    raise RuntimeError(f"only {len(selected)} complete paired examples")

global_df = global_df[global_df.example.isin(selected)].sort_values(["example", "source", "kind"])
layer_df = layer_df[layer_df.example.isin(selected)].sort_values(
    ["example", "source", "kind", "layer", "rank"]
)
expected = N * 4 * 64 * 5
if len(layer_df) != expected:
    raise RuntimeError(f"layer rows {len(layer_df)} != {expected}")
if not (global_df.tokens == global_df.selected).all():
    raise RuntimeError("rate=1 did not select every document token")

global_path = OUT / "qwen3_32b_50ex_rate1_adapter_rank_distortion_global.csv"
layer_path = OUT / "qwen3_32b_50ex_rate1_adapter_rank_distortion_layers.csv"
global_df.to_csv(global_path, index=False)
layer_df.to_csv(layer_path, index=False)

global_long = global_df.melt(
    id_vars=["example", "source", "kind"],
    value_vars=["original_gap"],
    var_name="metric",
    value_name="value",
)
# Equal example weighting: average all 64 layers inside each example first.
example_rank = (
    layer_df.groupby(["example", "source", "kind", "rank"], as_index=False)[
        ["delta_recovery_error", "final_kv_error", "explained_variance"]
    ]
    .mean()
)
rank_long = example_rank.melt(
    id_vars=["example", "source", "kind", "rank"],
    var_name="metric",
    value_name="value",
)
summary = {
    "selected_examples": selected,
    "available_complete_examples": sorted(valid),
    "n": N,
    "aggregation": "mean over 64 layers per example, then distribution over examples",
    "global_gap": stats(global_long, ["source", "kind", "metric"]),
    "rank_distortion": stats(rank_long, ["source", "kind", "rank", "metric"]),
}
summary_path = OUT / "qwen3_32b_50ex_rate1_adapter_rank_distortion_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")

colors = {"k": "#2471A3", "v": "#C0392B"}
for metric, ylabel, filename in [
    ("delta_recovery_error", "Relative Delta L2 error", "adapter_rank50_delta_recovery.png"),
    ("final_kv_error", "Final KV error / base KV L2", "adapter_rank50_final_kv_error.png"),
    ("explained_variance", "Explained Delta energy", "adapter_rank50_explained_variance.png"),
]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, source in zip(axes, ["raw", "preprocess"]):
        for kind in ["k", "v"]:
            part = rank_long[(rank_long.source == source) & (rank_long.kind == kind) & (rank_long.metric == metric)]
            agg = part.groupby("rank").value.agg(["mean", lambda x: x.quantile(.1), lambda x: x.quantile(.9)])
            ranks = agg.index.to_numpy()
            mean, low, high = agg.iloc[:, 0].to_numpy(), agg.iloc[:, 1].to_numpy(), agg.iloc[:, 2].to_numpy()
            ax.plot(ranks, mean, marker="o", label=kind.upper(), color=colors[kind])
            ax.fill_between(ranks, low, high, alpha=.16, color=colors[kind])
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32, 64])
        ax.set_xticklabels([4, 8, 16, 32, 64])
        ax.set_title(source)
        ax.set_xlabel("Rank")
        ax.grid(alpha=.25)
    axes[0].set_ylabel(ylabel)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)

print(json.dumps(summary, indent=2))
