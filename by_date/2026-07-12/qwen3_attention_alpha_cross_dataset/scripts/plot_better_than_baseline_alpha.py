#!/usr/bin/env python3
from pathlib import Path
import csv
import math

ROOT = Path("MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Source: ROOT/README.md completed cross-dataset table + qwen3_draft_attention_ablation_rate015/musique_final_or_current_summary.csv.
# Baseline is native online DraftModel at rate=0.15 for each dataset.
rows = []

def add(dataset, method, correct, total, baseline_correct, baseline_total, f1=None, em=None):
    acc = correct / total if total else 0.0
    base_acc = baseline_correct / baseline_total if baseline_total else 0.0
    rows.append({
        "dataset": dataset,
        "method": method,
        "correct": correct,
        "total": total,
        "acc": acc,
        "baseline_correct": baseline_correct,
        "baseline_total": baseline_total,
        "baseline_acc": base_acc,
        "delta_pp": (acc - base_acc) * 100.0,
        "avg_f1": f1,
        "avg_em": em,
    })

# Cross-dataset completed alpha results from qwen3_attention_alpha_cross_dataset/README.md.
cross_baseline = {
    "2WikiMQA": (101, 200),
    "HotpotQA": (207, 260),
    "TriviaQA": (214, 270),
}
cross_methods = {
    "uniform alpha=0.1": {"2WikiMQA": (96, 200, 0.3669), "HotpotQA": (226, 260, 0.6133), "TriviaQA": (242, 270, 0.6398)},
    "uniform alpha=0.25": {"2WikiMQA": (99, 200, 0.3419), "HotpotQA": (223, 260, 0.5893), "TriviaQA": (237, 270, 0.6043)},
    "random alpha=0.05": {"2WikiMQA": (94, 200, 0.3745), "HotpotQA": (231, 260, 0.6184), "TriviaQA": (241, 270, 0.6398)},
    "random alpha=0.1": {"2WikiMQA": (95, 200, 0.3652), "HotpotQA": (228, 260, 0.6129), "TriviaQA": (242, 270, 0.6367)},
    "random alpha=0.25": {"2WikiMQA": (100, 200, 0.3422), "HotpotQA": (224, 260, 0.5990), "TriviaQA": (238, 270, 0.6063)},
}
for method, ds_map in cross_methods.items():
    for dataset, vals in ds_map.items():
        correct, total, f1 = vals
        bc, bt = cross_baseline[dataset]
        add(dataset, method, correct, total, bc, bt, f1=f1)

# MuSiQue completed alpha results from qwen3_draft_attention_ablation_rate015/musique_final_or_current_summary.csv.
musique_base = (99, 135)
musique_methods = {
    "uniform alpha=0.1": (107, 135, 0.6021606301988138, 0.25),
    "uniform alpha=0.25": (107, 135, 0.5782605789813382, 0.23387096774193547),
    "random alpha=0.05": (105, 135, 0.5940690903206387, 0.24193548387096775),
    "random alpha=0.1": (105, 135, 0.5898234041139603, 0.24193548387096775),
    "random alpha=0.25": (106, 135, 0.5684650386719329, 0.21774193548387097),
}
for method, (correct, total, f1, em) in musique_methods.items():
    add("MuSiQue", method, correct, total, musique_base[0], musique_base[1], f1=f1, em=em)

# Micro rows over 2WikiMQA/HotpotQA/TriviaQA only, matching README.
for method in cross_methods:
    correct = sum(cross_methods[method][ds][0] for ds in cross_baseline)
    total = sum(cross_methods[method][ds][1] for ds in cross_baseline)
    bc = sum(v[0] for v in cross_baseline.values())
    bt = sum(v[1] for v in cross_baseline.values())
    add("Micro-3sets", method, correct, total, bc, bt)

all_csv = ROOT / "alpha_vs_native_baseline_summary.csv"
with all_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["dataset","method","correct","total","acc","baseline_correct","baseline_total","baseline_acc","delta_pp","avg_f1","avg_em"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

better_csv = ROOT / "better_than_native_baseline_alpha.csv"
with better_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["dataset","method","correct","total","acc","baseline_correct","baseline_total","baseline_acc","delta_pp","avg_f1","avg_em"])
    writer.writeheader()
    for r in rows:
        if r["delta_pp"] > 0:
            writer.writerow(r)

# Plot heatmap of delta pp.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

methods = ["uniform alpha=0.1", "uniform alpha=0.25", "random alpha=0.05", "random alpha=0.1", "random alpha=0.25"]
datasets = ["2WikiMQA", "HotpotQA", "TriviaQA", "MuSiQue", "Micro-3sets"]
lookup = {(r["dataset"], r["method"]): r["delta_pp"] for r in rows}
data = np.array([[lookup.get((ds, m), np.nan) for m in methods] for ds in datasets], dtype=float)

fig, ax = plt.subplots(figsize=(10.5, 4.8))
max_abs = np.nanmax(np.abs(data))
im = ax.imshow(data, cmap="RdYlGn", vmin=-max_abs, vmax=max_abs, aspect="auto")
ax.set_xticks(np.arange(len(methods)), labels=methods, rotation=25, ha="right")
ax.set_yticks(np.arange(len(datasets)), labels=datasets)
ax.set_title("Attention alpha ablation: accuracy delta vs native DraftModel rate=0.15")
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        txt = f"{val:+.2f} pp"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Accuracy delta (percentage points)")
fig.tight_layout()
fig.savefig(FIG_DIR / "alpha_vs_native_baseline_delta_heatmap.png", dpi=200)

# Plot grouped bars for only positive rows by dataset.
pos_rows = [r for r in rows if r["dataset"] != "Micro-3sets" and r["delta_pp"] > 0]
pos_datasets = ["HotpotQA", "TriviaQA", "MuSiQue"]
fig, ax = plt.subplots(figsize=(11, 4.8))
x = np.arange(len(pos_datasets))
width = 0.15
for idx, method in enumerate(methods):
    vals = [lookup.get((ds, method), np.nan) for ds in pos_datasets]
    ax.bar(x + (idx - 2) * width, vals, width, label=method)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x, labels=pos_datasets)
ax.set_ylabel("Accuracy gain over native baseline (pp)")
ax.set_title("Alpha settings that beat native DraftModel on completed datasets")
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "better_than_native_baseline_alpha_bars.png", dpi=200)

print(all_csv)
print(better_csv)
print(FIG_DIR / "alpha_vs_native_baseline_delta_heatmap.png")
print(FIG_DIR / "better_than_native_baseline_alpha_bars.png")

# Additional publication-style figures with baseline and every dataset.
# Color palette follows selection_cost_figures/qk_ttft_breakdown_by_rate.png style.
style_colors = {
    "native DraftModel": "#4B5563",
    "uniform alpha=0.1": "#66C2A5",
    "uniform alpha=0.25": "#FC8D62",
    "random alpha=0.05": "#8DA0CB",
    "random alpha=0.1": "#E78AC3",
    "random alpha=0.25": "#A6D854",
}

baseline_acc = {
    "2WikiMQA": 101/200,
    "HotpotQA": 207/260,
    "TriviaQA": 214/270,
    "MuSiQue": 99/135,
}
acc_lookup = {(r["dataset"], r["method"]): r["acc"] for r in rows}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#D1D5DB",
    "axes.labelcolor": "#111827",
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "font.size": 10,
})

plot_datasets = ["2WikiMQA", "HotpotQA", "TriviaQA", "MuSiQue"]
plot_methods = ["native DraftModel"] + methods
x = np.arange(len(plot_datasets))
width = 0.12
fig, ax = plt.subplots(figsize=(13.2, 5.6))
for idx, method in enumerate(plot_methods):
    vals = []
    for ds in plot_datasets:
        if method == "native DraftModel":
            vals.append(baseline_acc[ds] * 100.0)
        else:
            vals.append(acc_lookup[(ds, method)] * 100.0)
    offset = (idx - (len(plot_methods)-1)/2) * width
    bars = ax.bar(
        x + offset,
        vals,
        width,
        label=method,
        color=style_colors[method],
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.6,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            rotation=90,
            color="#374151",
        )
ax.set_ylim(40, 94)
ax.set_ylabel("Main accuracy (%)")
ax.set_title("DraftModel baseline vs attention alpha variants by dataset", fontsize=13, weight="bold", color="#111827")
ax.set_xticks(x, plot_datasets)
ax.grid(axis="y", color="#E5E7EB", linewidth=0.9)
ax.set_axisbelow(True)
ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13))
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(FIG_DIR / "alpha_accuracy_by_dataset_with_draft_baseline.png", dpi=220)

# Delta figure with all datasets, including negative 2WikiMQA, using the same style.
fig, ax = plt.subplots(figsize=(13.2, 5.4))
for idx, method in enumerate(methods):
    vals = [lookup[(ds, method)] for ds in plot_datasets]
    offset = (idx - (len(methods)-1)/2) * 0.15
    bars = ax.bar(
        x + offset,
        vals,
        0.15,
        label=method,
        color=style_colors[method],
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            val + (0.35 if val >= 0 else -0.55),
            f"{val:+.1f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8,
            rotation=90,
            color="#374151",
        )
ax.axhline(0, color="#111827", linewidth=0.9)
ax.set_ylabel("Delta vs native DraftModel (percentage points)")
ax.set_title("Attention alpha gains/losses vs native DraftModel by dataset", fontsize=13, weight="bold", color="#111827")
ax.set_xticks(x, plot_datasets)
ax.set_ylim(-5.5, 12.0)
ax.grid(axis="y", color="#E5E7EB", linewidth=0.9)
ax.set_axisbelow(True)
ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13))
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(FIG_DIR / "alpha_delta_by_dataset_with_all_methods.png", dpi=220)

print(FIG_DIR / "alpha_accuracy_by_dataset_with_draft_baseline.png")
print(FIG_DIR / "alpha_delta_by_dataset_with_all_methods.png")
