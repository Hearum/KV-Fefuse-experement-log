#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(
    "/home/hming/FusionRAG-pca-analysis/"
    "MOTIVATION_EXPERIMENTS/qwen3_235b_param_scaling_cross_dataset/figures"
)

METHODS = [
    ("full_rate1", "Full recompute", "#444444"),
    ("online_qk_rate015", "Online QK", "#66c2a5"),
    ("online_draft_rate015", "Online Draft", "#fc8d62"),
    ("offline3b_mean_rate015", "Offline 3B mean", "#8da0cb"),
    ("offline3b_freq_boundary2_rate015", "Offline 3B + boundary", "#b3b3b3"),
    ("offline32b_top2_rate015", "Offline 32B top2", "#a6d854"),
]

DATA = {
    "MuSiQue": {
        "full_rate1": {"main": 78.52, "sub": 86.69, "status": "done"},
        "online_qk_rate015": {"main": 54.81, "sub": 72.18, "status": "done"},
        "online_draft_rate015": {"main": 68.89, "sub": 82.26, "status": "done"},
        "offline3b_mean_rate015": {"main": 61.48, "sub": 77.82, "status": "done"},
        "offline3b_freq_boundary2_rate015": {"main": 65.93, "sub": 77.02, "status": "done"},
        "offline32b_top2_rate015": {"main": 63.70, "sub": 76.61, "status": "done"},
    },
    "2WikiMQA": {
        "full_rate1": {"main": 56.50, "sub": 56.50, "status": "done"},
        "online_qk_rate015": {"main": 49.00, "sub": 49.00, "status": "done"},
        "online_draft_rate015": {"main": 53.00, "sub": 53.00, "status": "done"},
        "offline3b_mean_rate015": {"main": 51.00, "sub": 51.00, "status": "done"},
        "offline3b_freq_boundary2_rate015": {"main": 52.50, "sub": 52.50, "status": "done"},
        "offline32b_top2_rate015": {"main": 51.50, "sub": 51.50, "status": "done"},
    },
    "HotpotQA": {
        "full_rate1": {"main": 85.00, "sub": 85.00, "status": "done"},
        "online_qk_rate015": {"main": 80.38, "sub": 80.38, "status": "done"},
        "online_draft_rate015": {"main": 81.15, "sub": 81.15, "status": "done"},
        "offline3b_mean_rate015": {"main": 83.85, "sub": 83.85, "status": "done"},
        "offline3b_freq_boundary2_rate015": {"main": 78.71, "sub": 78.71, "status": "partial"},
    },
}


def style_axes(ax):
    ax.grid(True, axis="y", color="#d0d7de", linewidth=0.8, alpha=0.65)
    ax.grid(True, axis="x", color="#eaeef2", linewidth=0.6, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)


def plot_metric(ax, metric, ylabel):
    datasets = list(DATA)
    x = np.arange(len(datasets))
    width = 0.12
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2.0) * width

    for idx, (method, label, color) in enumerate(METHODS):
        values = []
        statuses = []
        for dataset in datasets:
            item = DATA[dataset].get(method)
            values.append(np.nan if item is None else item[metric])
            statuses.append(None if item is None else item["status"])
        bars = ax.bar(
            x + offsets[idx],
            values,
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, status in zip(bars, statuses):
            if status == "partial":
                bar.set_hatch("///")
                bar.set_edgecolor("#555555")
                bar.set_linewidth(0.6)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 100)
    style_axes(ax)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 4.7), sharey=True)
    plot_metric(axes[0], "main", "Main accuracy (%)")
    plot_metric(axes[1], "sub", "Sub-question accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10)
    fig.text(0.985, 0.03, "hatched = partial result", ha="right", va="bottom", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 0.88))

    png = OUT_DIR / "qwen235b_non_trivia_accuracy.png"
    pdf = OUT_DIR / "qwen235b_non_trivia_accuracy.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
