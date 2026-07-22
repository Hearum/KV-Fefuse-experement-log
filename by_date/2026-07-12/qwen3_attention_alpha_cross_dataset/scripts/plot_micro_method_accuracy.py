#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/raid/home/hming/FusionRAG-pca-analysis")
ROOT = REPO / "MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset"
FIG_DIR = ROOT / "figures"
OUT_CSV = ROOT / "micro_all_datasets_method_accuracy.csv"

METHODS = [
    ("Full attention r=1", 636, 865),
    ("Online QK r=0.15", 609, 865),
    ("Online Draft r=0.15", 621, 865),
    ("Uniform a=0.1", 671, 865),
    ("Uniform a=0.25", 666, 865),
    ("Random a=0.05", 671, 865),
    ("Random a=0.1", 670, 865),
    ("Random a=0.25", 668, 865),
]
DRAFT_ACC = 621 / 865


def style_axes(ax):
    ax.grid(True, axis="y", color="#d0d7de", linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, correct, total in METHODS:
        acc = correct / total
        rows.append({
            "method": method,
            "correct": correct,
            "total": total,
            "acc": acc,
            "delta_vs_online_draft_pp": (acc - DRAFT_ACC) * 100.0,
        })
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    labels = [r["method"] for r in rows]
    values = [r["acc"] * 100 for r in rows]
    deltas = [r["delta_vs_online_draft_pp"] for r in rows]
    best = max(values)

    colors = []
    for label, val in zip(labels, values):
        if label == "Full attention r=1":
            colors.append("#374151")
        elif label == "Online QK r=0.15":
            colors.append("#6b7280")
        elif label == "Online Draft r=0.15":
            colors.append("#9ca3af")
        elif abs(val - best) < 1e-9:
            colors.append("#2a9d8f")
        else:
            colors.append("#8ecdc1")

    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, width=0.62, color=colors, edgecolor="white", linewidth=0.7)
    ax.axhline(DRAFT_ACC * 100, color="#444444", linestyle="--", linewidth=1.2, label="Online Draft r=0.15")

    for bar, acc, delta in zip(bars, values, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.45,
            f"{acc:.1f}%\n{delta:+.1f} pp",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#374151",
        )

    ax.set_ylabel("Main accuracy (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylim(68, 79.5)
    style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()

    png = FIG_DIR / "micro_all_datasets_method_accuracy.png"
    pdf = FIG_DIR / "micro_all_datasets_method_accuracy.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(OUT_CSV)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
