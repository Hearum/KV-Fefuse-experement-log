#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/raid/home/hming/FusionRAG-pca-analysis")
ROOT = REPO / "MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset"
FIG_DIR = ROOT / "figures"
SUMMARY = ROOT / "alpha_vs_native_baseline_summary.csv"
OUT_CSV = ROOT / "alpha_with_full_qk_draft_accuracy_summary.csv"

DATASETS = ["2WikiMQA", "HotpotQA", "TriviaQA", "MuSiQue"]
METHODS = [
    "Full attention r=1",
    "Online QK r=0.15",
    "Online Draft r=0.15",
    "Uniform a=0.1",
    "Uniform a=0.25",
    "Random a=0.05",
    "Random a=0.1",
    "Random a=0.25",
]
COLORS = {
    "Full attention r=1": "#111827",
    "Online QK r=0.15": "#2563eb",
    "Online Draft r=0.15": "#dc2626",
    "Uniform a=0.1": "#8da0cb",
    "Uniform a=0.25": "#fc8d62",
    "Random a=0.05": "#66c2a5",
    "Random a=0.1": "#b3b3b3",
    "Random a=0.25": "#a6d854",
}

# Sources:
# - 2Wiki/Hotpot/Trivia full/qk/draft: cross_dataset_offline_generalization/cross_dataset_summary.csv
# - MuSiQue qk/draft: qwen3_rate015_online_offline/summary.csv
# - MuSiQue full: qwen3_hybrid70_online_baselines/summary.csv
BASELINES = {
    "2WikiMQA": {
        "Full attention r=1": (113, 200),
        "Online QK r=0.15": (107, 200),
        "Online Draft r=0.15": (101, 200),
    },
    "HotpotQA": {
        "Full attention r=1": (207, 260),
        "Online QK r=0.15": (206, 260),
        "Online Draft r=0.15": (207, 260),
    },
    "TriviaQA": {
        "Full attention r=1": (211, 270),
        "Online QK r=0.15": (212, 270),
        "Online Draft r=0.15": (214, 270),
    },
    "MuSiQue": {
        "Full attention r=1": (105, 135),
        "Online QK r=0.15": (84, 135),
        "Online Draft r=0.15": (99, 135),
    },
}
NAME_MAP = {
    "uniform alpha=0.1": "Uniform a=0.1",
    "uniform alpha=0.25": "Uniform a=0.25",
    "random alpha=0.05": "Random a=0.05",
    "random alpha=0.1": "Random a=0.1",
    "random alpha=0.25": "Random a=0.25",
}


def style_axes(ax):
    ax.grid(True, axis="y", color="#d0d7de", linewidth=0.8, alpha=0.65)
    ax.grid(True, axis="x", color="#eaeef2", linewidth=0.6, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)


def save(fig, name):
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)


def load_alpha():
    values = {}
    with SUMMARY.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            if ds not in DATASETS:
                continue
            method = NAME_MAP.get(row["method"])
            if not method:
                continue
            values[(ds, method)] = {
                "correct": int(row["correct"]),
                "total": int(row["total"]),
                "acc": float(row["acc"]),
            }
    return values


def build_rows():
    alpha = load_alpha()
    rows = []
    for ds in DATASETS:
        draft_correct, draft_total = BASELINES[ds]["Online Draft r=0.15"]
        draft_acc = draft_correct / draft_total
        for method in METHODS:
            if method in BASELINES[ds]:
                correct, total = BASELINES[ds][method]
                acc = correct / total
            else:
                rec = alpha[(ds, method)]
                correct, total, acc = rec["correct"], rec["total"], rec["acc"]
            rows.append({
                "dataset": ds,
                "method": method,
                "correct": correct,
                "total": total,
                "acc": acc,
                "draft_baseline_acc": draft_acc,
                "delta_vs_draft_pp": (acc - draft_acc) * 100.0,
            })
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_accuracy(rows):
    lookup = {(r["dataset"], r["method"]): r["acc"] * 100.0 for r in rows}
    x = np.arange(len(DATASETS))
    width = 0.09
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    for i, method in enumerate(METHODS):
        y = [lookup[(ds, method)] for ds in DATASETS]
        ax.bar(
            x + (i - (len(METHODS) - 1) / 2) * width,
            y,
            width,
            color=COLORS[method],
            label=method,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_ylabel("Main accuracy (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylim(40, 94)
    style_axes(ax)
    ax.legend(loc="upper left", ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, "alpha_accuracy_with_full_qk_draft_reference_style")


def plot_delta(rows):
    lookup = {(r["dataset"], r["method"]): r["delta_vs_draft_pp"] for r in rows}
    x = np.arange(len(DATASETS))
    width = 0.10
    methods = [m for m in METHODS if m != "Online Draft r=0.15"]
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    for i, method in enumerate(methods):
        y = [lookup[(ds, method)] for ds in DATASETS]
        ax.bar(
            x + (i - (len(methods) - 1) / 2) * width,
            y,
            width,
            color=COLORS[method],
            label=method,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axhline(0, color="#444444", linestyle="--", linewidth=1.2, label="Online Draft r=0.15")
    ax.set_ylabel("Accuracy delta vs Online Draft (pp)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylim(-13.5, 12.0)
    style_axes(ax)
    ax.legend(loc="lower left", ncol=4, frameon=False, fontsize=8.3)
    fig.tight_layout()
    save(fig, "alpha_delta_with_full_qk_draft_reference_style")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    plot_accuracy(rows)
    plot_delta(rows)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
