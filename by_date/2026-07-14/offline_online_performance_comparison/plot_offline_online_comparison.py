#!/usr/bin/env python3
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path("/raid/home/hming/FusionRAG-pca-analysis")
if not REPO.exists():
    REPO = Path("/home/hming/FusionRAG-pca-analysis")

ROOT = REPO / "MOTIVATION_EXPERIMENTS/offline_online_performance_comparison"
CSV_PATH = ROOT / "offline_online_performance_summary.csv"
FIG_DIR = ROOT / "figures"

COLORS = {
    "full": "#444444",
    "online_qk": "#2563eb",
    "online_draft": "#fc8d62",
    "offline": "#66c2a5",
    "offline_residual": "#8da0cb",
}


def pct(s):
    if not s:
        return np.nan
    m = re.search(r"\(([-+0-9.]+)%\)", s)
    return float(m.group(1)) if m else np.nan


def load_rows():
    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["main_pct"] = pct(row["main_acc"])
        row["sub_pct"] = pct(row["sub_acc"])
    return rows


def style_axes(ax):
    ax.grid(True, axis="y", color="#d0d7de", linewidth=0.8, alpha=0.65)
    ax.grid(True, axis="x", color="#eaeef2", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10.5)


def color_for(row):
    method = row["method"]
    category = row["category"]
    if category == "full":
        return COLORS["full"]
    if method.startswith("online_qk"):
        return COLORS["online_qk"]
    if method.startswith("online_draft"):
        return COLORS["online_draft"]
    if category == "offline+online":
        return COLORS["offline_residual"]
    return COLORS["offline"]


def short_method(name):
    mapping = {
        "full_rate1": "Full",
        "online_qk_rate015": "Online QK",
        "online_draft_rate015": "Online Draft",
        "offline_hybrid70_rate015": "Offline hybrid",
        "draft_smart_mean_score_global": "Offline 3B mean",
        "draft_smart_freq_boundary0p02_global": "Offline + boundary",
        "draft32b_smart_top2_mean_global": "Offline 32B top2",
        "offline10_draft005": "Offline10 + Draft5",
        "offline10_hybrid_old70_docgen30_draft005": "Offline10 docgen + Draft5",
        "offline20_only": "Offline20 only",
        "offline3b_mean": "Offline 3B mean",
        "offline3b_freq_boundary2": "Offline + boundary",
        "offline32b_top2": "Offline 32B top2",
        "offline3b_mean_rate015": "Offline 3B mean",
        "offline3b_freq_boundary2_rate015": "Offline + boundary",
        "offline32b_top2_rate015": "Offline 32B top2",
    }
    return mapping.get(name, name)


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)


def plot_musique_32b(rows):
    keep = [
        "full_rate1",
        "online_qk_rate015",
        "online_draft_rate015",
        "offline_hybrid70_rate015",
        "draft_smart_mean_score_global",
        "draft_smart_freq_boundary0p02_global",
        "draft32b_smart_top2_mean_global",
        "offline10_draft005",
        "offline10_hybrid_old70_docgen30_draft005",
        "offline20_only",
    ]
    data = [r for m in keep for r in rows if r["model"] == "Qwen3-32B" and r["dataset"] == "musique" and r["method"] == m]
    labels = [short_method(r["method"]) for r in data]
    x = np.arange(len(data))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    ax.bar(x - width / 2, [r["main_pct"] for r in data], width, color=[color_for(r) for r in data], alpha=0.92, label="Main Acc")
    ax.bar(x + width / 2, [r["sub_pct"] for r in data], width, color=[color_for(r) for r in data], alpha=0.48, label="Sub Acc")
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylim(50, 90)
    style_axes(ax)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    fig.tight_layout()
    save(fig, "qwen3_32b_musique_offline_online_accuracy")


def plot_cross_dataset_32b(rows):
    methods = ["full_rate1", "online_qk_rate015", "online_draft_rate015", "offline3b_mean", "offline3b_freq_boundary2", "offline32b_top2"]
    datasets = ["2wikimqa", "hotpotqa", "triviaqa"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), sharey=True)
    width = 0.68
    for ax, ds in zip(axes, datasets):
        data = [r for m in methods for r in rows if r["model"] == "Qwen3-32B" and r["dataset"] == ds and r["method"] == m]
        x = np.arange(len(data))
        ax.bar(x, [r["sub_pct"] for r in data], width, color=[color_for(r) for r in data], edgecolor="white", linewidth=0.5)
        ax.set_title(ds, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([short_method(r["method"]) for r in data], rotation=34, ha="right")
        ax.set_ylim(45, 86)
        style_axes(ax)
    axes[0].set_ylabel("Accuracy (%)", fontsize=12)
    fig.tight_layout()
    save(fig, "qwen3_32b_cross_dataset_offline_online_accuracy")


def plot_235b(rows):
    datasets = ["musique", "2wikimqa", "hotpotqa"]
    methods = ["full_rate1", "online_qk_rate015", "online_draft_rate015", "offline3b_mean_rate015", "offline3b_freq_boundary2_rate015", "offline32b_top2_rate015"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), sharey=True)
    for ax, ds in zip(axes, datasets):
        data = [r for m in methods for r in rows if r["model"] == "Qwen3-235B-A22B" and r["dataset"] == ds and r["method"] == m]
        x = np.arange(len(data))
        ax.bar(x, [r["sub_pct"] for r in data], 0.68, color=[color_for(r) for r in data], edgecolor="white", linewidth=0.5)
        ax.set_title(ds, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([short_method(r["method"]) for r in data], rotation=34, ha="right")
        ax.set_ylim(45, 90)
        style_axes(ax)
    axes[0].set_ylabel("Accuracy (%)", fontsize=12)
    fig.tight_layout()
    save(fig, "qwen235b_cross_dataset_offline_online_accuracy")


def plot_selection_time_quality(rows):
    data = [
        r for r in rows
        if r["model"] == "Qwen3-32B" and r["dataset"] == "musique"
        and r["method"] in {
            "full_rate1",
            "online_qk_rate015",
            "online_draft_rate015",
            "offline_hybrid70_rate015",
            "draft32b_smart_top2_mean_global",
            "offline10_draft005",
            "offline20_only",
        }
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for r in data:
        st = float(r["selection_time_s"]) if r["selection_time_s"] else 0.0
        ax.scatter(st, r["sub_pct"], s=95, color=color_for(r), edgecolor="white", linewidth=0.7)
        ax.text(st + 0.006, r["sub_pct"] + 0.2, short_method(r["method"]), fontsize=8.5)
    ax.set_xlabel("Online selection time (s)", fontsize=12)
    ax.set_ylabel("Sub accuracy (%)", fontsize=12)
    ax.set_xlim(-0.02, 0.36)
    ax.set_ylim(76, 87)
    style_axes(ax)
    fig.tight_layout()
    save(fig, "qwen3_32b_musique_quality_vs_selection_time")


def write_readme():
    text = """# Offline vs Online Performance Figures

Figures generated from `offline_online_performance_summary.csv`.

- `qwen3_32b_musique_offline_online_accuracy.png/pdf`: detailed Qwen3-32B MuSiQue offline/online comparison.
- `qwen3_32b_cross_dataset_offline_online_accuracy.png/pdf`: Qwen3-32B cross-dataset comparison.
- `qwen235b_cross_dataset_offline_online_accuracy.png/pdf`: Qwen3-235B-A22B comparison.
- `qwen3_32b_musique_quality_vs_selection_time.png/pdf`: quality vs online selection time for the key MuSiQue methods.
"""
    (FIG_DIR / "README.md").write_text(text, encoding="utf-8")


def main():
    rows = load_rows()
    plot_musique_32b(rows)
    plot_cross_dataset_32b(rows)
    plot_235b(rows)
    plot_selection_time_quality(rows)
    write_readme()


if __name__ == "__main__":
    main()
