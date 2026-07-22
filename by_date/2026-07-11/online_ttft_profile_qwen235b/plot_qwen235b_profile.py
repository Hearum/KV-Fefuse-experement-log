#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_qwen235b")
CSV = ROOT / "qwen235b_profile_summary.csv"
FIG_DIR = ROOT / "figures"


def load_rows():
    rows = []
    with CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "selector": row["selector"],
                "rate": float(row["rate"]),
                "e2e": float(row["end_to_end_ttft_s_mean"]),
                "kv": float(row["kv_load_copy_s_mean"]),
                "selection": float(row["selection_s_mean"]),
                "update": float(row["online_update_query_prefill_s_mean"]),
                "overhead": float(row["extra_impl_overhead_s_mean"]),
            })
    return sorted(rows, key=lambda r: (r["selector"], r["rate"]))


def by_selector(rows, selector):
    return [r for r in rows if r["selector"] == selector]


def non_full(rows):
    return [r for r in rows if r["rate"] < 1.0]


def full_e2e(rows):
    for r in rows:
        if abs(r["rate"] - 1.0) < 1e-9:
            return r["e2e"]
    return None


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


def plot_e2e(qk_rows, draft_rows):
    qk = non_full(qk_rows)
    draft = non_full(draft_rows)
    full = full_e2e(qk_rows) or full_e2e(draft_rows)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot([r["rate"] for r in qk], [r["e2e"] for r in qk],
            marker="o", linewidth=2.2, markersize=5.5, color="#2563eb",
            label="FusionRAG-QK")
    ax.plot([r["rate"] for r in draft], [r["e2e"] for r in draft],
            marker="s", linewidth=2.2, markersize=5.2, color="#dc2626",
            label="DraftModel selector")
    if full is not None:
        ax.axhline(full, color="#111827", linestyle="--", linewidth=1.7,
                   label="Full recompute")

    ax.set_xlabel("Update rate", fontsize=12)
    ax.set_ylabel("End-to-end TTFT (s)", fontsize=12)
    ax.set_xticks([r["rate"] for r in qk])
    ax.set_xticklabels([f"{r['rate']:.2g}" for r in qk])
    ax.set_ylim(bottom=0)
    style_axes(ax)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    fig.tight_layout()
    save(fig, "qwen235b_e2e_ttft_vs_rate")


def common_ylim(*row_groups):
    val = 0.0
    for rows in row_groups:
        for r in non_full(rows):
            val = max(val, r["kv"] + r["selection"] + r["update"] + r["overhead"])
        full = full_e2e(rows)
        if full is not None:
            val = max(val, full)
    return val * 1.15


def plot_breakdown(rows, name, selection_label, ylim_top):
    data = non_full(rows)
    full = full_e2e(rows)
    x = np.arange(len(data))
    labels = [f"{r['rate']:.2g}" for r in data]
    width = 0.64

    kv = np.array([r["kv"] for r in data])
    selection = np.array([r["selection"] for r in data])
    update = np.array([r["update"] for r in data])
    overhead = np.array([r["overhead"] for r in data])

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x, kv, width, color="#8da0cb", label="KV load/copy", edgecolor="white", linewidth=0.5)
    ax.bar(x, selection, width, bottom=kv, color="#fc8d62", label=selection_label, edgecolor="white", linewidth=0.5)
    ax.bar(x, update, width, bottom=kv + selection, color="#66c2a5", label="Update + query", edgecolor="white", linewidth=0.5)
    ax.bar(x, overhead, width, bottom=kv + selection + update, color="#b3b3b3", label="Other overhead", edgecolor="white", linewidth=0.5)
    if full is not None:
        ax.axhline(full, color="#444444", linestyle="--", linewidth=1.5,
                   label="Full recompute")

    ax.set_xlabel("Update rate", fontsize=12)
    ax.set_ylabel("End-to-end TTFT breakdown (s)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, ylim_top)
    style_axes(ax)
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, name)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    qk_rows = by_selector(rows, "qk")
    draft_rows = by_selector(rows, "draft")
    ylim = common_ylim(qk_rows, draft_rows)
    plot_e2e(qk_rows, draft_rows)
    plot_breakdown(qk_rows, "qwen235b_qk_ttft_breakdown_by_rate", "Selection", ylim)
    plot_breakdown(draft_rows, "qwen235b_draft_ttft_breakdown_by_rate", "Draft selection", ylim)


if __name__ == "__main__":
    main()
