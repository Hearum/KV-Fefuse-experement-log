#!/usr/bin/env python3
import csv
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path("/raid/home/hming/FusionRAG-pca-analysis")
FIG_DIR = REPO / "MOTIVATION_EXPERIMENTS/selection_cost_figures"
QK_CSV = REPO / "MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/clean_rate_sweep_summary.csv"
DRAFT_DIR = REPO / "MOTIVATION_EXPERIMENTS/online_ttft_profile_draft_select_sparse_update_rrf18_pptrue_sweep_fastkv"


def load_qk_rows():
    rows = []
    with QK_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "rate": float(row["rate"]),
                "e2e": float(row["end_to_end_ttft_s"]),
                "kv": float(row["kv_load_copy_s"]),
                "selection": float(row["selection_s"]),
                "update": float(row["online_update_query_prefill_s"]),
                "overhead": float(row["extra_impl_overhead_s"]),
            })
    return sorted(rows, key=lambda r: r["rate"])


def load_draft_rows():
    rows = []
    for path in glob.glob(str(DRAFT_DIR / "*_summary.json")):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = data["config"]
        rows.append({
            "rate": float(cfg["rate"]),
            "e2e": float(data["end_to_end_ttft_s_mean"]),
            "kv": float(data["kv_load_copy_s_mean"]),
            "selection": float(data["draft_selection_s_mean"]),
            "score": float(data["draft_score_forward_s_mean"]),
            "update": float(data["online_update_query_prefill_s_mean"]),
            "overhead": float(data["extra_impl_overhead_s_mean"]),
        })
    return sorted(rows, key=lambda r: r["rate"])


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


def breakdown_ymax(*row_groups):
    max_val = 0.0
    for rows in row_groups:
        for r in non_full(rows):
            max_val = max(max_val, r["kv"] + r["selection"] + r["update"] + r["overhead"])
        full = full_e2e(rows)
        if full is not None:
            max_val = max(max_val, full)
    return max_val * 1.18


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
    save(fig, "qk_e2e_ttft_vs_rate")


def plot_qk_breakdown(qk_rows, ylim_top=None):
    rows = non_full(qk_rows)
    full = full_e2e(qk_rows)
    rates = [r["rate"] for r in rows]
    labels = [f"{r:.2g}" for r in rates]
    x = np.arange(len(rows))
    width = 0.64

    kv = np.array([r["kv"] for r in rows])
    selection = np.array([r["selection"] for r in rows])
    update = np.array([r["update"] for r in rows])
    overhead = np.array([r["overhead"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x, kv, width, color="#8da0cb", label="KV load/copy", edgecolor="white", linewidth=0.5)
    ax.bar(x, selection, width, bottom=kv, color="#fc8d62", label="Selection", edgecolor="white", linewidth=0.5)
    ax.bar(x, update, width, bottom=kv + selection, color="#66c2a5", label="Update + query", edgecolor="white", linewidth=0.5)
    ax.bar(x, overhead, width, bottom=kv + selection + update, color="#b3b3b3", label="Other overhead", edgecolor="white", linewidth=0.5)
    if full is not None:
        ax.axhline(full, color="#444444", linestyle="--", linewidth=1.5,
                   label="Full recompute")

    ax.set_xlabel("Update rate", fontsize=12)
    ax.set_ylabel("End-to-end TTFT breakdown (s)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, ylim_top if ylim_top is not None else ax.get_ylim()[1])
    style_axes(ax)
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "qk_ttft_breakdown_by_rate")


def plot_draft_breakdown(draft_rows, ylim_top=None):
    rows = non_full(draft_rows)
    full = full_e2e(draft_rows)
    rates = [r["rate"] for r in rows]
    labels = [f"{r:.2g}" for r in rates]
    x = np.arange(len(rows))
    width = 0.64

    kv = np.array([r["kv"] for r in rows])
    selection = np.array([r["selection"] for r in rows])
    update = np.array([r["update"] for r in rows])
    overhead = np.array([r["overhead"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x, kv, width, color="#8da0cb", label="KV load/copy", edgecolor="white", linewidth=0.5)
    ax.bar(x, selection, width, bottom=kv, color="#fc8d62", label="Draft selection", edgecolor="white", linewidth=0.5)
    ax.bar(x, update, width, bottom=kv + selection, color="#66c2a5", label="Update + query", edgecolor="white", linewidth=0.5)
    ax.bar(x, overhead, width, bottom=kv + selection + update, color="#b3b3b3", label="Other overhead", edgecolor="white", linewidth=0.5)
    if full is not None:
        ax.axhline(full, color="#444444", linestyle="--", linewidth=1.5,
                   label="Full recompute")

    ax.set_xlabel("Update rate", fontsize=12)
    ax.set_ylabel("End-to-end TTFT breakdown (s)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, ylim_top if ylim_top is not None else ax.get_ylim()[1])
    style_axes(ax)
    ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "draft_ttft_breakdown_by_rate")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    qk_rows = load_qk_rows()
    draft_rows = load_draft_rows()
    common_breakdown_ylim = breakdown_ymax(qk_rows, draft_rows)
    plot_e2e(qk_rows, draft_rows)
    plot_qk_breakdown(qk_rows, ylim_top=common_breakdown_ylim)
    plot_draft_breakdown(draft_rows, ylim_top=common_breakdown_ylim)

    readme = FIG_DIR / "README.md"
    readme.write_text(
        "# Selection Cost Figures\n\n"
        "Updated with the fast-KV DraftModel-selector + FusionRAG sparse-update sweep.\n\n"
        f"- QK source: `{QK_CSV}`\n"
        f"- Draft source: `{DRAFT_DIR}`\n"
        "- `qk_e2e_ttft_vs_rate.png/pdf`: end-to-end TTFT vs update rate; rate=1 is shown only as the full recompute dashed baseline.\n"
        "- `qk_ttft_breakdown_by_rate.png/pdf`: FusionRAG-QK TTFT breakdown; shares the same y-axis range as the DraftModel breakdown.\n"
        "- `draft_ttft_breakdown_by_rate.png/pdf`: DraftModel selector TTFT breakdown; rate=1 is shown only as the full recompute dashed baseline.\n",
        encoding="utf-8",
    )
    print(readme)


if __name__ == "__main__":
    main()
