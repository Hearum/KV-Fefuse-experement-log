from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
OUT = ROOT / "MOTIVATION_EXPERIMENTS" / "selection_cost_figures"
QK_CSV = (
    ROOT
    / "MOTIVATION_EXPERIMENTS"
    / "online_ttft_profile_clean_rate_sweep_v2"
    / "clean_rate_sweep_summary.csv"
)
DRAFT_CSV = (
    ROOT
    / "MOTIVATION_EXPERIMENTS"
    / "online_ttft_profile_draft_dispatch_rrf18_pptrue_sweep"
    / "rate_sweep_summary.csv"
)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.grid(True, axis="x", color="#eeeeee", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)


def plot_e2e_vs_rate(qk: pd.DataFrame, draft: pd.DataFrame) -> None:
    qk_nonfull = qk[qk["rate"].astype(float) < 1.0].copy()
    draft_nonfull = draft[draft["rate"].astype(float) < 1.0].copy()
    rates = qk_nonfull["rate"].astype(float)
    full = float(qk.loc[qk["rate"].astype(float) == 1.0, "end_to_end_ttft_s"].iloc[0])
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot(
        rates,
        qk_nonfull["end_to_end_ttft_s"],
        marker="o",
        linewidth=2.2,
        markersize=5,
        color="#1f77b4",
        label="FusionRAG-QK",
    )
    ax.plot(
        draft_nonfull["rate"].astype(float),
        draft_nonfull["end_to_end_ttft_s_mean"],
        marker="s",
        linewidth=2.2,
        markersize=5,
        color="#d62728",
        label="DraftModel",
    )
    ax.axhline(full, color="#444444", linestyle="--", linewidth=1.6, label="Full recompute")
    ax.set_xlabel("Update rate", fontsize=11)
    ax.set_ylabel("End-to-end TTFT (s)", fontsize=11)
    ax.set_xticks(rates)
    ax.set_xticklabels([f"{r:g}" for r in rates], rotation=0)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    style_axes(ax)
    save(fig, "qk_e2e_ttft_vs_rate")


def plot_qk_breakdown(qk: pd.DataFrame) -> None:
    full = float(qk.loc[qk["rate"].astype(float) == 1.0, "end_to_end_ttft_s"].iloc[0])
    qk = qk[qk["rate"].astype(float) < 1.0].copy()
    rates = qk["rate"].astype(float)
    labels = [f"{r:g}" for r in rates]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = range(len(qk))
    components = [
        ("KV load/copy", "kv_load_copy_s", "#8da0cb"),
        ("Selection", "selection_s", "#fc8d62"),
        ("Update + query", "online_update_query_prefill_s", "#66c2a5"),
        ("Other overhead", "extra_impl_overhead_s", "#b3b3b3"),
    ]
    bottom = [0.0] * len(qk)
    for label, col, color in components:
        vals = qk[col].astype(float).tolist()
        ax.bar(x, vals, bottom=bottom, width=0.72, label=label, color=color, edgecolor="white", linewidth=0.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.axhline(full, color="#444444", linestyle="--", linewidth=1.5, label="Full recompute")
    ax.set_xlabel("Update rate", fontsize=11)
    ax.set_ylabel("End-to-end TTFT breakdown (s)", fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    style_axes(ax)
    save(fig, "qk_ttft_breakdown_by_rate")


def plot_draft_breakdown(draft: pd.DataFrame, qk: pd.DataFrame) -> None:
    draft = draft[draft["rate"].astype(float) < 1.0].copy()
    rates = draft["rate"].astype(float)
    labels = [f"{r:g}" for r in rates]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = range(len(draft))
    components = [
        ("KV load/copy", "kv_load_copy_s_mean", "#8da0cb"),
        ("Draft selection", "draft_selection_s_mean", "#fc8d62"),
        ("Update + query", "online_update_query_prefill_s_mean", "#66c2a5"),
        ("Other overhead", "extra_impl_overhead_s_mean", "#b3b3b3"),
    ]
    bottom = [0.0] * len(draft)
    for label, col, color in components:
        vals = draft[col].astype(float).tolist()
        ax.bar(x, vals, bottom=bottom, width=0.72, label=label, color=color, edgecolor="white", linewidth=0.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    full = float(qk.loc[qk["rate"].astype(float) == 1.0, "end_to_end_ttft_s"].iloc[0])
    ax.axhline(full, color="#444444", linestyle="--", linewidth=1.5, label="Full recompute")
    ax.set_xlabel("Update rate", fontsize=11)
    ax.set_ylabel("End-to-end TTFT breakdown (s)", fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    style_axes(ax)
    save(fig, "draft_ttft_breakdown_by_rate")


def plot_selection_cost_compare(qk: pd.DataFrame, draft: pd.DataFrame) -> None:
    qk_nonfull = qk[qk["rate"].astype(float) < 1.0].copy()
    draft_nonfull = draft[draft["rate"].astype(float) < 1.0].copy()
    rates = qk_nonfull["rate"].astype(float)
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot(
        rates,
        qk_nonfull["selection_s"],
        marker="o",
        linewidth=2.2,
        markersize=5,
        color="#1f77b4",
        label="QK selector",
    )
    ax.plot(
        draft_nonfull["rate"].astype(float),
        draft_nonfull["draft_selection_s_mean"],
        marker="s",
        linewidth=2.2,
        markersize=5,
        color="#d62728",
        label="Draft selector",
    )
    full = float(qk.loc[qk["rate"].astype(float) == 1.0, "end_to_end_ttft_s"].iloc[0])
    ax.axhline(full, color="#444444", linestyle="--", linewidth=1.5, label="Full recompute TTFT")
    ax.set_xlabel("Update rate", fontsize=11)
    ax.set_ylabel("Selection cost (s)", fontsize=11)
    ax.set_xticks(rates)
    ax.set_xticklabels([f"{r:g}" for r in rates])
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    style_axes(ax)
    save(fig, "selection_cost_qk_vs_draft")


def main() -> None:
    qk = pd.read_csv(QK_CSV)
    draft = pd.read_csv(DRAFT_CSV)
    plot_e2e_vs_rate(qk, draft)
    plot_qk_breakdown(qk)
    plot_draft_breakdown(draft, qk)
    plot_selection_cost_compare(qk, draft)
    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Selection Cost Figures",
                "",
                "Generated from existing FusionRAG profile summaries.",
                "",
                "Sources:",
                f"- `{QK_CSV}`",
                f"- `{DRAFT_CSV}`",
                "",
                "Figures:",
                "- `qk_e2e_ttft_vs_rate.png/pdf`: FusionRAG-QK and DraftModel end-to-end TTFT vs update rate with full recompute baseline.",
                "- `qk_ttft_breakdown_by_rate.png/pdf`: FusionRAG-QK TTFT breakdown by rate.",
                "- `draft_ttft_breakdown_by_rate.png/pdf`: DraftModel TTFT breakdown by rate.",
                "- `selection_cost_qk_vs_draft.png/pdf`: online selection cost comparison between QK selector and DraftModel selector.",
                "",
                "All figures intentionally omit a title for paper/document embedding.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
