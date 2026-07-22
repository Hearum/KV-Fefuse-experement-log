import glob
import matplotlib.pyplot as plt
import pandas as pd

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, kind in zip(axes, ["k", "v"]):
    for source, marker in [("raw", "o"), ("preprocess", "s")]:
        d = pd.read_csv(f"results/strict_prefix_rank_{source}_{kind}.csv")
        d = d[d["rank"] > 0].groupby("rank").explained_delta_energy.mean()
        ax.plot(d.index, d.values, marker=marker, label=source)
    ax.set_title("RoPE-aligned Key" if kind == "k" else "Value")
    ax.set_xlabel("Per-document context-template rank")
    ax.set_ylabel("Heldout explained Delta energy")
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.grid(alpha=.25)
    ax.legend()
fig.tight_layout()
fig.savefig("figures/strict_prefix_document_rank_distortion.png", dpi=180)
