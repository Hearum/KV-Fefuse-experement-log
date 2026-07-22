import glob
import json

import matplotlib.pyplot as plt
import pandas as pd


files = glob.glob("results/predict_perdoc_coefficients_t*_*.csv")
df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
df.to_csv("results/predict_perdoc_coefficients_5targets_50heldout.csv", index=False)

metrics = [
    "original_gap",
    "delta_recovery_error",
    "final_kv_error",
    "explained_delta_energy",
    "cosine",
]
summary = (
    df.groupby(["source", "method"])[metrics]
    .agg(["mean", "std", "median"])
    .reset_index()
)
summary.columns = ["_".join(x).rstrip("_") for x in summary.columns]
summary.to_csv("results/predict_perdoc_coefficients_5targets_50heldout_summary.csv", index=False)
with open("results/predict_perdoc_coefficients_5targets_50heldout_summary.json", "w") as f:
    json.dump(summary.to_dict(orient="records"), f, indent=2)

order = ["mean", "position_r4", "position_r8", "cached_prefix_r4", "cached_prefix_r8", "oracle_r4", "oracle_r8"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, source in zip(axes, ["raw", "preprocess"]):
    part = df[df.source == source].groupby("method").agg(
        mean=("final_kv_error", "mean"), std=("final_kv_error", "std")
    ).reindex(order)
    ax.bar(range(len(order)), part["mean"], yerr=part["std"], capsize=3)
    ax.set_xticks(range(len(order)), order, rotation=35, ha="right")
    ax.set_title(f"{source}: Value KV final relative L2")
    ax.set_ylabel("||V_pred - V_full|| / ||V_full||")
    ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("figures/predict_perdoc_coefficients_value.png", dpi=180)

print(summary.to_string(index=False))
