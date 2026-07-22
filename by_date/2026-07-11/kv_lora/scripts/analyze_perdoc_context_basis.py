#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import torch


p = argparse.ArgumentParser()
p.add_argument("--part", required=True)
p.add_argument("--train", type=int, default=40)
p.add_argument("--device", default="cuda:0")
a = p.parse_args()
root = Path("/raid/home/hming/FusionRAG-pca-analysis")
base = root / "MOTIVATION_EXPERIMENTS/kv_lora"
data_dir = base / "results/perdoc_context_deltas" / a.part
manifest = json.loads((data_dir / "manifest.json").read_text())
ranks = [0, 1, 2, 4, 8, 16, 32]
rows = []

for source in ("raw", "preprocess"):
    paths = sorted(data_dir.glob(f"context_*_{source}.pt"))
    if len(paths) < 50:
        raise RuntimeError(f"{a.part}/{source}: expected >=50 contexts, got {len(paths)}")
    payloads = [torch.load(path, map_location="cpu") for path in paths[:50]]
    for kind in ("k", "v"):
        # [context, layer, target feature vector], retained in fp16 until each layer.
        deltas = torch.stack([x[f"delta_{kind}"][:, 0].flatten(1) for x in payloads])
        base_norm2 = torch.stack([x[f"base_{kind}_norm2_layer"] for x in payloads]).float()
        for layer in range(deltas.shape[1]):
            train = deltas[:a.train, layer].float().to(a.device)
            test = deltas[a.train:50, layer].float().to(a.device)
            mean = train.mean(0)
            centered = train - mean
            # Exact context-template PCA: at most train-1 nonzero context directions.
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            for test_offset, target in enumerate(test):
                context_index = a.train + test_offset
                target_norm2 = float(target.square().sum())
                gap = (target_norm2 / max(float(base_norm2[context_index, layer]), 1e-30)) ** 0.5
                residual = target - mean
                for rank in ranks:
                    if rank == 0:
                        prediction = mean
                    else:
                        basis = vh[:rank]
                        prediction = mean + (residual @ basis.T) @ basis
                    error = target - prediction
                    error_norm2 = float(error.square().sum())
                    prediction_norm2 = float(prediction.square().sum())
                    cosine = float(target @ prediction) / max((target_norm2 * prediction_norm2) ** 0.5, 1e-30)
                    rows.append({
                        "part": a.part, "example": manifest["example"], "target": manifest["target"],
                        "source": source, "kind": kind, "context_index": context_index,
                        "prefix_tokens": payloads[context_index]["meta"]["prefix_tokens"],
                        "layer": layer, "rank": rank, "original_gap": gap,
                        "delta_norm2": target_norm2,
                        "base_norm2": float(base_norm2[context_index, layer]),
                        "error_norm2": error_norm2,
                        "prediction_norm2": prediction_norm2,
                        "target_prediction_dot": float(target @ prediction),
                        "delta_recovery_error": (error_norm2 / max(target_norm2, 1e-30)) ** 0.5,
                        "final_kv_error": (error_norm2 / max(float(base_norm2[context_index, layer]), 1e-30)) ** 0.5,
                        "explained_delta_energy": 1.0 - error_norm2 / max(target_norm2, 1e-30),
                        "cosine": cosine,
                    })
        del deltas, base_norm2
    del payloads

out = base / "results" / f"perdoc_context_basis_{a.part}_40train10test.csv"
with out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

summary = []
metrics = ["original_gap", "delta_recovery_error", "final_kv_error", "explained_delta_energy", "cosine"]
for source in ("raw", "preprocess"):
    for kind in ("k", "v"):
        for rank in ranks:
            selected = [x for x in rows if x["source"] == source and x["kind"] == kind and x["rank"] == rank]
            # Equal context weighting after averaging the 64 layers of each held-out context.
            context_values = {m: [] for m in metrics}
            for context_index in range(a.train, 50):
                group = [x for x in selected if x["context_index"] == context_index]
                for metric in metrics:
                    context_values[metric].append(sum(x[metric] for x in group) / len(group))
            item = {"source": source, "kind": kind, "rank": rank, "n_test_contexts": 10}
            for metric, values in context_values.items():
                tensor = torch.tensor(values)
                item[f"{metric}_mean"] = float(tensor.mean())
                item[f"{metric}_std"] = float(tensor.std(unbiased=True))
                item[f"{metric}_median"] = float(tensor.median())
                item[f"{metric}_p10"] = float(torch.quantile(tensor, .1))
                item[f"{metric}_p90"] = float(torch.quantile(tensor, .9))
            summary.append(item)
(base / "results" / f"perdoc_context_basis_{a.part}_40train10test_summary.json").write_text(
    json.dumps({"manifest": manifest, "basis": "per-layer context-template PCA", "summary": summary}, indent=2) + "\n"
)
print(json.dumps(summary, indent=2))
