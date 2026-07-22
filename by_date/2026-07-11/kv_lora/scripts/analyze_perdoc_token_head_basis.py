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
root = Path("/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora")
data_dir = root / "results/perdoc_context_deltas" / a.part
manifest = json.loads((data_dir / "manifest.json").read_text())
ranks = [0, 1, 2, 4, 8, 16, 32, 64]
rows = []

for source in ("raw", "preprocess"):
    paths = sorted(data_dir.glob(f"context_*_{source}.pt"))[:50]
    payloads = [torch.load(path, map_location="cpu") for path in paths]
    for kind in ("k", "v"):
        # [context, layer, head, token, dim]
        deltas = torch.stack([x[f"delta_{kind}"][:, 0] for x in payloads])
        bases = torch.stack([x[f"base_{kind}_norm2_layer"] for x in payloads]).float()
        accum = {(context, rank): {"delta": 0., "base": 0., "error": 0., "pred": 0., "dot": 0.}
                 for context in range(a.train, 50) for rank in ranks}
        for layer in range(deltas.shape[1]):
            for head in range(deltas.shape[2]):
                train = deltas[:a.train, layer, head].float().to(a.device)
                test = deltas[a.train:50, layer, head].float().to(a.device)
                mean = train.mean(0)  # per-token document template
                centered = (train - mean).flatten(0, 1)  # context*token x 128
                _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                for offset, target in enumerate(test):
                    context = a.train + offset
                    residual = target - mean
                    target_norm2 = float(target.square().sum())
                    for rank in ranks:
                        prediction = mean if rank == 0 else mean + (residual @ vh[:rank].T) @ vh[:rank]
                        error = target - prediction
                        cell = accum[(context, rank)]
                        cell["delta"] += target_norm2
                        cell["error"] += float(error.square().sum())
                        cell["pred"] += float(prediction.square().sum())
                        cell["dot"] += float((target * prediction).sum())
            for context in range(a.train, 50):
                # Layer base norm is shared across heads and must be added once.
                for rank in ranks:
                    accum[(context, rank)]["base"] += float(bases[context, layer])
        for (context, rank), cell in accum.items():
            rows.append({
                "part": a.part, "target": manifest["target"], "source": source, "kind": kind,
                "context_index": context, "rank": rank, "delta_norm2": cell["delta"],
                "base_norm2": cell["base"], "error_norm2": cell["error"],
                "prediction_norm2": cell["pred"], "target_prediction_dot": cell["dot"],
                "original_gap": (cell["delta"] / cell["base"]) ** .5,
                "delta_recovery_error": (cell["error"] / cell["delta"]) ** .5,
                "final_kv_error": (cell["error"] / cell["base"]) ** .5,
                "explained_delta_energy": 1 - cell["error"] / cell["delta"],
                "cosine": cell["dot"] / max((cell["delta"] * cell["pred"]) ** .5, 1e-30),
            })
        del deltas, bases
    del payloads

out = root / "results" / f"perdoc_token_head_basis_{a.part}_40train10test.csv"
with out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
print(json.dumps({"part": a.part, "rows": len(rows), "output": str(out)}, indent=2))
