#!/usr/bin/env python3
"""Probe whether a cached low-dimensional hidden sidecar predicts DeltaH/DeltaV.

This tests the simplest hidden-sidecar hypothesis:

    z_l,t = PCA_l(offline h_l,t)
    z_l,t -> Delta h_l,t or Delta V_l,t

It deliberately does not use prefix features.  If this is weak, a deployable
hidden adaptor needs context-conditioned prefix features in addition to the
offline sidecar.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


L, T, D, H, F = 36, 8, 4096, 8, 128


def load_records(data_dir: Path):
    files = sorted(data_dir.glob("sample*.pt"))
    if not files:
        raise FileNotFoundError(f"no sample*.pt in {data_dir}")
    return [(path, torch.load(path, map_location="cpu", weights_only=False)) for path in files]


def stack_layer(records, layer: int, target: str):
    xs, ys = [], []
    for _, record in records:
        xs.append(record["own_h"][layer].float())
        if target == "h":
            ys.append(record["delta_h"][layer].float())
        elif target == "v":
            ys.append(record["delta_v"][layer].float().permute(1, 0, 2).reshape(T, H * F))
        else:
            raise KeyError(target)
    return torch.cat(xs, 0), torch.cat(ys, 0)


def pca_project(train_x, test_x, rank: int):
    mean = train_x.mean(0, keepdim=True)
    train_centered = train_x - mean
    test_centered = test_x - mean
    used_rank = min(rank, train_x.shape[0] - 1, train_x.shape[1])
    if used_rank <= 0:
        return train_centered[:, :0], test_centered[:, :0], 0
    _, _, vh = torch.linalg.svd(train_centered.double(), full_matrices=False)
    basis = vh[:used_rank].float().T.contiguous()
    return train_centered @ basis, test_centered @ basis, used_rank


def ridge_predict(train_z, train_y, test_z, alpha: float):
    z_mean = train_z.mean(0, keepdim=True)
    y_mean = train_y.mean(0, keepdim=True)
    z = train_z - z_mean
    y = train_y - y_mean
    xtx = z.T @ z
    reg = alpha * torch.eye(xtx.shape[0], dtype=xtx.dtype)
    weight = torch.linalg.solve(xtx + reg, z.T @ y)
    return (test_z - z_mean) @ weight + y_mean


def evaluate_prediction(pred, target, mean_pred):
    delta_sq = float(target.square().sum())
    err_sq = float((target - pred).square().sum())
    mean_err_sq = float((target - mean_pred).square().sum())
    dot = float((target * pred).sum())
    pred_sq = float(pred.square().sum())
    return {
        "remaining_delta": math.sqrt(err_sq / max(delta_sq, 1e-30)),
        "explained_delta_energy": 1.0 - err_sq / max(delta_sq, 1e-30),
        "mean_explained_delta_energy": 1.0 - mean_err_sq / max(delta_sq, 1e-30),
        "cosine": dot / math.sqrt(max(delta_sq * pred_sq, 1e-30)),
    }


def eval_rank(train, test, layer: int, target: str, rank: int, alpha: float):
    train_x, train_y = stack_layer(train, layer, target)
    test_x, test_y = stack_layer(test, layer, target)
    train_z, test_z, used_rank = pca_project(train_x, test_x, rank)
    pred = ridge_predict(train_z.float(), train_y.float(), test_z.float(), alpha)
    mean_pred = train_y.mean(0, keepdim=True).expand_as(test_y)
    row = evaluate_prediction(pred, test_y.float(), mean_pred.float())
    row.update(
        {
            "layer": layer,
            "target": target,
            "sidecar_rank": rank,
            "used_rank": used_rank,
            "train_rows": train_x.shape[0],
            "test_rows": test_x.shape[0],
        }
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--alpha", type=float, default=1e-2)
    args = parser.parse_args()

    records = load_records(Path(args.data_dir))
    if args.train_count <= 0 or args.train_count >= len(records):
        raise ValueError("--train-count must leave a heldout split")
    train, test = records[: args.train_count], records[args.train_count :]

    rows = []
    for rank in args.ranks:
        for layer in range(L):
            for target in ("h", "v"):
                rows.append(eval_rank(train, test, layer, target, rank, args.alpha))

    summary = {}
    for target in ("h", "v"):
        summary[target] = {}
        baseline_rows = [r for r in rows if r["target"] == target and r["sidecar_rank"] == args.ranks[0]]
        summary[target]["mean_template"] = {
            "mean_explained": sum(r["mean_explained_delta_energy"] for r in baseline_rows) / len(baseline_rows)
        }
        for rank in args.ranks:
            selected = [r for r in rows if r["target"] == target and r["sidecar_rank"] == rank]
            summary[target][f"own_h_sidecar_rank{rank}"] = {
                "mean_explained": sum(r["explained_delta_energy"] for r in selected) / len(selected),
                "mean_remaining": sum(r["remaining_delta"] for r in selected) / len(selected),
                "mean_cosine": sum(r["cosine"] for r in selected) / len(selected),
                "best_layer": max(selected, key=lambda r: r["explained_delta_energy"])["layer"],
                "best_layer_explained": max(r["explained_delta_energy"] for r in selected),
            }

    result = {
        "data_dir": args.data_dir,
        "records": len(records),
        "train_count": len(train),
        "test_count": len(test),
        "alpha": args.alpha,
        "summary": summary,
        "rows": rows,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
