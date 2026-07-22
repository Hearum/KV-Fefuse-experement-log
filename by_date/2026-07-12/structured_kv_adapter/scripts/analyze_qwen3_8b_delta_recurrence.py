#!/usr/bin/env python3
"""Analyze whether layer-l Delta-K/V is predictable from previous-layer deltas.

This is an intentionally small diagnostic.  It uses the existing
layer_update_audit artifacts and fits heldout linear/ridge probes per layer.
The goal is to test the causal modeling hypothesis:

    Delta state^{l-1} -> Delta K/V^l

rather than treating each layer/head Delta-KV as an unrelated regression target.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


L, H, T, F, D = 36, 8, 8, 128, 4096


def load_records(data_dir: Path):
    files = sorted(data_dir.glob("sample*.pt"))
    if not files:
        raise FileNotFoundError(f"no sample*.pt files in {data_dir}")
    return [(path, torch.load(path, map_location="cpu", weights_only=False)) for path in files]


def flatten_layer(record, key: str, layer: int) -> torch.Tensor:
    x = record[key].float()
    if key in ("delta_k", "delta_v"):
        return x[layer].permute(1, 0, 2).reshape(T, H * F)
    if key in ("delta_h", "delta_attn"):
        return x[layer].reshape(T, D)
    raise KeyError(key)


def stack_xy(records, layer: int, feature_name: str, target_key: str):
    xs, ys = [], []
    for _, record in records:
        if feature_name == "prev_dv":
            x = flatten_layer(record, "delta_v", layer - 1)
        elif feature_name == "prev_dk_dv":
            x = torch.cat(
                [
                    flatten_layer(record, "delta_k", layer - 1),
                    flatten_layer(record, "delta_v", layer - 1),
                ],
                dim=-1,
            )
        elif feature_name == "prev_attn":
            x = flatten_layer(record, "delta_attn", layer - 1)
        elif feature_name == "prev_h":
            x = flatten_layer(record, "delta_h", layer - 1)
        elif feature_name == "prev_h_attn_dv":
            x = torch.cat(
                [
                    flatten_layer(record, "delta_h", layer - 1),
                    flatten_layer(record, "delta_attn", layer - 1),
                    flatten_layer(record, "delta_v", layer - 1),
                ],
                dim=-1,
            )
        elif feature_name == "oracle_current_h":
            x = flatten_layer(record, "delta_h", layer)
        else:
            raise KeyError(feature_name)
        y = flatten_layer(record, target_key, layer)
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, 0), torch.cat(ys, 0)


def pca_project(train_x: torch.Tensor, test_x: torch.Tensor, rank: int):
    mean = train_x.mean(0, keepdim=True)
    train_centered = train_x - mean
    test_centered = test_x - mean
    max_rank = min(rank, train_x.shape[0] - 1, train_x.shape[1])
    if max_rank <= 0:
        return train_centered[:, :0], test_centered[:, :0], 0
    _, _, vh = torch.linalg.svd(train_centered.double(), full_matrices=False)
    basis = vh[:max_rank].float().T.contiguous()
    return train_centered @ basis, test_centered @ basis, max_rank


def ridge_fit_predict(train_x, train_y, test_x, alpha: float):
    x_mean = train_x.mean(0, keepdim=True)
    y_mean = train_y.mean(0, keepdim=True)
    x = train_x - x_mean
    y = train_y - y_mean
    xtx = x.T @ x
    reg = alpha * torch.eye(xtx.shape[0], dtype=xtx.dtype)
    weight = torch.linalg.solve(xtx + reg, x.T @ y)
    return (test_x - x_mean) @ weight + y_mean


def metrics(pred: torch.Tensor, target: torch.Tensor, mean_pred: torch.Tensor):
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


def evaluate_feature(train, test, layer: int, feature_name: str, target_key: str, rank: int, alpha: float):
    train_x, train_y = stack_xy(train, layer, feature_name, target_key)
    test_x, test_y = stack_xy(test, layer, feature_name, target_key)
    train_z, test_z, used_rank = pca_project(train_x, test_x, rank)
    pred = ridge_fit_predict(train_z.float(), train_y.float(), test_z.float(), alpha)
    mean_pred = train_y.mean(0, keepdim=True).expand_as(test_y)
    row = metrics(pred, test_y.float(), mean_pred.float())
    row.update(
        {
            "layer": layer,
            "target": target_key.replace("delta_", ""),
            "feature": feature_name,
            "input_dim": train_x.shape[1],
            "pca_rank": used_rank,
            "train_rows": train_x.shape[0],
            "test_rows": test_x.shape[0],
        }
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--train-count", type=int, default=48)
    parser.add_argument("--feature-rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1e-2)
    args = parser.parse_args()

    records = load_records(Path(args.data_dir))
    if args.train_count <= 0 or args.train_count >= len(records):
        raise ValueError("--train-count must leave a heldout test split")
    train, test = records[: args.train_count], records[args.train_count :]

    rows = []
    features = [
        "prev_dv",
        "prev_dk_dv",
        "prev_attn",
        "prev_h",
        "prev_h_attn_dv",
        "oracle_current_h",
    ]
    for layer in range(1, L):
        for target_key in ("delta_k", "delta_v"):
            for feature in features:
                rows.append(evaluate_feature(train, test, layer, feature, target_key, args.feature_rank, args.alpha))

    summary = {}
    for target in ("k", "v"):
        summary[target] = {}
        for feature in features:
            selected = [r for r in rows if r["target"] == target and r["feature"] == feature]
            summary[target][feature] = {
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
        "feature_rank": args.feature_rank,
        "alpha": args.alpha,
        "hypothesis": "predict layer-l Delta-K/V from previous-layer DeltaV, DeltaKV, DeltaAttnOut, DeltaH, or oracle current DeltaH",
        "summary": summary,
        "rows": rows,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
