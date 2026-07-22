#!/usr/bin/env python3
"""Probe whether prefix context makes hidden sidecars useful for DeltaV.

Joins two existing artifact sets by sample id:

* hidden_vproj_audit: own_h, DeltaH, DeltaV
* wikitext_delta: own_kv, prefix_features, DeltaKV

For each layer, fit a small heldout PCA+Ridge probe:

    feature(offline token, prefix summary) -> DeltaH or DeltaV

This is not a deployable adaptor; it is a diagnostic to decide whether the
next adaptor should combine a cached hidden sidecar with online prefix context.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


L, T, D, H, F = 36, 8, 4096, 8, 128


def load_pairs(hidden_dir: Path, delta_dir: Path):
    hidden_files = sorted(hidden_dir.glob("sample*.pt"))
    if not hidden_files:
        raise FileNotFoundError(f"no sample*.pt in {hidden_dir}")
    pairs = []
    for h_path in hidden_files:
        d_path = delta_dir / h_path.name
        if not d_path.exists():
            raise FileNotFoundError(f"missing joined delta record {d_path}")
        h = torch.load(h_path, map_location="cpu", weights_only=False)
        d = torch.load(d_path, map_location="cpu", weights_only=False)
        for key in ("sample", "token_start", "prefix_tokens", "target_tokens"):
            if h.get(key) != d.get(key):
                raise ValueError(f"{h_path.name}: mismatch {key}: {h.get(key)} vs {d.get(key)}")
        if not torch.equal(h["sampled_positions"], d["sampled_positions"]):
            raise ValueError(f"{h_path.name}: sampled_positions mismatch")
        pairs.append((h_path, h, d))
    return pairs


def stack_layer(pairs, layer: int, feature_name: str, target: str):
    xs, ys = [], []
    for _, hrec, drec in pairs:
        own_h = hrec["own_h"][layer].float()  # [T,D]
        own_kv = drec["own_kv"][layer].float().permute(1, 0, 2).reshape(T, H * 2 * F)
        prefix = drec["prefix_features"][layer].float().reshape(1, H * 4 * F).expand(T, -1)
        if feature_name == "own_h":
            x = own_h
        elif feature_name == "prefix":
            x = prefix
        elif feature_name == "own_h_prefix":
            x = torch.cat([own_h, prefix], -1)
        elif feature_name == "own_kv_prefix":
            x = torch.cat([own_kv, prefix], -1)
        elif feature_name == "own_h_own_kv_prefix":
            x = torch.cat([own_h, own_kv, prefix], -1)
        else:
            raise KeyError(feature_name)
        if target == "h":
            y = hrec["delta_h"][layer].float()
        elif target == "v":
            y = hrec["delta_v"][layer].float().permute(1, 0, 2).reshape(T, H * F)
        else:
            raise KeyError(target)
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, 0), torch.cat(ys, 0)


def pca_basis(train_x: torch.Tensor, max_rank: int):
    mean = train_x.mean(0, keepdim=True)
    centered = train_x - mean
    used = min(max_rank, train_x.shape[0] - 1, train_x.shape[1])
    if used <= 0:
        return mean, torch.empty(train_x.shape[1], 0)
    _, _, vh = torch.linalg.svd(centered.double(), full_matrices=False)
    return mean, vh[:used].float().T.contiguous()


def ridge_predict(train_z, train_y, test_z, alpha: float):
    z_mean = train_z.mean(0, keepdim=True)
    y_mean = train_y.mean(0, keepdim=True)
    z = train_z - z_mean
    y = train_y - y_mean
    xtx = z.T @ z
    weight = torch.linalg.solve(xtx + alpha * torch.eye(xtx.shape[0], dtype=xtx.dtype), z.T @ y)
    return (test_z - z_mean) @ weight + y_mean


def metrics(pred, target, mean_pred):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dir", required=True)
    parser.add_argument("--delta-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--alpha", type=float, default=1e-2)
    args = parser.parse_args()

    pairs = load_pairs(Path(args.hidden_dir), Path(args.delta_dir))
    if args.train_count <= 0 or args.train_count >= len(pairs):
        raise ValueError("--train-count must leave a heldout split")
    train, test = pairs[: args.train_count], pairs[args.train_count :]
    features = ["own_h", "prefix", "own_h_prefix", "own_kv_prefix", "own_h_own_kv_prefix"]
    max_rank = max(args.ranks)

    rows = []
    for layer in range(L):
        for feature in features:
            train_x, _ = stack_layer(train, layer, feature, "h")
            test_x, _ = stack_layer(test, layer, feature, "h")
            x_mean, basis = pca_basis(train_x, max_rank)
            train_centered = train_x - x_mean
            test_centered = test_x - x_mean
            for target in ("h", "v"):
                _, train_y = stack_layer(train, layer, feature, target)
                _, test_y = stack_layer(test, layer, feature, target)
                mean_pred = train_y.mean(0, keepdim=True).expand_as(test_y)
                for rank in args.ranks:
                    used = min(rank, basis.shape[1])
                    train_z = train_centered @ basis[:, :used]
                    test_z = test_centered @ basis[:, :used]
                    pred = ridge_predict(train_z.float(), train_y.float(), test_z.float(), args.alpha)
                    row = metrics(pred, test_y.float(), mean_pred.float())
                    row.update(
                        {
                            "layer": layer,
                            "target": target,
                            "feature": feature,
                            "rank": rank,
                            "used_rank": used,
                            "input_dim": train_x.shape[1],
                            "train_rows": train_x.shape[0],
                            "test_rows": test_x.shape[0],
                        }
                    )
                    rows.append(row)

    summary = {}
    for target in ("h", "v"):
        summary[target] = {}
        base = [r for r in rows if r["target"] == target and r["rank"] == args.ranks[0]]
        base_no0 = [r for r in base if r["layer"] > 0]
        summary[target]["mean_template_all_layers"] = sum(r["mean_explained_delta_energy"] for r in base) / len(base)
        summary[target]["mean_template_no_layer0"] = sum(r["mean_explained_delta_energy"] for r in base_no0) / len(base_no0)
        for feature in features:
            summary[target][feature] = {}
            for rank in args.ranks:
                selected = [r for r in rows if r["target"] == target and r["feature"] == feature and r["rank"] == rank and r["layer"] > 0]
                summary[target][feature][str(rank)] = {
                    "mean_explained_no_layer0": sum(r["explained_delta_energy"] for r in selected) / len(selected),
                    "mean_remaining_no_layer0": sum(r["remaining_delta"] for r in selected) / len(selected),
                    "mean_cosine_no_layer0": sum(r["cosine"] for r in selected) / len(selected),
                    "best_layer": max(selected, key=lambda r: r["explained_delta_energy"])["layer"],
                    "best_layer_explained": max(r["explained_delta_energy"] for r in selected),
                }

    result = {
        "hidden_dir": args.hidden_dir,
        "delta_dir": args.delta_dir,
        "records": len(pairs),
        "train_count": len(train),
        "test_count": len(test),
        "ranks": args.ranks,
        "alpha": args.alpha,
        "summary": summary,
        "rows": rows,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
