#!/usr/bin/env python3
"""Compare matched Dense and Sparse sampled Delta-KV traces."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", required=True)
    parser.add_argument("--sparse", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--load-kv", required=True, choices=["raw", "preprocess"])
    args = parser.parse_args()
    dense = torch.load(args.dense, map_location="cpu", weights_only=True)["records"]
    sparse = torch.load(args.sparse, map_location="cpu", weights_only=True)["records"]
    if len(dense) != len(sparse):
        raise ValueError(f"trace length mismatch: dense={len(dense)} sparse={len(sparse)}")

    rows = []
    layer_occurrence = defaultdict(int)
    for index, (d, s) in enumerate(zip(dense, sparse)):
        if d["layer"] != s["layer"] or d["tokens"] != s["tokens"]:
            raise ValueError(f"unmatched trace record {index}: {d['layer'], d['tokens']} vs {s['layer'], s['tokens']}")
        example = layer_occurrence[d["layer"]]
        layer_occurrence[d["layer"]] += 1
        for kind in ("k", "v"):
            target = d[f"delta_{kind}"].float().reshape(-1)
            candidate = s[f"delta_{kind}"].float().reshape(-1)
            target_norm = torch.linalg.vector_norm(target)
            candidate_norm = torch.linalg.vector_norm(candidate)
            dot = torch.dot(target, candidate)
            cosine = (
                float(dot / (target_norm * candidate_norm))
                if target_norm > 1e-8 and candidate_norm > 1e-8 else float("nan")
            )
            relative_l2 = torch.linalg.vector_norm(candidate - target) / (target_norm + 1e-12)
            oracle = torch.clamp(dot / (candidate_norm.square() + 1e-12), 0.0, 1.0)
            rows.append({
                "load_kv": args.load_kv,
                "record": index,
                "example": example,
                "layer": d["layer"],
                "kind": kind.upper(),
                "selected_tokens": d["tokens"],
                "sampled_elements": target.numel(),
                "cosine": cosine,
                "relative_l2": float(relative_l2),
                "oracle_alpha": float(oracle),
                "dense_delta_norm": float(target_norm),
                "sparse_delta_norm": float(candidate_norm),
                "dot": float(dot),
                "dense_sq": float(target_norm.square()),
                "sparse_sq": float(candidate_norm.square()),
                "error_sq": float(torch.sum((candidate - target).square())),
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["kind"], row["layer"])].append(row)
        grouped[(row["kind"], "ALL")].append(row)
    summaries = []
    for (kind, layer), values in sorted(grouped.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        dot = sum(row["dot"] for row in values)
        dense_sq = sum(row["dense_sq"] for row in values)
        sparse_sq = sum(row["sparse_sq"] for row in values)
        error_sq = sum(row["error_sq"] for row in values)
        oracle_alpha = min(1.0, max(0.0, dot / sparse_sq)) if sparse_sq else float("nan")
        oracle_error_sq = (
            dense_sq - 2.0 * oracle_alpha * dot + oracle_alpha * oracle_alpha * sparse_sq
            if sparse_sq else float("nan")
        )
        summaries.append({
            "load_kv": args.load_kv,
            "kind": kind,
            "layer": layer,
            "records": len(values),
            "cosine": dot / math.sqrt(dense_sq * sparse_sq) if dense_sq and sparse_sq else float("nan"),
            "relative_l2": math.sqrt(error_sq / dense_sq) if dense_sq else float("nan"),
            "oracle_alpha": oracle_alpha,
            "oracle_relative_l2": math.sqrt(max(0.0, oracle_error_sq) / dense_sq) if dense_sq else float("nan"),
            "sparse_to_dense_norm": math.sqrt(sparse_sq / dense_sq) if dense_sq else float("nan"),
        })
    summary_output = output.with_name(output.stem + "_summary.csv")
    with summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        if row["layer"] == "ALL":
            print(row)


if __name__ == "__main__":
    main()
