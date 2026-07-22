#!/usr/bin/env python3
"""Audit cached formal preprocess-to-rate1 residuals without modifying them."""

import argparse
import json
import re
from pathlib import Path

import torch


ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
DEFAULT_DATA = ROOT / "MOTIVATION_EXPERIMENTS/kv_lora/results/formal_preprocess_residual_50"
DEFAULT_CACHE = Path(
    "/raid/home/hming/fusionrag-reflect-qwen3-full-cache/"
    "Qwen3-32B/musique/preprocess_kv_cache_global_topk10_bge"
)
DEFAULT_OUT = ROOT / "MOTIVATION_EXPERIMENTS/structured_kv_adapter/results"
SHAPE_PREFIX = (64, 1, 8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit-examples", type=int, nargs="+", default=[0, 1])
    return parser.parse_args()


def example_id(path):
    match = re.match(r"ex(\d+)_sub\d+\.pt$", path.name)
    if not match:
        raise ValueError(f"unexpected residual filename: {path.name}")
    return int(match.group(1))


def split_for(example):
    if example < 30:
        return "train"
    if example < 40:
        return "validation"
    return "test"


def squared_norm(tensor):
    return float(tensor.float().square().sum())


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.data.glob("ex*_sub*.pt"))
    examples = sorted({example_id(path) for path in files})
    manifest = {
        "source": str(args.data),
        "split_policy": "main example id: train <30, validation 30-39, test >=40",
        "examples": {split_for(ex): [] for ex in examples},
        "files": [],
    }
    manifest["examples"] = {"train": [], "validation": [], "test": []}
    for ex in examples:
        manifest["examples"][split_for(ex)].append(ex)
    for path in files:
        manifest["files"].append(
            {"path": path.name, "example": example_id(path), "split": split_for(example_id(path))}
        )

    selected = [path for path in files if example_id(path) in set(args.audit_examples)]
    rows = []
    errors = []
    for path in selected:
        record = torch.load(path, map_location="cpu", weights_only=False)
        running_prefix = 0
        for rank, item in enumerate(record["items"]):
            ex, chunk, tokens = record["example"], item["chunk"], item["tokens"]
            if item["prefix_tokens"] != running_prefix:
                errors.append(
                    f"{path.name} chunk {chunk}: prefix {item['prefix_tokens']} != {running_prefix}"
                )
            row = {
                "file": path.name,
                "example": ex,
                "subquery": record["subquery"],
                "doc_rank": rank,
                "chunk": chunk,
                "prefix_tokens": item["prefix_tokens"],
                "tokens": tokens,
            }
            for kind, suffix in (("k", "key"), ("v", "value")):
                delta = item[f"delta_{kind}"]
                expected = SHAPE_PREFIX + (tokens, 128)
                if tuple(delta.shape) != expected:
                    errors.append(f"{path.name} chunk {chunk} delta_{kind}: {tuple(delta.shape)} != {expected}")
                cache_path = args.cache / f"{ex}_{chunk}_{suffix}.pt"
                if not cache_path.exists():
                    errors.append(f"missing cache: {cache_path}")
                    continue
                base = torch.load(cache_path, map_location="cpu", weights_only=True)
                if tuple(base.shape) != expected:
                    errors.append(f"{cache_path.name}: {tuple(base.shape)} != {expected}")
                if not torch.isfinite(delta).all() or not torch.isfinite(base).all():
                    errors.append(f"non-finite tensor: {path.name} chunk {chunk} {kind}")
                delta_sq = squared_norm(delta)
                base_sq = squared_norm(base)
                row[f"{kind}_delta_l2"] = delta_sq ** 0.5
                row[f"{kind}_base_l2"] = base_sq ** 0.5
                row[f"{kind}_original_gap"] = (delta_sq / base_sq) ** 0.5
            rows.append(row)
            running_prefix += tokens

    first = [row for row in rows if row["doc_rank"] == 0]
    later = [row for row in rows if row["doc_rank"] > 0]
    summary = {
        "status": "pass" if not errors else "fail",
        "residual_files": len(files),
        "available_examples": examples,
        "split_counts": {key: len(value) for key, value in manifest["examples"].items()},
        "audited_files": len(selected),
        "audited_documents": len(rows),
        "shape_contract": list(SHAPE_PREFIX + ("tokens", 128)),
        "first_doc_mean_gap": {
            kind: sum(row[f"{kind}_original_gap"] for row in first) / len(first)
            for kind in ("k", "v")
        },
        "later_doc_mean_gap": {
            kind: sum(row[f"{kind}_original_gap"] for row in later) / len(later)
            for kind in ("k", "v")
        },
        "errors": errors,
        "rows": rows,
    }
    (args.output_dir / "stage0_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "stage0_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
