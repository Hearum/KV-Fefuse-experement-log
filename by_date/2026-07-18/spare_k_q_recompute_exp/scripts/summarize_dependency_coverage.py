#!/usr/bin/env python3
"""Aggregate per-head sparse-router dependency and support records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    pairs = sum(row["dependency_pairs"] for row in records)
    covered = sum(row["dependency_covered"] for row in records)
    queries = sum(row["queries"] for row in records)
    summary = {
        "records": len(records),
        "queries": queries,
        "dependency_pairs": pairs,
        "dependency_covered": covered,
        "dependency_coverage": covered / pairs if pairs else 1.0,
        "effective_blocks_per_query": sum(row["effective_blocks"] for row in records) / queries,
        "effective_kv_tokens_per_query": sum(row["effective_kv_tokens"] for row in records) / queries,
        "causal_kv_tokens_per_query": sum(row["causal_kv_tokens"] for row in records) / queries,
        "kv_support_fraction": (
            sum(row["effective_kv_tokens"] for row in records)
            / sum(row["causal_kv_tokens"] for row in records)
        ),
        "preserved_dependency_blocks_per_query": sum(
            row["preserved_dependency_blocks"] for row in records
        ) / queries,
        "preserved_dependency_kv_tokens_per_query": sum(
            row["preserved_dependency_kv_tokens"] for row in records
        ) / queries,
        "preserved_dependency_support_fraction": (
            sum(row["preserved_dependency_kv_tokens"] for row in records)
            / sum(row["causal_kv_tokens"] for row in records)
        ),
        "attention_mass_recall_mean": sum(row["attention_mass_recall_mean"] * row["queries"] for row in records) / queries,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
