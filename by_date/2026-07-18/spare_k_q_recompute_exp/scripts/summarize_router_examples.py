#!/usr/bin/env python3
"""Summarize router diagnostics at the independent example level."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

METRICS = (
    "dependency_coverage",
    "attention_mass_recall",
    "kv_support_fraction",
    "preserved_dependency_support_fraction",
    "effective_kv_tokens_per_query",
    "causal_kv_tokens_per_query",
)


def summarize(records):
    queries = sum(row["queries"] for row in records)
    pairs = sum(row["dependency_pairs"] for row in records)
    covered = sum(row["dependency_covered"] for row in records)
    effective = sum(row["effective_kv_tokens"] for row in records)
    causal = sum(row["causal_kv_tokens"] for row in records)
    preserved = sum(row["preserved_dependency_kv_tokens"] for row in records)
    return {
        "dependency_coverage": covered / pairs,
        "attention_mass_recall": sum(row["attention_mass_recall_mean"] * row["queries"] for row in records) / queries,
        "kv_support_fraction": effective / causal,
        "preserved_dependency_support_fraction": preserved / causal,
        "effective_kv_tokens_per_query": effective / queries,
        "causal_kv_tokens_per_query": causal / queries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--records-per-example", type=int, default=4096)
    args = parser.parse_args()
    examples = []
    for input_path in args.inputs:
        records = [json.loads(line) for line in Path(input_path).read_text().splitlines() if line.strip()]
        if len(records) % args.records_per_example:
            raise ValueError(f"incomplete router file {input_path}: {len(records)} records")
        for offset in range(0, len(records), args.records_per_example):
            row = {"example": len(examples) + 1, "source": str(input_path)}
            row.update(summarize(records[offset:offset + args.records_per_example]))
            examples.append(row)
    output_csv = Path(args.output_csv)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=examples[0].keys())
        writer.writeheader()
        writer.writerows(examples)
    aggregate = {"examples": len(examples)}
    for metric in METRICS:
        values = [row[metric] for row in examples]
        aggregate[metric] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    Path(args.output_json).write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
