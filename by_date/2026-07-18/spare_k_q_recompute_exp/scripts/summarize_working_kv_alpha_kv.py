#!/usr/bin/env python3
"""Summarize K/V-alpha validation and preprocess confirmatory test results."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def metric(row: dict[str, str]) -> tuple[float, float, float]:
    return float(row["em"]), float(row["f1"]), float(row["glm_correct"].lower() == "true")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-n", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for path in Path(args.root).glob("**/metrics/metrics.csv"):
        parts = path.relative_to(args.root).parts
        method = parts[0]
        alpha = next(part for part in parts if part.startswith("alpha_k_"))
        key = f"{method}/{alpha}"
        for row in csv.DictReader(path.open(encoding="utf-8", newline="")):
            question = row["Question"]
            if question in grouped[key] and grouped[key][question] != row:
                raise ValueError(f"conflicting duplicate: {key} {question}")
            grouped[key][question] = row

    if not grouped:
        raise ValueError("no metrics")
    question_sets = {key: set(rows) for key, rows in grouped.items()}
    reference = next(iter(question_sets.values()))
    for key, questions in question_sets.items():
        if len(questions) != args.expected_n:
            raise ValueError(f"{key}: {len(questions)} != {args.expected_n}")
        if questions != reference:
            raise ValueError(f"question set mismatch: {key}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for key in sorted(grouped):
        values = [metric(grouped[key][question]) for question in sorted(reference)]
        rng = random.Random(20260719)
        record = {"method": key, "n": len(values)}
        for index, name in enumerate(("em", "f1", "glm")):
            point = sum(value[index] for value in values) / len(values)
            boot = [
                sum(values[rng.randrange(len(values))][index] for _ in values) / len(values)
                for _ in range(5000)
            ]
            record[name] = point
            record[f"{name}_ci_low"] = sorted(boot)[round(0.025 * (len(boot) - 1))]
            record[f"{name}_ci_high"] = sorted(boot)[round(0.975 * (len(boot) - 1))]
        rows_out.append(record)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows_out[0])
        writer.writeheader()
        writer.writerows(rows_out)
    for row in rows_out:
        print(row)


if __name__ == "__main__":
    main()
