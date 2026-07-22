#!/usr/bin/env python3
"""Aggregate complete working-KV metrics and paired bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, round(q * (len(values) - 1))))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-n", type=int, default=50)
    args = parser.parse_args()
    rows = []
    per_method = defaultdict(dict)
    metadata = {}
    signatures = defaultdict(dict)
    for root_arg in args.root:
        root = Path(root_arg)
        for path in sorted(root.glob("**/metrics/metrics.csv")):
            records = list(csv.DictReader(path.open(encoding="utf-8")))
            rel = path.relative_to(root).parts
            method, dataset, rate = rel[:3]
            alpha = next((part for part in rel if part.startswith("alpha_")), "alpha_none")
            key = (method, alpha)
            current_metadata = (dataset, rate)
            if key in metadata and metadata[key] != current_metadata:
                raise ValueError(f"metadata mismatch for {key}: {path}")
            metadata[key] = current_metadata
            for record in records:
                question = record["Question"]
                signature = tuple(sorted(record.items()))
                if question in signatures[key] and signatures[key][question] != signature:
                    raise ValueError(f"conflicting duplicate for {key}: {question}")
                signatures[key][question] = signature
                per_method[key][question] = {
                    "em": float(record["em"]),
                    "f1": float(record["f1"]),
                    "glm": float(record["glm_correct"].lower() == "true"),
                }

    if not per_method:
        raise RuntimeError("no metrics.csv files found")
    reference_questions = None
    for key, predictions in per_method.items():
        if len(predictions) != args.expected_n:
            raise ValueError(f"incomplete {key}: {len(predictions)} != {args.expected_n}")
        questions = set(predictions)
        if reference_questions is None:
            reference_questions = questions
        elif questions != reference_questions:
            raise ValueError(f"question set mismatch for {key}")

    for (method, alpha), predictions in sorted(per_method.items()):
        dataset, rate = metadata[(method, alpha)]
        records = list(predictions.values())
        values = {
            metric: [row[metric] for row in records] for metric in ("em", "f1", "glm")
        }
        rng = random.Random(20260719)
        summary = {"method": method, "dataset": dataset, "rate": rate, "alpha": alpha, "n": len(records)}
        for metric, data in values.items():
            means = []
            for _ in range(5000):
                means.append(sum(data[rng.randrange(len(data))] for _ in data) / len(data))
            summary[metric] = sum(data) / len(data)
            summary[f"{metric}_ci_low"] = percentile(means, 0.025)
            summary[f"{metric}_ci_high"] = percentile(means, 0.975)
        rows.append(summary)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    paired_rows = []
    for (method, alpha), predictions in sorted(per_method.items()):
        baseline = per_method.get((method, "alpha_0p0"))
        if baseline is None or alpha == "alpha_0p0":
            continue
        questions = sorted(set(baseline) & set(predictions))
        if len(questions) != args.expected_n:
            continue
        rng = random.Random(20260719)
        row = {"method": method, "alpha": alpha, "baseline": "alpha_0p0", "n": len(questions)}
        for metric in ("em", "f1", "glm"):
            diffs = [predictions[q][metric] - baseline[q][metric] for q in questions]
            boot = []
            for _ in range(5000):
                boot.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs))
            row[f"{metric}_delta"] = sum(diffs) / len(diffs)
            row[f"{metric}_delta_ci_low"] = percentile(boot, 0.025)
            row[f"{metric}_delta_ci_high"] = percentile(boot, 0.975)
        discordant_up = sum(predictions[q]["glm"] > baseline[q]["glm"] for q in questions)
        discordant_down = sum(predictions[q]["glm"] < baseline[q]["glm"] for q in questions)
        discordant = discordant_up + discordant_down
        if discordant:
            tail = sum(math.comb(discordant, k) for k in range(0, min(discordant_up, discordant_down) + 1)) / (2 ** discordant)
            mcnemar_p = min(1.0, 2.0 * tail)
        else:
            mcnemar_p = 1.0
        row.update({"glm_up": discordant_up, "glm_down": discordant_down, "mcnemar_exact_p": mcnemar_p})
        paired_rows.append(row)
    if paired_rows:
        paired_output = output.with_name(output.stem + "_paired.csv")
        with paired_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=paired_rows[0].keys())
            writer.writeheader()
            writer.writerows(paired_rows)
    for row in rows:
        print(row)
    for row in paired_rows:
        print("PAIRED", row)


if __name__ == "__main__":
    main()
