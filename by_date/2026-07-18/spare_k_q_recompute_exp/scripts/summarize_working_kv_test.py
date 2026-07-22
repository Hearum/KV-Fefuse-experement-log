#!/usr/bin/env python3
"""Aggregate sharded test metrics and frozen paired comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import hashlib
from collections import defaultdict
from pathlib import Path


def percentile(values, q):
    values = sorted(values)
    return values[round(q * (len(values) - 1))]


def metric_row(row):
    return {
        "em": float(row["em"]),
        "f1": float(row["f1"]),
        "glm": float(row["glm_correct"].lower() == "true"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="Result root to merge; repeat for independently sharded controls.",
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument("--paired", required=True)
    parser.add_argument("--integrity", required=True)
    parser.add_argument("--expected-n", type=int, default=150)
    args = parser.parse_args()
    grouped = defaultdict(dict)
    row_signatures = defaultdict(dict)
    provenance = defaultdict(list)
    duplicate_rows = defaultdict(int)
    for root_arg in args.root:
        root = Path(root_arg)
        for path in root.glob("**/metrics/metrics.csv"):
            rel = path.relative_to(root).parts
            method = rel[0]
            alpha = next((part for part in rel if part.startswith("alpha_")), "alpha_none")
            key = f"{method}/{alpha}"
            provenance[key].append(str(path))
            for row in csv.DictReader(path.open(encoding="utf-8")):
                question = row["Question"]
                metrics = metric_row(row)
                signature = tuple(sorted(row.items()))
                if question in grouped[key]:
                    if row_signatures[key][question] != signature:
                        raise ValueError(f"conflicting duplicate in {key}: {question}")
                    duplicate_rows[key] += 1
                    continue
                grouped[key][question] = metrics
                row_signatures[key][question] = signature
    for key, values in grouped.items():
        if len(values) != args.expected_n:
            raise ValueError(f"incomplete {key}: {len(values)} != {args.expected_n}")
    if not grouped:
        raise ValueError("no metrics found")
    question_sets = {key: set(values) for key, values in grouped.items()}
    reference_questions = next(iter(question_sets.values()))
    for key, questions in question_sets.items():
        if questions != reference_questions:
            raise ValueError(f"question set mismatch: {key}")

    integrity = {
        "expected_unique_questions_per_method": args.expected_n,
        "roots": args.root,
        "question_set_sha256": hashlib.sha256(
            "\n".join(sorted(reference_questions)).encode()
        ).hexdigest(),
        "methods": {
            key: {
                "unique_questions": len(grouped[key]),
                "identical_duplicate_rows_ignored": duplicate_rows[key],
                "source_metrics_csv": sorted(provenance[key]),
            }
            for key in sorted(grouped)
        },
    }
    Path(args.integrity).write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summaries = []
    for key, records in sorted(grouped.items()):
        row = {"method": key, "n": len(records)}
        rng = random.Random(20260719)
        for metric in ("em", "f1", "glm"):
            values = [item[metric] for item in records.values()]
            boot = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(5000)]
            row[metric] = sum(values) / len(values)
            row[f"{metric}_ci_low"] = percentile(boot, 0.025)
            row[f"{metric}_ci_high"] = percentile(boot, 0.975)
        summaries.append(row)
    with Path(args.summary).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    comparisons = [
        ("dense_working_kv_raw/alpha_0p75", "dense_working_kv_raw/alpha_0p0"),
        ("sparse_working_kv_raw/alpha_0p75", "dense_working_kv_raw/alpha_0p0"),
        ("dense_working_kv_raw/alpha_0p75", "full_rate1/alpha_none"),
        ("sparse_working_kv_raw/alpha_0p75", "full_rate1/alpha_none"),
    ]
    paired = []
    for candidate_key, baseline_key in comparisons:
        candidate, baseline = grouped[candidate_key], grouped[baseline_key]
        questions = sorted(set(candidate) & set(baseline))
        if len(questions) != args.expected_n:
            raise ValueError(f"unmatched comparison {candidate_key} vs {baseline_key}")
        row = {"candidate": candidate_key, "baseline": baseline_key, "n": len(questions)}
        rng = random.Random(20260719)
        for metric in ("em", "f1", "glm"):
            diffs = [candidate[q][metric] - baseline[q][metric] for q in questions]
            boot = [sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(5000)]
            row[f"{metric}_delta"] = sum(diffs) / len(diffs)
            row[f"{metric}_ci_low"] = percentile(boot, 0.025)
            row[f"{metric}_ci_high"] = percentile(boot, 0.975)
        up = sum(candidate[q]["glm"] > baseline[q]["glm"] for q in questions)
        down = sum(candidate[q]["glm"] < baseline[q]["glm"] for q in questions)
        discordant = up + down
        tail = sum(math.comb(discordant, k) for k in range(min(up, down) + 1)) / (2 ** discordant) if discordant else 0.5
        row.update({"glm_up": up, "glm_down": down, "mcnemar_exact_p": min(1.0, 2.0 * tail)})
        paired.append(row)
    with Path(args.paired).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=paired[0].keys())
        writer.writeheader()
        writer.writerows(paired)
    for row in summaries:
        print(row)
    for row in paired:
        print("PAIRED", row)


if __name__ == "__main__":
    main()
