#!/usr/bin/env python3
"""Sensitivity analysis using lexical near-duplicate question clusters."""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path


def read(root: Path) -> dict[str, tuple[float, float, float]]:
    out = {}
    paths = [root] if root.is_file() else list(root.glob("**/metrics/metrics.csv"))
    for path in paths:
        for row in csv.DictReader(path.open(encoding="utf-8", newline="")):
            out[row["Question"]] = (
                float(row["em"]), float(row["f1"]), float(row["glm_correct"].lower() == "true")
            )
    return out


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def clusters(questions: list[str]) -> list[list[str]]:
    parent = list(range(len(questions)))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    sets = [tokens(q) for q in questions]
    for i in range(len(questions)):
        for j in range(i):
            overlap = len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j]))
            if overlap >= 0.5:
                union(i, j)
    grouped: dict[int, list[str]] = {}
    for i, question in enumerate(questions):
        grouped.setdefault(find(i), []).append(question)
    return list(grouped.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidate, baseline = read(Path(args.candidate)), read(Path(args.baseline))
    if not candidate or not baseline:
        raise ValueError("candidate/baseline contains no metrics rows; pass a metrics.csv or its result directory")
    questions = sorted(set(candidate) & set(baseline))
    if len(questions) != len(candidate) or len(questions) != len(baseline):
        raise ValueError("candidate/baseline question sets differ")
    groups = clusters(questions)
    diffs = {q: tuple(candidate[q][i] - baseline[q][i] for i in range(3)) for q in questions}
    rng = random.Random(20260719)
    out = {"label": args.label, "n": len(questions), "clusters": len(groups), "threshold": 0.5}
    for index, name in enumerate(("em", "f1", "glm")):
        row_mean = sum(diffs[q][index] for q in questions) / len(questions)
        boot = []
        for _ in range(5000):
            sampled = [groups[rng.randrange(len(groups))] for _ in groups]
            vals = [diffs[q][index] for group in sampled for q in group]
            boot.append(sum(vals) / len(vals))
        boot.sort()
        out[f"{name}_delta"] = row_mean
        out[f"{name}_cluster_boot_ci_low"] = boot[round(0.025 * (len(boot) - 1))]
        out[f"{name}_cluster_boot_ci_high"] = boot[round(0.975 * (len(boot) - 1))]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out)
        writer.writeheader()
        writer.writerow(out)
    print(out)


if __name__ == "__main__":
    main()
