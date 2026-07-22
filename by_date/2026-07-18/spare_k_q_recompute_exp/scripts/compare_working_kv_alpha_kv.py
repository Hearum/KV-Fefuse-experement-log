#!/usr/bin/env python3
"""Paired comparisons for one-axis K/V alpha ablations."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path


def load(root: Path, method: str, alpha_dir: str) -> dict[str, tuple[float, float, float]]:
    files = list((root / method / "musique-v2" / "rate_0p15" / alpha_dir).glob("seg_*/metrics/metrics.csv"))
    out = {}
    for path in files:
        for row in csv.DictReader(path.open(encoding="utf-8", newline="")):
            out[row["Question"]] = (
                float(row["em"]), float(row["f1"]), float(row["glm_correct"].lower() == "true")
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--new-method", action="append", nargs=3, metavar=("METHOD", "ALPHA", "LABEL"), required=True)
    parser.add_argument("--baseline-method", action="append", nargs=3, metavar=("METHOD", "ALPHA", "LABEL"), required=True)
    args = parser.parse_args()
    if len(args.new_method) != len(args.baseline_method):
        raise ValueError("new/baseline pair count mismatch")
    rows = []
    for new_spec, base_spec in zip(args.new_method, args.baseline_method):
        new = load(Path(args.new_root), new_spec[0], new_spec[1])
        base = load(Path(args.baseline_root), base_spec[0], base_spec[1])
        questions = sorted(set(new) & set(base))
        if len(questions) != len(new) or len(questions) != len(base):
            raise ValueError(f"question mismatch {new_spec[2]} vs {base_spec[2]}")
        diffs = [[new[q][i] - base[q][i] for q in questions] for i in range(3)]
        rng = random.Random(20260719)
        row = {"candidate": new_spec[2], "baseline": base_spec[2], "n": len(questions)}
        for i, name in enumerate(("em", "f1", "glm")):
            boot = [sum(diffs[i][rng.randrange(len(questions))] for _ in questions) / len(questions) for _ in range(5000)]
            boot.sort()
            row[f"{name}_delta"] = sum(diffs[i]) / len(questions)
            row[f"{name}_ci_low"] = boot[round(0.025 * (len(boot) - 1))]
            row[f"{name}_ci_high"] = boot[round(0.975 * (len(boot) - 1))]
        row["glm_up"] = sum(new[q][2] > base[q][2] for q in questions)
        row["glm_down"] = sum(new[q][2] < base[q][2] for q in questions)
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
