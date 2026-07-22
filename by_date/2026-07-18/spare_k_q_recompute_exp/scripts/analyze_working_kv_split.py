#!/usr/bin/env python3
"""Describe contiguous validation/test split covariates for MuSiQue-v2."""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
DATA = ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/data/musique-v2.jsonl"
OUTPUT = ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/split_covariates.json"


def describe(rows):
    features = {
        "context_chars": [len(row["context"]) for row in rows],
        "question_chars": [len(row["input"]) for row in rows],
        "passages": [len(re.findall(r"Passage \d+\b", row["context"])) for row in rows],
        "answers": [len(row["answers"]) for row in rows],
        "declared_length": [row["length"] for row in rows],
    }
    return {
        "n": len(rows),
        **{
            name: {"mean": statistics.mean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}
            for name, values in features.items()
        },
    }


def main():
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    result = {
        "split_policy": "contiguous: validation rows 1-50, test rows 51-200",
        "warning": "not randomized or stratified; reported covariates do not eliminate order bias",
        "validation": describe(rows[:50]),
        "test": describe(rows[50:200]),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
