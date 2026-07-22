#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT_CANDIDATES = [Path("/raid/home/hming/FusionRAG-pca-analysis"), Path("/home/hming/FusionRAG-pca-analysis")]
ROOT = next((p for p in ROOT_CANDIDATES if p.exists()), ROOT_CANDIDATES[0])
EXP = ROOT / "MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset"
OLD = ROOT / "MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization"
DATASETS = ["2wikimqa-v2", "hotpotqa-v2", "triviaqa-v2", "musique-v2"]
OLD_NAME = {"2wikimqa-v2": "2wikimqa", "hotpotqa-v2": "hotpotqa", "triviaqa-v2": "triviaqa"}


def load_rows(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) == 1 and isinstance(rows[0], list):
        rows = rows[0]
    return rows


def setup_doc_count(row: dict, dataset: str) -> int:
    if dataset in {"hotpotqa-v2", "triviaqa-v2"}:
        return len(row["output"][0]["document"])
    return len(list(re.finditer(r"(?m)^Passage \d+\n", row["context"])))


def old_fixed_key_count(npz_path: Path) -> int | None:
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    chunks = set()
    for key in data.keys():
        match = re.match(r"chunk(\d+)_", key)
        if match:
            chunks.add(int(match.group(1)))
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit old fixed-set compatibility with setup-v2 document chunks.")
    parser.add_argument("--out-csv", default=str(EXP / "setup_v2_fixed_set_compat.csv"))
    args = parser.parse_args()
    rows_out = []
    for dataset in DATASETS:
        rows = load_rows(EXP / "data" / f"{dataset}.jsonl")
        counts = [setup_doc_count(row, dataset) for row in rows]
        old_counts = []
        old_dataset = OLD_NAME.get(dataset)
        if old_dataset:
            fixed_dir = OLD / f"fixed_sets_{old_dataset}_3b" / "chunk_fixed_sets_npz"
            for i in range(len(rows)):
                old_counts.append(old_fixed_key_count(fixed_dir / f"example{i:03d}_rate0p15_chunk_local_sets.npz"))
        compatible = bool(old_counts) and all(o == c for o, c in zip(old_counts, counts))
        non_none = [x for x in old_counts if x is not None]
        rows_out.append({
            "dataset": dataset,
            "examples": len(rows),
            "setup_doc_count_min": min(counts),
            "setup_doc_count_max": max(counts),
            "setup_doc_count_hist": json.dumps(dict(sorted(Counter(counts).items())), sort_keys=True),
            "old_fixed_available": bool(old_counts),
            "old_fixed_doc_count_min": min(non_none) if non_none else "",
            "old_fixed_doc_count_max": max(non_none) if non_none else "",
            "old_fixed_compatible": compatible,
            "action": "reuse old fixed-set" if compatible else "rebuild setup-v2 fixed-set",
        })
    out = Path(args.out_csv)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(out)
    for row in rows_out:
        print(row)


if __name__ == "__main__":
    main()
