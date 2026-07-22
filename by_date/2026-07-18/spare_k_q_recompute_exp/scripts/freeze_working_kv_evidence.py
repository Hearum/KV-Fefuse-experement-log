#!/usr/bin/env python3
"""Freeze complete per-example metrics into a small, hash-addressed evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def method_key(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    method = parts[0]
    alpha = next((part for part in parts if part.startswith("alpha_")), "alpha_none")
    return f"{method}/{alpha}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--expected-n", type=int, required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(dict)
    sources = defaultdict(list)
    fieldnames = {}
    for root_arg in args.root:
        root = Path(root_arg)
        for path in sorted(root.glob("**/metrics/metrics.csv")):
            key = method_key(path, root)
            rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
            if not rows:
                continue
            fieldnames.setdefault(key, list(rows[0]))
            if fieldnames[key] != list(rows[0]):
                raise ValueError(f"field mismatch for {key}: {path}")
            sources[key].append({
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": sha256(path),
            })
            for row in rows:
                question = row["Question"]
                if question in grouped[key] and grouped[key][question] != row:
                    raise ValueError(f"conflicting duplicate for {key}: {question}")
                grouped[key][question] = row

    if not grouped:
        raise ValueError("no metrics found")
    question_sets = {key: set(rows) for key, rows in grouped.items()}
    reference = next(iter(question_sets.values()))
    for key, questions in question_sets.items():
        if len(questions) != args.expected_n:
            raise ValueError(f"{key}: {len(questions)} != {args.expected_n}")
        if questions != reference:
            raise ValueError(f"question set mismatch: {key}")

    records = {}
    for key in sorted(grouped):
        name = key.replace("/", "__") + ".csv"
        path = output / name
        rows = [grouped[key][question] for question in sorted(grouped[key])]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames[key])
            writer.writeheader()
            writer.writerows(rows)
        records[key] = {
            "rows": len(rows),
            "file": name,
            "sha256": sha256(path),
            "source_metrics_csv": sources[key],
        }

    question_digest = hashlib.sha256("\n".join(sorted(reference)).encode()).hexdigest()
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "split": args.split,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_from_git_head": git_head,
        "expected_rows_per_method": args.expected_n,
        "question_set_sha256": question_digest,
        "methods": records,
        "provenance_limit": (
            "This post-hoc bundle cryptographically binds preserved CSVs, but no per-worker git HEAD "
            "was captured before launch; launch chronology comes from EXPERIMENT_LOG.md."
        ),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
