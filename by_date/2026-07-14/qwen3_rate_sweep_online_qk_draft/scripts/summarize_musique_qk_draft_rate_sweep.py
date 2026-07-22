#!/usr/bin/env python3
"""Summarize Qwen3 MuSiQue Online QK/DraftModel rate sweep CSVs."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict, OrderedDict
from pathlib import Path


EXP_ROOT = Path(os.environ.get(
    "FUSIONRAG_QWEN3_RATE_SWEEP_ROOT",
    "/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate_sweep_20260714",
))
OUT_DIR = Path("MOTIVATION_EXPERIMENTS/qwen3_rate_sweep_online_qk_draft")
RATES = ["0.0", "0.1", "0.3", "0.5", "0.8", "0.9"]
METHODS = ["online_qk", "online_draft"]


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "correct"}


def expected_csv(method: str, rate: str) -> str:
    if method == "online_qk":
        return f"rate_{rate}_revert_rope.csv"
    if method == "online_draft":
        return f"rate_{rate}_draft_Qwen2.5-3B-Instruct_revert_rope.csv"
    raise ValueError(method)


def label(method: str, rate: str) -> str:
    return f"{method}_rate{rate.replace('.', 'p')}"


def segment_finished(csv_path: Path) -> bool:
    for parent in csv_path.parents:
        log = parent / "run.log"
        if log.exists():
            return "FINAL RESULTS" in log.read_text(errors="ignore")
    return False


def read_rows(method: str, rate: str):
    rows = []
    root = EXP_ROOT / label(method, rate) / "musique"
    for path in sorted(root.rglob(expected_csv(method, rate))):
        if not segment_finished(path):
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["_csv"] = str(path)
                rows.append(row)
    return rows


def summarize_rows(rows):
    by_key = OrderedDict()
    for row in rows:
        key = ((row.get("Main Question") or "").strip(), (row.get("Sub Question") or "").strip())
        by_key[key] = row
    by_main = defaultdict(list)
    for (main, _sub), row in by_key.items():
        by_main[main].append(is_true(row.get("Correct")))
    sub_total = len(by_key)
    sub_correct = sum(is_true(row.get("Correct")) for row in by_key.values())
    main_total = len(by_main)
    main_correct = sum(all(vals) for vals in by_main.values())
    f1_vals = []
    em_vals = []
    for row in by_key.values():
        try:
            f1_vals.append(float(row.get("F1", 0) or 0))
        except ValueError:
            pass
        try:
            em_vals.append(float(row.get("EM", 0) or 0))
        except ValueError:
            pass
    return {
        "raw_rows": len(rows),
        "unique_sub_rows": sub_total,
        "main_correct": main_correct,
        "main_total": main_total,
        "main_acc": main_correct / main_total if main_total else None,
        "sub_correct": sub_correct,
        "sub_total": sub_total,
        "sub_acc": sub_correct / sub_total if sub_total else None,
        "avg_f1": sum(f1_vals) / len(f1_vals) if f1_vals else None,
        "avg_em": sum(em_vals) / len(em_vals) if em_vals else None,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for method in METHODS:
        for rate in RATES:
            rows = read_rows(method, rate)
            rec = {"method": method, "rate": rate, "csv_files": len({r["_csv"] for r in rows})}
            rec.update(summarize_rows(rows))
            records.append(rec)
    out_csv = OUT_DIR / "rate_sweep_current_summary.csv"
    fields = [
        "method", "rate", "csv_files", "raw_rows", "unique_sub_rows",
        "main_correct", "main_total", "main_acc", "sub_correct", "sub_total", "sub_acc", "avg_f1", "avg_em",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (OUT_DIR / "rate_sweep_current_summary.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    for rec in records:
        print(rec)


if __name__ == "__main__":
    main()
