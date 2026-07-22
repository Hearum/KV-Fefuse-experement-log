#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path

EXP_ROOT = Path("/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_rate0_cross_dataset_baseline")
RESULTS = EXP_ROOT / "results"
CSV_NAME = "rate_0.0_draft_Qwen2.5-3B-Instruct_revert_rope.csv"


def read_segment(seg_dir):
    log = seg_dir / "run.log"
    finished = log.exists() and "FINAL RESULTS" in log.read_text(errors="ignore")
    csvs = list(seg_dir.glob(f"**/{CSV_NAME}"))
    rows = []
    if finished and csvs:
        with csvs[0].open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    return finished, rows


def main():
    records = []
    for ds_dir in sorted(RESULTS.iterdir() if RESULTS.exists() else []):
        if not ds_dir.is_dir():
            continue
        group = {
            "segments": 0,
            "finished_segments": 0,
            "sub_total": 0,
            "sub_correct": 0,
            "f1_sum": 0.0,
            "em_sum": 0.0,
            "main": defaultdict(list),
        }
        for seg_dir in sorted(ds_dir.glob("seg_*")):
            finished, rows = read_segment(seg_dir)
            group["segments"] += 1
            group["finished_segments"] += int(finished)
            for row in rows:
                correct = row.get("Correct", "").strip().lower() == "true"
                group["sub_total"] += 1
                group["sub_correct"] += int(correct)
                try:
                    group["f1_sum"] += float(row.get("F1") or 0)
                except ValueError:
                    pass
                try:
                    group["em_sum"] += float(row.get("EM") or 0)
                except ValueError:
                    pass
                group["main"][row.get("Main Question", "")].append(correct)
        main_total = len(group["main"])
        main_correct = sum(1 for vals in group["main"].values() if vals and all(vals))
        sub_total = group["sub_total"]
        records.append(
            {
                "dataset": ds_dir.name,
                "rate": 0,
                "finished_segments": group["finished_segments"],
                "segments": group["segments"],
                "main_correct": main_correct,
                "main_total": main_total,
                "main_acc": main_correct / main_total if main_total else 0,
                "sub_correct": group["sub_correct"],
                "sub_total": sub_total,
                "sub_acc": group["sub_correct"] / sub_total if sub_total else 0,
                "avg_f1": group["f1_sum"] / sub_total if sub_total else 0,
                "avg_em": group["em_sum"] / sub_total if sub_total else 0,
            }
        )
    out_csv = EXP_ROOT / "rate0_summary.csv"
    fields = ["dataset", "rate", "finished_segments", "segments", "main_correct", "main_total", "main_acc", "sub_correct", "sub_total", "sub_acc", "avg_f1", "avg_em"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (EXP_ROOT / "rate0_summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(out_csv)
    for row in records:
        print(row)


if __name__ == "__main__":
    main()
