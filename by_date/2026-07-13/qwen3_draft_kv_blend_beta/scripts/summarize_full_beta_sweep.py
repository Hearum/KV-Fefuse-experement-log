#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path

EXP_ROOT = Path("/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_full_sweep")
RESULTS = EXP_ROOT / "results_full"
CSV_NAME = "rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv"


def read_segment(seg_dir):
    log = seg_dir / "run.log"
    finished = log.exists() and "FINAL RESULTS" in log.read_text(errors="ignore")
    blend_calls = log.read_text(errors="ignore").count("reprocess_kv_blend:") if log.exists() else 0
    csvs = list(seg_dir.glob(f"**/{CSV_NAME}"))
    rows = []
    if csvs:
        with csvs[0].open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    return finished, blend_calls, rows


def main():
    records = []
    by_group = defaultdict(lambda: {"sub_total": 0, "sub_correct": 0, "f1_sum": 0.0, "em_sum": 0.0, "main": defaultdict(list), "segments": 0, "finished_segments": 0, "blend_calls": 0})
    for beta_dir in sorted(RESULTS.glob("beta*_kv")):
        beta = beta_dir.name.removeprefix("beta").removesuffix("_kv").replace("p", ".")
        for ds_dir in sorted(beta_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            dataset = ds_dir.name
            for seg_dir in sorted(ds_dir.glob("seg_*")):
                finished, blend_calls, rows = read_segment(seg_dir)
                group = by_group[(beta, dataset)]
                group["segments"] += 1
                group["finished_segments"] += int(finished)
                group["blend_calls"] += blend_calls
                for row in rows:
                    q = row.get("Main Question", "")
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
                    group["main"][q].append(correct)
    for (beta, dataset), g in sorted(by_group.items(), key=lambda x: (x[0][1], float(x[0][0]))):
        main_total = len(g["main"])
        main_correct = sum(1 for vals in g["main"].values() if vals and all(vals))
        sub_total = g["sub_total"]
        records.append(
            {
                "beta": beta,
                "dataset": dataset,
                "finished_segments": g["finished_segments"],
                "segments": g["segments"],
                "main_correct": main_correct,
                "main_total": main_total,
                "main_acc": main_correct / main_total if main_total else 0,
                "sub_correct": g["sub_correct"],
                "sub_total": sub_total,
                "sub_acc": g["sub_correct"] / sub_total if sub_total else 0,
                "avg_f1": g["f1_sum"] / sub_total if sub_total else 0,
                "avg_em": g["em_sum"] / sub_total if sub_total else 0,
                "blend_calls": g["blend_calls"],
            }
        )
    out_csv = EXP_ROOT / "full_beta_summary.csv"
    fields = ["beta", "dataset", "finished_segments", "segments", "main_correct", "main_total", "main_acc", "sub_correct", "sub_total", "sub_acc", "avg_f1", "avg_em", "blend_calls"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (EXP_ROOT / "full_beta_summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(out_csv)
    for row in records:
        print(row)


if __name__ == "__main__":
    main()
