#!/usr/bin/env python3
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
EXP = ROOT / "MOTIVATION_EXPERIMENTS/offline_boundary_mix_rate015"
RUNS = EXP / "accuracy_runs"
METHODS = [
    "draft_smart_freq_boundary0p02_global",
    "draft_smart_mean_boundary0p02_global",
    "draft_smart_freq_boundary0p03_global",
    "draft_smart_mean_boundary0p03_global",
    "draft_smart_freq_boundary0p05_global",
    "draft_smart_mean_boundary0p05_global",
]


def as_bool(value):
    return str(value).strip().lower() == "true"


def as_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def summarize_method(method):
    paths = []
    complete_segments = 0
    for seg_dir in sorted((RUNS / method).glob("seg_*")):
        log_path = seg_dir / "run.log"
        csv_path = seg_dir / "Qwen3-32B/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv"
        if log_path.exists() and "FINAL RESULTS" in log_path.read_text(encoding="utf-8", errors="ignore"):
            complete_segments += 1
            if csv_path.exists():
                paths.append(csv_path)
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["_csv_path"] = str(path.relative_to(ROOT))
                rows.append(row)
    if not rows:
        return {
            "method": method,
            "segments": 0,
            "status": "missing",
            "main_correct": 0,
            "main_total": 0,
            "sub_correct": 0,
            "sub_total": 0,
            "sub_acc": 0.0,
            "main_acc": 0.0,
            "avg_f1": 0.0,
            "avg_em": 0.0,
        }

    by_main = defaultdict(list)
    sub_correct = 0
    f1_values, em_values = [], []
    for row in rows:
        by_main[row["Main Question"]].append(as_bool(row["Correct"]))
        sub_correct += int(as_bool(row["Correct"]))
        f1_values.append(as_float(row.get("F1", 0.0)))
        em_values.append(as_float(row.get("EM", 0.0)))

    main_correct = sum(1 for vals in by_main.values() if all(vals))
    main_total = len(by_main)
    sub_total = len(rows)
    return {
        "method": method,
        "segments": complete_segments,
        "status": "complete" if complete_segments == 8 else "partial",
        "main_correct": main_correct,
        "main_total": main_total,
        "main_acc": main_correct / main_total if main_total else 0.0,
        "sub_correct": sub_correct,
        "sub_total": sub_total,
        "sub_acc": sub_correct / sub_total if sub_total else 0.0,
        "avg_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "avg_em": sum(em_values) / len(em_values) if em_values else 0.0,
        "csv_count": len(paths),
    }


def write_csv(path, rows):
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(num, den):
    return f"{num}/{den} ({num / den:.2%})" if den else "0/0"


def main():
    rows = [summarize_method(method) for method in METHODS]
    write_csv(EXP / "accuracy_summary.csv", rows)
    (EXP / "accuracy_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = (EXP / "README.md").read_text(encoding="utf-8")
    marker = "\n## Accuracy 结果\n"
    lines = lines.split(marker)[0].rstrip() + marker + "\n"
    lines += "| method | status | segments | Main Acc | Sub Acc | F1 | EM |\n"
    lines += "|---|---|---:|---:|---:|---:|---:|\n"
    for row in rows:
        lines += (
            f"| {row['method']} | {row['status']} | {row['segments']}/8 | "
            f"{pct(row['main_correct'], row['main_total'])} | {pct(row['sub_correct'], row['sub_total'])} | "
            f"{row['avg_f1']:.4f} | {row['avg_em']:.4f} |\n"
        )
    lines += "\n参考纯 offline baseline：`draft_smart_frequency_global` = 94/135 Main, 203/250 Sub；`draft_smart_mean_score_global` = 95/135 Main, 204/250 Sub。\n"
    (EXP / "README.md").write_text(lines, encoding="utf-8")
    print(lines)


if __name__ == "__main__":
    main()
