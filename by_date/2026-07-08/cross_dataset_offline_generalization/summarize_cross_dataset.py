#!/usr/bin/env python3
import csv
import glob
import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
EXP = ROOT / "MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization"
MODEL_NAME = "Qwen3-32B"

DATASETS = OrderedDict([
    ("2wikimqa", 200),
    ("hotpotqa", 260),
    ("triviaqa", 270),
])

RUNS = OrderedDict([
    ("full_rate1", {
        "rate": 1.0,
        "subdir": "FusionRAG_global_topk10_bge",
        "note": "rate=1.0 full recompute baseline; no selector.",
    }),
    ("online_qk_rate015", {
        "rate": 0.15,
        "subdir": "FusionRAG_global_topk10_bge",
        "note": "online FusionRAG-QK selector, rate=0.15.",
    }),
    ("online_draft_rate015", {
        "rate": 0.15,
        "subdir": "DraftModel_global_topk10_bge",
        "note": "online DraftModel selector with Qwen2.5-3B-Instruct, rate=0.15.",
    }),
    ("offline3b_mean", {
        "rate": 0.15,
        "subdir": "FusionRAG_global_topk10_bge",
        "note": "offline fixed set from 3B teacher mean-score aggregation, rate=0.15.",
    }),
    ("offline3b_freq_boundary2", {
        "rate": 0.15,
        "subdir": "FusionRAG_global_topk10_bge",
        "note": "offline 3B frequency fixed set with 2% boundary replacement, total rate=0.15.",
    }),
    ("offline32b_top2", {
        "rate": 0.15,
        "subdir": "FusionRAG_global_topk10_bge",
        "note": "offline fixed set from 32B teacher top2-mean aggregation, rate=0.15.",
    }),
    ("offline_qk_mean", {
        "rate": 0.15,
        "subdir": "FusionRAG_global_topk10_bge",
        "note": "PENDING addendum: offline QK mean-score fixed set, rate=0.15.",
    }),
    ("offline_qk_mean_boundary2", {
        "rate": 0.15,
        "subdir": "FusionRAG_global_topk10_bge",
        "note": "PENDING addendum: offline QK mean-score fixed set with 2% boundary replacement, total rate=0.15.",
    }),
])


def segments(n, step=25):
    out = []
    for s in range(0, n, step):
        out.append((s, min(n, s + step)))
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def csv_candidates(label, dataset, start, end, subdir, rate):
    base = EXP / "results" / label / dataset / f"seg_{start}_{end}" / MODEL_NAME / dataset / subdir
    if not base.exists():
        return []
    patterns = [
        str(base / f"rate_{rate}*_revert_rope.csv"),
        str(base / f"rate_{rate}*.csv"),
        str(base / "rate_*_revert_rope.csv"),
        str(base / "rate_*.csv"),
    ]
    seen = []
    for pat in patterns:
        for p in glob.glob(pat):
            if p not in seen:
                seen.append(p)
    return [Path(p) for p in seen]


def read_rows(label, dataset, subdir, rate):
    rows = OrderedDict()
    missing = []
    found_files = []
    for start, end in segments(DATASETS[dataset]):
        candidates = csv_candidates(label, dataset, start, end, subdir, rate)
        if not candidates:
            missing.append(f"{dataset}/{label}/seg_{start}_{end}")
            continue
        p = candidates[0]
        found_files.append(str(p))
        with p.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row.get("Main Question", ""), row.get("Sub Question", ""))
                rows[key] = row
    return list(rows.values()), missing, found_files


def parse_logs(label, dataset):
    text = ""
    existing = 0
    for start, end in segments(DATASETS[dataset]):
        p = EXP / "results" / label / dataset / f"seg_{start}_{end}" / "run.log"
        if p.exists():
            existing += 1
            text += "\n" + p.read_text(encoding="utf-8", errors="ignore")
    prompt = [float(x) for x in re.findall(r"^prompt eval duration:\s*([0-9.]+)s", text, re.M)]
    storage = [float(x) for x in re.findall(r"^storage_time:\s*([0-9.]+)", text, re.M)]
    select = [float(x) for x in re.findall(r"^(?:select_time|draft_select_time):\s*([0-9.]+)", text, re.M)]
    return {
        "log_files": existing,
        "finished_segments": text.count("FINAL RESULTS"),
        "traceback": text.count("Traceback"),
        "killed": text.count("Killed") + text.count("Terminated"),
        "n_prefill_timing": len(prompt),
        "prompt_eval_mean_s": mean(prompt),
        "storage_time_mean_s": mean(storage),
        "selection_time_mean_s": mean(select),
    }


def summarize_one(dataset, label, cfg):
    rows, missing, found_files = read_rows(label, dataset, cfg["subdir"], cfg["rate"])
    by_main = defaultdict(list)
    for r in rows:
        by_main[r.get("Main Question", "")].append(r)
    main_total = len(by_main)
    main_correct = sum(
        1 for group in by_main.values()
        if group and all(str(r.get("Correct", "")).lower() == "true" for r in group)
    )
    sub_total = len(rows)
    sub_correct = sum(1 for r in rows if str(r.get("Correct", "")).lower() == "true")
    out = {
        "dataset": dataset,
        "label": label,
        "rate": cfg["rate"],
        "main_correct": main_correct,
        "main_total": main_total,
        "main_acc": main_correct / main_total if main_total else 0.0,
        "sub_correct": sub_correct,
        "sub_total": sub_total,
        "sub_acc": sub_correct / sub_total if sub_total else 0.0,
        "avg_f1": mean([float(r.get("F1") or 0.0) for r in rows]),
        "avg_em": mean([float(r.get("EM") or 0.0) for r in rows]),
        "missing_segments": len(missing),
        "found_csv_files": len(found_files),
        "note": cfg["note"],
    }
    out.update(parse_logs(label, dataset))
    return out


def write_csv(path, rows):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def render_readme(rows):
    lines = []
    lines.append("# Cross-Dataset Offline Selection Generalization\n\n")
    plan = (EXP / "EXPERIMENT_PLAN.md").read_text(encoding="utf-8")
    lines.append(plan)
    lines.append("\n\n## 当前结果汇总\n\n")
    lines.append("| dataset | method | rate | Main Acc | Sub Acc | F1 | EM | prefill(s) | selection(s) | rows | finished seg | missing seg | traceback/killed | 含义 |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['label']} | {r['rate']:.2f} | "
            f"{r['main_correct']}/{r['main_total']} ({r['main_acc']:.2%}) | "
            f"{r['sub_correct']}/{r['sub_total']} ({r['sub_acc']:.2%}) | "
            f"{r['avg_f1']:.4f} | {r['avg_em']:.4f} | "
            f"{r['prompt_eval_mean_s']:.4f} | {r['selection_time_mean_s']:.4f} | "
            f"{r['sub_total']} | {r['finished_segments']} | {r['missing_segments']} | "
            f"{r['traceback']}/{r['killed']} | {r['note']} |\n"
        )
    lines.append("\n## 结果文件\n\n")
    lines.append("- `cross_dataset_summary.csv`\n")
    lines.append("- `cross_dataset_summary.json`\n")
    return "".join(lines)


def main():
    rows = []
    for dataset in DATASETS:
        for label, cfg in RUNS.items():
            rows.append(summarize_one(dataset, label, cfg))
    write_csv(EXP / "cross_dataset_summary.csv", rows)
    (EXP / "cross_dataset_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXP / "README.md").write_text(render_readme(rows), encoding="utf-8")
    print(render_readme(rows))


if __name__ == "__main__":
    main()
