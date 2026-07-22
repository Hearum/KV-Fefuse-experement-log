#!/usr/bin/env python3
import csv
import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
EXP = ROOT / "MOTIVATION_EXPERIMENTS/offline_draft32b_teacher_rate015"
RUN_ROOT = EXP / "accuracy_runs_control_qwen3_32b_boundary_mix"
MODEL_DIR = "Qwen3-32B/musique"
SUBDIR = "FusionRAG_global_topk10_bge"
FILENAME = "rate_0.15_revert_rope.csv"
SEGMENTS = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 150), (150, 175), (175, 200)]
METHODS = [
    "draft32b_smart_freq_boundary0p02_global",
    "draft32b_smart_mean_boundary0p02_global",
    "draft32b_smart_top2_boundary0p02_global",
    "draft32b_smart_freq_boundary0p03_global",
    "draft32b_smart_mean_boundary0p03_global",
    "draft32b_smart_top2_boundary0p03_global",
    "draft32b_smart_freq_boundary0p05_global",
    "draft32b_smart_mean_boundary0p05_global",
    "draft32b_smart_top2_boundary0p05_global",
]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def completed_paths(method):
    out = []
    for s, e in SEGMENTS:
        seg = RUN_ROOT / method / f"seg_{s}_{e}"
        log = seg / "run.log"
        csv_path = seg / MODEL_DIR / SUBDIR / FILENAME
        done = log.exists() and "FINAL RESULTS" in log.read_text(encoding="utf-8", errors="ignore")
        out.append((csv_path, done, log))
    return out


def read_rows(paths):
    out = OrderedDict()
    missing = []
    for p, done, _log in paths:
        if not done:
            missing.append(str(p))
            continue
        if not p.exists():
            missing.append(str(p))
            continue
        with p.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[(row.get("Main Question", ""), row.get("Sub Question", ""))] = row
    return list(out.values()), missing


def parse_logs(paths):
    text, existing = "", 0
    for _p, _done, log in paths:
        if log.exists():
            existing += 1
            text += "\n" + log.read_text(encoding="utf-8", errors="ignore")
    prompt = [float(x) for x in re.findall(r"^prompt eval duration:\s*([0-9.]+)s", text, re.M)]
    select = [float(x) for x in re.findall(r"^(?:select_time|draft_select_time):\s*([0-9.]+)", text, re.M)]
    return {
        "log_files": existing,
        "finished_segments": text.count("FINAL RESULTS"),
        "traceback": text.count("Traceback"),
        "killed": text.count("Killed") + text.count("Terminated"),
        "prompt_eval_mean_s": mean(prompt),
        "selection_time_mean_s": mean(select),
    }


def summarize(method):
    paths = completed_paths(method)
    rows, missing = read_rows(paths)
    by_main = defaultdict(list)
    for r in rows:
        by_main[r.get("Main Question", "")].append(r)
    main_total = len(by_main)
    main_correct = sum(1 for g in by_main.values() if g and all(str(r.get("Correct", "")).lower() == "true" for r in g))
    sub_total = len(rows)
    sub_correct = sum(1 for r in rows if str(r.get("Correct", "")).lower() == "true")
    ans = {
        "method": method,
        "main_correct": main_correct,
        "main_total": main_total,
        "main_acc": main_correct / main_total if main_total else 0.0,
        "sub_correct": sub_correct,
        "sub_total": sub_total,
        "sub_acc": sub_correct / sub_total if sub_total else 0.0,
        "avg_f1": mean([float(r.get("F1") or 0.0) for r in rows]),
        "avg_em": mean([float(r.get("EM") or 0.0) for r in rows]),
        "missing_csv": len(missing),
    }
    ans.update(parse_logs(paths))
    return ans


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


def pct(c, t):
    return f"{c}/{t} ({c/t:.2%})" if t else "0/0"


def main():
    rows = [summarize(m) for m in METHODS]
    write_csv(EXP / "accuracy_summary_control_qwen3_32b_boundary_mix.csv", rows)
    (EXP / "accuracy_summary_control_qwen3_32b_boundary_mix.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["\n## 32B Teacher Boundary-Mix Accuracy 结果\n\n"]
    lines.append("| method | segments | Main Acc | Sub Acc | F1 | EM | selection(s) | prefill(s) | missing csv | traceback/killed |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['finished_segments']}/8 | {pct(r['main_correct'], r['main_total'])} | "
            f"{pct(r['sub_correct'], r['sub_total'])} | {r['avg_f1']:.4f} | {r['avg_em']:.4f} | "
            f"{r['selection_time_mean_s']:.4f} | {r['prompt_eval_mean_s']:.4f} | {r['missing_csv']} | {r['traceback']}/{r['killed']} |\n"
        )
    text = "".join(lines)
    readme = EXP / "README.md"
    old = readme.read_text(encoding="utf-8")
    marker = "## 32B Teacher Boundary-Mix Accuracy 结果"
    if marker in old:
        old = old.split(marker, 1)[0].rstrip() + "\n"
    readme.write_text(old + text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
