#!/usr/bin/env python3
"""Summarize Qwen3-235B parameter-scaling experiment results."""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RESULTS = ROOT / "results"
OUT_MD = ROOT / "RESULTS_SUMMARY.md"

DATASETS = {
    "musique": 200,
    "2wikimqa": 200,
    "hotpotqa": 260,
    "triviaqa": 270,
}

METHODS = [
    ("full_rate1", "rate_1.0_revert_rope.csv"),
    ("online_qk_rate015", "rate_0.15_revert_rope.csv"),
    ("online_draft_rate015", "rate_0.15_revert_rope.csv"),
    ("offline3b_mean_rate015", "rate_0.15_revert_rope.csv"),
    ("offline3b_freq_boundary2_rate015", "rate_0.15_revert_rope.csv"),
    ("offline32b_top2_rate015", "rate_0.15_revert_rope.csv"),
]

METHOD_NOTES = {
    "full_rate1": "rate=1.0，作为 full recompute/full attention 风格空白对照。",
    "online_qk_rate015": "FusionRAG online QK selector，online 按真实 query 选 15% token。",
    "online_draft_rate015": "online DraftModel selector，3B draft model 按真实 query 选 15% token。",
    "offline3b_mean_rate015": "纯 offline fixed set，3B draft smart mean-score global，15%。",
    "offline3b_freq_boundary2_rate015": "纯 offline fixed set，3B draft smart frequency + 2% boundary compensation，15%。",
    "offline32b_top2_rate015": "纯 offline fixed set，32B teacher top2-mean global，15%。",
}


def find_csv(dataset: str, method: str, end: int, csv_name: str) -> Path | None:
    bases = [RESULTS / dataset / method / f"full_0_{end}"]
    if dataset == "musique":
        if method in {"online_qk_rate015", "online_draft_rate015"}:
            bases.append(REPO / "MOTIVATION_EXPERIMENTS" / "qwen3_235b_current_rerun_20260709" / method / f"full_0_{end}")
        if method in {"full_rate1", "online_qk_rate015", "online_draft_rate015"}:
            bases.append(REPO / "MOTIVATION_EXPERIMENTS" / "qwen3_235b_three_groups_unified_prompt" / method / f"full_0_{end}")

    matches = []
    for base in bases:
        matches.extend(sorted(base.glob(f"**/{csv_name}")))
        if method == "online_draft_rate015":
            matches.extend(sorted(base.glob("**/rate_0.15*_revert_rope.csv")))
    if not matches:
        return None
    return max(matches, key=lambda p: (csv_unique_rows(p), int(run_completed_for_csv(p)), p.stat().st_mtime))


def find_log(dataset: str, method: str, end: int) -> Path | None:
    path = RESULTS / dataset / method / f"full_0_{end}" / "run.log"
    return path if path.exists() else None


def run_log_for_csv(path: Path) -> Path | None:
    for parent in path.parents:
        if parent.name.startswith("full_0_"):
            log = parent / "run.log"
            return log if log.exists() else None
    return None


def run_completed_for_csv(path: Path) -> bool:
    log = run_log_for_csv(path)
    if not log:
        return True
    try:
        text = log.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "RESULTS" in text and "Results saved to" in text


def csv_unique_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return 0
    return len({(r.get("Main Question", ""), r.get("Sub Question", "")) for r in rows})


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def summarize_csv(path: Path, min_rows: int = 0) -> dict:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # Some interrupted/retried jobs can duplicate rows. Keep the latest row for
    # the same main/sub question pair, matching the intended evaluation unit.
    dedup = {}
    for row in rows:
        key = (row.get("Main Question", ""), row.get("Sub Question", ""))
        dedup[key] = row
    rows = list(dedup.values())

    sub_total = len(rows)
    sub_correct = sum(parse_bool(r.get("Correct", "")) for r in rows)
    f1_vals = [float(r.get("F1", 0.0) or 0.0) for r in rows]
    em_vals = [float(r.get("EM", 0.0) or 0.0) for r in rows]

    by_main = {}
    for row in rows:
        by_main.setdefault(row.get("Main Question", ""), []).append(row)
    main_total = len(by_main)
    main_correct = sum(all(parse_bool(r.get("Correct", "")) for r in group) for group in by_main.values())

    return {
        "status": "done" if (run_completed_for_csv(path) or sub_total >= min_rows) else "partial/running",
        "rows": sub_total,
        "main": f"{main_correct}/{main_total} ({main_correct / main_total * 100:.2f}%)" if main_total else "-",
        "sub": f"{sub_correct}/{sub_total} ({sub_correct / sub_total * 100:.2f}%)" if sub_total else "-",
        "f1": f"{sum(f1_vals) / len(f1_vals):.4f}" if f1_vals else "-",
        "em": f"{sum(em_vals) / len(em_vals):.4f}" if em_vals else "-",
        "csv": relpath(path),
    }


def relpath(path: Path) -> str:
    for base in (ROOT, REPO):
        try:
            return str(path.relative_to(base))
        except ValueError:
            pass
    return str(path)


def progress_from_log(path: Path | None) -> str:
    if not path or not path.exists():
        return "pending"
    last = ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Traceback" in line or "FileNotFoundError" in line or "KeyError" in line or "RuntimeError" in line:
                    last = "error: " + line.strip()
                elif "Main Question" in line:
                    last = line.strip()
                elif "Loading checkpoint shards" in line:
                    last = "loading model"
                elif "Encoding " in line and "documents" in line:
                    last = line.strip()
    except OSError as exc:
        return f"log_read_error: {exc}"
    return last or "started"


def build_rows() -> list[dict]:
    rows = []
    for dataset, end in DATASETS.items():
        for method, csv_name in METHODS:
            csv_path = find_csv(dataset, method, end, csv_name)
            if csv_path:
                summary = summarize_csv(csv_path, min_rows=end)
            else:
                summary = {
                    "status": progress_from_log(find_log(dataset, method, end)),
                    "rows": "-",
                    "main": "-",
                    "sub": "-",
                    "f1": "-",
                    "em": "-",
                    "csv": "-",
                }
            rows.append({"dataset": dataset, "method": method, **summary})
    return rows


def write_md(rows: list[dict]) -> None:
    lines = []
    lines.append("# Qwen3-235B 参数量泛化实验结果汇总")
    lines.append("")
    lines.append(f"更新时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 主表")
    lines.append("")
    lines.append("| Dataset | Method | Status/Progress | Rows | Main Acc | Sub Acc | F1 | EM | CSV |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['method']} | {r['status']} | {r['rows']} | "
            f"{r['main']} | {r['sub']} | {r['f1']} | {r['em']} | `{r['csv']}` |"
        )
    lines.append("")
    lines.append("## 方法说明")
    lines.append("")
    for method, note in METHOD_NOTES.items():
        lines.append(f"- `{method}`：{note}")
    lines.append("")
    lines.append("## 统计口径")
    lines.append("")
    lines.append("- CSV 先按 `(Main Question, Sub Question)` 去重，避免中断重跑产生重复行。")
    lines.append("- `Sub Acc` 是去重后的 sub-question 级别正确率。")
    lines.append("- `Main Acc` 是同一 `Main Question` 下所有 sub-question 都正确才计为正确。")
    lines.append("- `F1/EM` 是去重后所有 sub-question 的平均值。")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_md(rows)
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
