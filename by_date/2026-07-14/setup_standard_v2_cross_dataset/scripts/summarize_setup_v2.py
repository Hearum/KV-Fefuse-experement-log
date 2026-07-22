#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT_CANDIDATES = [Path("/raid/home/hming/FusionRAG-pca-analysis"), Path("/home/hming/FusionRAG-pca-analysis")]
ROOT = next((path for path in ROOT_CANDIDATES if path.exists()), ROOT_CANDIDATES[0])
EXP = ROOT / "MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset"
MODEL = "/mnt/qjhs-sh-lab-01/models/Qwen3-32B"
EXPECTED_ROWS = {"musique-v2": 200, "2wikimqa-v2": 200, "hotpotqa-v2": 260, "triviaqa-v2": 270}
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ktransformers"))

from ktransformers.util.utils import _exact_match_score, compute_f1
from transformers import AutoTokenizer


def iter_csvs():
    for p in sorted((EXP / "results").glob("*/*/rate_*/*/csv/reprocess_method_*.csv")):
        parts = p.relative_to(EXP / "results").parts
        yield p, parts[0], parts[1], parts[2], parts[3]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize setup-v2 EM/F1 and run GLM clean rejudge for complete groups.")
    parser.add_argument("--skip-glm", action="store_true", help="Only write EM/F1 summary; do not run GLM rejudge.")
    parser.add_argument("--glm-output-dir", default=str(EXP / "rejudge_glm_clean_auto"), help="Output directory for automatic GLM rejudge.")
    parser.add_argument("--glm-workers", default=os.environ.get("GLM_REJUDGE_WORKERS", "10"))
    parser.add_argument("--glm-datasets", default=os.environ.get("SETUP_V2_REJUDGE_DATASETS", ""), help="Comma-separated dataset filter for automatic GLM rejudge.")
    parser.add_argument("--glm-methods", default=os.environ.get("SETUP_V2_REJUDGE_METHODS", ""), help="Comma-separated method filter for automatic GLM rejudge.")
    parser.add_argument("--glm-rates", default=os.environ.get("SETUP_V2_REJUDGE_RATES", ""), help="Comma-separated rate filter for automatic GLM rejudge, e.g. 0.05,0.15.")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_glm_rejudge(args: argparse.Namespace) -> Path | None:
    if args.skip_glm or os.environ.get("SETUP_V2_SUMMARY_SKIP_GLM", "0").lower() in {"1", "true", "yes"}:
        return None
    env = os.environ.copy()
    env["SETUP_V2_REJUDGE_OUT_DIR"] = args.glm_output_dir
    env["SETUP_V2_REJUDGE_COMPLETE_ONLY"] = "1"
    env["GLM_REJUDGE_WORKERS"] = str(args.glm_workers)
    if args.glm_datasets:
        env["SETUP_V2_REJUDGE_DATASETS"] = args.glm_datasets
    if args.glm_methods:
        env["SETUP_V2_REJUDGE_METHODS"] = args.glm_methods
    if args.glm_rates:
        env["SETUP_V2_REJUDGE_RATES"] = args.glm_rates
    cmd = [sys.executable, str(EXP / "scripts/rejudge_setup_v2_glm_clean.py")]
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    return Path(args.glm_output_dir) / "rejudged_summary.csv"


def merge_glm_summary(rows: list[dict], glm_summary: Path | None) -> None:
    if glm_summary is None or not glm_summary.exists():
        return
    glm = {}
    with glm_summary.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            glm[(row["dataset"], row["method"], row["rate"])] = row
    merged = []
    for row in rows:
        out = dict(row)
        g = glm.get((row["dataset"], row["method"], row["rate"]))
        out["glm_correct"] = g.get("glm_correct", "") if g else ""
        out["glm_acc"] = g.get("glm_acc", "") if g else ""
        out["glm_complete"] = g.get("complete", "") if g else ""
        merged.append(out)
    fields = ["dataset", "method", "rate", "rows", "raw_rows", "expected_rows", "complete", "em", "f1", "glm_correct", "glm_acc", "glm_complete"]
    write_csv(EXP / "setup_v2_summary_with_glm.csv", merged, fields)
    (EXP / "setup_v2_summary_with_glm.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    rows = []
    grouped = {}
    for path, method, dataset, rate_tag, segment in iter_csvs():
        key = (dataset, method, rate_tag)
        grouped.setdefault(key, [])
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                grouped[key].append(row)
    for (dataset, method, rate_tag), items in sorted(grouped.items()):
        dedup = {}
        for row in items:
            question = row.get("Question", "") or ""
            # Segment-level rescue runs may overlap monolithic runs. Keep one row per question.
            dedup.setdefault(question, row)
        ems = []
        f1s = []
        for row in dedup.values():
            pred = row.get("Pred Answer") or row.get("Answer") or row.get("Predicted") or ""
            gold = row.get("Real Answer") or row.get("Ground Truth") or row.get("Gold") or ""
            ems.append(float(_exact_match_score(pred, gold)))
            f1s.append(float(compute_f1(pred, gold, tokenizer)))
        n = len(dedup)
        expected = EXPECTED_ROWS.get(dataset, 0)
        rows.append({
            "dataset": dataset,
            "method": method,
            "rate": rate_tag.replace("rate_", "").replace("p", "."),
            "rows": n,
            "raw_rows": len(items),
            "expected_rows": expected,
            "complete": bool(expected and n >= expected),
            "em": sum(ems) / n if n else 0.0,
            "f1": sum(f1s) / n if n else 0.0,
        })
    fields = ["dataset", "method", "rate", "rows", "raw_rows", "expected_rows", "complete", "em", "f1"]
    write_csv(EXP / "setup_v2_summary.csv", rows, fields)
    (EXP / "setup_v2_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    glm_summary = run_glm_rejudge(args)
    merge_glm_summary(rows, glm_summary)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    if glm_summary:
        print(f"glm_summary={glm_summary}")
        print(f"merged_summary={EXP / 'setup_v2_summary_with_glm.csv'}")


if __name__ == "__main__":
    main()

