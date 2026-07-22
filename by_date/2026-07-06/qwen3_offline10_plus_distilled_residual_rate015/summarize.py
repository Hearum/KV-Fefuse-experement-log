#!/usr/bin/env python3
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
EXP = ROOT / "MOTIVATION_EXPERIMENTS/qwen3_offline10_plus_distilled_residual_rate015"
LABEL = "offline10_distilled005"
MODEL_DIR = "Qwen3-32B/musique/FusionRAG_global_topk10_bge"
CSV_NAME = "rate_0.15_revert_rope.csv"
SEGMENTS = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 150), (150, 175), (175, 200)]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def read_rows():
    rows_by_key = {}
    missing = []
    csv_files = []
    for start, end in SEGMENTS:
        path = EXP / LABEL / f"seg_{start}_{end}" / MODEL_DIR / CSV_NAME
        if not path.exists():
            missing.append(str(path))
            continue
        csv_files.append(str(path))
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows_by_key[(row.get("Main Question", ""), row.get("Sub Question", ""))] = row
    return list(rows_by_key.values()), missing, csv_files


def parse_logs():
    text = ""
    log_files = 0
    for start, end in SEGMENTS:
        path = EXP / LABEL / f"seg_{start}_{end}" / "run.log"
        if path.exists():
            log_files += 1
            text += "\n" + path.read_text(encoding="utf-8", errors="ignore")
    prompt = [float(x) for x in re.findall(r"^prompt eval duration:\s*([0-9.]+)s", text, re.M)]
    residual = [float(x) for x in re.findall(r"residual_select_time=([0-9.]+)", text)]
    added = [float(x) for x in re.findall(r"added_tokens=([0-9.]+)", text)]
    return {
        "log_files": log_files,
        "finished_segments": text.count("FINAL RESULTS"),
        "traceback": text.count("Traceback"),
        "killed": text.count("Killed") + text.count("Terminated"),
        "mean_prompt_eval_duration": mean(prompt),
        "mean_residual_select_time": mean(residual),
        "mean_added_tokens": mean(added),
        "n_prompt_eval": len(prompt),
        "n_residual_timing": len(residual),
    }


def main():
    rows, missing, csv_files = read_rows()
    by_main = defaultdict(list)
    for row in rows:
        by_main[row.get("Main Question", "")].append(row)

    main_total = len(by_main)
    main_correct = sum(
        1
        for group in by_main.values()
        if group and all(str(row.get("Correct", "")).lower() == "true" for row in group)
    )
    sub_total = len(rows)
    sub_correct = sum(1 for row in rows if str(row.get("Correct", "")).lower() == "true")

    out = {
        "method": "offline10_draft_smart_frequency_global + distilled_4layer_residual005",
        "main_correct": main_correct,
        "main_total": main_total,
        "main_acc": main_correct / main_total if main_total else 0.0,
        "sub_correct": sub_correct,
        "sub_total": sub_total,
        "sub_acc": sub_correct / sub_total if sub_total else 0.0,
        "avg_f1": mean([float(row.get("F1") or 0.0) for row in rows]),
        "avg_em": mean([float(row.get("EM") or 0.0) for row in rows]),
        "missing_csv": len(missing),
        "csv_files": len(csv_files),
    }
    out.update(parse_logs())

    with (EXP / "accuracy_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out.keys()))
        writer.writeheader()
        writer.writerow(out)

    (EXP / "summary.json").write_text(
        json.dumps({"summary": out, "missing": missing, "csv_files": csv_files}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Qwen3 Offline10 + Distilled Residual5\n\n",
        "实验设置：Qwen3-32B 主模型，MuSiQue/reflect 200 条样本，topk=10，preprocess=true，BGE recall，offline 固定集合使用 `draft_smart_frequency_global` 选 10% doc tokens；online residual 使用 WikiText-103 蒸馏得到的 4-layer selector 从剩余 token 中补 5%，总预算 15%。GLM-5.2 用作 judge。\n\n",
        "| method | Main Acc | Sub Acc | F1 | EM | added tokens | residual selection(s) | prefill(s) | csv files | missing | traceback/killed |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        f"| {out['method']} | {main_correct}/{main_total} ({out['main_acc']:.2%}) | "
        f"{sub_correct}/{sub_total} ({out['sub_acc']:.2%}) | {out['avg_f1']:.4f} | {out['avg_em']:.4f} | "
        f"{out['mean_added_tokens']:.2f} | {out['mean_residual_select_time']:.4f} | "
        f"{out['mean_prompt_eval_duration']:.4f} | {out['csv_files']} | {out['missing_csv']} | "
        f"{out['traceback']}/{out['killed']} |\n",
    ]
    (EXP / "README.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))


if __name__ == "__main__":
    main()
