#!/usr/bin/env python3
import csv
import glob
import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path("/home/hming/FusionRAG-pca-analysis")
EXP = ROOT / "MOTIVATION_EXPERIMENTS/rag_docs_preprocess_ablation"
MODEL_NAME = "Qwen3-32B"

DATASETS = OrderedDict([
    ("2wikimqa", 200),
    ("hotpotqa", 260),
    ("triviaqa", 270),
])

RUNS = OrderedDict([
    ("ragdocs_online_qk_rate015", {
        "rate": 0.15,
        "subdir": "FusionRAG_global_topk10_rag_docs",
        "baseline": {"2wikimqa": "107/200 (53.50%)", "hotpotqa": "206/260 (79.23%)", "triviaqa": "212/270 (78.52%)"},
        "note": "preprocess uses online RAG docs; online selector is FusionRAG-QK.",
    }),
    ("ragdocs_online_draft_rate015", {
        "rate": 0.15,
        "subdir": "DraftModel_global_topk10_rag_docs",
        "baseline": {"2wikimqa": "101/200 (50.50%)", "hotpotqa": "207/260 (79.62%)", "triviaqa": "214/270 (79.26%)"},
        "note": "preprocess uses online RAG docs; online selector is DraftModel 3B.",
    }),
])


def segments(n, step=25):
    return [(s, min(n, s + step)) for s in range(0, n, step)]


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
    select = [float(x) for x in re.findall(r"^(?:select_time|draft_select_time):\s*([0-9.]+)", text, re.M)]
    return {
        "log_files": existing,
        "finished_segments": text.count("FINAL RESULTS"),
        "traceback": text.count("Traceback"),
        "killed": text.count("Killed") + text.count("Terminated"),
        "prompt_eval_mean_s": mean(prompt),
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
        "baseline_native": cfg["baseline"].get(dataset, ""),
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
    lines.append("# RAG Docs Preprocess Ablation\n\n")
    lines.append("目的：把 preprocess 阶段的 offline top-k 文档替换为 online RAG 真实召回/使用的文档集合，观察 online QK 和 DraftModel 的性能是否接近 full_rate1。\n\n")
    lines.append("实现：`--recall_method rag_docs`。对每个 chunk，使用包含该 chunk 的 sub-question 的真实 `chunk_ids` 作为 preprocess recall；原 preprocess 逻辑会跳过 self 并最后追加当前 chunk，因此文档集合接近 online RAG，但顺序经过确定性轮转，不完全等同原 online 顺序。\n\n")
    lines.append("运行：`tmux new-session -d -s ragdocs_preprocess_ablation_qjy003 /home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/rag_docs_preprocess_ablation/launch_ragdocs_ablation_qjy003.sh`\n\n")
    lines.append("## 当前结果\n\n")
    lines.append("| dataset | method | native baseline | Main Acc | Sub Acc | F1 | EM | rows | finished seg | missing seg | traceback/killed | note |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['label']} | {r['baseline_native']} | "
            f"{r['main_correct']}/{r['main_total']} ({r['main_acc']:.2%}) | "
            f"{r['sub_correct']}/{r['sub_total']} ({r['sub_acc']:.2%}) | "
            f"{r['avg_f1']:.4f} | {r['avg_em']:.4f} | {r['sub_total']} | "
            f"{r['finished_segments']} | {r['missing_segments']} | {r['traceback']}/{r['killed']} | {r['note']} |\n"
        )
    lines.append("\n## 输出文件\n\n")
    lines.append("- `ragdocs_ablation_summary.csv`\n")
    lines.append("- `ragdocs_ablation_summary.json`\n")
    lines.append("- `logs/supervisor.log`\n")
    return "".join(lines)


def main():
    rows = []
    for dataset in DATASETS:
        for label, cfg in RUNS.items():
            rows.append(summarize_one(dataset, label, cfg))
    write_csv(EXP / "ragdocs_ablation_summary.csv", rows)
    (EXP / "ragdocs_ablation_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXP / "README.md").write_text(render_readme(rows), encoding="utf-8")
    print(render_readme(rows))


if __name__ == "__main__":
    main()
