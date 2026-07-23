#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path

ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
OUT = ROOT / "MOTIVATION_EXPERIMENTS/qwen3_layer4_selector_gap_to_full_draft"

METHODS = {
    "full_draft": ROOT / "MOTIVATION_EXPERIMENTS/qwen3_fair_budget_offline_vs_residual/offline10_draft005",
    "native_layer4": ROOT / "MOTIVATION_EXPERIMENTS/qwen3_offline10_plus_shallow_layer4_residual_rate015",
    "distilled_layer4": ROOT / "MOTIVATION_EXPERIMENTS/qwen3_offline10_plus_distilled_residual_rate015_trace_after_fix",
}


def key_from_name(path):
    m = re.search(r"example(\d+)_sub(\d+)_", path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def load_selected(root):
    data = {}
    times = {}
    for path in root.glob("seg_*/selected/*.json"):
        key = key_from_name(path)
        if key is None:
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        data[key] = set(int(x) for x in obj.get("selected_abs", []))
        times[key] = float(obj.get("selection_time", 0.0))
    return data, times


def load_offline(root):
    data = {}
    for path in root.glob("seg_*/offline_fixed_selected_indices/*.json"):
        key = key_from_name(path)
        if key is None:
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        data[key] = set(int(x) for x in obj.get("selected_abs", []))
    return data


def safe_div(a, b):
    return a / b if b else 0.0


def stats(candidate, teacher, offline, keys):
    rows = []
    for key in sorted(keys):
        cand_all = candidate[key]
        teach_all = teacher[key]
        off = offline[key]
        cand_res = cand_all - off
        teach_res = teach_all - off
        rows.append(
            {
                "example_id": key[0],
                "sub_q_idx": key[1],
                "offline_count": len(off),
                "candidate_count": len(cand_all),
                "teacher_count": len(teach_all),
                "candidate_residual_count": len(cand_res),
                "teacher_residual_count": len(teach_res),
                "final_intersection": len(cand_all & teach_all),
                "final_jaccard": safe_div(len(cand_all & teach_all), len(cand_all | teach_all)),
                "final_teacher_recall": safe_div(len(cand_all & teach_all), len(teach_all)),
                "residual_intersection": len(cand_res & teach_res),
                "residual_jaccard": safe_div(len(cand_res & teach_res), len(cand_res | teach_res)),
                "residual_teacher_recall": safe_div(len(cand_res & teach_res), len(teach_res)),
            }
        )
    return rows


def mean(rows, field):
    vals = [float(r[field]) for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selected = {}
    times = {}
    offline = {}
    for name, root in METHODS.items():
        selected[name], times[name] = load_selected(root)
        offline[name] = load_offline(root)

    teacher = selected["full_draft"]
    teacher_offline = offline["full_draft"]
    summaries = []
    detail_files = []
    for name in ["native_layer4", "distilled_layer4"]:
        common = set(teacher) & set(selected[name]) & set(teacher_offline) & set(offline[name])
        # The offline set definition is identical across these runs; use full_draft's copy as the reference.
        rows = stats(selected[name], teacher, teacher_offline, common)
        detail_path = OUT / f"{name}_vs_full_draft_detail.csv"
        with detail_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["example_id"])
            writer.writeheader()
            writer.writerows(rows)
        detail_files.append(detail_path)
        timing_values = [times[name][k] for k in common if k in times[name]]
        timing_steady = sorted(timing_values)[: max(0, len(timing_values) - 8)] if len(timing_values) > 8 else timing_values
        summaries.append(
            {
                "method": name,
                "common_items": len(common),
                "final_jaccard": mean(rows, "final_jaccard"),
                "final_teacher_recall": mean(rows, "final_teacher_recall"),
                "residual_jaccard": mean(rows, "residual_jaccard"),
                "residual_teacher_recall": mean(rows, "residual_teacher_recall"),
                "avg_candidate_count": mean(rows, "candidate_count"),
                "avg_teacher_count": mean(rows, "teacher_count"),
                "avg_candidate_residual_count": mean(rows, "candidate_residual_count"),
                "avg_teacher_residual_count": mean(rows, "teacher_residual_count"),
                "selection_time_mean_all": sum(timing_values) / len(timing_values) if timing_values else 0.0,
                "selection_time_mean_excluding_top8": sum(timing_steady) / len(timing_steady) if timing_steady else 0.0,
            }
        )

    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# Layer4 Selector Gap to Full DraftModel\n\n",
        "目标：在同一套 Qwen3/MuSiQue reflect 数据上，用 full DraftModel selector 作为 teacher，比较原生 layer4 selector 与 WikiText 蒸馏 4-layer selector 的选 token 差距。\n\n",
        "统计口径：`final` 是 offline 10% + residual 5% 的最终 15% selected set；`residual` 是从 final set 中扣掉共同 offline fixed 10% 后的在线补选 5%。`teacher recall` 表示候选方法覆盖了 full DraftModel 选中 token 的比例。\n\n",
        "| method | items | final Jaccard | final teacher recall | residual Jaccard | residual teacher recall | selection all(s) | selection steady(s) |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['common_items']} | {row['final_jaccard']:.4f} | "
            f"{row['final_teacher_recall']:.4f} | {row['residual_jaccard']:.4f} | "
            f"{row['residual_teacher_recall']:.4f} | {row['selection_time_mean_all']:.4f} | "
            f"{row['selection_time_mean_excluding_top8']:.4f} |\n"
        )
    (OUT / "README.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))


if __name__ == "__main__":
    main()
