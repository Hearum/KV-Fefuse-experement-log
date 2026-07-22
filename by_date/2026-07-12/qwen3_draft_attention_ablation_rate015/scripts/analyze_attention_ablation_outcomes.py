#!/usr/bin/env python3
import csv
import glob
from pathlib import Path


EXP = Path("MOTIVATION_EXPERIMENTS/qwen3_draft_attention_ablation_rate015")
BASELINE = Path("MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/online_draft_rate015")
OUT = EXP / "attention_ablation_outcome_flips.csv"
SUMMARY = EXP / "attention_ablation_outcome_summary.csv"

METHODS = {
    "uniform_alpha0p1": EXP / "uniform_alpha0p1_draft_rate015",
    "uniform_alpha0p25": EXP / "uniform_alpha0p25_draft_rate015",
    "random_alpha0p05": EXP / "random_alpha0p05_draft_rate015",
    "random_alpha0p1": EXP / "random_alpha0p1_draft_rate015",
    "random_alpha0p25": EXP / "random_alpha0p25_draft_rate015",
    "uniform_alpha0p5": EXP / "uniform_alpha0p5_draft_rate015",
    "random_alpha0p5": EXP / "random_alpha0p5_draft_rate015",
}


def load_rows(root: Path):
    rows = {}
    pattern = str(root / "seg_*" / "**" / "rate_0.15*draft*revert_rope.csv")
    for path in glob.glob(pattern, recursive=True):
        seg = Path(path).parts[-5] if "seg_" in Path(path).parts[-5] else ""
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row.get("Main Question", ""), row.get("Sub Question", ""))
                row = dict(row)
                row["segment"] = seg
                row["source_csv"] = path
                rows[key] = row
    return rows


def is_correct(row):
    return str(row.get("Correct", "")).strip().lower() == "true"


def main():
    baseline = load_rows(BASELINE)
    summary_rows = []
    flip_rows = []
    for method, root in METHODS.items():
        rows = load_rows(root)
        keys = sorted(set(baseline) & set(rows))
        w2r = r2w = same_right = same_wrong = 0
        for main_q, sub_q in keys:
            b = baseline[(main_q, sub_q)]
            cur = rows[(main_q, sub_q)]
            b_ok = is_correct(b)
            c_ok = is_correct(cur)
            if (not b_ok) and c_ok:
                flip = "wrong_to_right"
                w2r += 1
            elif b_ok and (not c_ok):
                flip = "right_to_wrong"
                r2w += 1
            elif b_ok and c_ok:
                flip = "same_right"
                same_right += 1
            else:
                flip = "same_wrong"
                same_wrong += 1
            if flip in {"wrong_to_right", "right_to_wrong"}:
                flip_rows.append({
                    "method": method,
                    "flip": flip,
                    "segment": cur.get("segment", ""),
                    "main_question": main_q,
                    "sub_question": sub_q,
                    "ground_truth": cur.get("Ground Truth", ""),
                    "baseline_predicted": b.get("Predicted", ""),
                    "method_predicted": cur.get("Predicted", ""),
                    "baseline_f1": b.get("F1", ""),
                    "method_f1": cur.get("F1", ""),
                    "baseline_em": b.get("EM", ""),
                    "method_em": cur.get("EM", ""),
                    "method_reason": cur.get("Reason", ""),
                })
        summary_rows.append({
            "method": method,
            "paired_sub_questions": len(keys),
            "wrong_to_right": w2r,
            "right_to_wrong": r2w,
            "same_right": same_right,
            "same_wrong": same_wrong,
            "net_sub_accuracy_gain": w2r - r2w,
        })

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        fields = [
            "method", "flip", "segment", "main_question", "sub_question",
            "ground_truth", "baseline_predicted", "method_predicted",
            "baseline_f1", "method_f1", "baseline_em", "method_em",
            "method_reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flip_rows)
    print(f"wrote {SUMMARY}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
