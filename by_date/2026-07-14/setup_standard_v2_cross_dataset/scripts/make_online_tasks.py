#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
EXP = ROOT / "MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset"

DATASETS = {
    "musique-v2": 200,
    "2wikimqa-v2": 200,
    "hotpotqa-v2": 260,
    "triviaqa-v2": 270,
}
RATES = [0, 0.15, 0.3, 0.5, 0.8, 1.0]
ALPHA_RATES = [0.15, 0.3, 0.5, 0.8, 1.0]


def main() -> None:
    out = EXP / "logs/online_tasks.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    idx = 0
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["task_id", "dataset", "method", "rate", "start", "end"])
        for dataset, n in DATASETS.items():
            for start in range(0, n, 25):
                end = min(n, start + 25)
                w.writerow([idx, dataset, "full_rate1", 1.0, start, end]); idx += 1
                for rate in RATES:
                    w.writerow([idx, dataset, "online_qk", rate, start, end]); idx += 1
                    w.writerow([idx, dataset, "online_draft", rate, start, end]); idx += 1
                for rate in ALPHA_RATES:
                    w.writerow([idx, dataset, "uniform_alpha0p1_draft", rate, start, end]); idx += 1
    print(f"wrote {idx} tasks to {out}")


if __name__ == "__main__":
    main()

