#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


REPO_CANDIDATES = [Path("/raid/home/hming/FusionRAG-pca-analysis"), Path("/home/hming/FusionRAG-pca-analysis")]
REPO = next((p for p in REPO_CANDIDATES if p.exists()), REPO_CANDIDATES[0])
EXP = REPO / "MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset"
PY = "/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python"
DEFAULT_CACHE_ROOT = "/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--slot", type=int, required=True)
    p.add_argument("--total-slots", type=int, default=24)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--tasks", default=str(EXP / "logs/online_tasks.tsv"))
    p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks_path = Path(args.tasks)
    log_dir = EXP / "logs/worker_task_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with tasks_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            task_id = int(row["task_id"])
            if task_id % args.total_slots != args.slot:
                continue
            log_path = log_dir / (
                f"task{task_id:04d}_{row['dataset']}_{row['method']}_"
                f"r{str(row['rate']).replace('.', 'p')}_seg{row['start']}_{row['end']}_gpu{args.gpu}.log"
            )
            done_marker = log_path.with_suffix(".done")
            if done_marker.exists():
                print(f"skip done task={task_id} log={log_path}", flush=True)
                continue
            cmd = [
                PY,
                str(EXP / "scripts/run_setup_v2_task.py"),
                "--dataset", row["dataset"],
                "--method", row["method"],
                "--rate", row["rate"],
                "--start", row["start"],
                "--end", row["end"],
                "--gpu", str(args.gpu),
                "--cache-root", args.cache_root,
            ]
            print("RUN", " ".join(cmd), flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode == 0:
                done_marker.write_text("ok\n", encoding="utf-8")
            else:
                print(f"task failed task={task_id} rc={proc.returncode} log={log_path}", flush=True)


if __name__ == "__main__":
    main()

