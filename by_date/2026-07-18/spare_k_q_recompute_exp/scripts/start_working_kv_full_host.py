#!/usr/bin/env python3
"""Start one host's eight local full-MuSiQue workers without nested SSH."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
EXP = ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp"
PYTHON = "/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python"
RUNNER = ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, choices=(0, 100), required=True)
    parser.add_argument("--gpu-count", type=int, default=8)
    args = parser.parse_args()
    methods = [
        ("dense_working_kv_raw", .75, .75, "dense_raw_shared"),
        ("sparse_working_kv_raw", .75, .75, "sparse_raw_shared"),
        ("dense_working_kv_preprocess", 1.0, 1.0, "dense_pp_shared"),
        ("sparse_working_kv_preprocess", .5, .5, "sparse_pp_shared"),
        ("dense_working_kv_raw", .75, 0.0, "dense_raw_k_only"),
        ("dense_working_kv_raw", 0.0, .75, "dense_raw_v_only"),
        ("sparse_working_kv_preprocess", .5, 0.0, "sparse_pp_k_only"),
        ("sparse_working_kv_preprocess", 0.0, .5, "sparse_pp_v_only"),
    ]
    end = args.start + 100
    log_dir = EXP / "logs_working_kv_full_musique"
    log_dir.mkdir(parents=True, exist_ok=True)
    for gpu, (method, alpha_k, alpha_v, label) in enumerate(methods[:args.gpu_count]):
        log = log_dir / f"{label}_{args.start}_{end}.log"
        command = [
            PYTHON, str(RUNNER), "--dataset", "musique-v2", "--method", method, "--rate", "0.15",
            "--working-kv-alpha-k", str(alpha_k), "--working-kv-alpha-v", str(alpha_v),
            "--start", str(args.start), "--end", str(end), "--gpu", str(gpu),
            "--result-root", str(EXP / "results_working_kv_full_musique"),
        ]
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        print(f"gpu={gpu} pid={process.pid}: {' '.join(shlex.quote(x) for x in command)}")


if __name__ == "__main__":
    main()
