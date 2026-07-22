#!/usr/bin/env python3
"""Launch fixed 50-sample raw/preprocess working-KV experiments."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

PYTHON = "/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python"
RUNNER = "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py"
EXP = "MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp"
RESULT_ROOT = f"{EXP}/results_working_kv_50"
LOG_ROOT = f"{EXP}/logs_working_kv_50"
HOST_ROOTS = {
    "qjy000": Path("/raid/home/hming/FusionRAG-pca-analysis"),
    "qjy001": Path("/home/hming/FusionRAG-pca-analysis"),
}

TASKS = [
    ("qjy000", 0, "dense_working_kv_preprocess", 0.25),
    ("qjy000", 1, "dense_working_kv_preprocess", 0.50),
    ("qjy000", 2, "dense_working_kv_preprocess", 0.75),
    ("qjy000", 3, "dense_working_kv_preprocess", 1.00),
    ("qjy000", 4, "sparse_working_kv_preprocess", 0.25),
    ("qjy000", 5, "sparse_working_kv_preprocess", 0.50),
    ("qjy000", 6, "sparse_working_kv_preprocess", 0.75),
    ("qjy001", 0, "sparse_working_kv_preprocess", 1.00),
    ("qjy001", 1, "dense_working_kv_raw", 0.25),
    ("qjy001", 2, "dense_working_kv_raw", 0.50),
    ("qjy001", 3, "dense_working_kv_raw", 0.75),
    ("qjy001", 4, "dense_working_kv_raw", 1.00),
    ("qjy001", 5, "sparse_working_kv_raw", 0.50),
    ("qjy001", 6, "sparse_working_kv_raw", 0.75),
    ("qjy001", 7, "sparse_working_kv_raw", 1.00),
]


def main() -> None:
    for host, gpu, method, alpha in TASKS:
        root = HOST_ROOTS[host]
        subprocess.run(["ssh", host, "mkdir", "-p", str(root / LOG_ROOT)], check=True)
        alpha_tag = str(alpha).replace(".", "p")
        log = root / LOG_ROOT / f"{method}_alpha{alpha_tag}.log"
        args = [
            PYTHON,
            RUNNER,
            "--dataset", "musique-v2",
            "--method", method,
            "--rate", "0.15",
            "--working-kv-alpha", str(alpha),
            "--sparse-block-size", "64",
            "--sparse-block-topk", "8",
            "--start", "0",
            "--end", "50",
            "--gpu", str(gpu),
            "--result-root", RESULT_ROOT,
        ]
        remote = (
            f"cd {shlex.quote(str(root))} && "
            f"CUDA_VISIBLE_DEVICES={gpu} SETUP_V2_AUTO_GLM=0 PYTHONUNBUFFERED=1 "
            f"nohup {shlex.join(args)} > {shlex.quote(str(log))} 2>&1 < /dev/null & echo $!"
        )
        result = subprocess.run(["ssh", host, remote], check=True, text=True, capture_output=True)
        print(f"{host} gpu={gpu} method={method} alpha={alpha}: pid={result.stdout.strip()}")


if __name__ == "__main__":
    main()
