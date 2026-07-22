#!/usr/bin/env python3
"""Launch the frozen-commit 50-sample working-KV validation matrix."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

PYTHON = "/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python"
RUNNER = "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py"
EXP = "MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp"
RESULT_ROOT = f"{EXP}/results_working_kv_fix2_validation_50"
LOG_ROOT = f"{EXP}/logs_working_kv_fix2_validation_50"
HOST_ROOTS = {
    "qjy000": Path("/raid/home/hming/FusionRAG-pca-analysis"),
    "qjy001": Path("/home/hming/FusionRAG-pca-analysis"),
}

# One persistent queue per physical GPU. No worker creates or owns a cache;
# every task reads the shared Qwen3-32B/MuSiQue-v2 cache configured by runner.
WORKERS = {
    ("qjy000", 0): [("dense_working_kv_preprocess", 0.00), ("dense_working_kv_preprocess", 0.25), ("dense_working_kv_preprocess", 0.50)],
    ("qjy000", 1): [("dense_working_kv_preprocess", 0.75), ("dense_working_kv_preprocess", 1.00)],
    ("qjy000", 2): [("dense_working_kv_raw", 0.00), ("dense_working_kv_raw", 0.25), ("dense_working_kv_raw", 0.50)],
    ("qjy000", 3): [("dense_working_kv_raw", 0.75), ("dense_working_kv_raw", 1.00)],
    ("qjy000", 7): [("sparse_working_kv_preprocess", 0.00), ("sparse_working_kv_preprocess", 0.25)],
    ("qjy001", 1): [("sparse_working_kv_preprocess", 0.50), ("sparse_working_kv_preprocess", 0.75)],
    ("qjy001", 2): [("sparse_working_kv_preprocess", 1.00), ("sparse_working_kv_raw", 0.00)],
    ("qjy001", 3): [("sparse_working_kv_raw", 0.25), ("sparse_working_kv_raw", 0.50)],
    ("qjy001", 4): [("sparse_working_kv_raw", 0.75), ("sparse_working_kv_raw", 1.00)],
}


def task_command(method: str, alpha: float, gpu: int) -> str:
    args = [
        PYTHON, RUNNER,
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
    return shlex.join(args)


def main() -> None:
    for (host, gpu), tasks in WORKERS.items():
        root = HOST_ROOTS[host]
        log_dir = root / LOG_ROOT
        subprocess.run(["ssh", host, "mkdir", "-p", str(log_dir)], check=True)
        commands = []
        for method, alpha in tasks:
            tag = str(alpha).replace(".", "p")
            task_log = log_dir / f"{method}_alpha{tag}.log"
            commands.append(
                f"{task_command(method, alpha, gpu)} > {shlex.quote(str(task_log))} 2>&1"
            )
        worker_log = log_dir / f"worker_{host}_gpu{gpu}.log"
        commands.append(f"echo PASS > {shlex.quote(str(worker_log))}")
        queue = " && ".join(commands)
        remote = (
            f"cd {shlex.quote(str(root))} && "
            f"nohup bash -lc {shlex.quote(queue)} >> {shlex.quote(str(worker_log))} 2>&1 "
            f"< /dev/null & echo $!"
        )
        result = subprocess.run(["ssh", host, remote], check=True, text=True, capture_output=True)
        print(f"{host} gpu={gpu} tasks={len(tasks)} pid={result.stdout.strip()}")


if __name__ == "__main__":
    main()
