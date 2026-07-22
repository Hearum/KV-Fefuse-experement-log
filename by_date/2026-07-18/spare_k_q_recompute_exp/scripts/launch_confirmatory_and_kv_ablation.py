#!/usr/bin/env python3
"""Launch the frozen preprocess candidates and one-axis K/V alpha ablations."""

from __future__ import annotations

import argparse
import shlex
import subprocess

ROOTS = {
    "qjy000": "/raid/home/hming/FusionRAG-pca-analysis",
    "qjy003": "/home/hming/FusionRAG-pca-analysis",
}
PYTHON = "/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python"
RUNNER = (
    "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/"
    "setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py"
)
EXP = "MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp"


def command(method: str, alpha_k: float, alpha_v: float, start: int, end: int, gpu: int, root: str) -> str:
    args = [
        PYTHON, RUNNER,
        "--dataset", "musique-v2",
        "--method", method,
        "--rate", "0.15",
        "--working-kv-alpha-k", str(alpha_k),
        "--working-kv-alpha-v", str(alpha_v),
        "--start", str(start),
        "--end", str(end),
        "--gpu", str(gpu),
        "--result-root", f"{EXP}/{root}",
    ]
    return " ".join(shlex.quote(value) for value in args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", action="append", choices=sorted(ROOTS))
    args = parser.parse_args()
    tasks = []
    test_segments = [(50, 88), (88, 126), (126, 163), (163, 200)]
    for gpu, (start, end) in enumerate(test_segments):
        tasks.append(("qjy000", gpu, command(
            "dense_working_kv_preprocess", 1.0, 1.0, start, end, gpu,
            "results_working_kv_confirmatory_preprocess_test",
        ), f"confirm_dense_pp_a1_{start}_{end}.log"))
    for offset, (start, end) in enumerate(test_segments):
        gpu = offset + 4
        tasks.append(("qjy000", gpu, command(
            "sparse_working_kv_preprocess", 0.5, 0.5, start, end, gpu,
            "results_working_kv_confirmatory_preprocess_test",
        ), f"confirm_sparse_pp_a0p5_{start}_{end}.log"))

    validation = [
        ("dense_working_kv_raw", 0.75, 0.0, "dense_raw_k_only"),
        ("dense_working_kv_raw", 0.0, 0.75, "dense_raw_v_only"),
        ("sparse_working_kv_raw", 0.75, 0.0, "sparse_raw_k_only"),
        ("sparse_working_kv_raw", 0.0, 0.75, "sparse_raw_v_only"),
        ("dense_working_kv_preprocess", 1.0, 0.0, "dense_pp_k_only"),
        ("dense_working_kv_preprocess", 0.0, 1.0, "dense_pp_v_only"),
        ("sparse_working_kv_preprocess", 0.5, 0.0, "sparse_pp_k_only"),
        ("sparse_working_kv_preprocess", 0.0, 0.5, "sparse_pp_v_only"),
    ]
    for gpu, (method, alpha_k, alpha_v, name) in enumerate(validation):
        tasks.append(("qjy003", gpu, command(
            method, alpha_k, alpha_v, 0, 50, gpu,
            "results_working_kv_alpha_kv_validation",
        ), f"{name}_0_50.log"))

    for host, gpu, task, log_name in tasks:
        if args.host and host not in args.host:
            continue
        root = ROOTS[host]
        log = f"{EXP}/logs_confirmatory_and_kv_ablation/{log_name}"
        remote = (
            f"cd {shlex.quote(root)} && mkdir -p {shlex.quote(EXP + '/logs_confirmatory_and_kv_ablation')} "
            f"&& nohup {task} > {shlex.quote(log)} 2>&1 < /dev/null &"
        )
        subprocess.run(["ssh", host, remote], check=True)
        print(f"{host} gpu={gpu}: {task}")


if __name__ == "__main__":
    main()
