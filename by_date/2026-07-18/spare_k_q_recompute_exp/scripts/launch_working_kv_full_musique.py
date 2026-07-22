#!/usr/bin/env python3
"""Launch full MuSiQue-v2 Working-KV comparisons on fixed GPUs."""

from __future__ import annotations

import argparse
import shlex
import subprocess

ROOTS = {"qjy000": "/raid/home/hming/FusionRAG-pca-analysis", "qjy003": "/home/hming/FusionRAG-pca-analysis"}
PY = "/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python"
RUNNER = "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py"
EXP = "MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", action="append", choices=sorted(ROOTS))
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
    tasks = []
    for method_index, spec in enumerate(methods):
        method, alpha_k, alpha_v, label = spec
        for start, end, suffix in ((0, 100, "0_100"), (100, 200, "100_200")):
            host = "qjy000" if start == 0 else "qjy003"
            local_gpu = method_index
            command = [
                PY, RUNNER, "--dataset", "musique-v2", "--method", method, "--rate", "0.15",
                "--working-kv-alpha-k", str(alpha_k), "--working-kv-alpha-v", str(alpha_v),
                "--start", str(start), "--end", str(end), "--gpu", str(local_gpu),
                "--result-root", f"{EXP}/results_working_kv_full_musique",
            ]
            task = " ".join(shlex.quote(v) for v in command)
            log = f"{EXP}/logs_working_kv_full_musique/{label}_{suffix}.log"
            tasks.append((host, local_gpu, task, log))
    for host, gpu, task, log in tasks:
        if args.host and host not in args.host:
            continue
        root = ROOTS[host]
        remote = f"cd {shlex.quote(root)} && mkdir -p {shlex.quote(EXP + '/logs_working_kv_full_musique')} && nohup {task} > {shlex.quote(log)} 2>&1 < /dev/null &"
        subprocess.run(["ssh", host, remote], check=True)
        print(f"{host} gpu={gpu}: {task}")


if __name__ == "__main__":
    main()
