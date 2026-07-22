#!/usr/bin/env python3
"""Launch fixed rate=0.15 sparse top-k shards as independent local workers."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
PYTHON = "/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python"
RUNNER = ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py"
EXP = ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--query-chunk", type=int, default=64)
    args = parser.parse_args()

    result_root = EXP / "results_sparse_rate015_sharded"
    log_root = EXP / "logs_sparse_rate015_sharded"
    log_root.mkdir(parents=True, exist_ok=True)
    log = log_root / f"topk{args.topk}_{args.start}_{args.end}.log"
    command = [
        PYTHON, str(RUNNER),
        "--dataset", "musique-v2",
        "--method", "sparse_working_kv_preprocess",
        "--rate", "0.15",
        "--working-kv-alpha-k", "1",
        "--working-kv-alpha-v", "1",
        "--sparse-block-size", "64",
        "--sparse-block-topk", str(args.topk),
        "--start", str(args.start),
        "--end", str(args.end),
        "--gpu", str(args.gpu),
        "--result-root", str(result_root),
    ]
    env = {"FUSIONRAG_SPARSE_QUERY_CHUNK": str(args.query_chunk)}
    import os
    child_env = os.environ.copy()
    child_env.update(env)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, env=child_env, stdout=handle,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    print(f"pid={process.pid}: {' '.join(shlex.quote(x) for x in command)}")


if __name__ == "__main__":
    main()
