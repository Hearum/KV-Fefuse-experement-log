#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
EXP = ROOT / "MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset"
PY = "/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python"
HOSTS = ["qjy000", "qjy001", "qjy003"]
REMOTE_REPO_EXPR = "if [ -d /raid/home/hming/FusionRAG-pca-analysis ]; then echo /raid/home/hming/FusionRAG-pca-analysis; else echo /home/hming/FusionRAG-pca-analysis; fi"


def main() -> None:
    subprocess.run([PY, str(EXP / "scripts/build_v2_data.py")], check=True, cwd=str(ROOT))
    subprocess.run([PY, str(EXP / "scripts/make_online_tasks.py")], check=True, cwd=str(ROOT))
    if os.environ.get("SETUP_V2_AUTO_GLM", "1").lower() not in {"0", "false", "no"}:
        finalizer = ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/auto_finalize_setup_v2.py"
        finalizer_log = ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/logs/auto_finalize_setup_v2.log"
        finalizer_pid = ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/logs/auto_finalize_setup_v2.pid"
        finalizer_log.parent.mkdir(parents=True, exist_ok=True)
        watcher = f"cd {ROOT} && nohup {PY} {finalizer} > {finalizer_log} 2>&1 < /dev/null & echo $! > {finalizer_pid}"
        subprocess.run(["bash", "-lc", watcher], check=True)
        print("launched automatic EM/F1 + GLM finalizer")
    for host_idx, host in enumerate(HOSTS):
        repo = subprocess.check_output(["ssh", host, REMOTE_REPO_EXPR], text=True).strip()
        for gpu in range(8):
            slot = host_idx * 8 + gpu
            outer = f"{repo}/MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/logs/worker_{host}_gpu{gpu}.outer.log"
            pid = f"{repo}/MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/logs/worker_{host}_gpu{gpu}.pid"
            cmd = (
                f"cd {repo} && nohup {PY} "
                f"MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/run_setup_v2_worker.py "
                f"--slot {slot} --total-slots 24 --gpu {gpu} "
                f"> {outer} 2>&1 < /dev/null & echo $! > {pid}"
            )
            subprocess.run(["ssh", host, cmd], check=True)
            print(f"launched host={host} gpu={gpu} slot={slot}")


if __name__ == "__main__":
    main()

