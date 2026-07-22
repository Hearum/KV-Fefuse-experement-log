#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, os, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
EXP = ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset"
PYTHON = "/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python"
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tasks",default=str(EXP/"logs/online_tasks.tsv"))
    p.add_argument("--poll-seconds",type=int,default=60)
    p.add_argument("--skip-glm",action="store_true")
    a=p.parse_args()
    if os.environ.get("SETUP_V2_AUTO_GLM","1").lower() in {"0","false","no"}: a.skip_glm=True
    with Path(a.tasks).open(encoding="utf-8",newline="") as f: tasks=list(csv.DictReader(f,delimiter="\t"))
    marker_dir=EXP/"logs/worker_task_logs"
    prefixes=[]
    for row in tasks:
        prefixes.append("task{:04d}_".format(int(row["task_id"])))
    while True:
        names=[x.name for x in marker_dir.glob("*.done")]
        complete=sum(any(n.startswith(prefix) for n in names) for prefix in prefixes)
        print(f"[auto-finalize] completed_tasks={complete}/{len(prefixes)}",flush=True)
        if complete>=len(prefixes): break
        time.sleep(max(5,a.poll_seconds))
    cmd=[PYTHON,str(EXP/"scripts/summarize_setup_v2.py")]
    if a.skip_glm: cmd.append("--skip-glm")
    subprocess.run(cmd,cwd=str(ROOT),check=True,env=os.environ.copy())
if __name__ == "__main__": main()
