#!/usr/bin/env python3
import csv
import os
import socket
import subprocess
import sys
import time
from multiprocessing import Process
from pathlib import Path

ROOT = Path.cwd()
DEFAULT_EXP_ROOT = Path("/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_alpha_cross_dataset_v2")
EXP = Path(os.environ.get("FUSIONRAG_CROSS_EXP_ROOT", str(DEFAULT_EXP_ROOT)))
LOGS = EXP / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

PYTHON = "/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python"
MODEL = "/mnt/qjhs-sh-lab-01/models/Qwen3-32B"
DRAFT = "/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct"
BGE = "/mnt/qjhs-sh-lab-01/models/bge-m3"
API_URL = "http://36.150.226.221:32355/v1"
API_KEY = "api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS"
DEFAULT_CACHE_ROOT = Path("/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-attn-alpha-cross-cache-v2")
CACHE_ROOT = Path(os.environ.get("FUSIONRAG_CROSS_CACHE_ROOT", str(DEFAULT_CACHE_ROOT)))
CSV_NAME = "rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv"
COMBOS = [("uniform", "0.1"), ("uniform", "0.25"), ("random", "0.05"), ("random", "0.1"), ("random", "0.25")]
WORKER_GPUS = [int(x) for x in os.environ.get("FUSIONRAG_CROSS_WORKER_GPUS", "0,1,2,3").split(",") if x.strip()]


def segments(start, end, step=25):
    out = []
    cur = start
    while cur < end:
        nxt = min(cur + step, end)
        out.append((cur, nxt))
        cur = nxt
    return out


def build_tasks():
    tasks = []
    if str(ROOT).startswith("/raid/"):
        datasets = [
            ("2wikimqa", "MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/2wikimqa_reflect.json", segments(0, 200)),
            ("hotpotqa", "MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/hotpotqa_reflect.json", segments(0, 125)),
        ]
    else:
        datasets = [
            ("hotpotqa", "MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/hotpotqa_reflect.json", segments(125, 260)),
            ("triviaqa", "MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/triviaqa_reflect.json", segments(0, 270)),
        ]
    for mode, alpha in COMBOS:
        for ds, data_path, segs in datasets:
            for start, end in segs:
                tasks.append({"dataset": ds, "data_path": data_path, "mode": mode, "alpha": alpha, "start": start, "end": end})
    return tasks


def label(mode, alpha):
    return f"{mode}_alpha{alpha.replace('.', 'p')}"


def out_dir(task):
    return EXP / label(task["mode"], task["alpha"]) / task["dataset"] / f"seg_{task['start']}_{task['end']}"


def done(task):
    d = out_dir(task)
    log = d / "run.log"
    if not log.exists():
        return False
    try:
        if "FINAL RESULTS" not in log.read_text(errors="ignore"):
            return False
    except Exception:
        return False
    return any(d.glob(f"**/{CSV_NAME}"))


def running(task):
    d = str(out_dir(task))
    proc = subprocess.run(["pgrep", "-af", d], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return proc.returncode == 0


def gpu_free(gpu):
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    for line in proc.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(gpu):
            return int(parts[1]) < 10000
    return False


def run_task(gpu, task):
    d = out_dir(task)
    d.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONUNBUFFERED": "1",
        "FUSIONRAG_REPROCESS_ATTENTION_ABLATION": task["mode"],
        "FUSIONRAG_REPROCESS_ATTENTION_ABLATION_ALPHA": task["alpha"],
    })
    cmd = [
        PYTHON, "test_fusionrag_reflect_preprocess_exp.py",
        "--model_type", "qwen3",
        "--model_path", MODEL,
        "--model_name", "Qwen3-32B",
        "--data_path", task["data_path"],
        "--dataset_name", task["dataset"],
        "--cache_path", str(CACHE_ROOT / f"worker_gpu{gpu}" / task["dataset"]),
        "--result_path", str(d),
        "--start_sample", str(task["start"]),
        "--end_sample", str(task["end"]),
        "--rate", "0.15",
        "--topk", "10",
        "--preprocess", "true",
        "--recall_method", "bge",
        "--reprocess_method", "DraftModel",
        "--draft_model_path", DRAFT,
        "--revert_rope", "true",
        "--preprocess_scope", "global",
        "--bge_model_path", BGE,
        "--device", "cuda:0",
        "--use_multi_gpu", "false",
        "--openai_base_url", API_URL,
        "--openai_api_key", API_KEY,
        "--openai_model", "GLM-5.2",
    ]
    with (d / "run.log").open("w", encoding="utf-8") as f:
        return subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def worker(gpu, tasks):
    host = socket.gethostname().split(".")[0]
    log_path = LOGS / f"fixed_worker_{host}_gpu{gpu}.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"[{time.strftime('%F %T')}] worker start gpu={gpu} tasks={len(tasks)} root={ROOT}\n")
        for task in tasks:
            name = f"{task['dataset']} {label(task['mode'], task['alpha'])} {task['start']}-{task['end']}"
            while running(task) and not done(task):
                log.write(f"[{time.strftime('%F %T')}] wait existing {name}\n")
                time.sleep(90)
            if done(task):
                log.write(f"[{time.strftime('%F %T')}] skip done {name}\n")
                continue
            while not gpu_free(gpu):
                log.write(f"[{time.strftime('%F %T')}] wait gpu {gpu} for {name}\n")
                time.sleep(90)
            log.write(f"[{time.strftime('%F %T')}] run {name}\n")
            rc = run_task(gpu, task)
            log.write(f"[{time.strftime('%F %T')}] done rc={rc} {name}\n")
        log.write(f"[{time.strftime('%F %T')}] worker complete gpu={gpu}\n")


def main():
    all_tasks = build_tasks()
    buckets = {gpu: [] for gpu in WORKER_GPUS}
    for i, task in enumerate(all_tasks):
        buckets[WORKER_GPUS[i % len(WORKER_GPUS)]].append(task)
    procs = []
    for gpu, tasks in buckets.items():
        p = Process(target=worker, args=(gpu, tasks), daemon=False)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
