#!/usr/bin/env python3
import os
import socket
import subprocess
import time
from multiprocessing import Process
from pathlib import Path

ROOT = Path.cwd()
EXP = Path(os.environ.get("FUSIONRAG_RATE1_SAME_PIPELINE_ROOT", "/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_alpha_rate1_same_pipeline"))
LOGS = EXP / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

PYTHON = "/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python"
MODEL = "/mnt/qjhs-sh-lab-01/models/Qwen3-32B"
DRAFT = "/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct"
BGE = "/mnt/qjhs-sh-lab-01/models/bge-m3"
API_URL = "http://36.150.226.221:32355/v1"
API_KEY = "api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS"
CACHE_ROOT = Path(os.environ.get("FUSIONRAG_RATE1_CACHE_ROOT", "/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-attn-alpha-cross-cache-v2"))
WORKER_GPUS = [int(x) for x in os.environ.get("FUSIONRAG_RATE1_WORKER_GPUS", "0,1,2,3,4,5,6,7").split(",") if x.strip()]
CSV_NAME = "rate_1.0_draft_Qwen2.5-3B-Instruct_revert_rope.csv"

DATASETS = {
    "2wikimqa": ("MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/2wikimqa_reflect.json", 200),
    "hotpotqa": ("MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/hotpotqa_reflect.json", 260),
    "triviaqa": ("MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/triviaqa_reflect.json", 270),
    "musique": ("data/result_reflect.json", 200),
}

def segments(n, step=25):
    out = []
    s = 0
    while s < n:
        e = min(s + step, n)
        out.append((s, e))
        s = e
    return out

def build_tasks():
    tasks = []
    include = os.environ.get("FUSIONRAG_RATE1_DATASETS", "2wikimqa,hotpotqa,triviaqa,musique")
    wanted = [x.strip() for x in include.split(",") if x.strip()]
    for ds in wanted:
        data_path, n = DATASETS[ds]
        for start, end in segments(n):
            tasks.append({"dataset": ds, "data_path": data_path, "start": start, "end": end})
    return tasks

def out_dir(task):
    return EXP / "full_rate1_draft_layout" / task["dataset"] / f"seg_{task['start']}_{task['end']}"

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
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"})
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
        "--rate", "1.0",
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
    log_path = LOGS / f"worker_{host}_gpu{gpu}.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"[{time.strftime('%F %T')}] start gpu={gpu} tasks={len(tasks)} root={ROOT}\n")
        for task in tasks:
            name = f"{task['dataset']} {task['start']}-{task['end']}"
            if done(task):
                log.write(f"[{time.strftime('%F %T')}] skip done {name}\n")
                continue
            while not gpu_free(gpu):
                log.write(f"[{time.strftime('%F %T')}] wait gpu={gpu} for {name}\n")
                time.sleep(60)
            log.write(f"[{time.strftime('%F %T')}] run {name}\n")
            rc = run_task(gpu, task)
            log.write(f"[{time.strftime('%F %T')}] done rc={rc} {name}\n")
        log.write(f"[{time.strftime('%F %T')}] complete gpu={gpu}\n")

def main():
    tasks = build_tasks()
    buckets = {gpu: [] for gpu in WORKER_GPUS}
    for i, task in enumerate(tasks):
        buckets[WORKER_GPUS[i % len(WORKER_GPUS)]].append(task)
    procs = []
    for gpu, bucket in buckets.items():
        p = Process(target=worker, args=(gpu, bucket), daemon=False)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

if __name__ == "__main__":
    main()
