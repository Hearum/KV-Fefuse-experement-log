#!/usr/bin/env python3
import csv
import os
import socket
import subprocess
import time
from multiprocessing import Process
from pathlib import Path

ROOT = Path.cwd()
EXP_ROOT = Path(os.environ.get("FUSIONRAG_KV_BLEND_FULL_EXP_ROOT", "/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_full_sweep"))
RESULTS = EXP_ROOT / "results_full"
LOGS = EXP_ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

PYTHON = "/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python"
MODEL = "/mnt/qjhs-sh-lab-01/models/Qwen3-32B"
DRAFT = "/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct"
BGE = "/mnt/qjhs-sh-lab-01/models/bge-m3"
API_URL = "http://36.150.226.221:32355/v1"
API_KEY = "api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS"
CSV_NAME = "rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv"
BETAS = [x.strip() for x in os.environ.get("FUSIONRAG_KV_BLEND_BETAS", "0,0.25,0.5,0.75,1.0").split(",") if x.strip()]
DATASET_FILTER = {x.strip() for x in os.environ.get("FUSIONRAG_KV_BLEND_DATASETS", "").split(",") if x.strip()}
WORKER_GPUS = [int(x) for x in os.environ.get("FUSIONRAG_KV_BLEND_GPUS", "0,1,2,3,4,5,6,7").split(",") if x.strip()]
SHARD_ID = int(os.environ.get("FUSIONRAG_KV_BLEND_SHARD_ID", "0"))
SHARD_COUNT = int(os.environ.get("FUSIONRAG_KV_BLEND_SHARD_COUNT", "1"))
STEP = int(os.environ.get("FUSIONRAG_KV_BLEND_SEGMENT_SIZE", "25"))
STRICT_QUERY_PREFILL = os.environ.get("FUSIONRAG_KV_BLEND_STRICT_QUERY_PREFILL", "0")
HOST = socket.gethostname().split(".")[0]
OLD_CROSS_CACHE = Path("/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-attn-alpha-cross-cache-v2")
FULL_CACHE = Path("/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-kv-blend-beta-full-cache")
SHARED_MUSIQUE_CACHE = os.environ.get("FUSIONRAG_KV_BLEND_SHARED_MUSIQUE_CACHE", "")

DATASETS = [
    {"name": "2wikimqa", "path": "MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/2wikimqa_reflect.json", "end": 200},
    {"name": "hotpotqa", "path": "MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/hotpotqa_reflect.json", "end": 260},
    {"name": "triviaqa", "path": "MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/triviaqa_reflect.json", "end": 270},
    {"name": "musique", "path": "data/result_reflect.json", "end": 135},
]


def beta_label(beta: str) -> str:
    return beta.replace(".", "p")


def segments(end):
    cur = 0
    while cur < end:
        nxt = min(cur + STEP, end)
        yield cur, nxt
        cur = nxt


def build_tasks():
    tasks = []
    for beta in BETAS:
        for ds in DATASETS:
            if DATASET_FILTER and ds["name"] not in DATASET_FILTER:
                continue
            for start, end in segments(ds["end"]):
                tasks.append({"beta": beta, "dataset": ds["name"], "data_path": ds["path"], "start": start, "end": end})
    return [t for i, t in enumerate(tasks) if i % SHARD_COUNT == SHARD_ID]


def out_dir(task):
    return RESULTS / f"beta{beta_label(task['beta'])}_kv" / task["dataset"] / f"seg_{task['start']}_{task['end']}"


def csv_path(task):
    return next(out_dir(task).glob(f"**/{CSV_NAME}"), None)


def done(task):
    log = out_dir(task) / "run.log"
    if not log.exists():
        return False
    if "FINAL RESULTS" not in log.read_text(errors="ignore"):
        return False
    return csv_path(task) is not None


def running(task):
    proc = subprocess.run(["pgrep", "-af", str(out_dir(task))], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
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


def cache_path(task, gpu):
    ds = task["dataset"]
    if ds == "musique" and SHARED_MUSIQUE_CACHE:
        return Path(SHARED_MUSIQUE_CACHE)
    if ds != "musique":
        old = OLD_CROSS_CACHE / f"worker_gpu{gpu}" / ds
        if old.exists():
            return old
    return FULL_CACHE / f"{HOST}_gpu{gpu}" / ds


def run_task(gpu, task):
    d = out_dir(task)
    d.mkdir(parents=True, exist_ok=True)
    c = cache_path(task, gpu)
    c.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "FUSIONRAG_REPROCESS_KV_BLEND_BETA": task["beta"],
            "FUSIONRAG_REPROCESS_KV_BLEND_MODE": "kv",
            "FUSIONRAG_STRICT_BETA_QUERY_PREFILL": STRICT_QUERY_PREFILL,
        }
    )
    cmd = [
        PYTHON,
        "test_fusionrag_reflect_preprocess_exp.py",
        "--model_type",
        "qwen3",
        "--model_path",
        MODEL,
        "--model_name",
        "Qwen3-32B",
        "--data_path",
        task["data_path"],
        "--dataset_name",
        task["dataset"],
        "--cache_path",
        str(c),
        "--result_path",
        str(d),
        "--start_sample",
        str(task["start"]),
        "--end_sample",
        str(task["end"]),
        "--rate",
        "0.15",
        "--topk",
        "10",
        "--preprocess",
        "true",
        "--recall_method",
        "bge",
        "--reprocess_method",
        "DraftModel",
        "--draft_model_path",
        DRAFT,
        "--revert_rope",
        "true",
        "--preprocess_scope",
        "global",
        "--bge_model_path",
        BGE,
        "--device",
        "cuda:0",
        "--use_multi_gpu",
        "false",
        "--openai_base_url",
        API_URL,
        "--openai_api_key",
        API_KEY,
        "--openai_model",
        "GLM-5.2",
    ]
    with (d / "task_meta.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["host", "gpu", "beta", "dataset", "data_path", "start", "end", "cache_path", "result_path", "strict_query_prefill"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({**task, "host": HOST, "gpu": gpu, "cache_path": str(c), "result_path": str(d), "strict_query_prefill": STRICT_QUERY_PREFILL})
    with (d / "run.log").open("w", encoding="utf-8") as f:
        return subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def worker(gpu, tasks):
    log_path = LOGS / f"worker_{HOST}_gpu{gpu}.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(
            f"[{time.strftime('%F %T')}] worker start host={HOST} gpu={gpu} "
            f"tasks={len(tasks)} shard={SHARD_ID}/{SHARD_COUNT} root={ROOT} exp_root={EXP_ROOT}\n"
        )
        for task in tasks:
            name = f"beta={task['beta']} {task['dataset']} {task['start']}-{task['end']}"
            while running(task) and not done(task):
                log.write(f"[{time.strftime('%F %T')}] wait existing {name}\n")
                time.sleep(60)
            if done(task):
                log.write(f"[{time.strftime('%F %T')}] skip done {name}\n")
                continue
            while not gpu_free(gpu):
                log.write(f"[{time.strftime('%F %T')}] wait gpu {gpu} for {name}\n")
                time.sleep(60)
            log.write(f"[{time.strftime('%F %T')}] run {name}\n")
            rc = run_task(gpu, task)
            log.write(f"[{time.strftime('%F %T')}] done rc={rc} {name}\n")
        log.write(f"[{time.strftime('%F %T')}] worker complete host={HOST} gpu={gpu}\n")


def main():
    if not WORKER_GPUS:
        raise SystemExit("No worker GPUs configured")
    tasks = build_tasks()
    buckets = {gpu: [] for gpu in WORKER_GPUS}
    for i, task in enumerate(tasks):
        buckets[WORKER_GPUS[i % len(WORKER_GPUS)]].append(task)
    manifest = EXP_ROOT / f"manifest_{HOST}_shard{SHARD_ID}_of_{SHARD_COUNT}.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        fields = ["beta", "dataset", "data_path", "start", "end", "assigned_gpu"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for gpu, items in buckets.items():
            for task in items:
                writer.writerow({**task, "assigned_gpu": gpu})
    procs = []
    for gpu, items in buckets.items():
        p = Process(target=worker, args=(gpu, items), daemon=False)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
