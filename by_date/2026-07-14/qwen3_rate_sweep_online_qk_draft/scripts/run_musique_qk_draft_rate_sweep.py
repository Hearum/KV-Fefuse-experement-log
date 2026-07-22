#!/usr/bin/env python3
"""Launch Qwen3-32B MuSiQue Online QK/DraftModel rate sweep.

Rates and methods are intentionally explicit so the experiment can be resumed
without changing the main FusionRAG pipeline.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from multiprocessing import Process
from pathlib import Path


ROOT_CANDIDATES = [Path("/raid/home/hming/FusionRAG-pca-analysis"), Path("/home/hming/FusionRAG-pca-analysis")]
ROOT = next((p for p in ROOT_CANDIDATES if (p / "test_fusionrag_reflect_preprocess_exp.py").exists()), ROOT_CANDIDATES[0])
PYTHON = "/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python"
MODEL = "/mnt/qjhs-sh-lab-01/models/Qwen3-32B"
DRAFT = "/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct"
BGE = "/mnt/qjhs-sh-lab-01/models/bge-m3"
API_URL = "http://36.150.226.221:32355/v1"
API_KEY = "api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS"

EXP_ROOT = Path(os.environ.get(
    "FUSIONRAG_QWEN3_RATE_SWEEP_ROOT",
    "/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate_sweep_20260714",
))
CACHE_ROOT = Path(os.environ.get(
    "FUSIONRAG_QWEN3_RATE_SWEEP_CACHE_ROOT",
    "/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache",
))
CACHE_LAYOUT = os.environ.get("FUSIONRAG_QWEN3_RATE_SWEEP_CACHE_LAYOUT", "shared").strip().lower()
CACHE_READONLY = os.environ.get("FUSIONRAG_QWEN3_RATE_SWEEP_CACHE_READONLY", "0").strip().lower()
LOG_DIR = EXP_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "musique"
DATA_PATH = "data/result_reflect.json"
TOTAL = 200
SEGMENT = 25
RATES = [x.strip() for x in os.environ.get("FUSIONRAG_QWEN3_RATE_SWEEP_RATES", "0.0,0.1,0.3,0.5,0.8,0.9").split(",") if x.strip()]
METHODS = [x.strip() for x in os.environ.get("FUSIONRAG_QWEN3_RATE_SWEEP_METHODS", "online_qk,online_draft").split(",") if x.strip()]
GPUS = [int(x) for x in os.environ.get("FUSIONRAG_QWEN3_RATE_SWEEP_GPUS", "0,1,2,3,4,5,6,7").split(",") if x.strip()]


def segments():
    start = 0
    while start < TOTAL:
        end = min(start + SEGMENT, TOTAL)
        yield start, end
        start = end


def method_cfg(method: str) -> dict:
    if method == "online_qk":
        return {"reprocess_method": "FusionRAG", "subdir": "FusionRAG_global_topk10_bge", "extra": []}
    if method == "online_draft":
        return {
            "reprocess_method": "DraftModel",
            "subdir": "DraftModel_global_topk10_bge",
            "extra": ["--draft_model_path", DRAFT],
        }
    raise ValueError(f"Unknown method: {method}")


def csv_name(method: str, rate: str) -> str:
    if method == "online_qk":
        return f"rate_{rate}_revert_rope.csv"
    if method == "online_draft":
        return f"rate_{rate}_draft_Qwen2.5-3B-Instruct_revert_rope.csv"
    raise ValueError(method)


def build_tasks():
    tasks = []
    for rate in RATES:
        for method in METHODS:
            method_cfg(method)
            for start, end in segments():
                tasks.append({"rate": rate, "method": method, "start": start, "end": end})
    shard_count = int(os.environ.get("FUSIONRAG_QWEN3_RATE_SWEEP_SHARD_COUNT", "1") or "1")
    shard_index = int(os.environ.get("FUSIONRAG_QWEN3_RATE_SWEEP_SHARD_INDEX", "0") or "0")
    if shard_count > 1:
        tasks = [task for idx, task in enumerate(tasks) if idx % shard_count == shard_index]
    return tasks


def out_dir(task: dict) -> Path:
    rate_label = task["rate"].replace(".", "p")
    return EXP_ROOT / f"{task['method']}_rate{rate_label}" / DATASET / f"seg_{task['start']}_{task['end']}"


def done(task: dict) -> bool:
    directory = out_dir(task)
    log = directory / "run.log"
    if not log.exists():
        return False
    try:
        if "FINAL RESULTS" not in log.read_text(errors="ignore"):
            return False
    except OSError:
        return False
    return any(directory.glob(f"**/{csv_name(task['method'], task['rate'])}"))


def gpu_free(gpu: int) -> bool:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    threshold = int(os.environ.get("FUSIONRAG_QWEN3_RATE_SWEEP_FREE_MEM_MB", "10000"))
    for line in proc.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(gpu):
            return int(parts[1]) < threshold
    return False


def cache_path(host: str, gpu: int) -> Path:
    if CACHE_LAYOUT == "shared":
        # test_fusionrag_reflect_preprocess_exp.py appends model_name/dataset_name
        # under --cache_path, so the shared semantic root must not include worker,
        # model, or dataset components.
        return CACHE_ROOT
    if CACHE_LAYOUT == "worker":
        return CACHE_ROOT / host / f"worker_gpu{gpu}" / DATASET
    raise ValueError(f"Unknown cache layout: {CACHE_LAYOUT}")


def run_task(host: str, gpu: int, task: dict) -> int:
    directory = out_dir(task)
    directory.mkdir(parents=True, exist_ok=True)
    cfg = method_cfg(task["method"])
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"})
    if CACHE_READONLY in ("1", "true", "yes"):
        env["FUSIONRAG_PREPROCESS_CACHE_READONLY"] = "1"
    else:
        env.pop("FUSIONRAG_PREPROCESS_CACHE_READONLY", None)
    cmd = [
        PYTHON,
        "test_fusionrag_reflect_preprocess_exp.py",
        "--model_type", "qwen3",
        "--model_path", MODEL,
        "--model_name", "Qwen3-32B",
        "--data_path", DATA_PATH,
        "--dataset_name", DATASET,
        "--cache_path", str(cache_path(host, gpu)),
        "--result_path", str(directory),
        "--start_sample", str(task["start"]),
        "--end_sample", str(task["end"]),
        "--rate", task["rate"],
        "--topk", "10",
        "--preprocess", "true",
        "--recall_method", "bge",
        "--reprocess_method", cfg["reprocess_method"],
        "--revert_rope", "true",
        "--preprocess_scope", "global",
        "--bge_model_path", BGE,
        "--device", "cuda:0",
        "--use_multi_gpu", "false",
        "--openai_base_url", API_URL,
        "--openai_api_key", API_KEY,
        "--openai_model", "GLM-5.2",
        *cfg["extra"],
    ]
    (directory / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        return subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def worker(gpu: int, tasks: list[dict]) -> None:
    host = socket.gethostname().split(".")[0]
    log_path = LOG_DIR / f"worker_{host}_gpu{gpu}.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"[{time.strftime('%F %T')}] start host={host} gpu={gpu} tasks={len(tasks)} root={ROOT}\n")
        log.write(f"[{time.strftime('%F %T')}] exp_root={EXP_ROOT} cache_root={CACHE_ROOT} cache_layout={CACHE_LAYOUT} cache_readonly={CACHE_READONLY}\n")
        for task in tasks:
            label = f"{task['method']} rate={task['rate']} {task['start']}-{task['end']}"
            if done(task):
                log.write(f"[{time.strftime('%F %T')}] skip done {label}\n")
                continue
            while not gpu_free(gpu):
                log.write(f"[{time.strftime('%F %T')}] wait gpu={gpu} {label}\n")
                time.sleep(60)
            log.write(f"[{time.strftime('%F %T')}] run {label}\n")
            rc = run_task(host, gpu, task)
            log.write(f"[{time.strftime('%F %T')}] done rc={rc} {label}\n")
        log.write(f"[{time.strftime('%F %T')}] complete host={host} gpu={gpu}\n")


def main() -> None:
    tasks = build_tasks()
    buckets = {gpu: [] for gpu in GPUS}
    for idx, task in enumerate(tasks):
        buckets[GPUS[idx % len(GPUS)]].append(task)
    procs = []
    for gpu, bucket in buckets.items():
        proc = Process(target=worker, args=(gpu, bucket), daemon=False)
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()
