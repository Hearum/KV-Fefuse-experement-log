#!/usr/bin/env python3
"""Run rate sweeps for attention alpha ablation configs from the micro figure.

This launcher intentionally reuses the canonical Qwen3-32B cache root in
readonly mode. It does not create worker-local cache directories.
"""
from __future__ import annotations

import os
import socket
import subprocess
import time
from multiprocessing import Process
from pathlib import Path

ROOT_CANDIDATES = [Path("/raid/home/hming/FusionRAG-pca-analysis"), Path("/home/hming/FusionRAG-pca-analysis")]
ROOT = next((p for p in ROOT_CANDIDATES if (p / "test_fusionrag_reflect_preprocess_exp.py").exists()), Path.cwd())

PYTHON = "/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python"
MODEL = "/mnt/qjhs-sh-lab-01/models/Qwen3-32B"
DRAFT = "/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct"
BGE = "/mnt/qjhs-sh-lab-01/models/bge-m3"
API_URL = "http://36.150.226.221:32355/v1"
API_KEY = "api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS"

EXP_ROOT = Path(os.environ.get(
    "FUSIONRAG_ALPHA_RATE_SWEEP_ROOT",
    "/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_alpha_config_rate_sweep_canonical_20260714",
))
CACHE_ROOT = Path(os.environ.get(
    "FUSIONRAG_ALPHA_RATE_SWEEP_CACHE_ROOT",
    "/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache",
))
LOG_DIR = EXP_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "2wikimqa": ("MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/2wikimqa_reflect.json", 200),
    "hotpotqa": ("MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/hotpotqa_reflect.json", 260),
    "triviaqa": ("MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/triviaqa_reflect.json", 270),
    "musique": ("data/result_reflect.json", 200),
}
SEGMENT = int(os.environ.get("FUSIONRAG_ALPHA_RATE_SWEEP_SEGMENT", "25"))
RATES = [x.strip() for x in os.environ.get("FUSIONRAG_ALPHA_RATE_SWEEP_RATES", "0.3,0.5,0.8,1.0").split(",") if x.strip()]
COMBOS = [tuple(x.strip().split(":", 1)) for x in os.environ.get(
    "FUSIONRAG_ALPHA_RATE_SWEEP_COMBOS",
    "uniform:0.1,uniform:0.25,random:0.05,random:0.1,random:0.25",
).split(",") if x.strip()]
WANTED_DATASETS = [x.strip() for x in os.environ.get(
    "FUSIONRAG_ALPHA_RATE_SWEEP_DATASETS",
    "2wikimqa,hotpotqa,triviaqa,musique",
).split(",") if x.strip()]
GPUS = [int(x) for x in os.environ.get("FUSIONRAG_ALPHA_RATE_SWEEP_GPUS", "0,1,2,3").split(",") if x.strip()]
FREE_MEM_MB = int(os.environ.get("FUSIONRAG_ALPHA_RATE_SWEEP_FREE_MEM_MB", "10000"))
ATTN_CHUNK = os.environ.get("FUSIONRAG_ALPHA_RATE_SWEEP_ATTENTION_CHUNK", "64")
CACHE_READONLY = os.environ.get("FUSIONRAG_ALPHA_RATE_SWEEP_CACHE_READONLY", "1").strip().lower() in ("1", "true", "yes")


def segments(total: int):
    start = 0
    while start < total:
        end = min(start + SEGMENT, total)
        yield start, end
        start = end


def combo_label(mode: str, alpha: str) -> str:
    return f"{mode}_alpha{alpha.replace('.', 'p')}"


def rate_label(rate: str) -> str:
    return rate.replace(".", "p")


def csv_name(rate: str) -> str:
    return f"rate_{rate}_draft_Qwen2.5-3B-Instruct_revert_rope.csv"


def build_tasks() -> list[dict]:
    tasks = []
    for mode, alpha in COMBOS:
        if mode not in ("uniform", "random"):
            raise ValueError(f"Unsupported attention ablation mode: {mode}")
        for rate in RATES:
            for dataset in WANTED_DATASETS:
                data_path, total = DATASETS[dataset]
                for start, end in segments(total):
                    tasks.append({"mode": mode, "alpha": alpha, "rate": rate, "dataset": dataset, "data_path": data_path, "start": start, "end": end})
    return tasks


def out_dir(task: dict) -> Path:
    return EXP_ROOT / combo_label(task["mode"], task["alpha"]) / f"rate{rate_label(task['rate'])}" / task["dataset"] / f"seg_{task['start']}_{task['end']}"


def done(task: dict) -> bool:
    d = out_dir(task)
    log = d / "run.log"
    if not log.exists():
        return False
    try:
        if "FINAL RESULTS" not in log.read_text(errors="ignore"):
            return False
    except OSError:
        return False
    return any(d.glob(f"**/{csv_name(task['rate'])}"))


def running(task: dict) -> bool:
    d = str(out_dir(task))
    proc = subprocess.run(["pgrep", "-af", d], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    if proc.returncode != 0:
        return False
    own = str(os.getpid())
    lines = [line for line in proc.stdout.splitlines() if own not in line]
    return bool(lines)


def gpu_free(gpu: int) -> bool:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    for line in proc.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(gpu):
            return int(parts[1]) < FREE_MEM_MB
    return False


def run_task(host: str, gpu: int, task: dict) -> int:
    d = out_dir(task)
    d.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONUNBUFFERED": "1",
        "FUSIONRAG_REPROCESS_ATTENTION_ABLATION": task["mode"],
        "FUSIONRAG_REPROCESS_ATTENTION_ABLATION_ALPHA": task["alpha"],
        "FUSIONRAG_REPROCESS_ATTENTION_CHUNK": ATTN_CHUNK,
    })
    if CACHE_READONLY:
        env["FUSIONRAG_PREPROCESS_CACHE_READONLY"] = "1"
    cmd = [
        PYTHON,
        "test_fusionrag_reflect_preprocess_exp.py",
        "--model_type", "qwen3",
        "--model_path", MODEL,
        "--model_name", "Qwen3-32B",
        "--data_path", task["data_path"],
        "--dataset_name", task["dataset"],
        "--cache_path", str(CACHE_ROOT),
        "--result_path", str(d),
        "--start_sample", str(task["start"]),
        "--end_sample", str(task["end"]),
        "--rate", task["rate"],
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
    (d / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    with (d / "run.log").open("w", encoding="utf-8") as log:
        return subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def worker(gpu: int, tasks: list[dict]) -> None:
    host = socket.gethostname().split(".")[0]
    log_path = LOG_DIR / f"alpha_rate_sweep_worker_{host}_gpu{gpu}.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"[{time.strftime('%F %T')}] start host={host} gpu={gpu} tasks={len(tasks)} root={ROOT}\n")
        log.write(f"[{time.strftime('%F %T')}] exp_root={EXP_ROOT} cache_root={CACHE_ROOT} cache_readonly={CACHE_READONLY} rates={RATES} combos={COMBOS} datasets={WANTED_DATASETS} attn_chunk={ATTN_CHUNK} shard={os.environ.get('FUSIONRAG_ALPHA_RATE_SWEEP_SHARD_INDEX', '0')}/{os.environ.get('FUSIONRAG_ALPHA_RATE_SWEEP_SHARD_COUNT', '1')}\n")
        for task in tasks:
            name = f"{task['dataset']} {combo_label(task['mode'], task['alpha'])} rate={task['rate']} {task['start']}-{task['end']}"
            if running(task) and not done(task):
                log.write(f"[{time.strftime('%F %T')}] skip running {name}\n")
                continue
            if done(task):
                log.write(f"[{time.strftime('%F %T')}] skip done {name}\n")
                continue
            while not gpu_free(gpu):
                log.write(f"[{time.strftime('%F %T')}] wait gpu={gpu} {name}\n")
                time.sleep(60)
            log.write(f"[{time.strftime('%F %T')}] run {name}\n")
            rc = run_task(host, gpu, task)
            log.write(f"[{time.strftime('%F %T')}] done rc={rc} {name}\n")
        log.write(f"[{time.strftime('%F %T')}] complete host={host} gpu={gpu}\n")


def main() -> None:
    tasks = build_tasks()
    shard_count = int(os.environ.get("FUSIONRAG_ALPHA_RATE_SWEEP_SHARD_COUNT", "1"))
    shard_index = int(os.environ.get("FUSIONRAG_ALPHA_RATE_SWEEP_SHARD_INDEX", "0"))
    if shard_count < 1:
        raise ValueError(f"Invalid shard_count={shard_count}")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard_index={shard_index} for shard_count={shard_count}")
    if shard_count > 1:
        tasks = [task for idx, task in enumerate(tasks) if idx % shard_count == shard_index]
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
