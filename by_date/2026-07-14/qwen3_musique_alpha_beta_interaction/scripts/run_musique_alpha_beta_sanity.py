#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import socket
import subprocess
import time
from multiprocessing import Process
from pathlib import Path

ROOT_CANDIDATES = [Path('/raid/home/hming/FusionRAG-pca-analysis'), Path('/home/hming/FusionRAG-pca-analysis')]
ROOT = next((p for p in ROOT_CANDIDATES if (p / 'test_fusionrag_reflect_preprocess_exp.py').exists()), Path.cwd())

PYTHON = '/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python'
MODEL = '/mnt/qjhs-sh-lab-01/models/Qwen3-32B'
DRAFT = '/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct'
BGE = '/mnt/qjhs-sh-lab-01/models/bge-m3'
API_URL = 'http://36.150.226.221:32355/v1'
API_KEY = 'api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS'

EXP_ROOT = Path(os.environ.get('FUSIONRAG_MUSIQUE_ALPHA_BETA_ROOT', '/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_alpha_beta_interaction_20260714'))
CACHE_ROOT = Path(os.environ.get('FUSIONRAG_MUSIQUE_ALPHA_BETA_CACHE_ROOT', '/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache'))
GPUS = [int(x) for x in os.environ.get('FUSIONRAG_MUSIQUE_ALPHA_BETA_GPUS', '4,5,6').split(',') if x.strip()]
START = int(os.environ.get('FUSIONRAG_MUSIQUE_ALPHA_BETA_START', '0'))
END = int(os.environ.get('FUSIONRAG_MUSIQUE_ALPHA_BETA_END', '50'))
FREE_MEM_MB = int(os.environ.get('FUSIONRAG_MUSIQUE_ALPHA_BETA_FREE_MEM_MB', '10000'))

TASKS = [
    {'label': 'native_beta0', 'mode': '', 'alpha': '', 'beta': '0'},
    {'label': 'native_beta0p5', 'mode': '', 'alpha': '', 'beta': '0.5'},
    {'label': 'native_beta1', 'mode': '', 'alpha': '', 'beta': '1'},
    {'label': 'uniform_a0p1_b0', 'mode': 'uniform', 'alpha': '0.1', 'beta': '0'},
    {'label': 'uniform_a0p1_b0p5', 'mode': 'uniform', 'alpha': '0.1', 'beta': '0.5'},
    {'label': 'uniform_a0p25_b0', 'mode': 'uniform', 'alpha': '0.25', 'beta': '0'},
    {'label': 'uniform_a0p25_b0p5', 'mode': 'uniform', 'alpha': '0.25', 'beta': '0.5'},
    {'label': 'random_a0p1_b0', 'mode': 'random', 'alpha': '0.1', 'beta': '0'},
    {'label': 'random_a0p1_b0p5', 'mode': 'random', 'alpha': '0.1', 'beta': '0.5'},
]

CSV_NAME = 'rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv'


def out_dir(task: dict) -> Path:
    return EXP_ROOT / 'results' / f"{task['label']}_{START}_{END}"


def done(task: dict) -> bool:
    d = out_dir(task)
    log = d / 'run.log'
    if not log.exists():
        return False
    if 'FINAL RESULTS' not in log.read_text(errors='ignore'):
        return False
    return any(d.glob(f'**/{CSV_NAME}'))


def running(task: dict) -> bool:
    proc = subprocess.run(['pgrep', '-af', str(out_dir(task))], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return proc.returncode == 0


def gpu_free(gpu: int) -> bool:
    proc = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    for line in proc.stdout.splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) >= 2 and parts[0] == str(gpu):
            return int(parts[1]) < FREE_MEM_MB
    return False


def run_task(gpu: int, task: dict) -> int:
    d = out_dir(task)
    d.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES': str(gpu),
        'PYTHONUNBUFFERED': '1',
        'PYTHONDONTWRITEBYTECODE': '1',
        'FUSIONRAG_STRICT_BETA_QUERY_PREFILL': '1',
        'FUSIONRAG_REPROCESS_KV_BLEND_BETA': task['beta'],
        'FUSIONRAG_REPROCESS_KV_BLEND_MODE': 'kv',
        'FUSIONRAG_PREPROCESS_CACHE_READONLY': '1',
        'FUSIONRAG_REPROCESS_ATTENTION_CHUNK': '64',
    })
    if task['mode']:
        env['FUSIONRAG_REPROCESS_ATTENTION_ABLATION'] = task['mode']
        env['FUSIONRAG_REPROCESS_ATTENTION_ABLATION_ALPHA'] = task['alpha']
    cmd = [
        PYTHON, 'test_fusionrag_reflect_preprocess_exp.py',
        '--model_type', 'qwen3',
        '--model_path', MODEL,
        '--model_name', 'Qwen3-32B',
        '--data_path', 'data/result_reflect.json',
        '--dataset_name', 'musique',
        '--cache_path', str(CACHE_ROOT),
        '--result_path', str(d),
        '--start_sample', str(START),
        '--end_sample', str(END),
        '--rate', '0.15',
        '--topk', '10',
        '--preprocess', 'true',
        '--recall_method', 'bge',
        '--reprocess_method', 'DraftModel',
        '--draft_model_path', DRAFT,
        '--revert_rope', 'true',
        '--preprocess_scope', 'global',
        '--bge_model_path', BGE,
        '--device', 'cuda:0',
        '--use_multi_gpu', 'false',
        '--openai_base_url', API_URL,
        '--openai_api_key', API_KEY,
        '--openai_model', 'GLM-5.2',
    ]
    with (d / 'task_meta.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['host', 'gpu', 'label', 'mode', 'alpha', 'beta', 'start', 'end', 'cache_root', 'strict_query_prefill', 'command']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            **task,
            'host': socket.gethostname().split('.')[0],
            'gpu': gpu,
            'start': START,
            'end': END,
            'cache_root': str(CACHE_ROOT),
            'strict_query_prefill': '1',
            'command': ' '.join(cmd),
        })
    (d / 'command.txt').write_text(' '.join(cmd) + '\n', encoding='utf-8')
    with (d / 'run.log').open('w', encoding='utf-8') as log:
        return subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def worker(gpu: int, tasks: list[dict]) -> None:
    log_dir = EXP_ROOT / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname().split('.')[0]
    with (log_dir / f'worker_{host}_gpu{gpu}.log').open('a', encoding='utf-8', buffering=1) as log:
        log.write(f'[{time.strftime("%F %T")}] start gpu={gpu} tasks={len(tasks)} range={START}-{END}\n')
        for task in tasks:
            name = task['label']
            while running(task) and not done(task):
                log.write(f'[{time.strftime("%F %T")}] wait existing {name}\n')
                time.sleep(60)
            if done(task):
                log.write(f'[{time.strftime("%F %T")}] skip done {name}\n')
                continue
            while not gpu_free(gpu):
                log.write(f'[{time.strftime("%F %T")}] wait gpu {gpu} for {name}\n')
                time.sleep(60)
            log.write(f'[{time.strftime("%F %T")}] run {name}\n')
            rc = run_task(gpu, task)
            log.write(f'[{time.strftime("%F %T")}] done rc={rc} {name}\n')
        log.write(f'[{time.strftime("%F %T")}] complete gpu={gpu}\n')


def main() -> None:
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    buckets = {gpu: [] for gpu in GPUS}
    for idx, task in enumerate(TASKS):
        buckets[GPUS[idx % len(GPUS)]].append(task)
    procs = []
    for gpu, tasks in buckets.items():
        proc = Process(target=worker, args=(gpu, tasks), daemon=False)
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()


if __name__ == '__main__':
    main()

