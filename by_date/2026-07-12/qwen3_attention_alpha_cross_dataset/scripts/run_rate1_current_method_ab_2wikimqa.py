#!/usr/bin/env python3
import os
import socket
import subprocess
import time
from multiprocessing import Process
from pathlib import Path

ROOT = Path.cwd()
EXP = Path(os.environ.get('FUSIONRAG_RATE1_AB_ROOT', '/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_rate1_current_method_ab'))
LOGS = EXP / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)
PYTHON = '/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python'
MODEL = '/mnt/qjhs-sh-lab-01/models/Qwen3-32B'
DRAFT = '/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct'
BGE = '/mnt/qjhs-sh-lab-01/models/bge-m3'
API_URL = 'http://36.150.226.221:32355/v1'
API_KEY = 'api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS'
CACHE_ROOT = Path(os.environ.get('FUSIONRAG_RATE1_AB_CACHE_ROOT', '/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-attn-alpha-cross-cache-v2'))
GPUS = [int(x) for x in os.environ.get('FUSIONRAG_RATE1_AB_GPUS', '0,1,2,4,5,6,7').split(',') if x.strip()]
DATA_PATH = 'MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/2wikimqa_reflect.json'
DATASET = '2wikimqa'
METHODS = [x.strip() for x in os.environ.get('FUSIONRAG_RATE1_AB_METHODS', 'FusionRAG,DraftModel').split(',') if x.strip()]

def segments(n=200, step=25):
    return [(s, min(s + step, n)) for s in range(0, n, step)]

def csv_name(method):
    return 'rate_1.0_draft_Qwen2.5-3B-Instruct_revert_rope.csv' if method == 'DraftModel' else 'rate_1.0_revert_rope.csv'

def build_tasks():
    out = []
    for method in METHODS:
        for start, end in segments():
            out.append({'method': method, 'start': start, 'end': end})
    return out

def out_dir(task):
    return EXP / task['method'] / DATASET / f"seg_{task['start']}_{task['end']}"

def done(task):
    d = out_dir(task)
    log = d / 'run.log'
    if not log.exists():
        return False
    try:
        if 'FINAL RESULTS' not in log.read_text(errors='ignore'):
            return False
    except Exception:
        return False
    return any(d.glob(f"**/{csv_name(task['method'])}"))

def gpu_free(gpu):
    proc = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for line in proc.stdout.splitlines():
        a = [x.strip() for x in line.split(',')]
        if len(a) >= 2 and a[0] == str(gpu):
            return int(a[1]) < 10000
    return False

def run_task(gpu, task):
    d = out_dir(task)
    d.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED': '1', 'FUSIONRAG_PREPROCESS_CACHE_READONLY': '1'})
    cmd = [
        PYTHON, 'test_fusionrag_reflect_preprocess_exp.py',
        '--model_type', 'qwen3', '--model_path', MODEL, '--model_name', 'Qwen3-32B',
        '--data_path', DATA_PATH, '--dataset_name', DATASET,
        '--cache_path', str(CACHE_ROOT / f"worker_gpu{gpu}" / DATASET),
        '--result_path', str(d),
        '--start_sample', str(task['start']), '--end_sample', str(task['end']),
        '--rate', '1.0', '--topk', '10', '--preprocess', 'true', '--recall_method', 'bge',
        '--reprocess_method', task['method'], '--revert_rope', 'true', '--preprocess_scope', 'global',
        '--bge_model_path', BGE, '--device', 'cuda:0', '--use_multi_gpu', 'false',
        '--openai_base_url', API_URL, '--openai_api_key', API_KEY, '--openai_model', 'GLM-5.2',
    ]
    if task['method'] == 'DraftModel':
        cmd.extend(['--draft_model_path', DRAFT])
    with (d / 'run.log').open('w', encoding='utf-8') as f:
        return subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode

def worker(gpu, tasks):
    host = socket.gethostname().split('.')[0]
    with (LOGS / f'worker_{host}_gpu{gpu}.log').open('a', encoding='utf-8', buffering=1) as log:
        log.write(f"[{time.strftime('%F %T')}] start gpu={gpu} tasks={len(tasks)}\n")
        for task in tasks:
            label = f"{task['method']} {DATASET} {task['start']}-{task['end']}"
            if done(task):
                log.write(f"[{time.strftime('%F %T')}] skip done {label}\n")
                continue
            while not gpu_free(gpu):
                log.write(f"[{time.strftime('%F %T')}] wait gpu={gpu} for {label}\n")
                time.sleep(60)
            log.write(f"[{time.strftime('%F %T')}] run {label}\n")
            rc = run_task(gpu, task)
            log.write(f"[{time.strftime('%F %T')}] done rc={rc} {label}\n")
        log.write(f"[{time.strftime('%F %T')}] complete gpu={gpu}\n")

def main():
    tasks = build_tasks()
    buckets = {g: [] for g in GPUS}
    for i, task in enumerate(tasks):
        buckets[GPUS[i % len(GPUS)]].append(task)
    procs = []
    for gpu, bucket in buckets.items():
        p = Process(target=worker, args=(gpu, bucket), daemon=False)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

if __name__ == '__main__':
    main()
