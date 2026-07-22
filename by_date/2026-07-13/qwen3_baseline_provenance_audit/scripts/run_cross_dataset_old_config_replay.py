#!/usr/bin/env python3
import os
import subprocess
import time
from multiprocessing import Process
from pathlib import Path

ROOT = Path('/raid/home/hming/FusionRAG-pca-analysis')
EXP = Path('/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_baseline_provenance_audit/cross_dataset_old_config_replay_current_head')
PYTHON = '/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python'
MODEL = '/mnt/qjhs-sh-lab-01/models/Qwen3-32B'
DRAFT = '/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct'
BGE = '/mnt/qjhs-sh-lab-01/models/bge-m3'
CACHE = Path('/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-attn-alpha-cross-cache-v2')
API_URL = os.environ.get('API_URL', 'http://36.150.226.221:32355/v1')
API_KEY = os.environ.get('API_KEY', 'api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS')
GPUS = [int(x) for x in os.environ.get('FUSIONRAG_REPLAY_GPUS', '0,1,2,3,4,5,6,7').split(',') if x.strip()]
DATASETS = [('2wikimqa', 200), ('hotpotqa', 260), ('triviaqa', 270)]
CONFIGS = ['online_qk_rate015', 'online_draft_rate015']

def segments(n, step=25):
    s = 0
    while s < n:
        e = min(s + step, n)
        yield s, e
        s = e

def build_tasks():
    tasks = []
    idx = 0
    for dataset, n in DATASETS:
        for cfg in CONFIGS:
            for start, end in segments(n):
                tasks.append({'idx': idx, 'dataset': dataset, 'cfg': cfg, 'start': start, 'end': end})
                idx += 1
    return tasks

def csv_name(task):
    if task['cfg'] == 'online_draft_rate015':
        return 'rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv'
    return 'rate_0.15_revert_rope.csv'

def out_dir(task):
    return EXP / 'results' / task['cfg'] / task['dataset'] / f"seg_{task['start']}_{task['end']}"

def done(task):
    d = out_dir(task)
    log = d / 'run.log'
    return log.exists() and 'FINAL RESULTS' in log.read_text(errors='ignore') and any(d.glob(f"**/{csv_name(task)}"))

def run_task(gpu, task):
    d = out_dir(task)
    d.mkdir(parents=True, exist_ok=True)
    if done(task):
        return
    method = 'DraftModel' if task['cfg'] == 'online_draft_rate015' else 'FusionRAG'
    cmd = [
        PYTHON, 'test_fusionrag_reflect_preprocess_exp.py',
        '--model_type', 'qwen3',
        '--model_path', MODEL,
        '--model_name', 'Qwen3-32B',
        '--data_path', str(ROOT / 'MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data' / f"{task['dataset']}_reflect.json"),
        '--dataset_name', task['dataset'],
        '--cache_path', str(CACHE / f"worker_gpu{gpu}" / task['dataset']),
        '--result_path', str(d),
        '--start_sample', str(task['start']),
        '--end_sample', str(task['end']),
        '--rate', '0.15',
        '--topk', '10',
        '--preprocess', 'true',
        '--recall_method', 'bge',
        '--reprocess_method', method,
        '--revert_rope', 'true',
        '--preprocess_scope', 'global',
        '--bge_model_path', BGE,
        '--device', 'cuda:0',
        '--use_multi_gpu', 'false',
        '--openai_base_url', API_URL,
        '--openai_api_key', API_KEY,
        '--openai_model', 'GLM-5.2',
    ]
    if method == 'DraftModel':
        cmd.extend(['--draft_model_path', DRAFT])
    env = os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES': str(gpu),
        'PYTHONUNBUFFERED': '1',
        'FUSIONRAG_PREPROCESS_CACHE_READONLY': '1',
    })
    (d / 'command.txt').write_text(' '.join(cmd) + '\n', encoding='utf-8')
    with (d / 'run.log').open('w', encoding='utf-8') as f:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    (d / 'returncode.txt').write_text(str(rc) + '\n', encoding='utf-8')

def worker(gpu, tasks):
    log_dir = EXP / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f'worker_gpu{gpu}.log').open('a', encoding='utf-8', buffering=1) as log:
        log.write(f"[{time.strftime('%F %T')}] start gpu={gpu}\n")
        for task in tasks:
            if task['idx'] % len(GPUS) != GPUS.index(gpu):
                continue
            label = f"{task['cfg']} {task['dataset']} {task['start']}-{task['end']}"
            log.write(f"[{time.strftime('%F %T')}] run {label}\n")
            run_task(gpu, task)
            log.write(f"[{time.strftime('%F %T')}] done {label}\n")
        log.write(f"[{time.strftime('%F %T')}] complete gpu={gpu}\n")

def main():
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / 'versions.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True), encoding='utf-8')
    tasks = build_tasks()
    procs = []
    for gpu in GPUS:
        p = Process(target=worker, args=(gpu, tasks), daemon=False)
        p.start(); procs.append(p)
    for p in procs:
        p.join()

if __name__ == '__main__':
    main()
