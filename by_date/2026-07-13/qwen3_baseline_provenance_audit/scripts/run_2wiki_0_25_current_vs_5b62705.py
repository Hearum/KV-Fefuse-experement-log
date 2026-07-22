#!/usr/bin/env python3
import os
import subprocess
import time
from multiprocessing import Process
from pathlib import Path

CURRENT_REPO = Path('/raid/home/hming/FusionRAG-pca-analysis')
OLD_COMMIT = '5b62705'
OLD_REPO = Path('/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag_worktrees/fusionrag_5b62705')
EXP = Path('/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_baseline_provenance_audit/2wiki_0_25_current_vs_5b62705')
PYTHON = '/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python'
MODEL = '/mnt/qjhs-sh-lab-01/models/Qwen3-32B'
DRAFT = '/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct'
BGE = '/mnt/qjhs-sh-lab-01/models/bge-m3'
DATA = str(CURRENT_REPO / 'MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/2wikimqa_reflect.json')
API_URL = 'http://36.150.226.221:32355/v1'
API_KEY = 'api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS'

TASKS = [
    {'label': 'current_qk', 'repo': CURRENT_REPO, 'gpu': 0, 'method': 'FusionRAG'},
    {'label': 'current_draft', 'repo': CURRENT_REPO, 'gpu': 1, 'method': 'DraftModel'},
    {'label': 'old5b_qk', 'repo': OLD_REPO, 'gpu': 2, 'method': 'FusionRAG'},
    {'label': 'old5b_draft', 'repo': OLD_REPO, 'gpu': 4, 'method': 'DraftModel'},
]

def ensure_old_worktree():
    if (OLD_REPO / 'test_fusionrag_reflect_preprocess_exp.py').exists():
        return
    OLD_REPO.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'worktree', 'add', '--detach', str(OLD_REPO), OLD_COMMIT], cwd=CURRENT_REPO, check=True)

def run_task(task):
    out_dir = EXP / task['label']
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = EXP / 'cache' / task['label']
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': str(task['gpu']), 'PYTHONUNBUFFERED': '1'})
    cmd = [
        PYTHON, 'test_fusionrag_reflect_preprocess_exp.py',
        '--model_type', 'qwen3',
        '--model_path', MODEL,
        '--model_name', 'Qwen3-32B',
        '--data_path', DATA,
        '--dataset_name', '2wikimqa',
        '--cache_path', str(cache),
        '--result_path', str(out_dir),
        '--start_sample', '0',
        '--end_sample', '25',
        '--rate', '0.15',
        '--topk', '10',
        '--preprocess', 'true',
        '--recall_method', 'bge',
        '--reprocess_method', task['method'],
        '--revert_rope', 'true',
        '--preprocess_scope', 'global',
        '--bge_model_path', BGE,
        '--device', 'cuda:0',
        '--use_multi_gpu', 'false',
        '--openai_base_url', API_URL,
        '--openai_api_key', API_KEY,
        '--openai_model', 'GLM-5.2',
    ]
    if task['method'] == 'DraftModel':
        cmd.extend(['--draft_model_path', DRAFT])
    meta = out_dir / 'command.txt'
    meta.write_text('cwd=' + str(task['repo']) + '\n' + ' '.join(cmd) + '\n', encoding='utf-8')
    with (out_dir / 'run.log').open('w', encoding='utf-8') as f:
        rc = subprocess.run(cmd, cwd=task['repo'], env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    (out_dir / 'returncode.txt').write_text(str(rc) + '\n', encoding='utf-8')

def main():
    EXP.mkdir(parents=True, exist_ok=True)
    ensure_old_worktree()
    (EXP / 'versions.txt').write_text(
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CURRENT_REPO, text=True).strip() + '\n' +
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=OLD_REPO, text=True).strip() + '\n',
        encoding='utf-8',
    )
    procs = []
    for task in TASKS:
        p = Process(target=run_task, args=(task,), daemon=False)
        p.start(); procs.append(p)
        time.sleep(5)
    for p in procs:
        p.join()

if __name__ == '__main__':
    main()
