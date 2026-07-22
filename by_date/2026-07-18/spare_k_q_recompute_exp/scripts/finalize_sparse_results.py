#!/usr/bin/env python3
import csv
import glob
import json
import sys
import subprocess
import time
from pathlib import Path

ROOT = Path('/raid/home/hming/FusionRAG-pca-analysis')
EXP = ROOT / 'MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp'
MODEL = '/mnt/qjhs-sh-lab-01/models/Qwen3-32B'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ktransformers'))
from transformers import AutoTokenizer
from ktransformers.util.utils import _exact_match_score, compute_f1
CONFIGS = {
    'sparse_block_preprocess_k2': EXP / 'results_final_k2/sparse_block_preprocess',
    'sparse_block_raw_k2': EXP / 'results_final_k2/sparse_block_raw',
    'sparse_block_preprocess_k4': EXP / 'results_final_k4/sparse_block_preprocess',
    'sparse_block_raw_k4': EXP / 'results_final_k4/sparse_block_raw',
    'sparse_block_preprocess_k8': EXP / 'results_final3_k8/sparse_block_preprocess',
    'sparse_block_raw_k8': EXP / 'results_final3_k8/sparse_block_raw',
}


def rows_for(path: Path):
    rows = {}
    for csv_path in sorted(path.glob('musique-v2/rate_0p15/*/csv/*.csv')):
        with csv_path.open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                q = row.get('Question', '')
                if q:
                    rows.setdefault(q, row)
    return rows


def main():
    while True:
        counts = {name: len(rows_for(path)) for name, path in CONFIGS.items()}
        print(json.dumps(counts), flush=True)
        if all(value >= 200 for value in counts.values()):
            break
        time.sleep(60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    output = EXP / 'final_summary.csv'
    fields = ['method', 'rate', 'rows', 'em', 'f1']
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, path in CONFIGS.items():
            rows = rows_for(path)
            em = []
            f1 = []
            for row in rows.values():
                pred = row.get('Pred Answer') or row.get('Answer') or ''
                gold = row.get('Real Answer') or row.get('Ground Truth') or ''
                em.append(float(_exact_match_score(pred, gold)))
                f1.append(float(compute_f1(pred, gold, tokenizer)))
            writer.writerow({'method': name, 'rate': '0.15', 'rows': len(rows),
                             'em': sum(em) / len(em), 'f1': sum(f1) / len(f1)})

    input_root = EXP / 'rejudge_input_final'
    input_root.mkdir(exist_ok=True)
    for name, path in CONFIGS.items():
        link = input_root / name
        link.unlink(missing_ok=True)
        link.symlink_to(path)
    out_dir = EXP / 'rejudge_glm_final'
    out_dir.mkdir(exist_ok=True)
    script = EXP / 'scripts/rejudge_sparse_glm.py'
    env = {
        **__import__('os').environ,
        'SETUP_V2_REJUDGE_RESULTS_ROOT': str(input_root),
        'SETUP_V2_REJUDGE_OUT_DIR': str(out_dir),
        'GLM_REJUDGE_WORKERS': '24',
    }
    subprocess.run(['/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python', str(script)],
                   cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
