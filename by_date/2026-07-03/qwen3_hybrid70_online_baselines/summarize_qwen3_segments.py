#!/usr/bin/env python3
import csv
import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path('/raid/home/hming/FusionRAG-pca-analysis')
EXP = ROOT / 'MOTIVATION_EXPERIMENTS/qwen3_hybrid70_online_baselines'
MODEL_DIR = 'Qwen3-32B/musique'
RUNS = [
    ('full_rate1', 1.0, 'FusionRAG_global_topk10_bge', 'rate_1.0_revert_rope.csv', 'rate=1 full document-token recompute upper baseline'),
    ('online_qk_rate050', 0.5, 'FusionRAG_global_topk10_bge', 'rate_0.5_revert_rope.csv', 'online FusionRAG-QK selector at rate=0.5'),
    ('online_draft_rate050', 0.5, 'DraftModel_global_topk10_bge', 'rate_0.5_draft_Qwen2.5-3B-Instruct_revert_rope.csv', 'online draft-model selector at rate=0.5'),
    ('offline_hybrid70_rate050', 0.5, 'FusionRAG_global_topk10_bge', 'rate_0.5_revert_rope.csv', 'offline fixed set, hybrid draft70/qk30 score per chunk, rate=0.5'),
]
SEGMENTS = [(0,25),(25,50),(50,75),(75,100),(100,125),(125,150),(150,175),(175,200)]

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def csv_paths(label, subdir, filename):
    return [EXP / label / f'seg_{start}_{end}' / MODEL_DIR / subdir / filename for start, end in SEGMENTS]

def log_paths(label):
    return [EXP / label / f'seg_{start}_{end}' / 'run.log' for start, end in SEGMENTS]

def read_rows(paths):
    out = OrderedDict()
    missing = []
    for p in paths:
        if not p.exists():
            missing.append(str(p))
            continue
        with p.open('r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                out[(row.get('Main Question',''), row.get('Sub Question',''))] = row
    return list(out.values()), missing

def parse_logs(paths):
    text = ''
    existing = 0
    for p in paths:
        if p.exists():
            existing += 1
            text += '\n' + p.read_text(encoding='utf-8', errors='ignore')
    prompt = [float(x) for x in re.findall(r'^prompt eval duration:\s*([0-9.]+)s', text, re.M)]
    storage = [float(x) for x in re.findall(r'^storage_time:\s*([0-9.]+)', text, re.M)]
    select = [float(x) for x in re.findall(r'^(?:select_time|draft_select_time):\s*([0-9.]+)', text, re.M)]
    return {
        'log_files': existing,
        'finished_segments': text.count('FINAL RESULTS'),
        'traceback': text.count('Traceback'),
        'killed': text.count('Killed') + text.count('Terminated'),
        'n_prefill_timing': len(prompt),
        'prompt_eval_mean_s': mean(prompt),
        'storage_time_mean_s': mean(storage),
        'selection_time_mean_s': mean(select),
    }

def summarize(label, rate, subdir, filename, note):
    rows, missing = read_rows(csv_paths(label, subdir, filename))
    by_main = defaultdict(list)
    for r in rows:
        by_main[r.get('Main Question','')].append(r)
    main_total = len(by_main)
    main_correct = sum(1 for group in by_main.values() if group and all(str(r.get('Correct','')).lower() == 'true' for r in group))
    sub_total = len(rows)
    sub_correct = sum(1 for r in rows if str(r.get('Correct','')).lower() == 'true')
    ans = {
        'label': label,
        'rate': rate,
        'main_correct': main_correct,
        'main_total': main_total,
        'main_acc': main_correct / main_total if main_total else 0.0,
        'sub_correct': sub_correct,
        'sub_total': sub_total,
        'sub_acc': sub_correct / sub_total if sub_total else 0.0,
        'avg_f1': mean([float(r.get('F1') or 0.0) for r in rows]),
        'avg_em': mean([float(r.get('EM') or 0.0) for r in rows]),
        'missing_csv': len(missing),
        'note': note,
    }
    ans.update(parse_logs(log_paths(label)))
    return ans

def write_csv(path, rows):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def main():
    rows = [summarize(*r) for r in RUNS]
    write_csv(EXP / 'summary.csv', rows)
    (EXP / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = []
    lines.append('# Qwen3-32B Hybrid70/Online Baseline Experiment\n\n')
    lines.append('实验设置：reflect/musique 数据，Qwen3-32B 主模型，preprocess=true，topk=10，BGE recall，GLM-5.2 judge。样本按 8 个 segment 并行运行，汇总时按 `(Main Question, Sub Question)` 去重。\n\n')
    lines.append('| label | rate | Main Acc | Sub Acc | F1 | EM | prefill(s) | selection(s) | rows | finished | missing csv | traceback/killed | note |\n')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n')
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['rate']:.2f} | {r['main_correct']}/{r['main_total']} ({r['main_acc']:.2%}) | "
            f"{r['sub_correct']}/{r['sub_total']} ({r['sub_acc']:.2%}) | {r['avg_f1']:.4f} | {r['avg_em']:.4f} | "
            f"{r['prompt_eval_mean_s']:.4f} | {r['selection_time_mean_s']:.4f} | {r['sub_total']} | {r['finished_segments']} | "
            f"{r['missing_csv']} | {r['traceback']}/{r['killed']} | {r['note']} |\n"
        )
    (EXP / 'README.md').write_text(''.join(lines), encoding='utf-8')
    print(''.join(lines))

if __name__ == '__main__':
    main()
