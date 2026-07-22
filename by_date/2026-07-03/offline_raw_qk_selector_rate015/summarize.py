#!/usr/bin/env python3
import csv
import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path('/raid/home/hming/FusionRAG-pca-analysis')
EXP = ROOT / 'MOTIVATION_EXPERIMENTS/offline_raw_qk_selector_rate015'
BASE_SUMMARY = ROOT / 'MOTIVATION_EXPERIMENTS/full_accuracy_offline_selector_reflect_summary/summary.json'
RAW_PREV = ROOT / 'MOTIVATION_EXPERIMENTS/raw_vs_preprocess_kv_fixed_set/raw_vs_preprocess_summary.json'

RUNS = [
    ('rawqk_freq_preprocess_runtime', 'raw', 'frequency', 'preprocess', 'qk_frequency_per_chunk', True),
    ('rawqk_mean_preprocess_runtime', 'raw', 'mean_score', 'preprocess', 'qk_mean_score_per_chunk', True),
    ('rawqk_freq_raw_runtime', 'raw', 'frequency', 'raw', 'qk_frequency_per_chunk', False),
    ('rawqk_mean_raw_runtime', 'raw', 'mean_score', 'raw', 'qk_mean_score_per_chunk', False),
]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def is_true(x):
    return str(x).strip().lower() == 'true'


def csv_paths(label, preprocess):
    sub = 'FusionRAG_global_topk10_bge' if preprocess else 'nopreprocess'
    return sorted((EXP / label / 'Qwen2.5-7B-Instruct/musique' / sub).glob('rate_0.15_revert_rope.csv'))


def read_rows(paths):
    rows = OrderedDict()
    for path in paths:
        if not path.exists():
            continue
        with path.open('r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('Main Question') and row.get('Sub Question'):
                    rows[(row['Main Question'], row['Sub Question'])] = row
    return list(rows.values())


def parse_timing(label):
    path = EXP / 'generation_logs' / f'{label}.log'
    text = path.read_text(encoding='utf-8', errors='ignore') if path.exists() else ''
    prompt = [float(x) for x in re.findall(r'^prompt eval duration:\s*([0-9.]+)s', text, re.M)]
    storage = [float(x) for x in re.findall(r'^storage_time:\s*([0-9.]+)', text, re.M)]
    select = [float(x) for x in re.findall(r'^select_time:\s*([0-9.]+)', text, re.M)]
    return {
        'n_timing': len(prompt),
        'prompt_eval_mean_s': mean(prompt),
        'storage_time_mean_s': mean(storage),
        'selection_time_mean_s': mean(select),
        'finished': 'FINAL RESULTS' in text,
        'terminated_count': text.count('Terminated') + text.count('Killed'),
    }


def summarize_rows(label, selector_kv, selector_method, runtime_kv, fixed_key, rows):
    by_main = defaultdict(list)
    for row in rows:
        by_main[row['Main Question']].append(row)
    main_total = len(by_main)
    main_correct = sum(1 for group in by_main.values() if all(is_true(r.get('Correct')) for r in group))
    sub_total = len(rows)
    sub_correct = sum(1 for row in rows if is_true(row.get('Correct')))
    out = {
        'label': label,
        'selector_kv': selector_kv,
        'selector_method': selector_method,
        'runtime_kv': runtime_kv,
        'fixed_set_key': fixed_key,
        'rate': 0.15,
        'main_correct': main_correct,
        'main_total': main_total,
        'main_acc': main_correct / main_total if main_total else 0.0,
        'sub_correct': sub_correct,
        'sub_total': sub_total,
        'sub_acc': sub_correct / sub_total if sub_total else 0.0,
        'avg_f1': mean([float(r.get('F1') or 0.0) for r in rows]),
        'avg_em': mean([float(r.get('EM') or 0.0) for r in rows]),
    }
    out.update(parse_timing(label))
    return out


def load_existing_rows():
    rows = []
    if BASE_SUMMARY.exists():
        for r in json.loads(BASE_SUMMARY.read_text()):
            if r.get('label') == 'offline_qk_frequency':
                rows.append({
                    **r,
                    'selector_kv': 'preprocess',
                    'selector_method': 'frequency',
                    'runtime_kv': 'preprocess',
                    'fixed_set_key': 'qk_frequency_per_chunk',
                    'finished': True,
                })
            if r.get('label') == 'offline_qk_mean':
                rows.append({
                    **r,
                    'selector_kv': 'preprocess',
                    'selector_method': 'mean_score',
                    'runtime_kv': 'preprocess',
                    'fixed_set_key': 'qk_mean_score_per_chunk',
                    'finished': True,
                })
            if r.get('label') == 'online_qk_rate015':
                rows.append({**r, 'selector_kv': 'online_preprocess', 'selector_method': 'online_qk', 'runtime_kv': 'preprocess', 'fixed_set_key': 'N/A', 'finished': True})
    if RAW_PREV.exists():
        for r in json.loads(RAW_PREV.read_text()):
            if r.get('label') == 'raw_offline_qk_frequency':
                rows.append({**r, 'selector_kv': 'preprocess', 'selector_method': 'frequency', 'runtime_kv': 'raw', 'fixed_set_key': 'qk_frequency_per_chunk', 'finished': True})
            if r.get('label') == 'raw_offline_qk_mean':
                rows.append({**r, 'selector_kv': 'preprocess', 'selector_method': 'mean_score', 'runtime_kv': 'raw', 'fixed_set_key': 'qk_mean_score_per_chunk', 'finished': True})
    return rows


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    new_rows = []
    for label, selector_kv, selector_method, runtime_kv, fixed_key, preprocess in RUNS:
        new_rows.append(summarize_rows(label, selector_kv, selector_method, runtime_kv, fixed_key, read_rows(csv_paths(label, preprocess))))
    all_rows = load_existing_rows() + new_rows
    write_csv(EXP / 'summary.csv', all_rows)
    (EXP / 'summary.json').write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding='utf-8')

    base = (EXP / 'README.md').read_text(encoding='utf-8').split('## 当前结果', 1)[0]
    lines = [base, '## 当前结果\n\n']
    lines.append('| label | selector KV | method | runtime KV | finished | Main Acc | Sub Acc | F1 | EM | prefill(s) | storage(s) | selection(s) | rows |\n')
    lines.append('|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for r in all_rows:
        lines.append(
            f"| {r.get('label')} | {r.get('selector_kv')} | {r.get('selector_method')} | {r.get('runtime_kv')} | {r.get('finished', '')} | "
            f"{int(r.get('main_correct', 0))}/{int(r.get('main_total', 0))} ({float(r.get('main_acc', 0)):.2%}) | "
            f"{int(r.get('sub_correct', 0))}/{int(r.get('sub_total', 0))} ({float(r.get('sub_acc', 0)):.2%}) | "
            f"{float(r.get('avg_f1', 0)):.4f} | {float(r.get('avg_em', 0)):.4f} | "
            f"{float(r.get('prompt_eval_mean_s', 0)):.4f} | {float(r.get('storage_time_mean_s', 0)):.4f} | {float(r.get('selection_time_mean_s', 0)):.4f} | {int(r.get('sub_total', 0))} |\n"
        )
    lines.append('\n## 结果解读\n\n')
    if all(r.get('finished') for r in new_rows):
        lines.append('四组新增 raw-QK selector 推理均已完成。raw-QK selector 指的是 offline 阶段用 raw document KV 计算 calibration-query 的 QK/attention score，再固定选 token。\n')
    else:
        lines.append('新增 raw-QK selector 推理仍在进行中；当前表为增量结果，finished=False 的行不能作为最终结论。\n')
    (EXP / 'README.md').write_text(''.join(lines), encoding='utf-8')
    print('wrote', EXP / 'summary.csv')

if __name__ == '__main__':
    main()
