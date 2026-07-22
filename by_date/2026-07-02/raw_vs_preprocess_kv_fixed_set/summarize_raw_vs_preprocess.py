#!/usr/bin/env python3
import csv
import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path('/raid/home/hming/FusionRAG-pca-analysis')
EXP = ROOT / 'MOTIVATION_EXPERIMENTS/raw_vs_preprocess_kv_fixed_set'
SEL_SUMMARY = ROOT / 'MOTIVATION_EXPERIMENTS/full_accuracy_offline_selector_reflect_summary/summary.json'
HYBRID_SUMMARY = ROOT / 'MOTIVATION_EXPERIMENTS/full_accuracy_offline_hybrid70_rate_sweep_reflect_summary/summary.json'
DRAFT_AUDIT = ROOT / 'MOTIVATION_EXPERIMENTS/online_draft_impl_audit/full_profile_smart_sparse_rate015_summary.json'

RAW_RUNS = [
    ('raw_rate0_no_doc_recompute', 0.0),
    ('raw_online_qk_rate015', 0.15),
    ('raw_offline_qk_frequency', 0.15),
    ('raw_offline_qk_mean', 0.15),
    ('raw_offline_draft_frequency', 0.15),
    ('raw_offline_draft_mean', 0.15),
    ('raw_hybrid_draft70_qk30', 0.15),
    ('raw_online_draft_profile_sparse', 0.15),
]

PREPROCESS_LABEL_MAP = {
    'rate0_no_doc_recompute': 'raw_rate0_no_doc_recompute',
    'online_qk_rate015': 'raw_online_qk_rate015',
    'offline_qk_frequency': 'raw_offline_qk_frequency',
    'offline_qk_mean': 'raw_offline_qk_mean',
    'offline_draft_frequency': 'raw_offline_draft_frequency',
    'offline_draft_mean': 'raw_offline_draft_mean',
}


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def read_rows(paths):
    out = OrderedDict()
    for path in paths:
        if not path.exists():
            continue
        with path.open('r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if not row.get('Main Question') or not row.get('Sub Question'):
                    continue
                out[(row['Main Question'], row['Sub Question'])] = row
    return list(out.values())


def parse_timing(paths):
    text = ''
    for path in paths:
        if path.exists():
            text += '\n' + path.read_text(encoding='utf-8', errors='ignore')
    prompt = [float(x) for x in re.findall(r'^prompt eval duration:\s*([0-9.]+)s', text, re.M)]
    storage = [float(x) for x in re.findall(r'^storage_time:\s*([0-9.]+)', text, re.M)]
    select = [float(x) for x in re.findall(r'^select_time:\s*([0-9.]+)', text, re.M)]
    select.extend(float(x) for x in re.findall(r'^draft_select_time:\s*([0-9.]+)', text, re.M))
    return {
        'n_timing': len(prompt),
        'prompt_eval_mean_s': mean(prompt),
        'storage_time_mean_s': mean(storage),
        'selection_time_mean_s': mean(select),
        'finished': 'FINAL RESULTS' in text,
        'killed': text.count('Killed') + text.count('Terminated'),
    }


def summarize_rows(label, kv_mode, rate, rows, logs):
    by_main = defaultdict(list)
    for row in rows:
        by_main[row['Main Question']].append(row)
    main_total = len(by_main)
    main_correct = sum(1 for group in by_main.values() if all(str(r.get('Correct', '')).lower() == 'true' for r in group))
    sub_total = len(rows)
    sub_correct = sum(1 for row in rows if str(row.get('Correct', '')).lower() == 'true')
    out = {
        'label': label,
        'kv_mode': kv_mode,
        'rate': rate,
        'main_correct': main_correct,
        'main_total': main_total,
        'main_acc': main_correct / main_total if main_total else 0.0,
        'sub_correct': sub_correct,
        'sub_total': sub_total,
        'sub_acc': sub_correct / sub_total if sub_total else 0.0,
        'avg_f1': mean([float(r.get('F1') or 0.0) for r in rows]),
        'avg_em': mean([float(r.get('EM') or 0.0) for r in rows]),
    }
    out.update(parse_timing(logs))
    return out


def raw_csvs(label):
    return sorted((EXP / label).glob('Qwen2.5-7B-Instruct/musique/nopreprocess/rate_*.csv'))


def existing_preprocess_rows():
    rows = []
    if SEL_SUMMARY.exists():
        for r in json.loads(SEL_SUMMARY.read_text()):
            if r['label'] in PREPROCESS_LABEL_MAP:
                rows.append({**r, 'kv_mode': 'preprocess', 'paired_raw_label': PREPROCESS_LABEL_MAP[r['label']]})
    if HYBRID_SUMMARY.exists():
        for r in json.loads(HYBRID_SUMMARY.read_text()):
            if abs(float(r.get('rate', -1)) - 0.15) < 1e-9:
                rows.append({**r, 'label': 'hybrid_draft70_qk30_rate015', 'kv_mode': 'preprocess', 'paired_raw_label': 'raw_hybrid_draft70_qk30'})
    if DRAFT_AUDIT.exists():
        for r in json.loads(DRAFT_AUDIT.read_text()):
            if r.get('method') == 'online_draft_profile_sparse_full':
                rows.append({
                    'label': 'online_draft_profile_sparse',
                    'rate': 0.15,
                    'main_correct': r['main_correct'],
                    'main_total': r['main_total'],
                    'main_acc': r['main_acc'],
                    'sub_correct': r['sub_correct'],
                    'sub_total': r['sub_total'],
                    'sub_acc': r['sub_acc'],
                    'avg_f1': r['f1'],
                    'avg_em': r['em'],
                    'kv_mode': 'preprocess',
                    'paired_raw_label': 'raw_online_draft_profile_sparse',
                })
    return rows


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
    raw = []
    for label, rate in RAW_RUNS:
        raw.append(summarize_rows(label, 'raw', rate, read_rows(raw_csvs(label)), [EXP / f'{label}.log']))

    preprocess = existing_preprocess_rows()
    all_rows = [*preprocess, *raw]
    write_csv(EXP / 'raw_vs_preprocess_summary.csv', all_rows)
    (EXP / 'raw_vs_preprocess_summary.json').write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding='utf-8')

    by_raw = {r['label']: r for r in raw}
    paired = []
    for p in preprocess:
        rr = by_raw.get(p.get('paired_raw_label'))
        if not rr:
            continue
        paired.append({
            'pair': p['paired_raw_label'].replace('raw_', ''),
            'rate': rr.get('rate', p.get('rate')),
            'preprocess_main_acc': p.get('main_acc', 0.0),
            'raw_main_acc': rr.get('main_acc', 0.0),
            'delta_raw_minus_preprocess_main_acc': rr.get('main_acc', 0.0) - p.get('main_acc', 0.0),
            'preprocess_sub_acc': p.get('sub_acc', 0.0),
            'raw_sub_acc': rr.get('sub_acc', 0.0),
            'delta_raw_minus_preprocess_sub_acc': rr.get('sub_acc', 0.0) - p.get('sub_acc', 0.0),
            'preprocess_f1': p.get('avg_f1', 0.0),
            'raw_f1': rr.get('avg_f1', 0.0),
            'delta_raw_minus_preprocess_f1': rr.get('avg_f1', 0.0) - p.get('avg_f1', 0.0),
            'raw_finished': rr.get('finished', False),
            'raw_rows': rr.get('sub_total', 0),
        })
    write_csv(EXP / 'paired_delta_summary.csv', paired)
    (EXP / 'paired_delta_summary.json').write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding='utf-8')

    base = (EXP / 'README.md').read_text(encoding='utf-8').split('## 当前结果', 1)[0]
    lines = [
        base,
        '## 当前结果\n\n',
        '### Raw KV 单独结果\n\n',
        '| label | finished | Main Acc | Sub Acc | F1 | EM | prefill(s) | storage(s) | selection(s) | rows |\n',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n',
    ]
    for r in raw:
        lines.append(
            f"| {r.get('label')} | {r.get('finished')} | {r.get('main_correct')}/{r.get('main_total')} ({r.get('main_acc', 0):.2%}) | "
            f"{r.get('sub_correct')}/{r.get('sub_total')} ({r.get('sub_acc', 0):.2%}) | {r.get('avg_f1', 0):.4f} | {r.get('avg_em', 0):.4f} | "
            f"{r.get('prompt_eval_mean_s', 0):.4f} | {r.get('storage_time_mean_s', 0):.4f} | {r.get('selection_time_mean_s', 0):.4f} | {r.get('sub_total')} |\n"
        )
    lines += [
        '\n### Paired raw - preprocess 差值\n\n',
        '| pair | rate | preprocess Main | raw Main | ΔMain | preprocess Sub | raw Sub | ΔSub | preprocess F1 | raw F1 | ΔF1 | raw rows |\n',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n',
    ]
    for r in paired:
        lines.append(
            f"| {r.get('pair')} | {r.get('rate', 0):.2f} | {r.get('preprocess_main_acc', 0):.2%} | {r.get('raw_main_acc', 0):.2%} | {r.get('delta_raw_minus_preprocess_main_acc', 0):+.2%} | "
            f"{r.get('preprocess_sub_acc', 0):.2%} | {r.get('raw_sub_acc', 0):.2%} | {r.get('delta_raw_minus_preprocess_sub_acc', 0):+.2%} | "
            f"{r.get('preprocess_f1', 0):.4f} | {r.get('raw_f1', 0):.4f} | {r.get('delta_raw_minus_preprocess_f1', 0):+.4f} | {r.get('raw_rows')} |\n"
        )
    lines += [
        '\n## 初步解释\n\n',
        '等待所有 raw KV 作业 finished=True 后再写最终结论。当前表会随着作业进度增量更新。\n',
    ]
    (EXP / 'README.md').write_text(''.join(lines), encoding='utf-8')
    print('wrote', EXP / 'raw_vs_preprocess_summary.csv')

if __name__ == '__main__':
    main()
