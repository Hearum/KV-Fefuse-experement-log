#!/usr/bin/env python3
import csv
import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent
MODEL_DIR = 'Qwen3-32B/musique/DraftModel_global_topk10_bge'
CSV_NAME = 'rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv'
SEGMENTS = [(0,25),(25,50),(50,75),(75,100),(100,125),(125,150),(150,175),(175,200)]
RUNS = [
    ('uniform_draft_rate015', 'selected doc-token recompute uses uniform attention over allowed prior tokens'),
    ('random_draft_rate015', 'selected doc-token recompute uses random softmax attention over allowed prior tokens'),
]
BASELINES = [
    {'label': 'online_draft_rate015_baseline', 'main_correct': 99, 'main_total': 135, 'main_acc': 99/135, 'sub_correct': 209, 'sub_total': 248, 'sub_acc': 209/248, 'avg_f1': None, 'avg_em': None, 'note': 'normal online DraftModel selector/recompute baseline from qwen3_rate015_online_offline'},
    {'label': 'online_qk_rate015_baseline', 'main_correct': 84, 'main_total': 135, 'main_acc': 84/135, 'sub_correct': 189, 'sub_total': 248, 'sub_acc': 189/248, 'avg_f1': None, 'avg_em': None, 'note': 'normal online QK selector baseline from qwen3_rate015_online_offline'},
]

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def fmt_num(x, digits=4):
    return '' if x is None else f'{x:.{digits}f}'

def fmt_pct(x):
    return '' if x is None else f'{x:.2%}'

def csv_paths(label):
    return [EXP / label / f'seg_{start}_{end}' / MODEL_DIR / CSV_NAME for start, end in SEGMENTS]

def log_paths(label):
    return [EXP / label / f'seg_{start}_{end}' / 'run.log' for start, end in SEGMENTS]

def read_rows(paths):
    out = OrderedDict(); missing = []
    for p in paths:
        if not p.exists():
            missing.append(str(p)); continue
        with p.open('r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                out[(row.get('Main Question',''), row.get('Sub Question',''))] = row
    return list(out.values()), missing

def parse_logs(paths):
    text = ''; existing = 0
    for p in paths:
        if p.exists():
            existing += 1
            text += '\n' + p.read_text(encoding='utf-8', errors='ignore')
    prompt = [float(x) for x in re.findall(r'^prompt eval duration:\s*([0-9.]+)s', text, re.M)]
    select = [float(x) for x in re.findall(r'^(?:select_time|draft_select_time):\s*([0-9.]+)', text, re.M)]
    doc_tokens = [int(x) for x in re.findall(r'reprocess_attention_ablation:\s*(?:uniform|random), doc_tokens=(\d+)', text)]
    return {
        'log_files': existing,
        'finished_segments': text.count('FINAL RESULTS'),
        'traceback': text.count('Traceback'),
        'killed': text.count('Killed') + text.count('Terminated'),
        'prompt_eval_mean_s': mean(prompt),
        'selection_time_mean_s': mean(select),
        'ablation_calls': len(doc_tokens),
        'doc_tokens_mean': mean(doc_tokens),
    }

def summarize(label, note):
    rows, missing = read_rows(csv_paths(label))
    by_main = defaultdict(list)
    for r in rows:
        by_main[r.get('Main Question','')].append(r)
    main_total = len(by_main)
    main_correct = sum(1 for group in by_main.values() if group and all(str(r.get('Correct','')).lower() == 'true' for r in group))
    sub_total = len(rows)
    sub_correct = sum(1 for r in rows if str(r.get('Correct','')).lower() == 'true')
    ans = {
        'label': label,
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
            if k not in fields: fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    rows = [summarize(*r) for r in RUNS]
    all_rows = BASELINES + rows
    write_csv(EXP / 'summary.csv', all_rows)
    (EXP / 'summary.json').write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Qwen3 DraftModel Rate=0.15 Recompute Attention Ablation\n\n']
    lines.append('目标：selection 仍使用原 online DraftModel，rate=0.15；只替换被选中 doc token 在主模型重算时的 attention 分布，测试效果是否依赖真实 main-model attention。\n\n')
    lines.append('对标 baseline：qwen3_rate015_online_offline 中的正常 online DraftModel/QK 结果。\n\n')
    lines.append('| label | Main Acc | Sub Acc | F1 | EM | prefill(s) | selection(s) | ablation calls | doc tokens/call | finished | missing csv | traceback/killed | note |\n')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n')
    for r in all_rows:
        doc_tokens = r.get('doc_tokens_mean')
        lines.append(
            f"| {r['label']} | {r['main_correct']}/{r['main_total']} ({fmt_pct(r['main_acc'])}) | "
            f"{r['sub_correct']}/{r['sub_total']} ({fmt_pct(r['sub_acc'])}) | "
            f"{fmt_num(r.get('avg_f1'))} | {fmt_num(r.get('avg_em'))} | "
            f"{r.get('prompt_eval_mean_s', 0.0):.4f} | {r.get('selection_time_mean_s', 0.0):.4f} | "
            f"{r.get('ablation_calls', '')} | {'' if doc_tokens is None else f'{doc_tokens:.1f}'} | "
            f"{r.get('finished_segments', '')} | {r.get('missing_csv', '')} | {r.get('traceback', '')}/{r.get('killed', '')} | {r['note']} |\n"
        )
    lines.append('\n## Interpretation\n\n')
    lines.append('- 如果 uniform/random 接近 normal DraftModel，说明主要收益来自 selection 或 token mixing 的粗粒度上下文，真实 attention 权重不是关键。\n')
    lines.append('- 如果明显掉到 QK/rate0 附近，说明选中 token 以后仍需要真实 main-model attention，未来 adapter 至少要近似这一层 attention 聚合，而不能只加静态 bias。\n')
    (EXP / 'README.md').write_text(''.join(lines), encoding='utf-8')
    print(''.join(lines))

if __name__ == '__main__':
    main()
