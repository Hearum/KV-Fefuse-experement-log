#!/usr/bin/env python3
import csv, json
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]
CSV_NAME = "rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv"

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

rows = []
for d in sorted((EXP / 'results').glob('beta*_0_25')):
    csvs = list(d.glob(f'**/{CSV_NAME}'))
    data = []
    for p in csvs:
        with p.open(newline='', encoding='utf-8') as f:
            data.extend(csv.DictReader(f))
    by_main = {}
    for r in data:
        by_main.setdefault(r.get('Main Question', ''), []).append(r)
    main_total = len(by_main)
    main_correct = sum(1 for g in by_main.values() if g and all(str(r.get('Correct', '')).lower() == 'true' for r in g))
    sub_total = len(data)
    sub_correct = sum(1 for r in data if str(r.get('Correct', '')).lower() == 'true')
    log = (d / 'run.log').read_text(errors='ignore') if (d / 'run.log').exists() else ''
    rows.append({
        'label': d.name,
        'finished': 'FINAL RESULTS' in log,
        'main_correct': main_correct,
        'main_total': main_total,
        'main_acc': main_correct / main_total if main_total else 0,
        'sub_correct': sub_correct,
        'sub_total': sub_total,
        'sub_acc': sub_correct / sub_total if sub_total else 0,
        'avg_f1': mean([float(r.get('F1') or 0) for r in data]),
        'avg_em': mean([float(r.get('EM') or 0) for r in data]),
        'traceback': log.count('Traceback'),
        'blend_calls': log.count('reprocess_kv_blend:'),
    })
out_csv = EXP / 'results' / 'summary.csv'
fields = list(rows[0].keys()) if rows else ['label']
with out_csv.open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
(EXP / 'results' / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
print(out_csv)
for r in rows:
    print(r)
