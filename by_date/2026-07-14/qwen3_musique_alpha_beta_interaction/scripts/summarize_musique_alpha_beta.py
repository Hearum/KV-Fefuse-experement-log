#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get('FUSIONRAG_MUSIQUE_ALPHA_BETA_ROOT', '/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_alpha_beta_interaction_20260714'))
CSV_NAME = 'rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv'


def is_true(x: str) -> bool:
    return str(x).strip().lower() in {'1', 'true', 'yes', 'correct'}


def read_meta(d: Path) -> dict:
    p = d / 'task_meta.csv'
    if not p.exists():
        return {}
    with p.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def summarize_dir(d: Path) -> dict | None:
    csvs = list(d.glob(f'**/{CSV_NAME}'))
    if not csvs:
        return None
    csv_path = csvs[0]
    rows = list(csv.DictReader(csv_path.open(newline='', encoding='utf-8')))
    if not rows:
        return None
    meta = read_meta(d)
    log = d / 'run.log'
    finished = log.exists() and 'FINAL RESULTS' in log.read_text(errors='ignore')
    total = len(rows)
    correct = sum(1 for r in rows if is_true(r.get('Correct', '')))
    f1_vals = [float(r.get('F1') or 0) for r in rows]
    em_vals = [float(r.get('EM') or 0) for r in rows]
    return {
        'label': meta.get('label') or d.name,
        'mode': meta.get('mode', ''),
        'alpha': meta.get('alpha', ''),
        'beta': meta.get('beta', ''),
        'start': meta.get('start', ''),
        'end': meta.get('end', ''),
        'finished': finished,
        'rows': total,
        'correct': correct,
        'accuracy': correct / total if total else 0,
        'avg_f1': sum(f1_vals) / total if total else 0,
        'avg_em': sum(em_vals) / total if total else 0,
        'csv': str(csv_path),
    }


def main() -> None:
    results = []
    for d in sorted((ROOT / 'results').glob('*')):
        if d.is_dir():
            row = summarize_dir(d)
            if row:
                results.append(row)
    out_csv = ROOT / 'summary.csv'
    out_json = ROOT / 'summary.json'
    if results:
        fields = list(results[0].keys())
        with out_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(out_csv)
    for r in results:
        print(f"{r['label']}: {r['correct']}/{r['rows']} = {r['accuracy']:.2%}, f1={r['avg_f1']:.4f}, em={r['avg_em']:.4f}, finished={r['finished']}")


if __name__ == '__main__':
    main()

