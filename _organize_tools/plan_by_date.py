#!/usr/bin/env python3
from __future__ import annotations
import csv, json, os, re
from pathlib import Path
from datetime import datetime

ROOT = Path('MOTIVATION_EXPERIMENTS')
OUT = ROOT / '_organize_tools'
DATE_RE = re.compile(r'(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)')
SKIP = {'by_date', '_organize_tools'}
TOP_FILES = {p.name for p in ROOT.iterdir() if p.is_file()}

def mtime_date(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d')

def infer_date(path: Path) -> tuple[str, str]:
    # Prefer explicit dates in the experiment directory name.
    m = DATE_RE.search(path.name)
    if m:
        return f'{m.group(1)}-{m.group(2)}-{m.group(3)}', 'name'
    # Then explicit dates in nearby README/log files, if any.
    candidates = []
    for rel in ['README.md', 'EXPERIMENT_LOG.md', 'goal.md', 'RUN_STATUS.md']:
        p = path / rel
        if p.exists() and p.is_file():
            try:
                text = p.read_text(errors='ignore')[:20000]
            except Exception:
                continue
            for mm in DATE_RE.finditer(text):
                candidates.append(f'{mm.group(1)}-{mm.group(2)}-{mm.group(3)}')
    if candidates:
        return sorted(candidates)[0], 'doc'
    return mtime_date(path), 'mtime'

def summarize(path: Path) -> str:
    for rel in ['README.md', 'EXPERIMENT_LOG.md', 'goal.md', 'RUN_STATUS.md']:
        p = path / rel
        if not p.exists() or not p.is_file():
            continue
        try:
            lines = p.read_text(errors='ignore').splitlines()
        except Exception:
            continue
        for line in lines:
            s = line.strip().lstrip('#').strip()
            if not s or s.startswith('```') or len(s) < 8:
                continue
            return s[:180]
    return path.name.replace('_', ' ')

rows = []
for path in sorted(ROOT.iterdir(), key=lambda p: p.name):
    if not path.is_dir() or path.name in SKIP or path.is_symlink():
        continue
    date, source = infer_date(path)
    rows.append({
        'experiment': path.name,
        'date': date,
        'date_source': source,
        'old_path': str(path),
        'new_path': str(ROOT / 'by_date' / date / path.name),
        'summary': summarize(path),
    })
OUT.mkdir(parents=True, exist_ok=True)
with (OUT / 'by_date_manifest.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['experiment','date','date_source','old_path','new_path','summary'])
    w.writeheader(); w.writerows(rows)
(OUT / 'by_date_manifest.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'wrote {len(rows)} experiments to {OUT / "by_date_manifest.csv"}')
from collections import Counter
print('by date:')
for date,count in sorted(Counter(r['date'] for r in rows).items()):
    print(date, count)
print('date source:', dict(Counter(r['date_source'] for r in rows)))
