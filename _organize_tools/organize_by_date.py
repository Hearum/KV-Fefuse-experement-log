#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os, re, shutil, subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path('MOTIVATION_EXPERIMENTS')
BY_DATE = ROOT / 'by_date'
TOOLS = ROOT / '_organize_tools'
DATE_RE = re.compile(r'(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)')
SKIP = {'by_date', '_organize_tools'}
DATE_FILES = ['README.md', 'EXPERIMENT_LOG.md', 'goal.md', 'RUN_STATUS.md']

def valid_date(y: str, m: str, d: str) -> str | None:
    try:
        return datetime(int(y), int(m), int(d)).strftime('%Y-%m-%d')
    except ValueError:
        return None

def extract_dates(text: str) -> list[str]:
    out = []
    for mm in DATE_RE.finditer(text):
        date = valid_date(mm.group(1), mm.group(2), mm.group(3))
        if date:
            out.append(date)
    return out

def birth_date(path: Path) -> str | None:
    try:
        raw = subprocess.check_output(['stat', '-c', '%W', str(path)], text=True).strip()
        ts = int(raw)
    except Exception:
        return None
    if ts <= 0:
        return None
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d')

def mtime_date(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d')

def infer_date(path: Path) -> tuple[str, str]:
    bdate = birth_date(path)
    if bdate:
        return bdate, 'birth_time'
    dates = extract_dates(path.name)
    if dates:
        return dates[0], 'name'
    candidates = []
    for rel in DATE_FILES:
        p = path / rel
        if p.exists() and p.is_file():
            try:
                candidates.extend(extract_dates(p.read_text(errors='ignore')[:50000]))
            except Exception:
                pass
    if candidates:
        # Use earliest documented experiment date, not latest edit date.
        return sorted(candidates)[0], 'doc'
    return mtime_date(path), 'mtime'

def summarize(path: Path) -> str:
    for rel in DATE_FILES:
        p = path / rel
        if not p.exists() or not p.is_file():
            continue
        try:
            lines = p.read_text(errors='ignore').splitlines()
        except Exception:
            continue
        for line in lines:
            s = line.strip().lstrip('#').strip()
            if not s or s.startswith('```') or s.startswith('|') or len(s) < 8:
                continue
            return s.replace('|', '/').replace('\n', ' ')[:180]
    return path.name.replace('_', ' ')

def current_experiment_dirs() -> list[Path]:
    dirs = []
    for p in ROOT.iterdir():
        if not p.is_dir() or p.name in SKIP:
            continue
        # Do not re-plan compatibility symlinks after migration.
        if p.is_symlink():
            continue
        dirs.append(p)
    return sorted(dirs, key=lambda x: x.name)

def build_manifest() -> list[dict]:
    rows = []
    for p in current_experiment_dirs():
        date, source = infer_date(p)
        rows.append({
            'experiment': p.name,
            'date': date,
            'date_source': source,
            'old_path': str(p),
            'new_path': str(BY_DATE / date / p.name),
            'summary': summarize(p),
        })
    return rows

def write_manifest(rows: list[dict]) -> None:
    TOOLS.mkdir(parents=True, exist_ok=True)
    fields = ['experiment', 'date', 'date_source', 'old_path', 'new_path', 'summary']
    with (TOOLS / 'by_date_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    (TOOLS / 'by_date_manifest.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

def move_one(row: dict) -> None:
    old = Path(row['old_path'])
    new = Path(row['new_path'])
    if old.is_symlink():
        return
    if not old.exists():
        # Already moved. Ensure compatibility symlink exists.
        if new.exists() and not old.exists():
            rel = os.path.relpath(new, old.parent)
            old.symlink_to(rel, target_is_directory=True)
        return
    if new.exists():
        raise FileExistsError(f'target exists: {new}')
    new.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old), str(new))
    rel = os.path.relpath(new, old.parent)
    old.symlink_to(rel, target_is_directory=True)

def readme_for_date(date: str, rows: list[dict]) -> str:
    lines = [f'# {date} 实验目录', '', '本目录按推断出的实验创建日期汇总 MOTIVATION_EXPERIMENTS 下的实验。', '', '| 实验 | 简要说明 | 路径 | 日期来源 |', '|---|---|---|---|']
    for r in sorted(rows, key=lambda x: x['experiment']):
        lines.append(f"| `{r['experiment']}` | {r['summary']} | `{r['new_path']}` | `{r['date_source']}` |")
    lines += ['', '兼容性说明：原顶层实验路径会尽量保留为符号链接，因此旧命令和旧 README 中的路径仍可解析。', '']
    return '\n'.join(lines)

def write_indexes(rows: list[dict]) -> None:
    by_date = defaultdict(list)
    for r in rows:
        by_date[r['date']].append(r)
    BY_DATE.mkdir(parents=True, exist_ok=True)
    for date, items in sorted(by_date.items()):
        d = BY_DATE / date
        d.mkdir(parents=True, exist_ok=True)
        (d / 'README.md').write_text(readme_for_date(date, items), encoding='utf-8')
    top = ['# MOTIVATION_EXPERIMENTS 实验索引', '', '实验目录按推断日期整理到 `by_date/YYYY-MM-DD/` 下。', '', '原顶层实验目录名会尽量保留为兼容符号链接，旧命令和旧文档路径仍可继续解析。', '', '日期推断优先级：目录创建时间（`stat %W`）、目录名中的显式日期、README/EXPERIMENT_LOG/goal/RUN_STATUS 中的有效日期、目录修改时间。', '', '## 日期汇总', '', '| 日期 | 实验数 | 路径 |', '|---|---:|---|']
    for date, items in sorted(by_date.items()):
        top.append(f"| `{date}` | {len(items)} | `MOTIVATION_EXPERIMENTS/by_date/{date}/` |")
    top += ['', '## 实验明细', '', '| 日期 | 实验 | 简要说明 | 新路径 | 旧兼容路径 |', '|---|---|---|---|---|']
    for r in sorted(rows, key=lambda x: (x['date'], x['experiment'])):
        top.append(f"| `{r['date']}` | `{r['experiment']}` | {r['summary']} | `{r['new_path']}` | `{r['old_path']}` |")
    top.append('')
    (ROOT / 'README.md').write_text('\n'.join(top), encoding='utf-8')
    by = ['# MOTIVATION_EXPERIMENTS 按日期归档', '', '本目录包含按日期分组的实验集合。每个日期目录都有 README，用于快速说明当天包含哪些实验。', '', '| 日期 | 实验数 | README |', '|---|---:|---|']
    for date, items in sorted(by_date.items()):
        by.append(f"| `{date}` | {len(items)} | `{date}/README.md` |")
    by.append('')
    (BY_DATE / 'README.md').write_text('\n'.join(by), encoding='utf-8')

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    rows = build_manifest()
    write_manifest(rows)
    print(f'planned {len(rows)} experiments')
    print('by date:', dict(sorted(Counter(r['date'] for r in rows).items())))
    print('date source:', dict(Counter(r['date_source'] for r in rows)))
    if args.execute:
        for row in rows:
            move_one(row)
        write_indexes(rows)
        print('migration complete')
    else:
        print('dry run only; pass --execute to move directories')

if __name__ == '__main__':
    main()
