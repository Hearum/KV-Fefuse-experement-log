#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path('/raid/home/hming/FusionRAG-pca-analysis')
NATIVE_SUMMARY = REPO / 'MOTIVATION_EXPERIMENTS/qwen3_rate_sweep_online_qk_draft/rate_sweep_current_summary.csv'
ALPHA_ROOT = Path('/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_alpha_config_rate_sweep_canonical_20260714')
OUT_DIR = REPO / 'MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/figures'
OUT_CSV = REPO / 'MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/musique_rate_sweep_qk_draft_uniform_random.csv'
CSV_NAME_TMPL = 'rate_{rate}_draft_Qwen2.5-3B-Instruct_revert_rope.csv'


def truth(x: str) -> bool:
    return str(x).strip().lower() in {'1', 'true', 'yes', 'correct'}


def dedup_rows(rows: list[dict]) -> list[dict]:
    seen = {}
    for row in rows:
        key = (row.get('Main Question', ''), row.get('Sub Question', ''), row.get('Ground Truth', ''))
        seen[key] = row
    return list(seen.values())


def summarize_rows(rows: list[dict]) -> dict:
    rows = dedup_rows(rows)
    sub_total = len(rows)
    sub_correct = sum(truth(r.get('Correct', '')) for r in rows)
    by_main: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_main[row.get('Main Question', '')].append(truth(row.get('Correct', '')))
    main_total = len(by_main)
    main_correct = sum(all(v) for v in by_main.values())
    f1 = sum(float(r.get('F1') or 0) for r in rows) / sub_total if sub_total else 0.0
    em = sum(float(r.get('EM') or 0) for r in rows) / sub_total if sub_total else 0.0
    return {
        'main_correct': main_correct,
        'main_total': main_total,
        'main_acc': main_correct / main_total if main_total else 0.0,
        'sub_correct': sub_correct,
        'sub_total': sub_total,
        'sub_acc': sub_correct / sub_total if sub_total else 0.0,
        'avg_f1': f1,
        'avg_em': em,
        'unique_rows': sub_total,
    }


def load_native() -> list[dict]:
    rows = []
    with NATIVE_SUMMARY.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['method'] not in {'online_qk', 'online_draft'}:
                continue
            rows.append({
                'dataset': 'musique',
                'method': row['method'],
                'rate': float(row['rate']),
                'main_correct': int(row['main_correct']),
                'main_total': int(row['main_total']),
                'main_acc': float(row['main_acc']),
                'sub_correct': int(row['sub_correct']),
                'sub_total': int(row['sub_total']),
                'sub_acc': float(row['sub_acc']),
                'avg_f1': float(row['avg_f1']),
                'avg_em': float(row['avg_em']),
                'unique_rows': int(row.get('unique_sub_rows') or row.get('sub_total') or 0),
                'segments_done': int(row['csv_files']),
                'segments_expected': 8,
                'status': 'complete',
                'source': str(NATIVE_SUMMARY),
            })
    return rows


def alpha_rows(combo: str, display: str, rates: list[str]) -> list[dict]:
    out = []
    for rate in rates:
        rows = []
        done = 0
        for start in range(0, 200, 25):
            end = min(start + 25, 200)
            d = ALPHA_ROOT / combo / f"rate{rate.replace('.', 'p')}" / 'musique' / f'seg_{start}_{end}'
            log = d / 'run.log'
            if log.exists() and 'FINAL RESULTS' in log.read_text(errors='ignore'):
                done += 1
            for p in d.glob(f"**/{CSV_NAME_TMPL.format(rate=rate)}"):
                with p.open(newline='', encoding='utf-8') as f:
                    rows.extend(csv.DictReader(f))
        if not rows:
            continue
        s = summarize_rows(rows)
        out.append({
            'dataset': 'musique',
            'method': display,
            'rate': float(rate),
            **s,
            'segments_done': done,
            'segments_expected': 8,
            'status': 'complete' if done == 8 else 'partial',
            'source': str(ALPHA_ROOT / combo),
        })
    return out




def write_csv(rows: list[dict]) -> None:
    fields = ['dataset', 'method', 'rate', 'main_correct', 'main_total', 'main_acc', 'sub_correct', 'sub_total', 'sub_acc', 'avg_f1', 'avg_em', 'unique_rows', 'segments_done', 'segments_expected', 'status', 'source']
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda x: (x['method'], x['rate'])):
            writer.writerow(row)


def plot(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    colors = {
        'online_qk': '#4C78A8',
        'online_draft': '#F58518',
        'uniform_alpha0p1': '#54A24B',
        'random_alpha0p1': '#B279A2',
    }
    labels = {
        'online_qk': 'Online QK',
        'online_draft': 'Online DraftModel',
        'uniform_alpha0p1': 'Uniform alpha=0.1',
        'random_alpha0p1': 'Random alpha=0.1',
    }
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 6.4), sharex=True)
    panels = [('main_acc', 'Main accuracy (%)', (60, 90)), ('sub_acc', 'Sub accuracy (%)', (76, 94))]
    for ax, (metric, ylabel, ylim) in zip(axes, panels):
        for method in ['online_qk', 'online_draft', 'uniform_alpha0p1', 'random_alpha0p1']:
            pts = sorted([r for r in rows if r['method'] == method], key=lambda x: x['rate'])
            if not pts:
                continue
            xs = [p['rate'] for p in pts]
            ys = [100 * p[metric] for p in pts]
            partial = any(p['status'] == 'partial' for p in pts)
            ax.plot(xs, ys, marker='o', linewidth=2.2, markersize=5.5, color=colors[method], label=labels[method], linestyle='--' if partial else '-')
            for p in pts:
                if p['status'] == 'partial':
                    ax.scatter([p['rate']], [100 * p[metric]], s=78, facecolors='none', edgecolors=colors[method], linewidths=1.8, zorder=5)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis='y', color='#D0D0D0', linewidth=0.8, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[-1].set_xlabel('Recompute rate')
    axes[-1].set_xticks([0, 0.1, 0.15, 0.3, 0.5, 0.8, 0.9, 1.0])
    axes[0].legend(frameon=False, ncol=2, loc='lower right')
    axes[-1].text(0.01, -0.34, 'Open markers/dashed line: partial random-alpha sweep. Current-only plot; historical alpha=0.15 points excluded.', transform=axes[-1].transAxes, fontsize=8, color='#555555')
    fig.tight_layout()
    png = OUT_DIR / 'musique_rate_sweep_qk_draft_uniform_random.png'
    pdf = OUT_DIR / 'musique_rate_sweep_qk_draft_uniform_random.pdf'
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    print(png)
    print(pdf)


def main() -> None:
    rows = []
    rows.extend(load_native())
    rows.extend(alpha_rows('uniform_alpha0p1', 'uniform_alpha0p1', ['0.3', '0.5', '0.8', '1.0']))
    rows.extend(alpha_rows('random_alpha0p1', 'random_alpha0p1', ['0.3', '0.5', '0.8', '1.0']))
    write_csv(rows)
    plot(rows)
    print(OUT_CSV)


if __name__ == '__main__':
    main()

