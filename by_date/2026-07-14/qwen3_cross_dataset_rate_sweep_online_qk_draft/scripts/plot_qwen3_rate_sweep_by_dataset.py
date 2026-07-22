#!/usr/bin/env python3
"""Plot Qwen3-32B Online QK/DraftModel rate sweep by dataset.

Sources:
- Cross-dataset canonical sweep: rates 0.0,0.1,0.3,0.5,0.8,0.9
- Cross-dataset rate=0.15 add-on
- MuSiQue canonical/reusecache sweep: rates 0.0,0.1,0.3,0.5,0.8,0.9
- MuSiQue rate=0.15 add-on
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path('/raid/home/hming/FusionRAG-pca-analysis')
OUT_DIR = REPO / 'MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft'
FIG_DIR = OUT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = [
    Path('/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_canonical_20260714'),
    Path('/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate015_canonical_20260714'),
    Path('/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate_sweep_reusecache_20260714'),
    Path('/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate015_canonical_20260714'),
]
DATASET_ORDER = ['2wikimqa', 'hotpotqa', 'triviaqa', 'musique']
DATASET_LABEL = {'2wikimqa': '2WikiMQA', 'hotpotqa': 'HotpotQA', 'triviaqa': 'TriviaQA', 'musique': 'MuSiQue'}
METHOD_ORDER = ['online_qk', 'online_draft']
METHOD_LABEL = {'online_qk': 'Online QK', 'online_draft': 'Online Draft'}
METHOD_COLOR = {'online_qk': '#2a9d8f', 'online_draft': '#e76f51'}
METHOD_MARKER = {'online_qk': 'o', 'online_draft': 's'}
METRICS = [
    ('main_acc', 'Main Acc'),
    ('sub_acc', 'Sub Acc'),
    ('avg_f1', 'F1'),
    ('avg_em', 'EM'),
]


def parse_path(path: Path):
    parts = path.parts
    dataset = next((p for p in parts if p in DATASET_ORDER), None)
    method = None
    rate = None
    for p in parts:
        m = re.match(r'(online_qk|online_draft)_rate([0-9]+p[0-9]+)', p)
        if m:
            method = m.group(1)
            rate = m.group(2).replace('p', '.')
    return dataset, method, rate


def parse_final_log(log: Path):
    text = log.read_text(errors='ignore')
    if 'FINAL RESULTS' not in text:
        return None
    tail = text[text.rfind('FINAL RESULTS'):]
    m = re.search(
        r'Main Questions:\s*(\d+)/(\d+).*?Sub Questions:\s*(\d+)/(\d+).*?Average F1:\s*([0-9.]+).*?Average EM:\s*([0-9.]+)',
        tail,
        re.S,
    )
    if not m:
        return None
    main_correct, main_total, sub_correct, sub_total, f1, em = m.groups()
    return {
        'main_correct': int(main_correct),
        'main_total': int(main_total),
        'sub_correct': int(sub_correct),
        'sub_total': int(sub_total),
        'f1_weighted_sum': float(f1) * int(sub_total),
        'em_weighted_sum': float(em) * int(sub_total),
        'segments': 1,
    }


def aggregate_records():
    agg = defaultdict(lambda: {
        'main_correct': 0,
        'main_total': 0,
        'sub_correct': 0,
        'sub_total': 0,
        'f1_weighted_sum': 0.0,
        'em_weighted_sum': 0.0,
        'segments': 0,
        'sources': set(),
    })
    for root in SOURCES:
        if not root.exists():
            continue
        for log in root.rglob('run.log'):
            dataset, method, rate = parse_path(log)
            if not (dataset and method and rate):
                continue
            parsed = parse_final_log(log)
            if not parsed:
                continue
            key = (dataset, method, float(rate))
            rec = agg[key]
            for k in ['main_correct', 'main_total', 'sub_correct', 'sub_total', 'f1_weighted_sum', 'em_weighted_sum', 'segments']:
                rec[k] += parsed[k]
            rec['sources'].add(str(root))
    rows = []
    for (dataset, method, rate), rec in sorted(agg.items(), key=lambda x: (DATASET_ORDER.index(x[0][0]), METHOD_ORDER.index(x[0][1]), x[0][2])):
        sub_total = rec['sub_total']
        main_total = rec['main_total']
        rows.append({
            'dataset': dataset,
            'method': method,
            'rate': rate,
            'segments': rec['segments'],
            'main_correct': rec['main_correct'],
            'main_total': main_total,
            'main_acc': rec['main_correct'] / main_total if main_total else math.nan,
            'sub_correct': rec['sub_correct'],
            'sub_total': sub_total,
            'sub_acc': rec['sub_correct'] / sub_total if sub_total else math.nan,
            'avg_f1': rec['f1_weighted_sum'] / sub_total if sub_total else math.nan,
            'avg_em': rec['em_weighted_sum'] / sub_total if sub_total else math.nan,
            'sources': ';'.join(sorted(rec['sources'])),
        })
    return rows


def write_tables(rows):
    out_csv = OUT_DIR / 'qwen3_online_qk_draft_rate_sweep_by_dataset_with_rate015.csv'
    fields = ['dataset', 'method', 'rate', 'segments', 'main_correct', 'main_total', 'main_acc', 'sub_correct', 'sub_total', 'sub_acc', 'avg_f1', 'avg_em', 'sources']
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    out_json = OUT_DIR / 'qwen3_online_qk_draft_rate_sweep_by_dataset_with_rate015.json'
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    return out_csv, out_json


def plot_dataset(rows, dataset):
    ds_rows = [r for r in rows if r['dataset'] == dataset]
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.2), sharex=True)
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, METRICS):
        for method in METHOD_ORDER:
            mr = sorted([r for r in ds_rows if r['method'] == method], key=lambda x: x['rate'])
            if not mr:
                continue
            x = [r['rate'] for r in mr]
            y = [r[metric] * 100 for r in mr]
            ax.plot(
                x,
                y,
                marker=METHOD_MARKER[method],
                markersize=5.5,
                linewidth=2.0,
                color=METHOD_COLOR[method],
                label=METHOD_LABEL[method],
            )
            for xi, yi in zip(x, y):
                ax.text(xi, yi + 0.45, f'{yi:.1f}', ha='center', va='bottom', fontsize=7.5, color=METHOD_COLOR[method])
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_ylabel('%')
        ax.grid(True, axis='y', color='#d0d7de', linewidth=0.8, alpha=0.65)
        ax.grid(False, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        vals = [r[metric] * 100 for r in ds_rows if not math.isnan(r[metric])]
        if vals:
            lo, hi = min(vals), max(vals)
            pad = max(2.5, (hi - lo) * 0.18)
            ax.set_ylim(max(0, lo - pad), min(100, hi + pad + 1.0))
    axes[2].set_xlabel('Recompute rate')
    axes[3].set_xlabel('Recompute rate')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(DATASET_LABEL[dataset], y=0.965, fontsize=13, fontweight='semibold')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = FIG_DIR / f'qwen3_rate_sweep_{dataset}.png'
    pdf = FIG_DIR / f'qwen3_rate_sweep_{dataset}.pdf'
    fig.savefig(png, dpi=240, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    return png, pdf


def plot_main_grid(rows):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.2), sharex=True)
    axes = axes.ravel()
    for ax, dataset in zip(axes, DATASET_ORDER):
        ds_rows = [r for r in rows if r['dataset'] == dataset]
        for method in METHOD_ORDER:
            mr = sorted([r for r in ds_rows if r['method'] == method], key=lambda x: x['rate'])
            if not mr:
                continue
            ax.plot(
                [r['rate'] for r in mr],
                [r['main_acc'] * 100 for r in mr],
                marker=METHOD_MARKER[method],
                markersize=5.5,
                linewidth=2.0,
                color=METHOD_COLOR[method],
                label=METHOD_LABEL[method],
            )
        ax.set_title(DATASET_LABEL[dataset], fontsize=11, pad=6)
        ax.set_ylabel('Main Acc (%)')
        ax.grid(True, axis='y', color='#d0d7de', linewidth=0.8, alpha=0.65)
        ax.grid(False, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[2].set_xlabel('Recompute rate')
    axes[3].set_xlabel('Recompute rate')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = FIG_DIR / 'qwen3_rate_sweep_main_acc_all_datasets.png'
    pdf = FIG_DIR / 'qwen3_rate_sweep_main_acc_all_datasets.pdf'
    fig.savefig(png, dpi=240, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    return png, pdf


def main():
    rows = aggregate_records()
    out_csv, out_json = write_tables(rows)
    figures = []
    for dataset in DATASET_ORDER:
        figures.extend(plot_dataset(rows, dataset))
    figures.extend(plot_main_grid(rows))
    print(out_csv)
    print(out_json)
    for fig in figures:
        print(fig)


if __name__ == '__main__':
    main()
