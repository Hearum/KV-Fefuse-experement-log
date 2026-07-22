#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path('/raid/home/hming/FusionRAG-pca-analysis')
ROOT = REPO / 'MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset'
FIG_DIR = ROOT / 'figures'
ALPHA_SUMMARY = ROOT / 'alpha_with_full_qk_draft_accuracy_summary.csv'
CURRENT_ONLINE = ROOT / 'current_rate015_vs_micro_figure_baseline_comparison.csv'
OUT_CSV = ROOT / 'dataset_method_accuracy_corrected_full_current_online.csv'

METHOD_ORDER = [
    'Full attention r=1',
    'Online QK r=0.15',
    'Online Draft r=0.15',
    'Uniform a=0.1 r=0.15',
    'Uniform a=0.25 r=0.15',
    'Random a=0.05 r=0.15',
    'Random a=0.1 r=0.15',
    'Random a=0.25 r=0.15',
]

ALPHA_NAME_MAP = {
    'Full attention r=1': 'Full attention r=1',
    'Uniform a=0.1': 'Uniform a=0.1 r=0.15',
    'Uniform a=0.25': 'Uniform a=0.25 r=0.15',
    'Random a=0.05': 'Random a=0.05 r=0.15',
    'Random a=0.1': 'Random a=0.1 r=0.15',
    'Random a=0.25': 'Random a=0.25 r=0.15',
}

DATASET_LABELS = {
    '2wikimqa': '2WikiMQA',
    '2WikiMQA': '2WikiMQA',
    'hotpotqa': 'HotpotQA',
    'HotpotQA': 'HotpotQA',
    'triviaqa': 'TriviaQA',
    'TriviaQA': 'TriviaQA',
    'musique': 'MuSiQue',
    'MuSiQue': 'MuSiQue',
}

DATASET_ORDER = ['2WikiMQA', 'HotpotQA', 'TriviaQA', 'MuSiQue']

COLORS = {
    'Full attention r=1': '#2f3a4a',
    'Online QK r=0.15': '#6f7d8c',
    'Online Draft r=0.15': '#98a2ad',
    'Uniform a=0.1 r=0.15': '#2a9d8f',
    'Uniform a=0.25 r=0.15': '#58b7aa',
    'Random a=0.05 r=0.15': '#8ab6d6',
    'Random a=0.1 r=0.15': '#6a9ecf',
    'Random a=0.25 r=0.15': '#4f83bd',
}


def load_rows() -> list[dict]:
    rows: dict[tuple[str, str], dict] = {}

    with ALPHA_SUMMARY.open(newline='', encoding='utf-8') as f:
        for raw in csv.DictReader(f):
            dataset = DATASET_LABELS.get(raw['dataset'], raw['dataset'])
            method = ALPHA_NAME_MAP.get(raw['method'])
            if method is None:
                continue
            rows[(dataset, method)] = {
                'dataset': dataset,
                'method': method,
                'correct': int(raw['correct']),
                'total': int(raw['total']),
                'acc': float(raw['acc']),
                'source': 'alpha_with_full_qk_draft_accuracy_summary.csv',
                'note': 'full/alpha from original corrected-full micro figure source',
            }

    with CURRENT_ONLINE.open(newline='', encoding='utf-8') as f:
        for raw in csv.DictReader(f):
            if raw['dataset'] == 'micro_all':
                continue
            dataset = DATASET_LABELS.get(raw['dataset'], raw['dataset'])
            method = {
                'online_qk': 'Online QK r=0.15',
                'online_draft': 'Online Draft r=0.15',
            }[raw['method']]
            rows[(dataset, method)] = {
                'dataset': dataset,
                'method': method,
                'correct': int(raw['new_rate015_correct']),
                'total': int(raw['new_rate015_total']),
                'acc': float(raw['new_rate015_acc']),
                'source': 'current_rate015_vs_micro_figure_baseline_comparison.csv',
                'note': 'current online rerun replacing historical online rows',
            }

    out = []
    for dataset in DATASET_ORDER:
        full = rows[(dataset, 'Full attention r=1')]
        ref = full['acc']
        for method in METHOD_ORDER:
            row = dict(rows[(dataset, method)])
            row['delta_vs_full_pp'] = (row['acc'] - ref) * 100
            out.append(row)
    return out


def write_csv(rows: list[dict]) -> None:
    fields = ['dataset', 'method', 'correct', 'total', 'acc', 'delta_vs_full_pp', 'source', 'note']
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot(rows: list[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), sharey=False)
    axes = axes.flatten()
    for ax, dataset in zip(axes, DATASET_ORDER):
        ds_rows = [r for r in rows if r['dataset'] == dataset]
        values = [r['acc'] * 100 for r in ds_rows]
        labels = [r['method'] for r in ds_rows]
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=[COLORS[l] for l in labels], width=0.68, edgecolor='white', linewidth=0.7)
        full_acc = ds_rows[0]['acc'] * 100
        ax.axhline(full_acc, color='#2f3a4a', linestyle='--', linewidth=1.0, alpha=0.9)
        for bar, row, val in zip(bars, ds_rows, values):
            text = f"{row['correct']}/{row['total']}\n{val:.1f}%"
            color = 'white' if row['method'].startswith(('Uniform', 'Random', 'Full')) else '#26313f'
            y = val - 1.0 if val > 55 else val + 1.0
            va = 'top' if val > 55 else 'bottom'
            ax.text(bar.get_x() + bar.get_width() / 2, y, text, ha='center', va=va, fontsize=7.5, color=color)
        ax.set_title(dataset, fontsize=12, weight='normal')
        ax.set_ylim(max(35, min(values) - 7), min(96, max(values) + 7))
        ax.grid(True, axis='y', color='#d0d7de', linewidth=0.8, alpha=0.6)
        ax.grid(False, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels([])
    for ax in axes[::2]:
        ax.set_ylabel('Main accuracy (%)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in METHOD_ORDER]
    fig.legend(handles, METHOD_ORDER, loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    png = FIG_DIR / 'dataset_method_accuracy_corrected_full_current_online.png'
    pdf = FIG_DIR / 'dataset_method_accuracy_corrected_full_current_online.pdf'
    fig.savefig(png, dpi=240, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    print(OUT_CSV)
    print(png)
    print(pdf)


def main() -> None:
    rows = load_rows()
    write_csv(rows)
    plot(rows)


if __name__ == '__main__':
    main()

