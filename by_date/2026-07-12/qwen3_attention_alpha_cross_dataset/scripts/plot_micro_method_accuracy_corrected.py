#!/usr/bin/env python3
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO = Path('/raid/home/hming/FusionRAG-pca-analysis')
ROOT = REPO / 'MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset'
FIG_DIR = ROOT / 'figures'
OUT_CSV = ROOT / 'micro_all_datasets_method_accuracy_corrected_full.csv'

METHODS = [
    ('Full attention r=1', 680, 865, 'corrected same-pipeline rerun'),
    ('Online QK r=0.15', 609, 865, 'historical baseline; rerun pending'),
    ('Online Draft r=0.15', 621, 865, 'historical baseline; rerun pending'),
    ('Uniform a=0.1 r=0.15', 671, 865, 'previous alpha sweep'),
    ('Uniform a=0.25 r=0.15', 666, 865, 'previous alpha sweep'),
    ('Random a=0.05 r=0.15', 671, 865, 'previous alpha sweep'),
    ('Random a=0.1 r=0.15', 670, 865, 'previous alpha sweep'),
    ('Random a=0.25 r=0.15', 668, 865, 'previous alpha sweep'),
]
REF = 680 / 865

FIG_DIR.mkdir(parents=True, exist_ok=True)
rows=[]
for method, correct, total, note in METHODS:
    acc=correct/total
    rows.append({'method':method,'correct':correct,'total':total,'acc':acc,'delta_vs_corrected_full_pp':(acc-REF)*100,'note':note})
with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

labels=[r['method'] for r in rows]
values=[r['acc']*100 for r in rows]
deltas=[r['delta_vs_corrected_full_pp'] for r in rows]
colors=[]
for label in labels:
    if label.startswith('Full attention'):
        colors.append('#2f3a4a')
    elif label.startswith('Online QK'):
        colors.append('#6f7d8c')
    elif label.startswith('Online Draft'):
        colors.append('#98a2ad')
    elif label.startswith('Uniform'):
        colors.append('#2a9d8f')
    else:
        colors.append('#88c7bc')

fig, ax = plt.subplots(figsize=(9.2, 4.1))
x=np.arange(len(labels))
bars=ax.bar(x, values, width=0.62, color=colors, edgecolor='white', linewidth=0.8)
ax.axhline(REF*100, color='#2f3a4a', linestyle='--', linewidth=1.1)
for bar, label, acc, delta in zip(bars, labels, values, deltas):
    if label.startswith(('Uniform', 'Random')):
        ax.text(
            bar.get_x()+bar.get_width()/2,
            bar.get_height()-0.55,
            f'{acc:.1f}%\n{delta:+.1f} pp',
            ha='center', va='top', fontsize=8, color='white'
        )
    else:
        ax.text(
            bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.35,
            f'{acc:.1f}%\n{delta:+.1f} pp',
            ha='center', va='bottom', fontsize=8, color='#26313f'
        )
ax.set_ylabel('Main accuracy (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=24, ha='right')
ax.set_ylim(69, 80.4)
ax.grid(True, axis='y', color='#d0d7de', linewidth=0.8, alpha=0.65)
ax.grid(False, axis='x')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
png=FIG_DIR/'micro_all_datasets_method_accuracy_corrected_full.png'
pdf=FIG_DIR/'micro_all_datasets_method_accuracy_corrected_full.pdf'
fig.savefig(png, dpi=240, bbox_inches='tight')
fig.savefig(pdf, bbox_inches='tight')
print(OUT_CSV)
print(png)
print(pdf)
