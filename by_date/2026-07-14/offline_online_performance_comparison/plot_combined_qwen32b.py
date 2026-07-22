#!/usr/bin/env python3
import csv, re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
repo=Path('/raid/home/hming/FusionRAG-pca-analysis')
if not repo.exists(): repo=Path('/home/hming/FusionRAG-pca-analysis')
root=repo/'MOTIVATION_EXPERIMENTS/offline_online_performance_comparison'
fig_dir=root/'figures'
rows=[]
with (root/'offline_online_performance_summary.csv').open(encoding='utf-8', newline='') as f:
    for r in csv.DictReader(f):
        m=re.search(r'\(([-0-9.]+)%\)', r['sub_acc'] or '')
        r['sub_pct']=float(m.group(1)) if m else np.nan
        rows.append(r)
colors={
 'full_rate1':'#444444',
 'online_qk_rate015':'#2563eb',
 'online_draft_rate015':'#fc8d62',
 'offline3b_mean':'#66c2a5',
 'draft_smart_mean_score_global':'#66c2a5',
 'offline3b_freq_boundary2':'#9bd8c4',
 'draft_smart_freq_boundary0p02_global':'#9bd8c4',
 'offline32b_top2':'#8da0cb',
 'draft32b_smart_top2_mean_global':'#8da0cb',
}
labels={
 'full_rate1':'Full',
 'online_qk_rate015':'Online QK',
 'online_draft_rate015':'Online Draft',
 'offline3b_mean':'Offline 3B mean',
 'draft_smart_mean_score_global':'Offline 3B mean',
 'offline3b_freq_boundary2':'Offline + boundary',
 'draft_smart_freq_boundary0p02_global':'Offline + boundary',
 'offline32b_top2':'Offline 32B top2',
 'draft32b_smart_top2_mean_global':'Offline 32B top2',
}
# Use only common comparable methods. For MuSiQue map its best analogous rows into the same method family.
method_by_dataset={
 'musique':['full_rate1','online_qk_rate015','online_draft_rate015','draft_smart_mean_score_global','draft_smart_freq_boundary0p02_global','draft32b_smart_top2_mean_global'],
 '2wikimqa':['full_rate1','online_qk_rate015','online_draft_rate015','offline3b_mean','offline3b_freq_boundary2','offline32b_top2'],
 'hotpotqa':['full_rate1','online_qk_rate015','online_draft_rate015','offline3b_mean','offline3b_freq_boundary2','offline32b_top2'],
 'triviaqa':['full_rate1','online_qk_rate015','online_draft_rate015','offline3b_mean','offline3b_freq_boundary2'],
}
datasets=['musique','2wikimqa','hotpotqa','triviaqa']
fig,axes=plt.subplots(1,4,figsize=(17.2,4.4),sharey=True)
for ax,ds in zip(axes,datasets):
    data=[]
    for method in method_by_dataset[ds]:
        matches=[r for r in rows if r['model']=='Qwen3-32B' and r['dataset']==ds and r['method']==method]
        if matches: data.append(matches[0])
    x=np.arange(len(data))
    ax.bar(x,[r['sub_pct'] for r in data],0.68,color=[colors.get(r['method'],'#66c2a5') for r in data],edgecolor='white',linewidth=.5)
    ax.set_title(ds,fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(r['method'],r['method']) for r in data],rotation=34,ha='right')
    ax.grid(True,axis='y',color='#d0d7de',linewidth=.8,alpha=.65)
    ax.grid(True,axis='x',color='#eaeef2',linewidth=.6,alpha=.35)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)
axes[0].set_ylabel('Sub accuracy (%)',fontsize=12)
axes[0].set_ylim(45,88)
fig.tight_layout()
fig_dir.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']:
    out=fig_dir/f'qwen3_32b_all_datasets_common_methods_accuracy.{ext}'
    fig.savefig(out,dpi=240 if ext=='png' else None,bbox_inches='tight')
    print(out)
