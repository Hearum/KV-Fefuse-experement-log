import csv, json, statistics
from pathlib import Path
p=Path('/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/query_attention_mass_by_selection_rank/attention_mass_by_query.csv')
rows=list(csv.DictReader(open(p, encoding='utf-8')))
rates=[0.05,0.1,0.15,0.2,0.3,0.5,0.8]
print('rows',len(rows))
for rate in rates:
    col=f'top{rate:g}_mass'
    vals=[float(r[col]) for r in rows]
    ge50=sum(v>=0.5 for v in vals)/len(vals)
    ge70=sum(v>=0.7 for v in vals)/len(vals)
    ge80=sum(v>=0.8 for v in vals)/len(vals)
    print(rate, 'mean', statistics.mean(vals), 'p10', sorted(vals)[int(0.1*len(vals))], 'p50', statistics.median(vals), 'p90', sorted(vals)[int(0.9*len(vals))], '>=0.5', ge50, '>=0.7', ge70, '>=0.8', ge80, 'enrichment', statistics.mean(vals)/rate)
