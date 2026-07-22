#!/usr/bin/env python3
import csv,json
from collections import OrderedDict,defaultdict
from pathlib import Path
EXP=Path('/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/qwen3_online_draft_trace_oracle_accuracy_rate015')
SEG=[(0,25),(25,50),(50,75),(75,100),(100,125),(125,150),(150,175),(175,200)]
def mean(xs): return sum(xs)/len(xs) if xs else 0.0
rows=OrderedDict(); text=''; missing=0
for s,e in SEG:
 p=EXP/f'seg_{s}_{e}/Qwen3-32B/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv'
 if not p.exists(): missing+=1; continue
 with p.open(newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f): rows[(r['Main Question'],r['Sub Question'])]=r
 lp=EXP/f'seg_{s}_{e}/run.log'
 if lp.exists(): text+='\n'+lp.read_text(encoding='utf-8',errors='ignore')
g=defaultdict(list)
for (mq,sq),r in rows.items(): g[mq].append(r)
out={'label':'online_draft_trace_oracle_fixed_rate015','main_correct':sum(1 for v in g.values() if all(str(r['Correct']).lower()=='true' for r in v)),'main_total':len(g),'sub_correct':sum(1 for r in rows.values() if str(r['Correct']).lower()=='true'),'sub_total':len(rows),'avg_f1':mean([float(r.get('F1') or 0) for r in rows.values()]),'avg_em':mean([float(r.get('EM') or 0) for r in rows.values()]),'missing_csv':missing,'finished':text.count('FINAL RESULTS'),'traceback':text.count('Traceback'),'killed':text.count('Killed')+text.count('Terminated')}
out['main_acc']=out['main_correct']/out['main_total'] if out['main_total'] else 0; out['sub_acc']=out['sub_correct']/out['sub_total'] if out['sub_total'] else 0
(EXP/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
md=f"# Online Draft Trace Oracle Fixed Accuracy\n\n| label | Main Acc | Sub Acc | F1 | EM | rows | finished | missing | traceback/killed |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n| {out['label']} | {out['main_correct']}/{out['main_total']} ({out['main_acc']:.2%}) | {out['sub_correct']}/{out['sub_total']} ({out['sub_acc']:.2%}) | {out['avg_f1']:.4f} | {out['avg_em']:.4f} | {out['sub_total']} | {out['finished']} | {out['missing_csv']} | {out['traceback']}/{out['killed']} |\n"
(EXP/'README.md').write_text(md,encoding='utf-8'); print(md)
