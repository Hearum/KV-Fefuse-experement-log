#!/usr/bin/env python3
import csv,json,re
from collections import OrderedDict,defaultdict
from pathlib import Path
ROOT=Path('/raid/home/hming/FusionRAG-pca-analysis')
EXP=ROOT/'MOTIVATION_EXPERIMENTS/qwen3_offline_draft_aggregate_rate015'
MODEL='Qwen3-32B/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv'
SEG=[(0,25),(25,50),(50,75),(75,100),(100,125),(125,150),(150,175),(175,200)]
METHODS=['draft_frequency_per_chunk','draft_mean_score_per_chunk','draft_max_score_per_chunk','draft_top2_mean_score_per_chunk','draft_top4_mean_score_per_chunk']
def mean(xs): return sum(xs)/len(xs) if xs else 0.0
def rows(method):
 out=OrderedDict(); miss=[]
 for s,e in SEG:
  p=EXP/method/f'seg_{s}_{e}'/MODEL
  if not p.exists(): miss.append(str(p)); continue
  with p.open(newline='',encoding='utf-8') as f:
   for r in csv.DictReader(f): out[(r['Main Question'],r['Sub Question'])]=r
 return list(out.values()), miss
def logs(method):
 text=''; n=0
 for s,e in SEG:
  p=EXP/method/f'seg_{s}_{e}'/'run.log'
  if p.exists(): text+='\n'+p.read_text(encoding='utf-8',errors='ignore'); n+=1
 prompt=[float(x) for x in re.findall(r'^prompt eval duration:\s*([0-9.]+)s',text,re.M)]
 return {'log_files':n,'finished_segments':text.count('FINAL RESULTS'),'traceback':text.count('Traceback'),'killed':text.count('Killed')+text.count('Terminated'),'prefill_s':mean(prompt)}
def summarize(method):
 rs,miss=rows(method); g=defaultdict(list)
 for r in rs: g[r['Main Question']].append(r)
 ans={'method':method,'main_correct':sum(1 for v in g.values() if all(str(r['Correct']).lower()=='true' for r in v)),'main_total':len(g),'sub_correct':sum(1 for r in rs if str(r['Correct']).lower()=='true'),'sub_total':len(rs),'avg_f1':mean([float(r.get('F1') or 0) for r in rs]),'avg_em':mean([float(r.get('EM') or 0) for r in rs]),'missing_csv':len(miss)}
 ans['main_acc']=ans['main_correct']/ans['main_total'] if ans['main_total'] else 0
 ans['sub_acc']=ans['sub_correct']/ans['sub_total'] if ans['sub_total'] else 0
 ans.update(logs(method)); return ans
def main():
 data=[summarize(m) for m in METHODS]
 fields=list(data[0]) if data else []
 with (EXP/'summary.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)
 (EXP/'summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 lines=['# Qwen3 Offline Draft Aggregate Rate=0.15\n\n','| method | Main Acc | Sub Acc | F1 | EM | prefill(s) | rows | finished | missing | traceback/killed |\n','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n']
 for r in data:
  lines.append(f"| {r['method']} | {r['main_correct']}/{r['main_total']} ({r['main_acc']:.2%}) | {r['sub_correct']}/{r['sub_total']} ({r['sub_acc']:.2%}) | {r['avg_f1']:.4f} | {r['avg_em']:.4f} | {r['prefill_s']:.4f} | {r['sub_total']} | {r['finished_segments']} | {r['missing_csv']} | {r['traceback']}/{r['killed']} |\n")
 (EXP/'README.md').write_text(''.join(lines),encoding='utf-8')
 print(''.join(lines))
if __name__=='__main__': main()
