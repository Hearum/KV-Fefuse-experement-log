import csv, math
base='MOTIVATION_EXPERIMENTS/kv_update_rate_sweep_nojudge'
paths={
'0.0':'MOTIVATION_EXPERIMENTS/reflect_pipeline_full_rate0/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.0_revert_rope.csv',
'0.15':'MOTIVATION_EXPERIMENTS/reflect_pipeline_strict_v_only_clean/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv',
'0.3':f'{base}/v_only_rate0p3/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.3_revert_rope.csv',
'0.5':f'{base}/v_only_rate0p5/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.5_revert_rope.csv',
'0.8':f'{base}/v_only_rate0p8/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.8_revert_rope.csv',
}
rows={}
for r,p in paths.items():
  with open(p, newline='', encoding='utf-8') as f: rows[r]=list(csv.DictReader(f))
base_rate='0.0'
for r in ['0.15','0.3','0.5','0.8']:
  diffs=[float(b['F1'])-float(a['F1']) for a,b in zip(rows[base_rate],rows[r])]
  bins={'<-0.5':0,'-0.5~-0.1':0,'-0.1~0':0,'0':0,'0~0.1':0,'0.1~0.5':0,'>0.5':0}
  for d in diffs:
    if d < -0.5: bins['<-0.5']+=1
    elif d < -0.1: bins['-0.5~-0.1']+=1
    elif d < -1e-9: bins['-0.1~0']+=1
    elif abs(d)<=1e-9: bins['0']+=1
    elif d <=0.1: bins['0~0.1']+=1
    elif d <=0.5: bins['0.1~0.5']+=1
    else: bins['>0.5']+=1
  print('\nvs rate0 ->',r,'mean',sum(diffs)/len(diffs),'improved',sum(d>1e-9 for d in diffs),'worse',sum(d<-1e-9 for d in diffs),'same',sum(abs(d)<=1e-9 for d in diffs))
  print(bins)
