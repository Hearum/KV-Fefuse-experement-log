import csv, os
base='MOTIVATION_EXPERIMENTS/kv_update_rate_sweep_nojudge'
paths={
'0.15':'MOTIVATION_EXPERIMENTS/reflect_pipeline_strict_v_only_clean/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv',
'0.3':f'{base}/v_only_rate0p3/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.3_revert_rope.csv',
'0.5':f'{base}/v_only_rate0p5/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.5_revert_rope.csv',
'0.8':f'{base}/v_only_rate0p8/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.8_revert_rope.csv',
}
rows={}
for r,p in paths.items():
    print(r, os.path.exists(p), p)
    with open(p, newline='', encoding='utf-8') as f:
        data=list(csv.DictReader(f))
    rows[r]=data
    print(' rows',len(data),'avg_f1',sum(float(x['F1']) for x in data)/len(data),'avg_em',sum(float(x['EM']) for x in data)/len(data))
rates=list(paths)
for i in range(len(rates)):
  for j in range(i+1,len(rates)):
    r1,r2=rates[i],rates[j]
    same=sum(a.get('Predicted','')==b.get('Predicted','') for a,b in zip(rows[r1], rows[r2]))
    print(f'pair_same_pred {r1} {r2}: {same}/{len(rows[r1])}={same/len(rows[r1]):.3f}')
for r in rates[1:]:
    diffs=[float(b['F1'])-float(a['F1']) for a,b in zip(rows['0.15'], rows[r])]
    print(f'f1_delta 0.15->{r}: mean={sum(diffs)/len(diffs):.4f} improved={sum(d>1e-9 for d in diffs)} worse={sum(d<-1e-9 for d in diffs)} same={sum(abs(d)<=1e-9 for d in diffs)}')
# compare selected exact answer changed examples
for r in rates[1:]:
    changed=[]
    for idx,(a,b) in enumerate(zip(rows['0.15'], rows[r])):
        if a['Predicted'] != b['Predicted']:
            changed.append((idx,a['F1'],b['F1'],a['Predicted'][:60],b['Predicted'][:60],a['Sub Question'][:80]))
    print('\nexamples changed 0.15 ->', r, 'count', len(changed))
    for item in changed[:8]:
        print(item)
