import csv
from pathlib import Path

ROOT=Path('/raid/home/hming/FusionRAG-pca-analysis')
base=ROOT/'MOTIVATION_EXPERIMENTS/kv_update_rate_sweep_nojudge'
paths={
    ('kv',0.3): base/'kv_rate0p3/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.3_revert_rope.csv',
    ('v_only',0.3): base/'v_only_rate0p3/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.3_revert_rope.csv',
    ('k_only',0.3): base/'k_only_rate0p3/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.3_revert_rope.csv',
    ('kv',0.5): base/'kv_rate0p5/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.5_revert_rope.csv',
    ('v_only',0.5): base/'v_only_rate0p5/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.5_revert_rope.csv',
    ('k_only',0.5): base/'k_only_rate0p5/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.5_revert_rope.csv',
}

def load(p):
    with open(p, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def key(r):
    return (r['Main Question'], r['Sub Question'])

def trim(s,n=180):
    s=' '.join((s or '').split())
    return s if len(s)<=n else s[:n]+'...'

for rate in [0.3,0.5]:
    data={m:{key(r):r for r in load(paths[(m,rate)])} for m in ['kv','v_only','k_only']}
    common=set.intersection(*(set(x.keys()) for x in data.values()))
    cases=[]
    for k in common:
        rows={m:data[m][k] for m in data}
        f={m:float(rows[m]['F1']) for m in rows}
        pred={m:rows[m]['Predicted'] for m in rows}
        # prefer differentiated answers and clear score gaps
        score=(f['kv']-max(f['v_only'],f['k_only'])) + 0.5*(f['k_only']-f['v_only'])
        distinct=len(set(pred.values()))
        if distinct>=2:
            cases.append((score, abs(f['kv']-f['v_only'])+abs(f['kv']-f['k_only'])+abs(f['k_only']-f['v_only']), k, rows, f))
    cases=sorted(cases, key=lambda x:(x[0],x[1]), reverse=True)
    print(f'\n===== rate={rate} selected examples =====')
    picked=[]
    # mix: kv best, k_only > v_only, v_only sometimes best/worse
    for item in cases:
        _,_,k,rows,f=item
        if len(picked)>=8: break
        # avoid too many zero-zero cases
        if max(f.values())==0: continue
        picked.append(item)
    for idx,item in enumerate(picked,1):
        _,_,k,rows,f=item
        gt=rows['kv']['Ground Truth']
        print(f'\nCASE {idx}')
        print('Main:', trim(k[0],140))
        print('Sub :', trim(k[1],140))
        print('GT  :', trim(gt,220))
        for m in ['kv','k_only','v_only']:
            print(f'{m:6s} F1={float(rows[m]["F1"]):.4f} EM={float(rows[m]["EM"]):.1f} Pred: {trim(rows[m]["Predicted"],260)}')
