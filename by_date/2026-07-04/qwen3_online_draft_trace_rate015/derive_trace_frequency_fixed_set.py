#!/usr/bin/env python3
import json, re, csv
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path('/raid/home/hming/FusionRAG-pca-analysis')
TRACE=ROOT/'MOTIVATION_EXPERIMENTS/qwen3_online_draft_trace_rate015/selected_indices'
OUT=ROOT/'MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/online_draft_trace_oracle_rate015'
RATE=0.15

def parse_ids(p):
    m=re.search(r'example(\d+)_sub(\d+)_',p.name); return (int(m.group(1)),int(m.group(2)))

def main():
    counts=defaultdict(lambda: defaultdict(lambda: None))
    lengths=defaultdict(dict)
    rows=[]
    for p in sorted(TRACE.glob('*.json')):
        d=json.loads(p.read_text(encoding='utf-8'))
        ex=int(d['example_id']); selected=set(map(int,d['selected_abs']))
        lens=list(map(int,d['passages_len']))
        chunk_ids=list(map(int,d['chunk_ids']))
        starts=np.cumsum([0]+lens).tolist()
        # passages[:-1] includes system plus docs; chunk_id 0 is system.
        for pos_idx, chunk_id in enumerate(chunk_ids):
            if chunk_id == 0: continue
            start=starts[pos_idx]; end=starts[pos_idx+1]; L=end-start
            local_counts=counts[ex][chunk_id]
            if local_counts is None:
                local_counts=np.zeros(L,dtype=np.int32); counts[ex][chunk_id]=local_counts; lengths[ex][chunk_id]=L
            for abs_i in selected:
                if start <= abs_i < end:
                    local=abs_i-start
                    if 0 <= local < local_counts.shape[0]: local_counts[local]+=1
    outdir=OUT/'chunk_fixed_sets_npz'; outdir.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for ex, chunks in sorted(counts.items()):
        payload={}
        for chunk_id, cnt in sorted(chunks.items()):
            L=int(lengths[ex][chunk_id]); k=max(1,int(RATE*L))
            # tie-break by earlier local position for reproducibility
            order=sorted(range(L), key=lambda i:(-int(cnt[i]), int(i)))[:k]
            idx=np.array(sorted(order),dtype=np.int64)
            payload[f'chunk{chunk_id-1:02d}_online_draft_trace_frequency_per_chunk']=idx
            payload[f'chunk{chunk_id-1:02d}_online_draft_trace_frequency_count']=cnt.astype(np.int32)
            manifest.append({'example_id':ex,'chunk_id':chunk_id,'method':'online_draft_trace_frequency_per_chunk','rate':RATE,'chunk_len':L,'selected_count':len(idx),'max_count':int(cnt.max())})
        np.savez_compressed(outdir/f'example{ex:03d}_rate0p15_chunk_local_sets.npz',**payload)
    with (OUT/'fixed_set_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['example_id','chunk_id','method','rate','chunk_len','selected_count','max_count']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(manifest)
    (OUT/'README.md').write_text('# Online Draft Trace Oracle Fixed Set\n\nThis is a leaky diagnostic fixed set derived from true online draft selections on the evaluated questions. It is not a valid offline method; it estimates the upper bound if offline could predict online draft stable token choices.\n',encoding='utf-8')
    print('examples',len(counts),'npz',len(list(outdir.glob('*.npz'))),'manifest rows',len(manifest))
if __name__=='__main__': main()
