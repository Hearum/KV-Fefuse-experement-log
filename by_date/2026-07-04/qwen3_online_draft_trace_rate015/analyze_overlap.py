#!/usr/bin/env python3
import csv, json, re
from pathlib import Path
from collections import defaultdict

ROOT=Path('/raid/home/hming/FusionRAG-pca-analysis')
TRACE=ROOT/'MOTIVATION_EXPERIMENTS/qwen3_online_draft_trace_rate015/selected_indices'
OFF_ROOT=ROOT/'MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/offline_hybrid70_rate015'
DELTA_DIR=ROOT/'MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline'
OUT=ROOT/'MOTIVATION_EXPERIMENTS/qwen3_online_draft_trace_rate015'
DATA=ROOT/'data/result_reflect.json'

def parse_ids(path):
    m=re.search(r'example(\d+)_sub(\d+)_', path.name)
    return (int(m.group(1)), int(m.group(2))) if m else None

def load_jsons(paths):
    out={}
    for p in paths:
        ids=parse_ids(p)
        if ids is None: continue
        d=json.loads(p.read_text(encoding='utf-8'))
        out[ids]=set(map(int,d.get('selected_abs',[])))
    return out

def load_questions():
    data=json.loads(DATA.read_text(encoding='utf-8'))
    return {i:x.get('question','') for i,x in enumerate(data)}

def load_category_questions(name):
    p=DELTA_DIR/f'{name}.csv'
    if not p.exists(): return set()
    with p.open(newline='',encoding='utf-8') as f:
        return {r['main_question'] for r in csv.DictReader(f)}

def metrics(a,b):
    inter=len(a&b); union=len(a|b)
    return {
        'online_count':len(a),'offline_count':len(b),'intersection':inter,'union':union,
        'jaccard':inter/union if union else 0.0,
        'online_recall_by_offline':inter/len(a) if a else 0.0,
        'offline_precision_vs_online':inter/len(b) if b else 0.0,
    }

def mean(xs): return sum(xs)/len(xs) if xs else 0.0

def summarize(rows):
    keys=['jaccard','online_recall_by_offline','offline_precision_vs_online','online_count','offline_count','intersection']
    return {k:mean([float(r[k]) for r in rows]) for k in keys} | {'n':len(rows)}

def main():
    online=load_jsons(TRACE.glob('*.json'))
    offline=load_jsons(OFF_ROOT.glob('seg_*/offline_fixed_selected_indices/*.json'))
    qmap=load_questions()
    categories={
        'draft_correct_offline_wrong':load_category_questions('draft_correct_offline_wrong'),
        'offline_correct_draft_wrong':load_category_questions('offline_correct_draft_wrong'),
        'draft_correct_qk_wrong':load_category_questions('draft_correct_qk_wrong'),
        'qk_correct_draft_wrong':load_category_questions('qk_correct_draft_wrong'),
    }
    rows=[]
    for ids in sorted(set(online)&set(offline)):
        ex,sub=ids
        m=metrics(online[ids],offline[ids]); m.update({'example_id':ex,'sub_q_idx':sub,'main_question':qmap.get(ex,'')})
        for cat,qs in categories.items(): m[cat]=qmap.get(ex,'') in qs
        rows.append(m)
    fields=['example_id','sub_q_idx','main_question','online_count','offline_count','intersection','union','jaccard','online_recall_by_offline','offline_precision_vs_online']+list(categories)
    with (OUT/'online_draft_vs_offline_hybrid_overlap.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summaries={'all':summarize(rows)}
    for cat in categories:
        summaries[cat]=summarize([r for r in rows if r[cat]])
    (OUT/'online_draft_vs_offline_hybrid_overlap_summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Online Draft vs Offline Hybrid Token Overlap\n\n']
    lines.append('| subset | n subq | Jaccard | online recall by offline | offline precision vs online | online tokens | offline tokens |\n')
    lines.append('|---|---:|---:|---:|---:|---:|---:|\n')
    for name,s in summaries.items():
        lines.append(f"| {name} | {s['n']} | {s['jaccard']:.4f} | {s['online_recall_by_offline']:.4f} | {s['offline_precision_vs_online']:.4f} | {s['online_count']:.1f} | {s['offline_count']:.1f} |\n")
    lines.append('\nInterpretation: `online recall by offline` is the fraction of online draft-selected tokens also covered by offline hybrid. Low values mean offline misses many online draft choices.\n')
    (OUT/'OVERLAP_README.md').write_text(''.join(lines),encoding='utf-8')
    print(''.join(lines))

if __name__=='__main__': main()
