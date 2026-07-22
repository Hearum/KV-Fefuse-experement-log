#!/usr/bin/env python3
from __future__ import annotations
import csv, json, re, time, threading, urllib.request, glob
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL='http://36.150.226.221:32355/v1'
API_KEY='api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS'
MODEL='GLM-5.2'
ROOT=Path('/raid/home/hming/FusionRAG-pca-analysis')
OLD=ROOT/'MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/online_draft_rate015'
OUT=ROOT/'MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/rejudge_old_online_glm'
PAT=str(OLD/'seg_*/Qwen3-32B/musique/DraftModel_global_topk10_bge/rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv')
CACHE=OUT/'judge_cache_old_draft_rate015_glm52.jsonl'
OUTCSV=OUT/'old_online_draft_rate015_rejudged_glm52.csv'
SUMMARY=OUT/'summary.json'

def parse_judge(text):
    text=(text or '').strip(); correct=False; reason=text
    lines=text.splitlines()
    for i,line in enumerate(lines):
        s=line.strip()
        if (s.startswith('判断') or s.lower().startswith('judgment')) and (':' in s or '：' in s):
            value=re.split('[:：]', s, maxsplit=1)[-1].strip()
            if '正确' in value or '对' in value or value.lower().startswith('yes'):
                correct=True
            elif '错误' in value or '错' in value or value.lower().startswith('no'):
                correct=False
        if s.startswith('原因') or s.lower().startswith('reason'):
            value=re.split('[:：]', s, maxsplit=1)[-1].strip() if (':' in s or '：' in s) else ''
            tail='\n'.join(lines[i+1:]).strip()
            reason=(value+'\n'+tail).strip() if tail else value
            if reason: break
    return correct, reason or text

def make_prompt(q,gold,pred):
    return f'''你是一个答案评估专家。你的任务是判断预测答案是否正确地回答了问题。

问题: {q}

标准答案: {gold}

预测答案: {pred}

请判断预测答案是否正确回答了问题。判断标准：
1. 预测答案包含了标准答案的关键信息
2. 预测答案与标准答案在语义上等价
3. 允许措辞上的细微差异，只要意思保持一致即可

请按照以下格式回答：
判断: [正确/错误]
原因: [详细说明为什么正确或错误，至少30字]'''

def call(prompt, retries=4, timeout=60):
    payload={
        'model':MODEL,
        'stream':False,
        'temperature':0,
        'thinking':{'type':'disabled'},
        'max_tokens':300,
        'messages':[{'role':'system','content':'你是一个专业的答案评估专家。'}, {'role':'user','content':prompt}],
    }
    data=json.dumps(payload, ensure_ascii=False).encode('utf-8')
    last=None
    for i in range(retries+1):
        try:
            req=urllib.request.Request(BASE_URL.rstrip()+'/chat/completions', data=data, headers={'Authorization':f'Bearer {API_KEY}','Content-Type':'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                obj=json.loads(resp.read().decode('utf-8'))
            raw=obj['choices'][0]['message']['content'].strip()
            c,r=parse_judge(raw)
            return c,r,raw
        except Exception as e:
            last=e; time.sleep(min(2**i,8))
    return False, f'GLM judge failed: {last}', ''

def load_cache():
    d={}
    if CACHE.exists():
        with CACHE.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item=json.loads(line); d[item['cache_key']]=item
    return d

def append_cache(item, lock):
    with lock:
        with CACHE.open('a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False)+'\n')

def clean_pred(s):
    # This rejudge intentionally evaluates the raw old CSV prediction, not a cleaned answer.
    return (s or '').strip()

def read_rows():
    rows=[]
    for p in sorted(glob.glob(PAT)):
        seg=re.search(r'seg_(\d+)_(\d+)', p).group(0)
        with open(p, newline='', encoding='utf-8') as f:
            for i,r in enumerate(csv.DictReader(f)):
                r['_source_csv']=p; r['_seg']=seg; r['_row']=i
                rows.append(r)
    return rows

def is_true(v): return str(v).strip().lower() in ('true','1','yes','correct')

def summarize(rows, correct_key):
    # Match old summary: dedupe by (Main Question, Sub Question) with last row kept.
    by={}
    for r in rows:
        by[(r['Main Question'].strip(), r['Sub Question'].strip())]=r
    sub_total=len(by); sub_correct=sum(is_true(r[correct_key]) for r in by.values())
    groups=defaultdict(list)
    for (m,s),r in by.items(): groups[m].append(is_true(r[correct_key]))
    main_total=len(groups); main_correct=sum(all(v) for v in groups.values())
    return {'main_correct':main_correct,'main_total':main_total,'main_acc':main_correct/main_total,'sub_correct':sub_correct,'sub_total':sub_total,'sub_acc':sub_correct/sub_total}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows=read_rows(); cache=load_cache(); lock=threading.Lock()
    print('loaded rows', len(rows), 'cache', len(cache), flush=True)
    def work(idx_r):
        idx,r=idx_r
        q=(r.get('Sub Question') or r.get('Main Question') or '').strip()
        gold=(r.get('Ground Truth') or '').strip()
        pred=clean_pred(r.get('Predicted'))
        key=json.dumps({'q':q,'gold':gold,'pred':pred,'model':MODEL,'prompt':'main_judge_prompt_raw_pred_v1'}, ensure_ascii=False, sort_keys=True)
        if key in cache:
            item=cache[key]
        else:
            c,reason,raw=call(make_prompt(q,gold,pred))
            item={'cache_key':key,'correct':c,'reason':reason,'raw':raw}
            append_cache(item, lock)
        out=dict(r)
        out['Rejudge_Correct']='True' if item['correct'] else 'False'
        out['Rejudge_Reason']=item['reason']
        out['Rejudge_Raw']=item.get('raw','')
        return idx,out
    out=[None]*len(rows)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs=[ex.submit(work,x) for x in enumerate(rows)]
        done=0
        for fut in as_completed(futs):
            idx,row=fut.result(); out[idx]=row; done+=1
            if done%25==0: print('judged',done,'/',len(rows), flush=True)
    fields=list(out[0].keys())
    with OUTCSV.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
    old_summary=summarize(out,'Correct')
    new_summary=summarize(out,'Rejudge_Correct')
    flips=[]
    for r in out:
        if is_true(r['Correct']) != is_true(r['Rejudge_Correct']):
            flips.append({'main':r['Main Question'],'sub':r['Sub Question'],'old_correct':r['Correct'],'rejudge_correct':r['Rejudge_Correct'],'predicted':r['Predicted'],'old_reason':r.get('Reason',''),'rejudge_reason':r.get('Rejudge_Reason','')})
    summary={'input_pattern':PAT,'output_csv':str(OUTCSV),'old_summary':old_summary,'rejudge_summary':new_summary,'raw_rows':len(rows),'flips_raw_rows':len(flips),'old_true_to_rejudge_false':sum(is_true(x['old_correct']) and not is_true(x['rejudge_correct']) for x in flips),'old_false_to_rejudge_true':sum((not is_true(x['old_correct'])) and is_true(x['rejudge_correct']) for x in flips)}
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    with (OUT/'flips_sample.json').open('w', encoding='utf-8') as f: json.dump(flips[:80], f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
if __name__=='__main__': main()
