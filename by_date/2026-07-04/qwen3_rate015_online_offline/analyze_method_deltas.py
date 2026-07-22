#!/usr/bin/env python3
import csv
import json
from collections import defaultdict, OrderedDict
from pathlib import Path

ROOT=Path('/raid/home/hming/FusionRAG-pca-analysis')
EXP=ROOT/'MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline'
MODEL='Qwen3-32B/musique'
SEGMENTS=[(0,25),(25,50),(50,75),(75,100),(100,125),(125,150),(150,175),(175,200)]
METHODS={
 'online_qk':'online_qk_rate015/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv',
 'online_draft':'online_draft_rate015/DraftModel_global_topk10_bge/rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv',
 'offline_hybrid70':'offline_hybrid70_rate015/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv',
}

def read_method(method, rel):
    rows=OrderedDict()
    for s,e in SEGMENTS:
        p=EXP/method/f'seg_{s}_{e}'/MODEL/'/'.join(rel.split('/')[1:])
        # rel includes method prefix; rebuild explicitly below if needed
    return rows

def paths(method):
    if method=='online_qk': sub='FusionRAG_global_topk10_bge'; fn='rate_0.15_revert_rope.csv'
    elif method=='online_draft': sub='DraftModel_global_topk10_bge'; fn='rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv'
    elif method=='offline_hybrid70': sub='FusionRAG_global_topk10_bge'; fn='rate_0.15_revert_rope.csv'
    return [EXP/f'{method}_rate015'/f'seg_{s}_{e}'/MODEL/sub/fn for s,e in SEGMENTS]

def read_rows(method):
    out=OrderedDict()
    for p in paths(method):
        if not p.exists(): continue
        with p.open(newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                out[(r['Main Question'], r['Sub Question'])]=r
    return out

def main_status(rows):
    groups=defaultdict(list)
    for (mq,sq),r in rows.items(): groups[mq].append(r)
    return {mq: all(str(r['Correct']).lower()=='true' for r in rs) for mq,rs in groups.items()}

def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    data={m:read_rows(m) for m in METHODS}
    mstat={m:main_status(rows) for m,rows in data.items()}
    all_main=sorted(set().union(*[set(x) for x in mstat.values()]))
    comp=[]
    for mq in all_main:
        comp.append({'main_question':mq, **{m:mstat[m].get(mq, None) for m in METHODS}})
    fields=['main_question']+list(METHODS)
    write_csv(EXP/'method_main_correct_matrix.csv', comp, fields)
    cats={
      'draft_correct_offline_wrong':[r for r in comp if r['online_draft'] is True and r['offline_hybrid70'] is False],
      'offline_correct_draft_wrong':[r for r in comp if r['offline_hybrid70'] is True and r['online_draft'] is False],
      'draft_correct_qk_wrong':[r for r in comp if r['online_draft'] is True and r['online_qk'] is False],
      'qk_correct_draft_wrong':[r for r in comp if r['online_qk'] is True and r['online_draft'] is False],
    }
    for name,rows in cats.items():
        write_csv(EXP/f'{name}.csv', rows, fields)
    summary={name:len(rows) for name,rows in cats.items()}
    summary.update({m:sum(v is True for v in mstat[m].values()) for m in METHODS})
    (EXP/'method_delta_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[]
    lines.append('\n## Online Draft vs Offline Hybrid 差异诊断\n\n')
    lines.append('基于 Qwen3 rate=0.15 已完成结果，按 main question 聚合：一个 main question 的所有 sub-question 都正确才算正确。\n\n')
    lines.append('| category | count | meaning |\n|---|---:|---|\n')
    meaning={
      'draft_correct_offline_wrong':'online draft 答对、offline hybrid70 答错，offline 需要重点补的样本',
      'offline_correct_draft_wrong':'offline hybrid70 答对、online draft 答错，offline 固定集合反而有益的样本',
      'draft_correct_qk_wrong':'online draft 答对、online QK 答错，说明 draft selector 的优势样本',
      'qk_correct_draft_wrong':'online QK 答对、online draft 答错，说明 QK selector 的互补样本',
    }
    for k in ['draft_correct_offline_wrong','offline_correct_draft_wrong','draft_correct_qk_wrong','qk_correct_draft_wrong']:
        lines.append(f'| {k} | {summary[k]} | {meaning[k]} |\n')
    lines.append('\n差异样本 CSV：`draft_correct_offline_wrong.csv`、`offline_correct_draft_wrong.csv`、`draft_correct_qk_wrong.csv`、`qk_correct_draft_wrong.csv`。\n')
    with (EXP/'README.md').open('a',encoding='utf-8') as f: f.write(''.join(lines))
    print(''.join(lines))

if __name__=='__main__': main()
