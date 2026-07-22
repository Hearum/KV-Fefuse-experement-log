#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import re
import threading
import time
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT_CANDIDATES = [Path('/raid/home/hming/FusionRAG-pca-analysis'), Path('/home/hming/FusionRAG-pca-analysis')]
ROOT = next((path for path in ROOT_CANDIDATES if path.exists()), ROOT_CANDIDATES[0])
EXP = ROOT / 'MOTIVATION_EXPERIMENTS/by_date/2026-07-15/qwen3_2wiki_v2_layer_selective_recompute'
OUT = Path(os.environ.get('LAYER_SELECTIVE_REJUDGE_OUT_DIR', str(EXP / 'rejudge_glm_clean_full_20260715')))
BASE_URL = os.environ.get('GLM_BASE_URL', 'http://36.150.226.221:32355/v1')
API_KEY = os.environ.get('GLM_API_KEY')
if not API_KEY:
    # Reuse the existing local evaluator config so setup-v2 is judged with the same endpoint/key.
    import importlib.util
    old = ROOT / 'MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rejudge_glm_clean_20260714/rejudge_cross_dataset_glm_clean.py'
    spec = importlib.util.spec_from_file_location('old_glm_rejudge_config', old)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    API_KEY = mod.API_KEY
JUDGE_MODEL = os.environ.get('GLM_JUDGE_MODEL', 'GLM-5.2')
PROMPT_VERSION = 'setup_v2_clean_think_answer_v1'
METHOD_FILTER = {x.strip() for x in os.environ.get('SETUP_V2_REJUDGE_METHODS', '').split(',') if x.strip()}
DATASET_FILTER = {x.strip() for x in os.environ.get('SETUP_V2_REJUDGE_DATASETS', '').split(',') if x.strip()}
RATE_FILTER = {x.strip() for x in os.environ.get('SETUP_V2_REJUDGE_RATES', '').split(',') if x.strip()}
COMPLETE_ONLY = os.environ.get('SETUP_V2_REJUDGE_COMPLETE_ONLY', '0').lower() in {'1', 'true', 'yes'}
EXPECTED_ROWS = OrderedDict([('2wikimqa-v2', 200)])
CACHE = OUT / 'judge_cache_glm52_clean.jsonl'
ROWS_CSV = OUT / 'rejudged_rows.csv'
SUMMARY_CSV = OUT / 'rejudged_summary.csv'
SUMMARY_JSON = OUT / 'rejudged_summary.json'


def clean_pred(text: str | None) -> str:
    text = (text or '').strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'^\s*(?:answer|final answer|答案)\s*[:：]\s*', '', text, flags=re.IGNORECASE).strip()
    return text


def make_prompt(question: str, gold: str, pred: str) -> str:
    return f'''你是一个答案评估专家。你的任务是判断预测答案是否正确地回答了问题。

问题: {question}

标准答案: {gold}

预测答案: {pred}

请判断预测答案是否正确回答了问题。判断标准：
1. 预测答案包含了标准答案的关键信息。
2. 预测答案与标准答案在语义上等价。
3. 允许措辞上的细微差异，只要意思保持一致即可。
4. 如果预测答案为空、只包含思考标记、或没有给出可核验答案，判断为错误。

请严格按照以下格式回答：
判断: [正确/错误]
原因: [详细说明为什么正确或错误，至少30字]'''


def parse_judge(text: str):
    text = (text or '').strip()
    correct = False
    reason = text
    for i, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if (stripped.startswith('判断') or stripped.lower().startswith('judgment')) and (':' in stripped or '：' in stripped):
            value = re.split('[:：]', stripped, maxsplit=1)[-1].strip()
            if '正确' in value or value.lower().startswith('yes'):
                correct = True
            elif '错误' in value or value.lower().startswith('no'):
                correct = False
        if stripped.startswith('原因') or stripped.lower().startswith('reason'):
            value = re.split('[:：]', stripped, maxsplit=1)[-1].strip() if (':' in stripped or '：' in stripped) else ''
            tail = '\n'.join(text.splitlines()[i + 1:]).strip()
            reason = (value + '\n' + tail).strip() if tail else value
            break
    return correct, reason or text


def call_glm(prompt: str, retries: int = 4, timeout: int = 90):
    payload = {
        'model': JUDGE_MODEL,
        'stream': False,
        'temperature': 0,
        'top_p': 1.0,
        'top_k': 1,
        'seed': 0,
        'thinking': {'type': 'disabled'},
        'max_tokens': 300,
        'messages': [
            {'role': 'system', 'content': '你是一个专业的答案评估专家。'},
            {'role': 'user', 'content': prompt},
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    last_error = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                BASE_URL.rstrip() + '/chat/completions',
                data=data,
                headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},
                method='POST',
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                obj = json.loads(response.read().decode('utf-8'))
            raw = obj['choices'][0]['message']['content'].strip()
            correct, reason = parse_judge(raw)
            return correct, reason, raw
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2 ** attempt, 10))
    return False, f'GLM judge failed: {last_error}', ''


def cache_key(question: str, gold: str, pred: str) -> str:
    raw = json.dumps({'q': question, 'gold': gold, 'pred': pred, 'model': JUDGE_MODEL, 'prompt': PROMPT_VERSION}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def load_cache():
    cache = {}
    if CACHE.exists():
        with CACHE.open('r', encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    cache[item['cache_key']] = item
    return cache


def append_cache(item: dict, lock: threading.Lock):
    with lock:
        with CACHE.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')



def load_complete_keys():
    return None

def iter_prediction_rows():
    rows = []
    complete_keys = load_complete_keys()
    for path_str in sorted(glob.glob(str(EXP / 'results/*/online_qk/2wikimqa-v2/rate_*/*/csv/reprocess_method_*.csv'))):
        path = Path(path_str)
        condition, _online, dataset, rate_tag, segment = path.relative_to(EXP / 'results').parts[:5]
        method = condition
        rate = rate_tag.replace('rate_', '').replace('p', '.')
        if METHOD_FILTER and method not in METHOD_FILTER:
            continue
        if DATASET_FILTER and dataset not in DATASET_FILTER:
            continue
        if RATE_FILTER and rate not in RATE_FILTER:
            continue
        if complete_keys is not None and (dataset, method, rate) not in complete_keys:
            continue
        with path.open('r', encoding='utf-8', newline='') as handle:
            for source_row, row in enumerate(csv.DictReader(handle)):
                question = row.get('Question') or row.get('Main Question') or row.get('Sub Question') or ''
                gold = row.get('Real Answer') or row.get('Ground Truth') or row.get('Gold') or ''
                pred = row.get('Pred Answer') or row.get('Answer') or row.get('Predicted') or ''
                rows.append({
                    'dataset': dataset,
                    'method': method,
                    'rate': rate,
                    'segment': segment,
                    'source_csv': str(path),
                    'source_row': source_row,
                    'Question': question,
                    'Real Answer': gold,
                    'Pred Answer': pred,
                })
    dedup = OrderedDict()
    for row in rows:
        key = (row['dataset'], row['method'], row['rate'], row['Question'])
        dedup.setdefault(key, row)
    return list(dedup.values()), len(rows)


def write_csv(path: Path, rows: list[dict]):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, raw_rows = iter_prediction_rows()
    cache = load_cache()
    lock = threading.Lock()
    print(f'loaded_dedup_rows={len(rows)} raw_rows={raw_rows} cache={len(cache)}', flush=True)

    def work(index_row):
        index, row = index_row
        question = row['Question'].strip()
        gold = row['Real Answer'].strip()
        pred_clean = clean_pred(row['Pred Answer'])
        key = cache_key(question, gold, pred_clean)
        item = cache.get(key)
        if item is None:
            correct, reason, raw = call_glm(make_prompt(question, gold, pred_clean))
            item = {'cache_key': key, 'correct': correct, 'reason': reason, 'raw': raw}
            append_cache(item, lock)
        out = dict(row)
        out['Predicted_Clean'] = pred_clean
        out['Rejudge_Correct'] = 'True' if item['correct'] else 'False'
        out['Rejudge_Reason'] = item.get('reason', '')
        out['Rejudge_Raw'] = item.get('raw', '')
        return index, out

    out_rows = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=int(os.environ.get('GLM_REJUDGE_WORKERS', '10'))) as executor:
        futures = [executor.submit(work, item) for item in enumerate(rows)]
        for done, future in enumerate(as_completed(futures), 1):
            index, row = future.result()
            out_rows[index] = row
            if done % 100 == 0 or done == len(rows):
                print(f'judged={done}/{len(rows)}', flush=True)
    out_rows = [row for row in out_rows if row is not None]
    write_csv(ROWS_CSV, out_rows)

    summary_rows = []
    methods = sorted({row['method'] for row in out_rows})
    for dataset, expected in EXPECTED_ROWS.items():
        for method in methods:
            group = [row for row in out_rows if row['dataset'] == dataset and row['method'] == method]
            if not group:
                continue
            correct = sum(1 for row in group if row['Rejudge_Correct'] == 'True')
            summary_rows.append({
                'dataset': dataset,
                'method': method,
                'rate': group[0]['rate'],
                'rows': len(group),
                'expected_rows': expected,
                'complete': len(group) >= expected,
                'glm_correct': correct,
                'glm_acc': correct / len(group) if group else 0.0,
            })
    write_csv(SUMMARY_CSV, summary_rows)
    SUMMARY_JSON.write_text(json.dumps({'raw_rows': raw_rows, 'summary': summary_rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(SUMMARY_CSV)


if __name__ == '__main__':
    main()
