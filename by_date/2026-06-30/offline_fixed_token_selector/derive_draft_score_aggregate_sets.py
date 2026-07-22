#!/usr/bin/env python3
import csv
from pathlib import Path
import numpy as np

ROOT = Path('/raid/home/hming/FusionRAG-pca-analysis')
SRC = ROOT/'MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_draft_rate015_full/combined/score_cache_npz'
OUT = ROOT/'MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_draft_score_aggregates_rate015_full'
RATE = 0.15

def topk(scores, k):
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    k = max(1, min(int(k), int(scores.shape[0])))
    idx = np.argpartition(-scores, k-1)[:k]
    return np.array(sorted(idx), dtype=np.int64)

def select_frequency(chunk_scores, k):
    n, L = chunk_scores.shape
    freq = np.zeros(L, dtype=np.int32)
    mean = np.nan_to_num(chunk_scores, nan=0.0, posinf=0.0, neginf=0.0).mean(axis=0)
    for s in chunk_scores:
        freq[topk(s, k)] += 1
    order = sorted(range(L), key=lambda i: (-int(freq[i]), -float(mean[i]), int(i)))[:k]
    return np.array(sorted(order), dtype=np.int64)

def topn_mean(chunk_scores, n):
    cs = np.nan_to_num(chunk_scores, nan=0.0, posinf=0.0, neginf=0.0)
    if cs.shape[0] <= n:
        return cs.mean(axis=0)
    part = np.partition(cs, cs.shape[0]-n, axis=0)[-n:]
    return part.mean(axis=0)

def process(path):
    ex = int(path.stem.split('example')[1].split('_')[0])
    d = np.load(path, allow_pickle=True)
    scores = np.asarray(d['scores'], dtype=np.float32)  # [calib_q, doc_len]
    starts = np.asarray(d['starts'], dtype=np.int64)
    system_len = int(np.asarray(d['system_len']))
    payload = {}
    rows = []
    for chunk_idx, (abs_start, abs_end) in enumerate(zip(starts[1:-1], starts[2:])):
        doc_start = int(abs_start - system_len)
        doc_end = int(abs_end - system_len)
        chunk_scores = scores[:, doc_start:doc_end]
        L = int(chunk_scores.shape[1])
        k = max(1, int(RATE * L))
        methods = {
            'draft_frequency_per_chunk': select_frequency(chunk_scores, k),
            'draft_mean_score_per_chunk': topk(chunk_scores.mean(axis=0), k),
            'draft_max_score_per_chunk': topk(chunk_scores.max(axis=0), k),
            'draft_top2_mean_score_per_chunk': topk(topn_mean(chunk_scores, 2), k),
            'draft_top4_mean_score_per_chunk': topk(topn_mean(chunk_scores, 4), k),
        }
        for name, idx in methods.items():
            payload[f'chunk{chunk_idx:02d}_{name}'] = idx
            rows.append({'example_id': ex, 'chunk_id': chunk_idx, 'method': name, 'rate': RATE, 'chunk_len': L, 'selected_count': len(idx)})
    outp = OUT/'chunk_fixed_sets_npz'/f'example{ex:03d}_rate0p15_chunk_local_sets.npz'
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, **payload)
    return rows

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows=[]
    for p in sorted(SRC.glob('reflect_draft_example*_scores.npz')):
        all_rows.extend(process(p))
    fields=['example_id','chunk_id','method','rate','chunk_len','selected_count']
    with (OUT/'fixed_set_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(all_rows)
    (OUT/'README.md').write_text(
        '# Derived Draft Score Aggregate Fixed Sets\n\n'
        '- source score cache: `reflect_draft_rate015_full/combined/score_cache_npz`\n'
        '- rate: 0.15\n'
        '- calibration queries: existing 8 unrelated control queries per example\n'
        '- methods: `draft_frequency_per_chunk`, `draft_mean_score_per_chunk`, `draft_max_score_per_chunk`, `draft_top2_mean_score_per_chunk`, `draft_top4_mean_score_per_chunk`\n'
        '- no model forward was rerun; this is derived from saved score tensors.\n', encoding='utf-8')
    print(f'wrote {len(list((OUT/"chunk_fixed_sets_npz").glob("*.npz")))} npz files to {OUT}')

if __name__ == '__main__': main()
