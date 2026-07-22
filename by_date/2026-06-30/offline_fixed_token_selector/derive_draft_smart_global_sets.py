#!/usr/bin/env python3
"""Derive offline fixed sets from draft scores using online-style smart selection."""

import csv
from pathlib import Path
import numpy as np

ROOT = Path('/raid/home/hming/FusionRAG-pca-analysis')
SRC = ROOT / 'MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_draft_rate015_full/combined/score_cache_npz'
OUT = ROOT / 'MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_draft_smart_global_rate015_full'
RATE = 0.15
THRESHOLD_FACTOR = 0.5


def finite(x):
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def topn_mean(scores, n):
    scores = finite(scores)
    if scores.shape[0] <= n:
        return scores.mean(axis=0)
    part = np.partition(scores, scores.shape[0] - n, axis=0)[-n:]
    return part.mean(axis=0)


def smart_query_selection_local(attention_scores, target_ratio, threshold_factor=0.5):
    """Numpy equivalent of _fusionrag_smart_query_selection profile variant.

    Input/output indices are doc-local, i.e. 0..doc_len-1, without system offset.
    """
    attention_scores = finite(attention_scores)
    doc_len = int(attention_scores.shape[0])
    target_count = int(doc_len * target_ratio)
    if target_count <= 0:
        return np.array([], dtype=np.int64)
    target_count = min(target_count, doc_len)

    threshold = float(np.mean(attention_scores)) + threshold_factor * float(np.std(attention_scores))
    high_attn_positions = np.where(attention_scores > threshold)[0].astype(int).tolist()

    components = []
    current = []
    for pos in sorted(high_attn_positions):
        if not current or pos - current[-1] <= 2:
            current.append(pos)
        else:
            components.append(current)
            current = [pos]
    if current:
        components.append(current)

    component_scores = [(comp, float(sum(attention_scores[p] for p in comp))) for comp in components]
    component_scores.sort(key=lambda x: x[1], reverse=True)

    selected = set()
    for comp, _ in component_scores:
        extended = set()
        for p in comp:
            for offset in (-1, 0, 1):
                q = p + offset
                if 0 <= q < doc_len:
                    extended.add(q)
        new_positions = extended - selected
        if len(selected) + len(new_positions) <= target_count * 1.1:
            selected.update(extended)

    if len(selected) < target_count:
        for pos in np.argsort(attention_scores)[::-1]:
            selected.add(int(pos))
            if len(selected) >= target_count:
                break

    while len(selected) > target_count:
        selected.remove(min(selected, key=lambda p: attention_scores[p]))

    return np.array(sorted(selected), dtype=np.int64)


def split_doc_local_to_chunks(doc_indices, starts, system_len):
    doc_indices = np.asarray(sorted(set(map(int, doc_indices))), dtype=np.int64)
    payload = {}
    rows = []
    for chunk_idx, (abs_start, abs_end) in enumerate(zip(starts[1:-1], starts[2:])):
        local_doc_start = int(abs_start - system_len)
        local_doc_end = int(abs_end - system_len)
        mask = (doc_indices >= local_doc_start) & (doc_indices < local_doc_end)
        local = doc_indices[mask] - local_doc_start
        payload[chunk_idx] = np.asarray(local, dtype=np.int64)
        rows.append((chunk_idx, local_doc_end - local_doc_start, int(local.shape[0])))
    return payload, rows


def derive_sets(scores):
    scores = finite(scores)
    _, doc_len = scores.shape
    target_count = max(1, int(RATE * doc_len))

    per_query_selected = [smart_query_selection_local(s, RATE, THRESHOLD_FACTOR) for s in scores]
    freq = np.zeros(doc_len, dtype=np.int32)
    for idx in per_query_selected:
        freq[idx] += 1
    mean_score = scores.mean(axis=0)
    max_score = scores.max(axis=0)
    top2_score = topn_mean(scores, 2)

    def order_by(primary, secondary):
        order = sorted(range(doc_len), key=lambda i: (-float(primary[i]), -float(secondary[i]), int(i)))[:target_count]
        return np.array(sorted(order), dtype=np.int64)

    return {
        'draft_smart_frequency_global': order_by(freq, mean_score),
        'draft_smart_mean_score_global': order_by(mean_score, freq),
        'draft_smart_max_score_global': order_by(max_score, freq),
        'draft_smart_top2_mean_global': order_by(top2_score, freq),
    }, per_query_selected


def process(path):
    ex = int(path.stem.split('example')[1].split('_')[0])
    data = np.load(path, allow_pickle=True)
    scores = np.asarray(data['scores'], dtype=np.float32)
    starts = np.asarray(data['starts'], dtype=np.int64)
    system_len = int(np.asarray(data['system_len']))
    labels = [str(x) for x in data['labels']]
    queries = [str(x) for x in data['queries']]

    sets, per_query_selected = derive_sets(scores)
    out_payload = {}
    manifest_rows = []
    query_rows = []
    for q_idx, selected in enumerate(per_query_selected):
        query_rows.append({
            'example_id': ex,
            'query_idx': q_idx,
            'label': labels[q_idx] if q_idx < len(labels) else '',
            'query': queries[q_idx] if q_idx < len(queries) else '',
            'smart_selected_count': int(len(selected)),
        })

    for method, doc_idx in sets.items():
        chunk_payload, chunk_rows = split_doc_local_to_chunks(doc_idx, starts, system_len)
        for chunk_idx, local in chunk_payload.items():
            out_payload[f'chunk{chunk_idx:02d}_{method}'] = local
        for chunk_idx, chunk_len, selected_count in chunk_rows:
            manifest_rows.append({
                'example_id': ex,
                'chunk_id': chunk_idx,
                'method': method,
                'rate': RATE,
                'chunk_len': int(chunk_len),
                'selected_count': int(selected_count),
            })

    outp = OUT / 'chunk_fixed_sets_npz' / f'example{ex:03d}_rate0p15_chunk_local_sets.npz'
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, **out_payload)
    return manifest_rows, query_rows


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_manifest = []
    all_queries = []
    files = sorted(SRC.glob('reflect_draft_example*_scores.npz'))
    for path in files:
        manifest, queries = process(path)
        all_manifest.extend(manifest)
        all_queries.extend(queries)
    write_csv(OUT / 'fixed_set_manifest.csv', all_manifest)
    write_csv(OUT / 'calibration_query_manifest.csv', all_queries)
    (OUT / 'README.md').write_text(
        '# Draft Smart Global Offline Fixed Sets\n\n'
        '目的：避免和旧 `draft_frequency_per_chunk` 混淆，新增 `draft_smart_*` 命名。\n\n'
        '输入：`reflect_draft_rate015_full/combined/score_cache_npz` 中保存的 draft model query-to-doc scores。\n\n'
        '核心差异：每个 calibration query 先使用 online DraftModel 相同的 smart selection 后处理，'
        '再将多个 calibration query 的选择集合聚合成 offline fixed set。\n\n'
        '方法：\n\n'
        '- `draft_smart_frequency_global`: 对 smart-selected token 统计频率，频率优先，mean score 打破平局。\n'
        '- `draft_smart_mean_score_global`: 使用 calibration query 的 mean draft score 排序。\n'
        '- `draft_smart_max_score_global`: 使用 calibration query 的 max draft score 排序。\n'
        '- `draft_smart_top2_mean_global`: 使用每个 token top-2 calibration score 的均值排序。\n\n'
        '注意：这些方法是 global doc-level selection 后再拆成 chunk-local npz；不是旧的 per-chunk top-k。\n',
        encoding='utf-8',
    )
    print(f'wrote {len(files)} examples to {OUT}')


if __name__ == '__main__':
    main()
