#!/usr/bin/env python3
"""Derive 32B-teacher smart offline sets with small boundary-token replacement."""

import argparse
import csv
from pathlib import Path

import numpy as np


def finite(x):
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def topn_mean(scores, n):
    scores = finite(scores)
    if scores.shape[0] <= n:
        return scores.mean(axis=0)
    part = np.partition(scores, scores.shape[0] - n, axis=0)[-n:]
    return part.mean(axis=0)


def smart_query_selection_local(attention_scores, target_ratio, threshold_factor):
    attention_scores = finite(attention_scores)
    doc_len = int(attention_scores.shape[0])
    target_count = int(doc_len * target_ratio)
    if target_count <= 0:
        return np.array([], dtype=np.int64)
    target_count = min(target_count, doc_len)
    threshold = float(np.mean(attention_scores)) + threshold_factor * float(np.std(attention_scores))
    high_positions = np.where(attention_scores > threshold)[0].astype(int).tolist()
    components, current = [], []
    for pos in sorted(high_positions):
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


def base_orders(scores, rate, threshold_factor):
    scores = finite(scores)
    _, doc_len = scores.shape
    target_count = max(1, int(rate * doc_len))
    per_query_selected = [smart_query_selection_local(s, rate, threshold_factor) for s in scores]
    freq = np.zeros(doc_len, dtype=np.int32)
    for idx in per_query_selected:
        freq[idx] += 1
    mean_score = scores.mean(axis=0)
    max_score = scores.max(axis=0)
    top2_score = topn_mean(scores, 2)

    def order_all(primary, secondary):
        return np.asarray(
            sorted(range(doc_len), key=lambda i: (-float(primary[i]), -float(secondary[i]), int(i))),
            dtype=np.int64,
        )

    return {
        "freq": order_all(freq, mean_score),
        "mean": order_all(mean_score, freq),
        "max": order_all(max_score, freq),
        "top2": order_all(top2_score, freq),
    }, mean_score, target_count


def boundary_order(starts, system_len, doc_len, mean_score):
    boundaries = []
    for abs_start, abs_end in zip(starts[1:-1], starts[2:]):
        local_start = int(abs_start - system_len)
        local_end = int(abs_end - system_len) - 1
        if 0 <= local_start < doc_len:
            boundaries.append(local_start)
        if 0 <= local_end < doc_len:
            boundaries.append(local_end)
    if not boundaries:
        boundaries = [0, max(0, doc_len - 1)]
    boundaries = np.asarray(boundaries, dtype=np.int64)

    def dist(i):
        return int(np.min(np.abs(boundaries - int(i))))

    return np.asarray(
        sorted(range(doc_len), key=lambda i: (dist(i), -float(mean_score[i]), int(i))),
        dtype=np.int64,
    )


def mix_select(base_order, boundary_order_arr, total_budget, boundary_budget):
    boundary_budget = min(boundary_budget, total_budget)
    base_budget = max(0, total_budget - boundary_budget)
    selected = []
    seen = set()
    for pos in base_order:
        if len(selected) >= base_budget:
            break
        p = int(pos)
        if p not in seen:
            selected.append(p)
            seen.add(p)
    boundary_added = 0
    for pos in boundary_order_arr:
        if len(selected) >= total_budget or boundary_added >= boundary_budget:
            break
        p = int(pos)
        if p not in seen:
            selected.append(p)
            seen.add(p)
            boundary_added += 1
    if len(selected) < total_budget:
        for pos in base_order:
            if len(selected) >= total_budget:
                break
            p = int(pos)
            if p not in seen:
                selected.append(p)
                seen.add(p)
    return np.asarray(sorted(selected), dtype=np.int64)


def split_doc_local_to_chunks(doc_indices, starts, system_len):
    doc_indices = np.asarray(sorted(set(map(int, doc_indices))), dtype=np.int64)
    payload, rows = {}, []
    for chunk_idx, (abs_start, abs_end) in enumerate(zip(starts[1:-1], starts[2:])):
        local_doc_start = int(abs_start - system_len)
        local_doc_end = int(abs_end - system_len)
        mask = (doc_indices >= local_doc_start) & (doc_indices < local_doc_end)
        local = doc_indices[mask] - local_doc_start
        payload[chunk_idx] = np.asarray(local, dtype=np.int64)
        rows.append((chunk_idx, local_doc_end - local_doc_start, int(local.shape[0])))
    return payload, rows


def process(path, out_dir, rate, boundary_rates, threshold_factor):
    ex = int(path.stem.split("example")[1].split("_")[0])
    data = np.load(path, allow_pickle=True)
    scores = np.asarray(data["scores"], dtype=np.float32)
    starts = np.asarray(data["starts"], dtype=np.int64)
    system_len = int(np.asarray(data["system_len"]))
    orders, mean_score, total_budget = base_orders(scores, rate, threshold_factor)
    doc_len = int(scores.shape[1])
    b_order = boundary_order(starts, system_len, doc_len, mean_score)

    out_payload, manifest = {}, []
    method_defs = []
    for br in boundary_rates:
        tag = f"0p{int(round(br * 100)):02d}"
        method_defs.extend([
            (f"draft32b_smart_freq_boundary{tag}_global", orders["freq"], br),
            (f"draft32b_smart_mean_boundary{tag}_global", orders["mean"], br),
            (f"draft32b_smart_top2_boundary{tag}_global", orders["top2"], br),
        ])

    for method, order, br in method_defs:
        boundary_budget = int(br * doc_len)
        doc_idx = mix_select(order, b_order, total_budget, boundary_budget)
        chunk_payload, chunk_rows = split_doc_local_to_chunks(doc_idx, starts, system_len)
        for chunk_idx, local in chunk_payload.items():
            out_payload[f"chunk{chunk_idx:02d}_{method}"] = local
        for chunk_idx, chunk_len, selected_count in chunk_rows:
            manifest.append({
                "example_id": ex,
                "chunk_id": chunk_idx,
                "method": method,
                "rate": rate,
                "boundary_rate": br,
                "chunk_len": int(chunk_len),
                "selected_count": int(selected_count),
            })

    outp = out_dir / "chunk_fixed_sets_npz" / f"example{ex:03d}_rate0p15_chunk_local_sets.npz"
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, **out_payload)
    return manifest


def write_csv(path, rows):
    if not rows:
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rate", type=float, default=0.15)
    ap.add_argument("--boundary-rates", default="0.02,0.03,0.05")
    ap.add_argument("--threshold-factor", type=float, default=0.5)
    args = ap.parse_args()
    src = Path(args.score_cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    boundary_rates = [float(x) for x in args.boundary_rates.split(",") if x.strip()]
    rows = []
    for path in sorted(src.glob("reflect_draft_example*_scores.npz")):
        rows.extend(process(path, out_dir, args.rate, boundary_rates, args.threshold_factor))
    write_csv(out_dir / "fixed_set_manifest.csv", rows)
    (out_dir / "README.md").write_text(
        "# Draft-32B Boundary-Mix Offline Fixed Sets\n\n"
        f"- source score cache: `{src}`\n"
        f"- total rate: {args.rate}\n"
        f"- boundary rates: {boundary_rates}\n"
        "- methods: frequency/mean/top2 32B smart ranking with 2%/3%/5% boundary-near token replacement.\n",
        encoding="utf-8",
    )
    print(f"wrote {len(set(r['example_id'] for r in rows))} examples to {out_dir}")


if __name__ == "__main__":
    main()
