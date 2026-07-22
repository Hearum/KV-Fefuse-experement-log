#!/usr/bin/env python3
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
    return np.partition(scores, scores.shape[0] - n, axis=0)[-n:].mean(axis=0)


def smart_select(scores, rate, threshold_factor=0.5):
    scores = finite(scores)
    doc_len = len(scores)
    target = max(1, min(doc_len, int(rate * doc_len)))
    th = float(scores.mean()) + threshold_factor * float(scores.std())
    highs = np.where(scores > th)[0].astype(int).tolist()
    comps, cur = [], []
    for p in sorted(highs):
        if not cur or p - cur[-1] <= 2:
            cur.append(p)
        else:
            comps.append(cur)
            cur = [p]
    if cur:
        comps.append(cur)
    comps = sorted(comps, key=lambda c: -sum(float(scores[p]) for p in c))
    selected = set()
    for comp in comps:
        ext = {q for p in comp for q in (p - 1, p, p + 1) if 0 <= q < doc_len}
        if len(selected) + len(ext - selected) <= target * 1.1:
            selected.update(ext)
    if len(selected) < target:
        for p in np.argsort(scores)[::-1]:
            selected.add(int(p))
            if len(selected) >= target:
                break
    while len(selected) > target:
        selected.remove(min(selected, key=lambda p: scores[p]))
    return np.array(sorted(selected), dtype=np.int64)


def boundary_order(starts, system_len, doc_len, mean_score):
    bounds = []
    for a, b in zip(starts[1:-1], starts[2:]):
        s = int(a - system_len)
        e = int(b - system_len) - 1
        if 0 <= s < doc_len:
            bounds.append(s)
        if 0 <= e < doc_len:
            bounds.append(e)
    if not bounds:
        bounds = [0, max(0, doc_len - 1)]
    bounds = np.asarray(bounds)
    return np.asarray(sorted(range(doc_len), key=lambda i: (int(np.min(np.abs(bounds - i))), -float(mean_score[i]), i)), dtype=np.int64)


def split_chunks(doc_idx, starts, system_len):
    doc_idx = np.asarray(sorted(set(map(int, doc_idx))), dtype=np.int64)
    payload, rows = {}, []
    for ci, (a, b) in enumerate(zip(starts[1:-1], starts[2:])):
        ds, de = int(a - system_len), int(b - system_len)
        local = doc_idx[(doc_idx >= ds) & (doc_idx < de)] - ds
        payload[ci] = local.astype(np.int64)
        rows.append((ci, de - ds, int(local.shape[0])))
    return payload, rows


def mix(base_order, boundary_order_arr, total, boundary):
    selected, seen = [], set()
    base_budget = max(0, total - boundary)
    for p in base_order:
        if len(selected) >= base_budget:
            break
        p = int(p)
        if p not in seen:
            selected.append(p)
            seen.add(p)
    added = 0
    for p in boundary_order_arr:
        if len(selected) >= total or added >= boundary:
            break
        p = int(p)
        if p not in seen:
            selected.append(p)
            seen.add(p)
            added += 1
    for p in base_order:
        if len(selected) >= total:
            break
        p = int(p)
        if p not in seen:
            selected.append(p)
            seen.add(p)
    return np.array(sorted(selected), dtype=np.int64)


def process(path, out_dir, prefix, rate, boundary_rate):
    ex = int(path.stem.split("example")[1].split("_")[0])
    d = np.load(path, allow_pickle=True)
    scores = finite(d["scores"])
    starts = np.asarray(d["starts"], dtype=np.int64)
    system_len = int(np.asarray(d["system_len"]))
    doc_len = scores.shape[1]
    total = max(1, int(rate * doc_len))
    perq = [smart_select(s, rate) for s in scores]
    freq = np.zeros(doc_len, dtype=np.int32)
    for idx in perq:
        freq[idx] += 1
    mean_score = scores.mean(axis=0)
    top2 = topn_mean(scores, 2)
    orders = {
        f"{prefix}_mean_score_global": np.asarray(sorted(range(doc_len), key=lambda i: (-float(mean_score[i]), -int(freq[i]), i)), dtype=np.int64),
        f"{prefix}_top2_mean_global": np.asarray(sorted(range(doc_len), key=lambda i: (-float(top2[i]), -int(freq[i]), i)), dtype=np.int64),
        f"{prefix}_freq_boundary0p02_global": None,
    }
    freq_order = np.asarray(sorted(range(doc_len), key=lambda i: (-int(freq[i]), -float(mean_score[i]), i)), dtype=np.int64)
    b_order = boundary_order(starts, system_len, doc_len, mean_score)
    orders[f"{prefix}_freq_boundary0p02_global"] = mix(freq_order, b_order, total, int(boundary_rate * doc_len))

    payload, manifest = {}, []
    for method, order in orders.items():
        selected = order[:total] if "boundary" not in method else order
        chunks, rows = split_chunks(selected, starts, system_len)
        for ci, local in chunks.items():
            payload[f"chunk{ci:02d}_{method}"] = local
        for ci, clen, scount in rows:
            manifest.append({"example_id": ex, "chunk_id": ci, "method": method, "rate": rate, "chunk_len": clen, "selected_count": scount})
    outp = out_dir / "chunk_fixed_sets_npz" / f"example{ex:03d}_rate0p15_chunk_local_sets.npz"
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, **payload)
    return manifest


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--rate", type=float, default=0.15)
    ap.add_argument("--boundary-rate", type=float, default=0.02)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(Path(args.score_cache_dir).glob("reflect_draft_example*_scores.npz")):
        rows.extend(process(path, out_dir, args.prefix, args.rate, args.boundary_rate))
    write_csv(out_dir / "fixed_set_manifest.csv", rows)
    (out_dir / "README.md").write_text(
        f"# Generic smart fixed sets\n\nsource={args.score_cache_dir}\nprefix={args.prefix}\nrate={args.rate}\n",
        encoding="utf-8",
    )
    print(f"wrote {len(set(r['example_id'] for r in rows))} examples to {out_dir}")


if __name__ == "__main__":
    main()
