#!/usr/bin/env python3
from __future__ import annotations

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


def boundary_order(starts, system_len, doc_len, mean_score):
    bounds = []
    for start, end in zip(starts[1:-1], starts[2:]):
        local_start = int(start - system_len)
        local_end = int(end - system_len) - 1
        if 0 <= local_start < doc_len:
            bounds.append(local_start)
        if 0 <= local_end < doc_len:
            bounds.append(local_end)
    if not bounds:
        bounds = [0, max(0, doc_len - 1)]
    bounds = np.asarray(bounds)
    return np.asarray(sorted(range(doc_len), key=lambda i: (int(np.min(np.abs(bounds - i))), -float(mean_score[i]), i)), dtype=np.int64)


def split_chunks(doc_idx, starts, system_len):
    doc_idx = np.asarray(sorted(set(map(int, doc_idx))), dtype=np.int64)
    payload, rows = {}, []
    for chunk_idx, (start, end) in enumerate(zip(starts[1:-1], starts[2:])):
        doc_start = int(start - system_len)
        doc_end = int(end - system_len)
        local = doc_idx[(doc_idx >= doc_start) & (doc_idx < doc_end)] - doc_start
        payload[chunk_idx] = local.astype(np.int64)
        rows.append((chunk_idx, doc_end - doc_start, int(local.shape[0])))
    return payload, rows


def mix(base_order, boundary_order_arr, total, boundary):
    selected, seen = [], set()
    base_budget = max(0, total - boundary)
    for pos in base_order:
        if len(selected) >= base_budget:
            break
        pos = int(pos)
        if pos not in seen:
            selected.append(pos)
            seen.add(pos)
    added = 0
    for pos in boundary_order_arr:
        if len(selected) >= total or added >= boundary:
            break
        pos = int(pos)
        if pos not in seen:
            selected.append(pos)
            seen.add(pos)
            added += 1
    for pos in base_order:
        if len(selected) >= total:
            break
        pos = int(pos)
        if pos not in seen:
            selected.append(pos)
            seen.add(pos)
    return np.asarray(sorted(selected), dtype=np.int64)


def example_id_from_path(path: Path) -> int:
    tail = path.stem.split("example", 1)[1]
    digits = ""
    for ch in tail:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits)


def process(path: Path, out_dir: Path, prefix: str, rate: float, boundary_rate: float):
    example_id = example_id_from_path(path)
    data = np.load(path, allow_pickle=True)
    scores = finite(data["scores"])
    starts = np.asarray(data["starts"], dtype=np.int64)
    system_len = int(np.asarray(data["system_len"]))
    doc_len = int(scores.shape[1])
    total = max(1, int(rate * doc_len))
    mean_score = scores.mean(axis=0)
    top2 = topn_mean(scores, 2)
    freq = np.zeros(doc_len, dtype=np.int32)
    for score in scores:
        k = max(1, min(doc_len, int(rate * doc_len)))
        freq[np.argpartition(-finite(score), k - 1)[:k]] += 1
    freq_order = np.asarray(sorted(range(doc_len), key=lambda i: (-int(freq[i]), -float(mean_score[i]), i)), dtype=np.int64)
    b_order = boundary_order(starts, system_len, doc_len, mean_score)
    orders = {
        f"{prefix}_mean_score_global": np.asarray(sorted(range(doc_len), key=lambda i: (-float(mean_score[i]), -int(freq[i]), i)), dtype=np.int64)[:total],
        f"{prefix}_top2_mean_global": np.asarray(sorted(range(doc_len), key=lambda i: (-float(top2[i]), -int(freq[i]), i)), dtype=np.int64)[:total],
        f"{prefix}_freq_boundary0p02_global": mix(freq_order, b_order, total, int(boundary_rate * doc_len)),
    }
    payload, manifest = {}, []
    for method, order in orders.items():
        chunks, rows = split_chunks(order, starts, system_len)
        for chunk_idx, local in chunks.items():
            payload[f"chunk{chunk_idx:02d}_{method}"] = local
        for chunk_idx, chunk_len, selected_count in rows:
            manifest.append({"example_id": example_id, "chunk_id": chunk_idx, "method": method, "rate": rate, "chunk_len": chunk_len, "selected_count": selected_count})
    out_path = out_dir / "chunk_fixed_sets_npz" / f"example{example_id:03d}_rate{str(rate).replace('.', 'p')}_chunk_local_sets.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return manifest


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive setup-v2 fixed sets from setup-v2 score caches.")
    parser.add_argument("--score-cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--rate", type=float, default=0.15)
    parser.add_argument("--boundary-rate", type=float, default=0.02)
    args = parser.parse_args()
    files = sorted(Path(args.score_cache_dir).glob("*example*_scores.npz"))
    if not files:
        raise FileNotFoundError(f"no *example*_scores.npz files under {args.score_cache_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in files:
        rows.extend(process(path, out_dir, args.prefix, args.rate, args.boundary_rate))
    write_csv(out_dir / "fixed_set_manifest.csv", rows)
    readme = (
        "# setup-v2 fixed sets\n\n"
        f"source={args.score_cache_dir}\n"
        f"prefix={args.prefix}\n"
        f"rate={args.rate}\n"
        f"boundary_rate={args.boundary_rate}\n"
        "\nThis derives chunk-local fixed sets from setup-v2 score tensors. It does not run model forward.\n"
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote {len(files)} examples to {out_dir}")


if __name__ == "__main__":
    main()
