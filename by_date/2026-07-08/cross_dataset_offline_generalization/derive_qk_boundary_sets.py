#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np


def top_boundary_order(chunk_len, scores):
    if chunk_len <= 0:
        return []
    left = np.arange(chunk_len)
    right = chunk_len - 1 - left
    dist = np.minimum(left, right)
    return sorted(range(chunk_len), key=lambda i: (int(dist[i]), -float(scores[i]), int(i)))


def mix_qk_boundary(mean_scores, rate, boundary_rate):
    scores = np.nan_to_num(np.asarray(mean_scores, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    chunk_len = int(scores.shape[0])
    total = max(1, int(rate * chunk_len))
    boundary = max(1, int(boundary_rate * chunk_len)) if chunk_len > 1 else 0
    boundary = min(boundary, total)
    base_budget = max(0, total - boundary)
    qk_order = sorted(range(chunk_len), key=lambda i: (-float(scores[i]), int(i)))
    selected = []
    seen = set()
    for i in qk_order:
        if len(selected) >= base_budget:
            break
        selected.append(int(i))
        seen.add(int(i))
    added_boundary = 0
    for i in top_boundary_order(chunk_len, scores):
        if len(selected) >= total or added_boundary >= boundary:
            break
        if int(i) in seen:
            continue
        selected.append(int(i))
        seen.add(int(i))
        added_boundary += 1
    for i in qk_order:
        if len(selected) >= total:
            break
        if int(i) not in seen:
            selected.append(int(i))
            seen.add(int(i))
    return np.asarray(sorted(selected), dtype=np.int64)


def process_file(path, out_path, rate, boundary_rate, method_name):
    data = np.load(path, allow_pickle=True)
    payload = {k: data[k] for k in data.files}
    rows = []
    for key in data.files:
        if not key.endswith("_qk_mean_score"):
            continue
        chunk_prefix = key.removesuffix("_qk_mean_score")
        if not chunk_prefix.startswith("chunk"):
            continue
        selected = mix_qk_boundary(data[key], rate, boundary_rate)
        out_key = f"{chunk_prefix}_{method_name}"
        payload[out_key] = selected
        rows.append({
            "file": path.name,
            "chunk_key": chunk_prefix,
            "method": method_name,
            "rate": rate,
            "boundary_rate": boundary_rate,
            "chunk_len": int(np.asarray(data[key]).shape[0]),
            "selected_count": int(selected.shape[0]),
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return rows


def write_csv(path, rows):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rate", type=float, default=0.15)
    parser.add_argument("--boundary-rate", type=float, default=0.02)
    parser.add_argument("--method-name", default="qk_mean_boundary0p02_per_chunk")
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    rows = []
    for path in sorted(inp.glob("example*_rate0p15_chunk_local_sets.npz")):
        rows.extend(process_file(path, out / path.name, args.rate, args.boundary_rate, args.method_name))
    write_csv(out.parent / "qk_boundary_manifest.csv", rows)
    (out.parent / "README_boundary.md").write_text(
        "# QK Boundary Fixed Set Derivation\n\n"
        f"- source: `{inp}`\n"
        f"- output: `{out}`\n"
        f"- method: `{args.method_name}`\n"
        f"- total rate: {args.rate}\n"
        f"- boundary replacement rate: {args.boundary_rate}\n",
        encoding="utf-8",
    )
    print(f"processed {len(rows)} chunks from {inp} -> {out}")


if __name__ == "__main__":
    main()
