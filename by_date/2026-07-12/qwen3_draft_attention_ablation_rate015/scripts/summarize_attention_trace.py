#!/usr/bin/env python3
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: summarize_attention_trace.py TRACE_JSONL OUT_CSV")
    trace_path = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])
    groups = defaultdict(lambda: defaultdict(list))
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            layer = int(row["layer"])
            for key in [
                "doc_q",
                "kv_len",
                "allowed_mean",
                "real_norm_entropy_mean",
                "real_top1_mass_mean",
                "real_top5_mass_mean",
                "real_top10_mass_mean",
                "ablated_vs_real_rel_l2",
                "ablated_vs_real_cosine",
                "ablated_norm_over_real_norm",
                "mixed_vs_real_rel_l2",
            ]:
                groups[layer][key].append(float(row[key]))

    fields = [
        "layer", "calls", "doc_q_mean", "kv_len_mean", "allowed_mean",
        "real_norm_entropy_mean", "real_top1_mass_mean", "real_top5_mass_mean",
        "real_top10_mass_mean", "ablated_vs_real_rel_l2_mean",
        "ablated_vs_real_cosine_mean", "ablated_norm_over_real_norm_mean",
        "mixed_vs_real_rel_l2_mean",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for layer in sorted(groups):
            g = groups[layer]
            writer.writerow({
                "layer": layer,
                "calls": len(g["doc_q"]),
                "doc_q_mean": mean(g["doc_q"]),
                "kv_len_mean": mean(g["kv_len"]),
                "allowed_mean": mean(g["allowed_mean"]),
                "real_norm_entropy_mean": mean(g["real_norm_entropy_mean"]),
                "real_top1_mass_mean": mean(g["real_top1_mass_mean"]),
                "real_top5_mass_mean": mean(g["real_top5_mass_mean"]),
                "real_top10_mass_mean": mean(g["real_top10_mass_mean"]),
                "ablated_vs_real_rel_l2_mean": mean(g["ablated_vs_real_rel_l2"]),
                "ablated_vs_real_cosine_mean": mean(g["ablated_vs_real_cosine"]),
                "ablated_norm_over_real_norm_mean": mean(g["ablated_norm_over_real_norm"]),
                "mixed_vs_real_rel_l2_mean": mean(g["mixed_vs_real_rel_l2"]),
            })
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
