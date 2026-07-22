#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


L = 36


def cosine(a, b, dim):
    return (a * b).sum(dim) / (a.square().sum(dim).sqrt() * b.square().sum(dim).sqrt()).clamp_min(1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    files = sorted(Path(args.data_dir).glob("sample*.pt"))
    if not files:
        raise FileNotFoundError(args.data_dir)
    layer_gap = {name: [] for name in ("k", "v", "h", "attn")}
    metrics = []
    prev_attn_to_h = []
    same_attn_to_h = []
    h_to_next_h = []
    h_to_v_norm = []
    for path in files:
        record = torch.load(path, map_location="cpu", weights_only=False)
        metrics.append(record["metrics"])
        for name, key in (("k", "layer_k_gap"), ("v", "layer_v_gap"), ("h", "layer_h_gap"), ("attn", "layer_attn_gap")):
            layer_gap[name].append(record[key].float())
        dh = record["delta_h"].float()  # [L,T,D]
        da = record["delta_attn"].float()  # [L,T,D]
        dv = record["delta_v"].float()  # [L,H,T,F]
        prev_attn_to_h.append(cosine(da[:-1], dh[1:], dim=-1).mean(-1))
        same_attn_to_h.append(cosine(da, dh, dim=-1).mean(-1))
        h_to_next_h.append(cosine(dh[:-1], dh[1:], dim=-1).mean(-1))
        dh_norm = dh.square().sum(dim=(1, 2)).sqrt()
        dv_norm = dv.square().sum(dim=(1, 2, 3)).sqrt()
        h_to_v_norm.append(torch.stack([dh_norm, dv_norm], -1))

    gap_rows = []
    for name, values in layer_gap.items():
        stacked = torch.stack(values)
        mean = stacked.mean(0)
        gap_rows.append({
            "kind": name,
            "layer0": float(mean[0]),
            "layer1": float(mean[1]),
            "layer8": float(mean[8]),
            "layer16": float(mean[16]),
            "layer24": float(mean[24]),
            "layer35": float(mean[35]),
            "mean_all_layers": float(mean.mean()),
            "max_layer": int(mean.argmax()),
            "max_gap": float(mean.max()),
        })

    prev_attn = torch.stack(prev_attn_to_h).mean(0)
    same_attn = torch.stack(same_attn_to_h).mean(0)
    h_next = torch.stack(h_to_next_h).mean(0)
    norm_pairs = torch.stack(h_to_v_norm)
    dh_norm = norm_pairs[..., 0]
    dv_norm = norm_pairs[..., 1]
    norm_corr = []
    for layer in range(L):
        x = dh_norm[:, layer] - dh_norm[:, layer].mean()
        y = dv_norm[:, layer] - dv_norm[:, layer].mean()
        norm_corr.append(float((x * y).sum() / (x.square().sum().sqrt() * y.square().sum().sqrt()).clamp_min(1e-12)))

    result = {
        "data_dir": args.data_dir,
        "samples": len(files),
        "metrics_mean": {
            key: sum(m[key] for m in metrics) / len(metrics)
            for key in metrics[0]
        },
        "layer_gap_summary": gap_rows,
        "cross_layer_summary": {
            "prev_attn_to_next_h_mean": float(prev_attn.mean()),
            "prev_attn_to_next_h_layer0_to1": float(prev_attn[0]),
            "prev_attn_to_next_h_layer15_to16": float(prev_attn[15]),
            "prev_attn_to_next_h_layer34_to35": float(prev_attn[34]),
            "same_layer_attn_to_h_mean": float(same_attn.mean()),
            "h_to_next_h_mean": float(h_next.mean()),
            "delta_h_norm_to_delta_v_norm_corr_mean": sum(norm_corr) / len(norm_corr),
        },
        "per_layer_prev_attn_to_next_h": [float(x) for x in prev_attn],
        "per_layer_same_attn_to_h": [float(x) for x in same_attn],
        "per_layer_h_to_next_h": [float(x) for x in h_next],
        "per_layer_delta_h_norm_to_delta_v_norm_corr": norm_corr,
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
