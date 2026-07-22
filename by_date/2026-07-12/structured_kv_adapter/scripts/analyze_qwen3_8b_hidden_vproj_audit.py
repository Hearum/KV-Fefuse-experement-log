#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import torch


L, H, F, D = 36, 8, 128, 4096
RANKS = (8, 16, 32, 64, 128)


def load_records(directory: Path):
    files = sorted(directory.glob("sample*.pt"))
    if not files:
        raise FileNotFoundError(f"no sample*.pt in {directory}")
    return [(path, torch.load(path, map_location="cpu", weights_only=False)) for path in files]


def split_records(records, train_count: int):
    return records[:train_count], records[train_count:]


def pca_model_h(train):
    y = torch.cat([record["delta_h"].double() for _, record in train], dim=1)
    mean = y.mean(1)
    centered = y - mean.unsqueeze(1)
    cov = torch.einsum("ltd,lte->lde", centered, centered)
    evals, vec = torch.linalg.eigh(cov)
    evals = evals.flip(-1).clamp_min(0)
    vec = vec.flip(-1)
    total = evals.sum(-1).clamp_min(1e-12)
    energy = {str(r): float((evals[:, :r].sum(-1) / total).mean()) for r in RANKS}
    return {"mean": mean.float(), "basis": vec.float(), "energy": energy}


def pca_model_v(train):
    y = torch.cat([record["delta_v"].double() for _, record in train], dim=2)
    mean = y.mean(2)
    centered = y - mean.unsqueeze(2)
    cov = torch.einsum("lhti,lhtj->lhij", centered, centered)
    evals, vec = torch.linalg.eigh(cov)
    evals = evals.flip(-1).clamp_min(0)
    vec = vec.flip(-1)
    total = evals.sum(-1).clamp_min(1e-12)
    energy = {str(r): float((evals[..., :r].sum(-1) / total).mean()) for r in RANKS}
    return {"mean": mean.float(), "basis": vec.float(), "energy": energy}


def eval_h(model, test):
    rows = []
    for rank in RANKS:
        delta_sq = error_sq = dot = pred_sq = base_sq = 0.0
        basis = model["basis"][:, :, :rank]
        mean = model["mean"]
        for _, record in test:
            y = record["delta_h"].float()
            base = record["own_h"].float()
            centered = y - mean.unsqueeze(1)
            pred = mean.unsqueeze(1) + torch.einsum("ltd,ldr,ler->lte", centered, basis, basis)
            delta_sq += float(y.square().sum())
            error_sq += float((y - pred).square().sum())
            dot += float((y * pred).sum())
            pred_sq += float(pred.square().sum())
            base_sq += float(base.square().sum())
        remaining = math.sqrt(error_sq / max(delta_sq, 1e-30))
        rows.append({
            "kind": "h",
            "rank": rank,
            "remaining_delta": remaining,
            "explained_delta_energy": 1 - remaining ** 2,
            "original_gap": math.sqrt(delta_sq / max(base_sq, 1e-30)),
            "final_h_error": math.sqrt(error_sq / max(base_sq, 1e-30)),
            "delta_cosine": dot / math.sqrt(max(delta_sq * pred_sq, 1e-30)),
        })
    return rows


def eval_v(model, test):
    rows = []
    for rank in RANKS:
        delta_sq = error_sq = dot = pred_sq = 0.0
        basis = model["basis"][..., :rank]
        mean = model["mean"]
        for _, record in test:
            y = record["delta_v"].float()
            centered = y - mean.unsqueeze(2)
            pred = mean.unsqueeze(2) + torch.einsum("lhtf,lhfr,lhgr->lhtg", centered, basis, basis)
            delta_sq += float(y.square().sum())
            error_sq += float((y - pred).square().sum())
            dot += float((y * pred).sum())
            pred_sq += float(pred.square().sum())
        remaining = math.sqrt(error_sq / max(delta_sq, 1e-30))
        rows.append({
            "kind": "v",
            "rank": rank,
            "remaining_delta": remaining,
            "explained_delta_energy": 1 - remaining ** 2,
            "delta_cosine": dot / math.sqrt(max(delta_sq * pred_sq, 1e-30)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    records = load_records(Path(args.data_dir))
    train, test = split_records(records, args.train_count)
    if not test:
        raise ValueError("need heldout records; reduce --train-count")
    h_model = pca_model_h(train)
    v_model = pca_model_v(train)
    metrics = {
        "h_gap_mean": sum(r["metrics"]["h_original_gap"] for _, r in records) / len(records),
        "v_gap_sampled_mean": sum(r["metrics"]["v_original_gap_sampled"] for _, r in records) / len(records),
        "vproj_cache_mismatch_mean": sum(r["metrics"]["vproj_cache_mismatch"] for _, r in records) / len(records),
        "vproj_cache_mismatch_max": max(r["metrics"]["vproj_cache_mismatch"] for _, r in records),
    }
    result = {
        "data_dir": args.data_dir,
        "records": len(records),
        "train_count": len(train),
        "test_count": len(test),
        "metrics": metrics,
        "train_rank_energy": {"h": h_model["energy"], "v": v_model["energy"]},
        "heldout_rows": eval_h(h_model, test) + eval_v(v_model, test),
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
