#!/usr/bin/env python3
"""Train-only per-layer/head feature bases and strict-test oracle reconstruction."""

import json
from pathlib import Path

import torch


ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
DATA = ROOT / "MOTIVATION_EXPERIMENTS/kv_lora/results/formal_preprocess_residual_50"
CACHE = Path("/raid/home/hming/fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique/preprocess_kv_cache_global_topk10_bge")
OUT = ROOT / "MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/stage1_feature_basis.json"
RANKS = (8, 16, 32, 64, 128)
L, H, F = 64, 8, 128


def split(example):
    return "train" if example < 30 else "validation" if example < 40 else "test"


def load_base(example, chunk, kind):
    suffix = "key" if kind == "k" else "value"
    return torch.load(CACHE / f"{example}_{chunk}_{suffix}.pt", map_location="cpu", weights_only=True)[:, 0].float().cuda()


def main():
    torch.set_num_threads(4)
    files = sorted(DATA.glob("ex*_sub*.pt"))
    stats = {
        kind: {
            "sum": torch.zeros(L, H, F, device="cuda"),
            "yty": torch.zeros(L, H, F, F, device="cuda"),
            "n": 0,
        }
        for kind in ("k", "v")
    }
    train_examples = set()
    for path in files:
        record = torch.load(path, map_location="cpu", weights_only=False)
        if split(record["example"]) != "train":
            continue
        train_examples.add(record["example"])
        for item in record["items"]:
            for kind in ("k", "v"):
                y = item[f"delta_{kind}"][:, 0].float().cuda()
                if y.shape[2] > 32:
                    index = torch.linspace(0, y.shape[2] - 1, 32, device="cuda").long()
                    y = y[:, :, index]
                stats[kind]["sum"] += y.sum(2)
                stats[kind]["yty"] += torch.einsum("lhti,lhtj->lhij", y, y)
                stats[kind]["n"] += y.shape[2]

    models = {}
    for kind, stat in stats.items():
        mean = stat["sum"] / stat["n"]
        covariance = stat["yty"] - stat["n"] * torch.einsum("lhi,lhj->lhij", mean, mean)
        _, vectors = torch.linalg.eigh(covariance)
        models[kind] = {"mean": mean, "basis": vectors.flip(-1)}

    totals = {
        (kind, rank): {"delta_sq": 0.0, "error_sq": 0.0, "base_sq": 0.0, "dot": 0.0, "pred_sq": 0.0}
        for kind in ("k", "v") for rank in RANKS
    }
    test_examples = set()
    test_documents = 0
    for path in files:
        record = torch.load(path, map_location="cpu", weights_only=False)
        if split(record["example"]) != "test":
            continue
        test_examples.add(record["example"])
        for item in record["items"]:
            test_documents += 1
            for kind in ("k", "v"):
                y = item[f"delta_{kind}"][:, 0].float().cuda()
                base = load_base(record["example"], item["chunk"], kind)
                centered = y - models[kind]["mean"].unsqueeze(2)
                for rank in RANKS:
                    basis = models[kind]["basis"][..., :rank]
                    pred = models[kind]["mean"].unsqueeze(2) + torch.einsum(
                        "lhtf,lhfr,lhgr->lhtg", centered, basis, basis
                    )
                    total = totals[(kind, rank)]
                    total["delta_sq"] += float(y.square().sum())
                    total["error_sq"] += float((y - pred).square().sum())
                    total["base_sq"] += float(base.square().sum())
                    total["dot"] += float((y * pred).sum())
                    total["pred_sq"] += float(pred.square().sum())

    rows = []
    for (kind, rank), total in totals.items():
        remaining = (total["error_sq"] / total["delta_sq"]) ** 0.5
        rows.append({
            "kind": kind,
            "rank": rank,
            "remaining_delta": remaining,
            "explained_delta_energy": 1 - remaining ** 2,
            "original_gap": (total["delta_sq"] / total["base_sq"]) ** 0.5,
            "final_kv_error": (total["error_sq"] / total["base_sq"]) ** 0.5,
            "delta_cosine": total["dot"] / (total["delta_sq"] * total["pred_sq"]) ** 0.5,
        })
    result = {
        "source": "BGE preprocess",
        "basis": "train-only per-layer/per-head feature PCA; test uses oracle projection",
        "train_token_cap_per_document": 32,
        "train_examples": sorted(train_examples),
        "test_examples": sorted(test_examples),
        "test_document_instances": test_documents,
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
