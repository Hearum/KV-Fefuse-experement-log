#!/usr/bin/env python3
"""Predict rank-64 Delta coefficients without using held-out Delta at inference."""

import json
from pathlib import Path

import torch


ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
DATA_DIRS = [
    ROOT / "MOTIVATION_EXPERIMENTS/kv_lora/results/formal_preprocess_residual_50",
    ROOT / "MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/formal_residual_50_100",
]
CACHE = Path("/raid/home/hming/fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique/preprocess_kv_cache_global_topk10_bge")
OUT = ROOT / "MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/stage2_oracle_updated_prefix.json"
L, H, F, R = 64, 8, 128, 64
MAX_FEATURES = 4 * F + 4
VARIANTS = {
    "position": list(range(4 * F, 4 * F + 4)),
    "token_kv_position": list(range(0, 2 * F)) + list(range(4 * F, 4 * F + 4)),
    "token_oracle_updated_prefix_kv_position": list(range(MAX_FEATURES)),
}


def split(example):
    return "train" if example < 70 else "validation" if example < 85 else "test"


def cache_tensor(example, chunk, suffix):
    path = CACHE / f"{example}_{chunk}_{suffix}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)[:, 0].float().cuda()


def item_inputs(record, item, rank, prefix_k_sum, prefix_v_sum, prefix_tokens, sample=False):
    k = cache_tensor(record["example"], item["chunk"], "key")
    v = cache_tensor(record["example"], item["chunk"], "value")
    tokens = k.shape[2]
    if prefix_tokens:
        pk = prefix_k_sum / prefix_tokens
        pv = prefix_v_sum / prefix_tokens
    else:
        pk = torch.zeros(L, H, F, device="cuda")
        pv = torch.zeros(L, H, F, device="cuda")
    positions = torch.arange(tokens, device="cuda", dtype=torch.float32)
    positions = positions / max(tokens - 1, 1)
    pos = torch.stack(
        [
            positions,
            torch.full_like(positions, rank / 10.0),
            torch.full_like(positions, prefix_tokens / 8192.0),
            torch.full_like(positions, tokens / 512.0),
        ],
        dim=-1,
    ).view(1, 1, tokens, 4).expand(L, H, tokens, 4)
    features = torch.cat(
        [
            k,
            v,
            pk.unsqueeze(2).expand(L, H, tokens, F),
            pv.unsqueeze(2).expand(L, H, tokens, F),
            pos,
        ],
        dim=-1,
    )
    index = None
    if sample and tokens > 32:
        index = torch.linspace(0, tokens - 1, 32, device="cuda").long()
        features = features[:, :, index]
    return features, k, v, index


def iter_records(files, wanted_split):
    for path in files:
        record = torch.load(path, map_location="cpu", weights_only=False)
        if split(record["example"]) == wanted_split:
            yield record


def main():
    torch.set_num_threads(4)
    files = sorted(path for directory in DATA_DIRS for path in directory.glob("ex*_sub*.pt"))

    # Pass 1: train-only mean and rank-64 feature basis for each layer/head and K/V.
    moments = {
        kind: {"sum": torch.zeros(L, H, F, device="cuda"), "yty": torch.zeros(L, H, F, F, device="cuda"), "n": 0}
        for kind in ("k", "v")
    }
    train_examples = set()
    for record in iter_records(files, "train"):
        train_examples.add(record["example"])
        for item in record["items"]:
            for kind in ("k", "v"):
                y = item[f"delta_{kind}"][:, 0].float().cuda()
                if y.shape[2] > 32:
                    index = torch.linspace(0, y.shape[2] - 1, 32, device="cuda").long()
                    y = y[:, :, index]
                moments[kind]["sum"] += y.sum(2)
                moments[kind]["yty"] += torch.einsum("lhti,lhtj->lhij", y, y)
                moments[kind]["n"] += y.shape[2]
    models = {}
    for kind, stat in moments.items():
        mean = stat["sum"] / stat["n"]
        covariance = stat["yty"] - stat["n"] * torch.einsum("lhi,lhj->lhij", mean, mean)
        _, vectors = torch.linalg.eigh(covariance)
        models[kind] = {"mean": mean, "basis": vectors[..., -R:]}

    # Pass 2: one maximal feature Gram matrix; variants are fitted by slicing it.
    xtx = torch.zeros(L, H, MAX_FEATURES + 1, MAX_FEATURES + 1, device="cuda")
    xty = {kind: torch.zeros(L, H, MAX_FEATURES + 1, R, device="cuda") for kind in ("k", "v")}
    for record in iter_records(files, "train"):
        prefix_k_sum = prefix_v_sum = None
        prefix_tokens = 0
        for rank, item in enumerate(record["items"]):
            features, k, v, index = item_inputs(record, item, rank, prefix_k_sum, prefix_v_sum, prefix_tokens, sample=True)
            ones = torch.ones(L, H, features.shape[2], 1, device="cuda")
            xa = torch.cat([features, ones], dim=-1)
            xtx += torch.einsum("lhti,lhtj->lhij", xa, xa)
            for kind in ("k", "v"):
                y = item[f"delta_{kind}"][:, 0].float().cuda()
                if index is not None:
                    y = y[:, :, index]
                coefficient = torch.einsum("lhtf,lhfr->lhtr", y - models[kind]["mean"].unsqueeze(2), models[kind]["basis"])
                xty[kind] += torch.einsum("lhti,lhtr->lhir", xa, coefficient)
            full_k = k + item["delta_k"][:, 0].float().cuda()
            full_v = v + item["delta_v"][:, 0].float().cuda()
            prefix_k_sum = full_k.sum(2) if prefix_k_sum is None else prefix_k_sum + full_k.sum(2)
            prefix_v_sum = full_v.sum(2) if prefix_v_sum is None else prefix_v_sum + full_v.sum(2)
            prefix_tokens += k.shape[2]

    weights = {kind: {} for kind in ("k", "v")}
    for name, raw_indices in VARIANTS.items():
        indices = raw_indices + [MAX_FEATURES]
        gram = xtx[:, :, indices][:, :, :, indices]
        scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).unsqueeze(-1)
        eye = torch.eye(len(indices), device="cuda").view(1, 1, len(indices), len(indices))
        for kind in ("k", "v"):
            weights[kind][name] = torch.linalg.solve(gram + 1e-3 * scale * eye, xty[kind][:, :, indices])

    # Pass 3: strict test prediction. Oracle is reported only as representation ceiling.
    methods = ["mean", "oracle_rank64"] + list(VARIANTS)
    totals = {(kind, method): {key: 0.0 for key in ("delta", "error", "base", "dot", "pred")} for kind in ("k", "v") for method in methods}
    test_examples = set()
    documents = 0
    for record in iter_records(files, "test"):
        test_examples.add(record["example"])
        prefix_k_sum = prefix_v_sum = None
        prefix_tokens = 0
        for rank, item in enumerate(record["items"]):
            documents += 1
            features, k, v, _ = item_inputs(record, item, rank, prefix_k_sum, prefix_v_sum, prefix_tokens)
            xa = torch.cat([features, torch.ones(L, H, features.shape[2], 1, device="cuda")], -1)
            for kind, base in (("k", k), ("v", v)):
                y = item[f"delta_{kind}"][:, 0].float().cuda()
                mean = models[kind]["mean"].unsqueeze(2)
                centered = y - mean
                basis = models[kind]["basis"]
                predictions = {
                    "mean": mean.expand_as(y),
                    "oracle_rank64": mean + torch.einsum("lhtf,lhfr,lhgr->lhtg", centered, basis, basis),
                }
                for name, raw_indices in VARIANTS.items():
                    indices = raw_indices + [MAX_FEATURES]
                    coefficient = torch.einsum("lhti,lhir->lhtr", xa[..., indices], weights[kind][name])
                    predictions[name] = mean + torch.einsum("lhtr,lhfr->lhtf", coefficient, basis)
                for name, pred in predictions.items():
                    total = totals[(kind, name)]
                    total["delta"] += float(y.square().sum())
                    total["error"] += float((y - pred).square().sum())
                    total["base"] += float(base.square().sum())
                    total["dot"] += float((y * pred).sum())
                    total["pred"] += float(pred.square().sum())
            full_k = k + item["delta_k"][:, 0].float().cuda()
            full_v = v + item["delta_v"][:, 0].float().cuda()
            prefix_k_sum = full_k.sum(2) if prefix_k_sum is None else prefix_k_sum + full_k.sum(2)
            prefix_v_sum = full_v.sum(2) if prefix_v_sum is None else prefix_v_sum + full_v.sum(2)
            prefix_tokens += k.shape[2]

    rows = []
    for (kind, method), total in totals.items():
        remaining = (total["error"] / total["delta"]) ** 0.5
        rows.append({"kind": kind, "method": method, "remaining_delta": remaining,
                     "explained_delta_energy": 1 - remaining ** 2,
                     "original_gap": (total["delta"] / total["base"]) ** 0.5,
                     "final_kv_error": (total["error"] / total["base"]) ** 0.5,
                     "delta_cosine": total["dot"] / (total["delta"] * total["pred"]) ** 0.5})
    result = {"source": "BGE preprocess", "rank": R, "ridge": 1e-3,
              "train_examples": sorted(train_examples), "test_examples": sorted(test_examples),
              "test_document_instances": documents, "rows": rows}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
