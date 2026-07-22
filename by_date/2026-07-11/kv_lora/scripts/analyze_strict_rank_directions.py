#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoConfig


p = argparse.ArgumentParser()
p.add_argument("--source", choices=["raw", "preprocess"], required=True)
p.add_argument("--kind", choices=["k", "v"], required=True)
p.add_argument("--device", default="cuda:0")
a = p.parse_args()

root = Path("/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora")
ranks = [1, 2, 4, 8, 16, 32]
config = AutoConfig.from_pretrained("/mnt/qjhs-sh-lab-01/models/Qwen3-32B", trust_remote_code=True)
head_dim = 128
inv_freq = 1.0 / (float(config.rope_theta) ** (torch.arange(0, head_dim, 2).float() / head_dim))


def local_rope(delta, prefix_tokens):
    if a.kind != "k" or prefix_tokens == 0:
        return delta
    angles = -float(prefix_tokens) * inv_freq
    cos = torch.cat([angles.cos(), angles.cos()]).view(1, 1, 1, head_dim)
    sin = torch.cat([angles.sin(), angles.sin()]).view(1, 1, 1, head_dim)
    x1, x2 = delta[..., : head_dim // 2], delta[..., head_dim // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    return delta * cos + rot * sin


rows = []
overlaps = []
for target in range(1, 6):
    dd = root / "results/perdoc_context_deltas" / f"strict_t{target}"
    manifest = json.loads((dd / "manifest.json").read_text())
    assert set(manifest["train_pool"]).isdisjoint(manifest["test_pool"])
    paths = sorted(dd.glob(f"context_*_{a.source}.pt"))
    payload = [torch.load(x, map_location="cpu") for x in paths]
    assert [x["meta"]["split"] for x in payload[:40]] == ["train"] * 40
    assert [x["meta"]["split"] for x in payload[40:]] == ["test"] * 10
    acc = {r: dict(delta=0.0, error=0.0, base=0.0, dot=0.0, pred=0.0) for r in [0] + ranks}
    eranks = []
    for layer in range(64):
        data = []
        for x in payload:
            d = x[f"delta_{a.kind}"][layer, 0].float()
            d = local_rope(d, x["meta"]["prefix_tokens"]).flatten()
            data.append(d)
        data = torch.stack(data).to(a.device)
        train, test = data[:40], data[40:]
        mean = train.mean(0)
        centered = train - mean
        _, s, vh = torch.linalg.svd(centered, full_matrices=False)
        prob = s.square() / s.square().sum().clamp_min(1e-30)
        eranks.append(float(torch.exp(-(prob * prob.clamp_min(1e-30).log()).sum())))
        test_centered = test - mean
        _, _, test_vh = torch.linalg.svd(test_centered, full_matrices=False)
        for rank in ranks:
            rr = min(rank, vh.shape[0])
            overlap = (vh[:rr] @ test_vh[:rr].T).square().sum() / rr
            overlaps.append(dict(target=target, source=a.source, kind=a.kind, layer=layer,
                                 rank=rank, subspace_overlap=float(overlap)))
        for i in range(10):
            target_delta = test[i]
            for rank in [0] + ranks:
                pred = mean if rank == 0 else mean + (test_centered[i] @ vh[:rank].T) @ vh[:rank]
                err = target_delta - pred
                z = acc[rank]
                z["delta"] += float(target_delta.square().sum())
                z["error"] += float(err.square().sum())
                z["base"] += float(payload[40 + i][f"base_{a.kind}_norm2_layer"][layer])
                z["dot"] += float(target_delta @ pred)
                z["pred"] += float(pred.square().sum())
    for rank, z in acc.items():
        rows.append(dict(
            target=target, source=a.source, kind=a.kind, method="mean" if rank == 0 else f"oracle_r{rank}",
            rank=rank, original_gap=(z["delta"] / z["base"]) ** 0.5,
            final_kv_error=(z["error"] / z["base"]) ** 0.5,
            remaining_delta=(z["error"] / z["delta"]) ** 0.5,
            explained_delta_energy=1-z["error"] / z["delta"],
            cosine=z["dot"] / max((z["delta"] * z["pred"]) ** 0.5, 1e-30),
            train_effective_rank=sum(eranks) / len(eranks),
        ))

tag = f"strict_prefix_rank_{a.source}_{a.kind}"
with (root / "results" / f"{tag}.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
with (root / "results" / f"{tag}_overlap.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=overlaps[0]); w.writeheader(); w.writerows(overlaps)
(root / "results" / f"{tag}.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
print(json.dumps(rows, indent=2))
