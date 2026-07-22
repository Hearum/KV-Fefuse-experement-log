#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
from transformers import AutoConfig


ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
EXP = ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter"
SOURCE = ROOT / "MOTIVATION_EXPERIMENTS/kv_lora/results/perdoc_context_deltas"
CACHE = Path("/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/musique")
MODEL = "/mnt/qjhs-sh-lab-01/models/Qwen3-32B"
LAYERS = (7, 15, 31, 47, 63)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--targets", default="1,2,3,4,5")
    p.add_argument("--topks", default="1,2,4,8")
    p.add_argument("--output-dir", default=str(EXP / "results/strict_anchor_transfer"))
    return p.parse_args()


def local_rope(delta: torch.Tensor, prefix_tokens: int, inv_freq: torch.Tensor) -> torch.Tensor:
    if prefix_tokens == 0:
        return delta
    angles = -float(prefix_tokens) * inv_freq
    cos = torch.cat([angles.cos(), angles.cos()]).view(1, 1, 1, -1)
    sin = torch.cat([angles.sin(), angles.sin()]).view(1, 1, 1, -1)
    half = delta.shape[-1] // 2
    rotated = torch.cat([-delta[..., half:], delta[..., :half]], dim=-1)
    return delta * cos + rotated * sin


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def pairwise_cosine_distance(train: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
    train_n = normalize_rows(train)
    test_n = normalize_rows(test)
    distance = 1.0 - test_n @ train_n.T
    train_zero = train.norm(dim=1) < 1e-12
    test_zero = test.norm(dim=1) < 1e-12
    if train_zero.any() or test_zero.any():
        distance[:, train_zero] = 1.0
        distance[test_zero] = 1.0
        distance[test_zero[:, None] & train_zero[None, :]] = 0.0
    return distance


def position_distance(train: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
    mean = train.mean(0)
    std = train.std(0).clamp_min(1.0)
    return torch.cdist((test - mean) / std, (train - mean) / std)


def make_weights(distance: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, distance.shape[1])
    values, indices = torch.topk(distance, k=k, largest=False, dim=1)
    scale = values.median(dim=1, keepdim=True).values.clamp_min(1e-4)
    local = torch.softmax(-values / scale, dim=1)
    weights = torch.zeros_like(distance)
    weights.scatter_(1, indices, local)
    return weights


def load_chunk_signatures(source: str) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
    cache_dir = CACHE / ("kv_cache" if source == "raw" else "preprocess_kv_cache_global_topk10_bge")
    signatures: dict[int, torch.Tensor] = {}
    lengths: dict[int, int] = {}
    for chunk_id in range(1, 11):
        path = cache_dir / f"0_{chunk_id}_value.pt"
        if not path.exists():
            continue
        value = torch.load(path, map_location="cpu", weights_only=True).float()
        lengths[chunk_id] = int(value.shape[3])
        # Token mean retains layer/head/channel structure without online recompute.
        signatures[chunk_id] = value[list(LAYERS), 0].mean(dim=2).flatten()
    return signatures, lengths


def context_features(payloads: list[dict], source: str) -> tuple[torch.Tensor, torch.Tensor]:
    signatures, lengths = load_chunk_signatures(source)
    pos, cached = [], []
    feature_dim = len(LAYERS) * 8 * 128
    for item in payloads:
        ids = item["meta"]["prefix_ids"]
        pos.append([item["meta"]["prefix_tokens"], len(ids)])
        if not ids:
            cached.append(torch.zeros(feature_dim))
            continue
        # Length weighted mean is cheap and order invariant; position is handled separately.
        total = sum(lengths[i] for i in ids)
        cached.append(sum(signatures[i] * lengths[i] for i in ids) / total)
    return torch.tensor(pos, dtype=torch.float32), torch.stack(cached)


def prediction_methods(pos_distance: torch.Tensor, cached_distance: torch.Tensor, topks: list[int]) -> dict[str, torch.Tensor]:
    pos_scale = pos_distance.median().clamp_min(1e-4)
    cached_scale = cached_distance.median().clamp_min(1e-4)
    hybrid = 0.5 * pos_distance / pos_scale + 0.5 * cached_distance / cached_scale
    methods: dict[str, torch.Tensor] = {}
    for name, distance in (("position", pos_distance), ("cached_v", cached_distance), ("hybrid", hybrid)):
        for k in topks:
            methods[f"{name}_top{k}"] = make_weights(distance, k)
    return methods


def accumulate(cell: dict[str, float], target: torch.Tensor, pred: torch.Tensor, base_norm2: float) -> None:
    error = target - pred
    cell["delta"] += float(target.square().sum())
    cell["error"] += float(error.square().sum())
    cell["base"] += float(base_norm2)
    cell["dot"] += float((target * pred).sum())
    cell["pred"] += float(pred.square().sum())


def main() -> None:
    args = parse_args()
    targets = [int(x) for x in args.targets.split(",")]
    topks = [int(x) for x in args.topks.split(",")]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    head_dim = int(config.head_dim)
    inv_freq = 1.0 / (float(config.rope_theta) ** (torch.arange(0, head_dim, 2).float() / head_dim))
    rows = []

    for target_id in targets:
        target_dir = SOURCE / f"strict_t{target_id}"
        manifest = json.loads((target_dir / "manifest.json").read_text())
        assert set(manifest["train_pool"]).isdisjoint(manifest["test_pool"])
        for source in ("raw", "preprocess"):
            # context_50 is a historical replacement artifact in these folders;
            # the canonical manifest split is context_00..49.
            paths = [target_dir / f"context_{index:02d}_{source}.pt" for index in range(50)]
            assert all(path.exists() for path in paths)
            payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
            assert len(payloads) == 50
            assert all(x["meta"]["split"] == "train" for x in payloads[:40])
            assert all(x["meta"]["split"] == "test" for x in payloads[40:])
            pos, cached = context_features(payloads, source)
            pdist = position_distance(pos[:40], pos[40:])
            cdist = pairwise_cosine_distance(cached[:40], cached[40:])
            feature_weights = prediction_methods(pdist, cdist, topks)
            rng = random.Random(88000 + target_id + (0 if source == "raw" else 100))
            random_indices = [rng.randrange(40) for _ in range(10)]

            for kind in ("k", "v"):
                oracle_dist = torch.zeros((10, 40), dtype=torch.float64)
                # Select oracle anchors globally across layers, not independently per layer.
                for layer in range(64):
                    layer_data = []
                    for item in payloads:
                        delta = item[f"delta_{kind}"][layer, 0].float()
                        if kind == "k":
                            delta = local_rope(delta, item["meta"]["prefix_tokens"], inv_freq)
                        layer_data.append(delta.flatten())
                    data = torch.stack(layer_data)
                    oracle_dist += torch.cdist(data[40:].double(), data[:40].double()).square()
                oracle_weights = torch.nn.functional.one_hot(oracle_dist.argmin(1), num_classes=40).float()

                names = ["no_update", "mean_anchor", "random_anchor", "oracle_anchor"] + list(feature_weights)
                acc = {(test_i, name): {k: 0.0 for k in ("delta", "error", "base", "dot", "pred")} for test_i in range(10) for name in names}
                for layer in range(64):
                    layer_data = []
                    for item in payloads:
                        delta = item[f"delta_{kind}"][layer, 0].float()
                        if kind == "k":
                            delta = local_rope(delta, item["meta"]["prefix_tokens"], inv_freq)
                        layer_data.append(delta.flatten())
                    data = torch.stack(layer_data).to(args.device)
                    train, test = data[:40], data[40:]
                    mean = train.mean(0)
                    weights = {name: value.to(args.device) for name, value in feature_weights.items()}
                    weights["oracle_anchor"] = oracle_weights.to(args.device)
                    for test_i in range(10):
                        predictions = {
                            "no_update": torch.zeros_like(mean),
                            "mean_anchor": mean,
                            "random_anchor": train[random_indices[test_i]],
                            "oracle_anchor": weights["oracle_anchor"][test_i] @ train,
                        }
                        for name, matrix in weights.items():
                            if name == "oracle_anchor":
                                continue
                            predictions[name] = matrix[test_i] @ train
                        for name, pred in predictions.items():
                            base = payloads[40 + test_i][f"base_{kind}_norm2_layer"][layer]
                            accumulate(acc[(test_i, name)], test[test_i], pred, float(base))

                for test_i in range(10):
                    for name in names:
                        z = acc[(test_i, name)]
                        rows.append({
                            "target": target_id,
                            "source": source,
                            "kind": kind,
                            "test_context": test_i,
                            "method": name,
                            "delta_norm2": z["delta"],
                            "error_norm2": z["error"],
                            "base_norm2": z["base"],
                            "dot": z["dot"],
                            "pred_norm2": z["pred"],
                            "original_gap": math.sqrt(z["delta"] / z["base"]),
                            "delta_recovery_error": math.sqrt(z["error"] / z["delta"]) if z["delta"] else float("nan"),
                            "final_kv_error": math.sqrt(z["error"] / z["base"]),
                            "explained_delta_energy": 1.0 - z["error"] / z["delta"] if z["delta"] else float("nan"),
                            "cosine": z["dot"] / max(math.sqrt(z["delta"] * z["pred"]), 1e-30),
                        })
                print(json.dumps({"target": target_id, "source": source, "kind": kind, "status": "done"}), flush=True)
            del payloads
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    csv_path = out / "per_case_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["source"], row["kind"], row["method"]), []).append(row)
    summary = []
    for (source, kind, method), items in sorted(groups.items()):
        delta = sum(x["delta_norm2"] for x in items)
        error = sum(x["error_norm2"] for x in items)
        base = sum(x["base_norm2"] for x in items)
        dot = sum(x["dot"] for x in items)
        pred = sum(x["pred_norm2"] for x in items)
        summary.append({
            "source": source,
            "kind": kind,
            "method": method,
            "n": len(items),
            "original_gap": math.sqrt(delta / base),
            "recovery": math.sqrt(error / delta),
            "final_error": math.sqrt(error / base),
            "explained_delta_energy": 1.0 - error / delta,
            "cosine": dot / max(math.sqrt(delta * pred), 1e-30),
            "original_gap_mean": sum(x["original_gap"] for x in items) / len(items),
        })
    summary_path = out / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (out / "summary.json").write_text(json.dumps({"rows": summary}, indent=2) + "\n")
    print(summary_path)


if __name__ == "__main__":
    main()
