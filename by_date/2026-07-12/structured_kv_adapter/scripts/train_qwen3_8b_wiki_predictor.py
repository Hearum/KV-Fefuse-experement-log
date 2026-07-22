#!/usr/bin/env python3
"""Train/evaluate a compact Qwen3-8B WikiText Delta-KV predictor.

Data format is produced by collect_qwen3_8b_wikitext_delta.py:
  own_kv:          [L, H, T, 2F]
  prefix_features: [L, H, 4F]
  delta_kv:        [L, H, T, 2F]

The model predicts PCA coefficients, not full KV deltas.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn


L, H, F = 36, 8, 128
SEED = 20260712


def sample_files(directory: Path):
    files = sorted(directory.glob("sample*.pt"))
    if not files:
        raise FileNotFoundError(f"no sample*.pt in {directory}")
    return files


def load_record(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def fit_basis(paths, rank: int, max_files: int | None = None):
    selected = paths[:max_files] if max_files else paths
    stats = {
        kind: {
            "sum": torch.zeros(L, H, F, dtype=torch.float64),
            "yty": torch.zeros(L, H, F, F, dtype=torch.float64),
            "n": 0,
        }
        for kind in ("k", "v")
    }
    gaps = {"k": [], "v": []}
    for path in selected:
        record = load_record(path)
        gaps["k"].append(record["metrics"]["k_original_gap"])
        gaps["v"].append(record["metrics"]["v_original_gap"])
        for offset, kind in ((0, "k"), (F, "v")):
            y = record["delta_kv"][..., offset : offset + F].double()
            stats[kind]["sum"] += y.sum(2)
            stats[kind]["yty"] += torch.einsum("lhti,lhtj->lhij", y, y)
            stats[kind]["n"] += y.shape[2]
    basis = {}
    for kind, stat in stats.items():
        mean = stat["sum"] / stat["n"]
        covariance = stat["yty"] - stat["n"] * torch.einsum("lhi,lhj->lhij", mean, mean)
        evals, vectors = torch.linalg.eigh(covariance)
        vectors = vectors.flip(-1)[..., :rank].contiguous().float()
        evals = evals.flip(-1).clamp_min(0)
        total = evals.sum(-1).clamp_min(1e-12)
        basis[kind] = {
            "mean": mean.float(),
            "basis": vectors,
            "train_rank_energy": {
                str(r): float((evals[..., : min(r, F)].sum(-1) / total).mean())
                for r in (8, 16, 32, 64, 128)
                if r <= F
            },
        }
    return basis, {"k": sum(gaps["k"]) / len(gaps["k"]), "v": sum(gaps["v"]) / len(gaps["v"])}


def project_coefficients(delta_kv, basis, rank: int):
    coefficients = []
    for offset, kind in ((0, "k"), (F, "v")):
        y = delta_kv[..., offset : offset + F].float()
        centered = y - basis[kind]["mean"].unsqueeze(2)
        coefficients.append(torch.einsum("lhtf,lhfr->lhtr", centered, basis[kind]["basis"][..., :rank]))
    return torch.cat(coefficients, -1)


def flatten_record(record, basis, rank: int):
    t = record["delta_kv"].shape[2]
    own = record["own_kv"].float().reshape(-1, 2 * F)
    prefix = record["prefix_features"].float().unsqueeze(2).expand(L, H, t, 4 * F).reshape(-1, 4 * F)
    token_pos = record.get("sampled_positions", torch.linspace(0, 127, t).long()).float() / 127.0
    layer_ids = torch.arange(L).view(L, 1, 1).expand(L, H, t).reshape(-1)
    head_ids = torch.arange(H).view(1, H, 1).expand(L, H, t).reshape(-1)
    position = torch.stack(
        [
            token_pos.view(1, 1, t).expand(L, H, t).reshape(-1),
            layer_ids.float() / max(L - 1, 1),
            head_ids.float() / max(H - 1, 1),
            torch.full((L * H * t,), record["sample"] / 100000.0),
        ],
        -1,
    )
    target = project_coefficients(record["delta_kv"], basis, rank).reshape(-1, 2 * rank)
    delta = record["delta_kv"].float()
    base = record["own_kv"].float()
    return own, prefix, position, layer_ids.long(), head_ids.long(), target, delta, base


def coefficient_std(paths, basis, rank: int):
    squares = torch.zeros(2, L, H, rank)
    counts = torch.zeros(L, H, 1)
    for path in paths:
        record = load_record(path)
        coeff = project_coefficients(record["delta_kv"], basis, rank)
        squares[0] += coeff[..., :rank].square().sum(2)
        squares[1] += coeff[..., rank:].square().sum(2)
        counts += coeff.shape[2]
    return (squares / counts.clamp_min(1)).sqrt().clamp_min(1e-5)


class Predictor(nn.Module):
    def __init__(self, rank: int, prefix_dim: int = 32, group_rank: int = 8):
        super().__init__()
        self.rank = rank
        self.own = nn.Sequential(nn.LayerNorm(2 * F), nn.Linear(2 * F, 128), nn.GELU())
        self.prefix = nn.Sequential(nn.LayerNorm(4 * F), nn.Linear(4 * F, prefix_dim), nn.GELU())
        feature_dim = 128 + prefix_dim + 4
        self.group_a = nn.Parameter(torch.empty(L, H, feature_dim, group_rank))
        self.group_b = nn.Parameter(torch.empty(L, H, group_rank, 2 * rank))
        self.group_bias = nn.Parameter(torch.zeros(L, H, 2 * rank))
        nn.init.normal_(self.group_a, std=0.02)
        nn.init.zeros_(self.group_b)

    def forward(self, own, prefix, position, layer, head):
        features = torch.cat([self.own(own), self.prefix(prefix), position], -1)
        hidden = torch.einsum("nf,nfr->nr", features, self.group_a[layer, head])
        hidden = torch.nn.functional.gelu(hidden)
        return torch.einsum("nr,nro->no", hidden, self.group_b[layer, head]) + self.group_bias[layer, head]


def normalized_loss(model, batch, std):
    own, prefix, position, layer, head, target = batch
    pred = model(own, prefix, position, layer, head)
    scale = torch.cat([std[0, layer, head], std[1, layer, head]], -1)
    return (pred - target / scale).square().mean()


def random_batch(files, basis, rank: int, batch_files: int, rows_per_file: int, device, std):
    rows = []
    for path in random.sample(files, batch_files):
        own, prefix, pos, layer, head, target, _, _ = flatten_record(load_record(path), basis, rank)
        n = own.shape[0]
        idx = torch.randperm(n)[: min(rows_per_file, n)]
        rows.append((own[idx], prefix[idx], pos[idx], layer[idx], head[idx], target[idx]))
    batch = [torch.cat(items, 0).to(device) for items in zip(*rows)]
    batch[3] = batch[3].long()
    batch[4] = batch[4].long()
    return tuple(x.float() if i in (0, 1, 2, 5) else x for i, x in enumerate(batch))


@torch.no_grad()
def evaluate_model(model, files, basis, rank: int, std, device, max_files: int | None = None):
    methods = ["mean", "mlp", f"oracle_rank{rank}"]
    totals = {
        (kind, method): {key: 0.0 for key in ("delta", "error", "base", "dot", "pred")}
        for kind in ("k", "v")
        for method in methods
    }
    selected = files[:max_files] if max_files else files
    model.eval()
    for path in selected:
        own, prefix, pos, layer, head, target, delta, base = flatten_record(load_record(path), basis, rank)
        pred_chunks = []
        for start in range(0, own.shape[0], 8192):
            stop = min(own.shape[0], start + 8192)
            x = (
                own[start:stop].to(device),
                prefix[start:stop].to(device),
                pos[start:stop].to(device),
                layer[start:stop].to(device),
                head[start:stop].to(device),
            )
            normalized = model(*x)
            scale = torch.cat([std[0, x[3], x[4]], std[1, x[3], x[4]]], -1)
            pred_chunks.append((normalized * scale).cpu())
        coeff = torch.cat(pred_chunks, 0).reshape(L, H, delta.shape[2], 2 * rank)
        oracle_coeff = project_coefficients(delta, basis, rank)
        for offset, kind, coeff_offset in ((0, "k", 0), (F, "v", rank)):
            y = delta[..., offset : offset + F].float()
            b = base[..., offset : offset + F].float()
            mean = basis[kind]["mean"].unsqueeze(2).expand_as(y)
            vec = basis[kind]["basis"][..., :rank]
            mlp = basis[kind]["mean"].unsqueeze(2) + torch.einsum(
                "lhtr,lhfr->lhtf", coeff[..., coeff_offset : coeff_offset + rank], vec
            )
            oracle = basis[kind]["mean"].unsqueeze(2) + torch.einsum(
                "lhtr,lhfr->lhtf", oracle_coeff[..., coeff_offset : coeff_offset + rank], vec
            )
            for method, pred in (("mean", mean), ("mlp", mlp), (f"oracle_rank{rank}", oracle)):
                total = totals[(kind, method)]
                total["delta"] += float(y.square().sum())
                total["error"] += float((y - pred).square().sum())
                total["base"] += float(b.square().sum())
                total["dot"] += float((y * pred).sum())
                total["pred"] += float(pred.square().sum())
    rows = []
    for (kind, method), total in totals.items():
        remaining = math.sqrt(total["error"] / max(total["delta"], 1e-30))
        cosine = total["dot"] / math.sqrt(max(total["delta"] * total["pred"], 1e-30))
        rows.append(
            {
                "kind": kind,
                "method": method,
                "remaining_delta": remaining,
                "explained_delta_energy": 1 - remaining**2,
                "original_gap": math.sqrt(total["delta"] / max(total["base"], 1e-30)),
                "final_kv_error": math.sqrt(total["error"] / max(total["base"], 1e-30)),
                "delta_cosine": cosine,
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-pt", required=True)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--steps-per-epoch", type=int, default=320)
    p.add_argument("--batch-files", type=int, default=8)
    p.add_argument("--rows-per-file", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eval-max-files", type=int, default=0)
    args = p.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_files = sample_files(Path(args.train_dir))
    val_files = sample_files(Path(args.val_dir))
    basis, train_gap = fit_basis(train_files, args.rank)
    std = coefficient_std(train_files, basis, args.rank).to(device)
    model = Predictor(args.rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best = None
    patience = 0
    eval_files = args.eval_max_files if args.eval_max_files > 0 else None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            batch = random_batch(train_files, basis, args.rank, args.batch_files, args.rows_per_file, device, std)
            loss = normalized_loss(model, batch, std)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        rows = evaluate_model(model, val_files, basis, args.rank, std, device, eval_files)
        score = sum(row["remaining_delta"] for row in rows if row["method"] == "mlp")
        record = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "validation_score": score, "rows": rows}
        history.append(record)
        print(json.dumps(record), flush=True)
        if best is None or score < best[0]:
            best = (score, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, rows)
            patience = 0
        else:
            patience += 1
        if patience >= 4:
            break

    model.load_state_dict(best[2])
    final_rows = evaluate_model(model, val_files, basis, args.rank, std, device, None)
    result = {
        "model": "Qwen3-8B",
        "task": "WikiText A=prefix256, X=target128, raw/offline X -> full A+X target Delta",
        "rank": args.rank,
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "train_files": len(train_files),
        "val_files": len(val_files),
        "train_gap_mean": train_gap,
        "basis_train_rank_energy": {kind: basis[kind]["train_rank_energy"] for kind in ("k", "v")},
        "predictor": "shared own/prefix encoder + per-layer/head rank8 output heads, predicts K/V PCA coefficients",
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best[1],
        "history": history,
        "final_rows": final_rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    torch.save(
        {
            "state_dict": best[2],
            "coefficient_std": std.cpu().half(),
            "basis": basis,
            "config": {"rank": args.rank, "layers": L, "kv_heads": H, "head_dim": F},
            "result": result,
        },
        args.output_pt,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
