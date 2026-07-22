#!/usr/bin/env python3
"""Train/evaluate a depth-recurrent Qwen3-8B Delta-KV predictor.

Compared with train_qwen3_8b_wiki_predictor.py, this model does not predict each
layer independently.  For each (sample, kv_head, sampled_token) it forms a
36-step layer sequence and uses a GRU to propagate a small state code across
depth before decoding K/V PCA coefficients.
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


def fit_basis(paths, rank: int):
    stats = {
        kind: {
            "sum": torch.zeros(L, H, F, dtype=torch.float64),
            "yty": torch.zeros(L, H, F, F, dtype=torch.float64),
            "n": 0,
        }
        for kind in ("k", "v")
    }
    gaps = {"k": [], "v": []}
    for path in paths:
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


def sequence_tensors(record, basis, rank: int):
    # Returns all H*T sequences for one record, each sequence is length L.
    t = record["delta_kv"].shape[2]
    own = record["own_kv"].float().permute(1, 2, 0, 3).reshape(H * t, L, 2 * F)
    prefix = record["prefix_features"].float().unsqueeze(2).expand(L, H, t, 4 * F)
    prefix = prefix.permute(1, 2, 0, 3).reshape(H * t, L, 4 * F)
    coeff = project_coefficients(record["delta_kv"], basis, rank)
    target = coeff.permute(1, 2, 0, 3).reshape(H * t, L, 2 * rank)
    token_pos = record.get("sampled_positions", torch.linspace(0, 127, t).long()).float() / 127.0
    head_ids = torch.arange(H).view(H, 1).expand(H, t).reshape(-1)
    token_ids = torch.arange(t).view(1, t).expand(H, t).reshape(-1)
    layer_ids = torch.arange(L).view(1, L).expand(H * t, L)
    head_seq = head_ids.view(-1, 1).expand(H * t, L)
    token_pos_seq = token_pos[token_ids].view(-1, 1).expand(H * t, L)
    sample_seq = torch.full((H * t, L), record["sample"] / 100000.0)
    pos = torch.stack([token_pos_seq, layer_ids.float() / (L - 1), head_seq.float() / (H - 1), sample_seq], -1)
    delta = record["delta_kv"].float()
    base = record["own_kv"].float()
    return own, prefix, pos, layer_ids.long(), head_seq.long(), target, delta, base


class DepthRecurrentPredictor(nn.Module):
    def __init__(self, rank: int, hidden_dim: int = 128, group_rank: int = 8):
        super().__init__()
        self.rank = rank
        self.own = nn.Sequential(nn.LayerNorm(2 * F), nn.Linear(2 * F, 128), nn.GELU())
        self.prefix = nn.Sequential(nn.LayerNorm(4 * F), nn.Linear(4 * F, 32), nn.GELU())
        self.layer_emb = nn.Embedding(L, 16)
        self.head_emb = nn.Embedding(H, 16)
        input_dim = 128 + 32 + 16 + 16 + 4
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.group_a = nn.Parameter(torch.empty(L, H, hidden_dim, group_rank))
        self.group_b = nn.Parameter(torch.empty(L, H, group_rank, 2 * rank))
        self.group_bias = nn.Parameter(torch.zeros(L, H, 2 * rank))
        nn.init.normal_(self.group_a, std=0.02)
        nn.init.zeros_(self.group_b)

    def forward(self, own, prefix, position, layer, head):
        own_h = self.own(own)
        prefix_h = self.prefix(prefix)
        layer_h = self.layer_emb(layer)
        head_h = self.head_emb(head)
        gru_in = torch.cat([own_h, prefix_h, layer_h, head_h, position], -1)
        state, _ = self.gru(gru_in)
        flat_state = state.reshape(-1, state.shape[-1])
        flat_layer = layer.reshape(-1)
        flat_head = head.reshape(-1)
        hidden = torch.einsum("nf,nfr->nr", flat_state, self.group_a[flat_layer, flat_head])
        hidden = torch.nn.functional.gelu(hidden)
        out = torch.einsum("nr,nro->no", hidden, self.group_b[flat_layer, flat_head]) + self.group_bias[flat_layer, flat_head]
        return out.reshape(own.shape[0], own.shape[1], -1)


def random_batch(files, basis, rank: int, batch_files: int, seqs_per_file: int, device):
    rows = []
    for path in random.sample(files, batch_files):
        tensors = sequence_tensors(load_record(path), basis, rank)[:6]
        n = tensors[0].shape[0]
        idx = torch.randperm(n)[: min(seqs_per_file, n)]
        rows.append(tuple(x[idx] for x in tensors))
    batch = [torch.cat(items, 0).to(device) for items in zip(*rows)]
    batch[3] = batch[3].long()
    batch[4] = batch[4].long()
    return tuple(x.float() if i in (0, 1, 2, 5) else x for i, x in enumerate(batch))


def normalized_loss(model, batch, std):
    own, prefix, pos, layer, head, target = batch
    pred = model(own, prefix, pos, layer, head)
    scale = torch.cat([std[0, layer, head], std[1, layer, head]], -1)
    return (pred - target / scale).square().mean()


@torch.no_grad()
def evaluate(model, files, basis, rank: int, std, device, max_files: int | None = None):
    methods = ["mean", "recurrent", f"oracle_rank{rank}"]
    totals = {
        (kind, method): {key: 0.0 for key in ("delta", "error", "base", "dot", "pred")}
        for kind in ("k", "v")
        for method in methods
    }
    selected = files[:max_files] if max_files else files
    model.eval()
    for path in selected:
        own, prefix, pos, layer, head, _, delta, base = sequence_tensors(load_record(path), basis, rank)
        pred_chunks = []
        for start in range(0, own.shape[0], 64):
            stop = min(own.shape[0], start + 64)
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
        coeff_seq = torch.cat(pred_chunks, 0)  # [H*T,L,2R]
        t = delta.shape[2]
        coeff = coeff_seq.reshape(H, t, L, 2 * rank).permute(2, 0, 1, 3).contiguous()
        oracle_coeff = project_coefficients(delta, basis, rank)
        for offset, kind, coeff_offset in ((0, "k", 0), (F, "v", rank)):
            y = delta[..., offset : offset + F].float()
            b = base[..., offset : offset + F].float()
            mean = basis[kind]["mean"].unsqueeze(2).expand_as(y)
            vec = basis[kind]["basis"][..., :rank]
            pred_rec = basis[kind]["mean"].unsqueeze(2) + torch.einsum(
                "lhtr,lhfr->lhtf", coeff[..., coeff_offset : coeff_offset + rank], vec
            )
            oracle = basis[kind]["mean"].unsqueeze(2) + torch.einsum(
                "lhtr,lhfr->lhtf", oracle_coeff[..., coeff_offset : coeff_offset + rank], vec
            )
            for method, pred in (("mean", mean), ("recurrent", pred_rec), (f"oracle_rank{rank}", oracle)):
                total = totals[(kind, method)]
                total["delta"] += float(y.square().sum())
                total["error"] += float((y - pred).square().sum())
                total["base"] += float(b.square().sum())
                total["dot"] += float((y * pred).sum())
                total["pred"] += float(pred.square().sum())
    rows = []
    for (kind, method), total in totals.items():
        remaining = math.sqrt(total["error"] / max(total["delta"], 1e-30))
        rows.append({
            "kind": kind,
            "method": method,
            "remaining_delta": remaining,
            "explained_delta_energy": 1 - remaining ** 2,
            "original_gap": math.sqrt(total["delta"] / max(total["base"], 1e-30)),
            "final_kv_error": math.sqrt(total["error"] / max(total["base"], 1e-30)),
            "delta_cosine": total["dot"] / math.sqrt(max(total["delta"] * total["pred"], 1e-30)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-pt", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=240)
    parser.add_argument("--batch-files", type=int, default=8)
    parser.add_argument("--seqs-per-file", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--eval-max-files", type=int, default=0)
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_files = sample_files(Path(args.train_dir))
    val_files = sample_files(Path(args.val_dir))
    basis, train_gap = fit_basis(train_files, args.rank)
    std = coefficient_std(train_files, basis, args.rank).to(device)
    model = DepthRecurrentPredictor(args.rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best = None
    patience = 0
    eval_files = args.eval_max_files if args.eval_max_files > 0 else None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(args.steps_per_epoch):
            batch = random_batch(train_files, basis, args.rank, args.batch_files, args.seqs_per_file, device)
            loss = normalized_loss(model, batch, std)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        rows = evaluate(model, val_files, basis, args.rank, std, device, eval_files)
        score = sum(row["remaining_delta"] for row in rows if row["method"] == "recurrent")
        record = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "validation_score": score, "rows": rows}
        history.append(record)
        print(json.dumps(record), flush=True)
        if best is None or score < best[0]:
            best = (score, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            patience = 0
        else:
            patience += 1
        if patience >= 3:
            break
    model.load_state_dict(best[2])
    final_rows = evaluate(model, val_files, basis, args.rank, std, device, None)
    result = {
        "model": "Qwen3-8B",
        "task": "WikiText depth-recurrent layer predictor",
        "rank": args.rank,
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "train_files": len(train_files),
        "val_files": len(val_files),
        "train_gap_mean": train_gap,
        "basis_train_rank_energy": {kind: basis[kind]["train_rank_energy"] for kind in ("k", "v")},
        "predictor": "GRU over 36 layers for each (sample, kv_head, sampled_token); output is K/V PCA coefficients",
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best[1],
        "history": history,
        "final_rows": final_rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    torch.save({
        "state_dict": best[2],
        "coefficient_std": std.cpu().half(),
        "basis": basis,
        "config": {"rank": args.rank, "layers": L, "kv_heads": H, "head_dim": F},
        "result": result,
    }, args.output_pt)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
