#!/usr/bin/env python3
"""Diagnostic probe for parallel per-layer KV update feasibility.

Allowed inputs:
  - offline/cached per-layer hidden sidecar (own_h)
  - offline/cached K/V for the target token (own_kv)
  - prefix K/V from the already available context (prefix_kv)

Forbidden as inputs:
  - full-context hidden
  - full-context K/V
  - true Delta coefficients

For each layer independently, we use cached hidden to produce Q, let it read
prefix K/V at that same layer, and test whether that layer-local readout can
linearly explain DeltaK/DeltaV. This is a mechanism probe, not a deployable
trained adapter.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoConfig


for candidate in (
    Path("/raid/home/hming/FusionRAG-pca-analysis"),
    Path("/home/hming/FusionRAG-pca-analysis"),
):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

from test_fusionrag_reflect_preprocess_exp import load_model


L, H_KV, H_Q, T, F, D = 36, 8, 32, 8, 128, 4096


def load_pairs(hidden_dir: Path, fullprefix_dir: Path):
    hidden_files = sorted(hidden_dir.glob("sample*.pt"))
    if not hidden_files:
        raise FileNotFoundError(hidden_dir)
    pairs = []
    for h_path in hidden_files:
        f_path = fullprefix_dir / h_path.name
        if not f_path.exists():
            continue
        h = torch.load(h_path, map_location="cpu", weights_only=False)
        f = torch.load(f_path, map_location="cpu", weights_only=False)
        for key in ("sample", "token_start", "prefix_tokens", "target_tokens"):
            if h.get(key) != f.get(key):
                raise ValueError(f"{h_path.name}: {key} mismatch {h.get(key)} vs {f.get(key)}")
        if not torch.equal(h["sampled_positions"], f["sampled_positions"]):
            raise ValueError(f"{h_path.name}: sampled_positions mismatch")
        pairs.append((h_path.name, h, f))
    if not pairs:
        raise FileNotFoundError(f"no joined samples in {hidden_dir} and {fullprefix_dir}")
    return pairs


def rotate_half(x: torch.Tensor):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_query(model, query_states: torch.Tensor, position_ids: torch.Tensor):
    # query_states: [1, H_Q, T, F], position_ids: [1, T]
    try:
        cos, sin = model.model.rotary_emb(query_states[0], position_ids)
    except AttributeError:
        cos, sin = model.model.layers[0].self_attn.rotary_emb(query_states[0], position_ids)
    cos = cos.to(query_states.device).unsqueeze(1)
    sin = sin.to(query_states.device).unsqueeze(1)
    return query_states * cos + rotate_half(query_states) * sin


@torch.no_grad()
def parallel_context_for_record(model, hidden_record, fullprefix_record, device: str):
    sampled_positions = hidden_record["sampled_positions"].to(device)
    prefix_tokens = int(hidden_record["prefix_tokens"])
    full_positions = (prefix_tokens + sampled_positions).view(1, -1)
    own_h = hidden_record["own_h"].float()
    prefix_kv = fullprefix_record["prefix_kv"].float()
    contexts = torch.empty(L, H_KV, T, F, dtype=torch.float16)
    scale = 1.0 / math.sqrt(F)
    for layer_idx, layer in enumerate(model.model.layers):
        h = own_h[layer_idx].to(device=device, dtype=model.dtype)  # [T,D]
        q = layer.self_attn.q_proj(h.unsqueeze(0))
        q = q.view(1, T, H_Q, F).transpose(1, 2)
        q = layer.self_attn.q_norm(q)
        q = rope_query(model, q, full_positions)
        k = prefix_kv[layer_idx, :, :, :F].to(device=device, dtype=q.dtype)
        v = prefix_kv[layer_idx, :, :, F:].to(device=device, dtype=q.dtype)
        k_rep = k.repeat_interleave(H_Q // H_KV, dim=0)
        v_rep = v.repeat_interleave(H_Q // H_KV, dim=0)
        scores = torch.einsum("bhtf,hpf->bhtp", q, k_rep) * scale
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.einsum("bhtp,hpf->bhtf", probs, v_rep)[0]  # [H_Q,T,F]
        grouped = out.view(H_KV, H_Q // H_KV, T, F).mean(1)
        contexts[layer_idx] = grouped.detach().cpu().half()
    return contexts


def pca_project(train_x, test_x, rank: int):
    mean = train_x.mean(0, keepdim=True)
    train_centered = train_x - mean
    test_centered = test_x - mean
    used = min(rank, train_x.shape[0] - 1, train_x.shape[1])
    if used <= 0:
        return train_centered[:, :0], test_centered[:, :0], 0
    _, _, vh = torch.linalg.svd(train_centered.double(), full_matrices=False)
    basis = vh[:used].float().T.contiguous()
    return train_centered @ basis, test_centered @ basis, used


def ridge_predict(train_x, train_y, test_x, alpha: float):
    x_mean = train_x.mean(0, keepdim=True)
    y_mean = train_y.mean(0, keepdim=True)
    x = train_x - x_mean
    y = train_y - y_mean
    xtx = x.T @ x
    reg = alpha * torch.eye(xtx.shape[0], dtype=xtx.dtype)
    weight = torch.linalg.solve(xtx + reg, x.T @ y)
    return (test_x - x_mean) @ weight + y_mean


def metric_row(pred, target, mean_pred):
    delta_sq = float(target.square().sum())
    err_sq = float((target - pred).square().sum())
    mean_err_sq = float((target - mean_pred).square().sum())
    dot = float((target * pred).sum())
    pred_sq = float(pred.square().sum())
    return {
        "remaining_delta": math.sqrt(err_sq / max(delta_sq, 1e-30)),
        "explained_delta_energy": 1.0 - err_sq / max(delta_sq, 1e-30),
        "mean_explained_delta_energy": 1.0 - mean_err_sq / max(delta_sq, 1e-30),
        "cosine": dot / math.sqrt(max(delta_sq * pred_sq, 1e-30)),
    }


def build_rows(records, contexts_by_name, layer_idx: int, head_idx: int, feature: str, target: str):
    xs, ys = [], []
    for name, _, f in records:
        own = f["own_kv"][layer_idx, head_idx].float()  # [T,256]
        context = contexts_by_name[name][layer_idx, head_idx].float()  # [T,128]
        if feature == "own_kv":
            x = own
        elif feature == "parallel_context":
            x = context
        elif feature == "own_kv_parallel_context":
            x = torch.cat([own, context], -1)
        else:
            raise KeyError(feature)
        delta = f["delta_kv"][layer_idx, head_idx].float()
        y = delta[:, :F] if target == "k" else delta[:, F:]
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, 0), torch.cat(ys, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dir", required=True)
    parser.add_argument("--fullprefix-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model-path", default="/home/hming/models/Qwen3-8B")
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=96)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1e-2)
    args = parser.parse_args()

    pairs = load_pairs(Path(args.hidden_dir), Path(args.fullprefix_dir))[: args.max_records]
    if args.train_count <= 0 or args.train_count >= len(pairs):
        raise ValueError("--train-count must leave a heldout split")
    train, test = pairs[: args.train_count], pairs[args.train_count :]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"
    model, _ = load_model("qwen3", args.model_path, cfg, device, False)
    model.eval()

    contexts = {}
    for idx, (name, h, f) in enumerate(pairs):
        contexts[name] = parallel_context_for_record(model, h, f, device)
        print(json.dumps({"computed_context": name, "index": idx + 1, "total": len(pairs)}), flush=True)

    features = ["own_kv", "parallel_context", "own_kv_parallel_context"]
    rows = []
    for layer_idx in range(L):
        for head_idx in range(H_KV):
            for target in ("k", "v"):
                for feature in features:
                    train_x, train_y = build_rows(train, contexts, layer_idx, head_idx, feature, target)
                    test_x, test_y = build_rows(test, contexts, layer_idx, head_idx, feature, target)
                    train_z, test_z, used_rank = pca_project(train_x, test_x, args.rank)
                    pred = ridge_predict(train_z.float(), train_y.float(), test_z.float(), args.alpha)
                    mean_pred = train_y.mean(0, keepdim=True).expand_as(test_y)
                    row = metric_row(pred, test_y.float(), mean_pred.float())
                    row.update({
                        "layer": layer_idx,
                        "head": head_idx,
                        "target": target,
                        "feature": feature,
                        "rank": args.rank,
                        "used_rank": used_rank,
                    })
                    rows.append(row)

    summary = {}
    for target in ("k", "v"):
        summary[target] = {}
        base = [r for r in rows if r["target"] == target and r["feature"] == "own_kv"]
        summary[target]["mean_template"] = sum(r["mean_explained_delta_energy"] for r in base) / len(base)
        for feature in features:
            selected = [r for r in rows if r["target"] == target and r["feature"] == feature]
            summary[target][feature] = {
                "mean_explained": sum(r["explained_delta_energy"] for r in selected) / len(selected),
                "mean_remaining": sum(r["remaining_delta"] for r in selected) / len(selected),
                "mean_cosine": sum(r["cosine"] for r in selected) / len(selected),
                "best_layer": max(selected, key=lambda r: r["explained_delta_energy"])["layer"],
                "best_layer_explained": max(r["explained_delta_energy"] for r in selected),
            }

    result = {
        "hidden_dir": args.hidden_dir,
        "fullprefix_dir": args.fullprefix_dir,
        "records": len(pairs),
        "train_count": len(train),
        "test_count": len(test),
        "rank": args.rank,
        "alpha": args.alpha,
        "input_policy": "offline own_h/own_kv + prefix_kv only; no full hidden/full target/delta as inputs",
        "summary": summary,
        "rows": rows,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
