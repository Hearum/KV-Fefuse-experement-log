#!/usr/bin/env python3
"""Functional probe for a lightweight predicted-K adapter.

Train a per-layer/per-KV-head PCA+Ridge predictor from offline own_kv to
RoPE-aligned DeltaK on train records, then evaluate whether predicted K reduces
future-query attention logit/output error over target document X.

No oracle coefficient is used at evaluation time: predicted DeltaK is produced
only from offline target K/V. Full K/V is used only as teacher/reference for
metrics.
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

from ktransformers.models.custom_cache import StaticCache
from test_fusionrag_reflect_preprocess_exp import load_model

L, H_KV, H_Q, F = 36, 8, 32, 128


def repeat_kv(x, groups):
    return x.repeat_interleave(groups, dim=1)


def rotate(model, keys, delta, device):
    z = keys.to(device)
    pos = torch.full((1, z.shape[-2]), delta, device=device)
    try:
        cos, sin = model.model.rotary_emb(z[0], pos)
    except AttributeError:
        cos, sin = model.model.layers[0].self_attn.rotary_emb(z[0], pos)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    half = z.shape[-1] // 2
    return (z * cos + torch.cat([-z[..., half:], z[..., :half]], -1) * sin).cpu()


def q_for_positions(model, h_in, positions, layer_idx, device):
    layer = model.model.layers[layer_idx]
    h = h_in.to(device=device, dtype=layer.self_attn.q_proj.weight.dtype)
    q = layer.self_attn.q_proj(h)
    q = q.view(q.shape[0], layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(0, 1).unsqueeze(0)
    q = layer.self_attn.q_norm(q)
    dummy_k = torch.zeros(1, layer.self_attn.num_key_value_heads, q.shape[-2], q.shape[-1], device=device, dtype=q.dtype)
    pos = positions.view(1, -1).to(device)
    cos, sin = layer.self_attn.rotary_emb(dummy_k, pos)
    from models.modeling_qwen3 import apply_rotary_pos_emb

    q_rot, _ = apply_rotary_pos_emb(q, dummy_k, cos, sin, pos)
    return q_rot[0].detach().cpu().float()


def pca_project(train_x, rank):
    mean = train_x.mean(0, keepdim=True)
    centered = train_x - mean
    used = min(rank, train_x.shape[0] - 1, train_x.shape[1])
    if used <= 0:
        return mean, torch.empty(train_x.shape[1], 0), centered[:, :0], 0
    _, _, vh = torch.linalg.svd(centered.double(), full_matrices=False)
    basis = vh[:used].float().T.contiguous()
    return mean.float(), basis, centered.float() @ basis, used


def fit_ridge(train_z, train_y, alpha):
    x_mean = train_z.mean(0, keepdim=True)
    y_mean = train_y.mean(0, keepdim=True)
    x = train_z - x_mean
    y = train_y - y_mean
    xtx = x.T @ x
    reg = alpha * torch.eye(xtx.shape[0], dtype=xtx.dtype)
    weight = torch.linalg.solve(xtx + reg, x.T @ y)
    return x_mean.float(), y_mean.float(), weight.float()


def load_train_tensors(train_paths):
    xs, ys = [], []
    for path in train_paths:
        rec = torch.load(path, map_location="cpu", weights_only=False)
        xs.append(rec["own_kv"].float())
        ys.append(rec["delta_kv"][..., :F].float())
    # [L,H,files*T,dim]
    return torch.cat(xs, 2), torch.cat(ys, 2)


def train_k_predictors(train_paths, rank: int, alpha: float):
    train_x, train_y = load_train_tensors(train_paths)
    predictors = [[None for _ in range(H_KV)] for _ in range(L)]
    rows = []
    for layer in range(L):
        for head in range(H_KV):
            x = train_x[layer, head]
            y = train_y[layer, head]
            x_mean, basis, z, used = pca_project(x, rank)
            z_mean, y_mean, weight = fit_ridge(z, y, alpha)
            pred = (z - z_mean) @ weight + y_mean
            err = float((pred - y).square().sum())
            delta = float(y.square().sum())
            predictors[layer][head] = {
                "x_mean": x_mean,
                "basis": basis,
                "z_mean": z_mean,
                "y_mean": y_mean,
                "weight": weight,
                "used_rank": used,
            }
            rows.append({"layer": layer, "head": head, "train_explained": 1.0 - err / max(delta, 1e-30)})
    return predictors, rows


def predict_delta_k(predictors, own_kv_all):
    # own_kv_all: [L,H,T,256], returns [L,H,T,F] in offline/RoPE-aligned coordinates.
    pred = torch.empty(own_kv_all.shape[:-1] + (F,), dtype=torch.float32)
    for layer in range(L):
        for head in range(H_KV):
            p = predictors[layer][head]
            x = own_kv_all[layer, head].float()
            z = (x - p["x_mean"]) @ p["basis"]
            pred[layer, head] = (z - p["z_mean"]) @ p["weight"] + p["y_mean"]
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--start", type=int, default=96)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--stride", type=int, default=416)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model-path", default="/home/hming/models/Qwen3-8B")
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--scales", default="0,0.001,0.003,0.01,0.03,0.1,0.3,1.0")
    args = parser.parse_args()
    scales = [float(x) for x in args.scales.split(",") if x.strip()]

    train_paths = sorted(Path(args.train_dir).glob("sample*.pt"))[: args.train_count]
    if not train_paths:
        raise FileNotFoundError(args.train_dir)
    predictors, train_rows = train_k_predictors(train_paths, args.rank, args.alpha)

    device = "cuda:0"
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"
    model, _ = load_model("qwen3", args.model_path, cfg, device, False)
    model.eval()
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=512, device=device, dtype=model.dtype, passage_len=512, model=model)
    captures = {}
    handles = []

    def make_v_hook(layer_idx):
        def hook(module, inputs, output):
            captures[layer_idx] = inputs[0].detach().cpu().half()
        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(layer_idx)))

    def capture(ids):
        captures.clear()
        for layer in range(len(cache.key_cache)):
            cache.past_tokens[layer] = 0
        ids = ids.unsqueeze(0).to(device)
        pos = torch.arange(ids.shape[1], device=device)
        with torch.no_grad():
            model(input_ids=ids, cache_position=pos, past_key_values=cache, use_cache=True, return_dict=False)
        keys = torch.stack([x[:, :, : ids.shape[1], :].detach().cpu().half() for x in cache.key_cache])
        values = torch.stack([x[:, :, : ids.shape[1], :].detach().cpu().half() for x in cache.value_cache])
        h_in = torch.stack([captures[i][0] for i in range(len(model.model.layers))]).half()
        return keys, values, h_in

    ids = torch.load(args.token_cache, map_location="cpu", weights_only=True)
    groups = cfg.num_attention_heads // cfg.num_key_value_heads
    variants = ["cachedK_cachedV", "fullK_cachedV", "cachedK_fullV", "fullK_fullV"] + [f"predK_s{scale:g}_cachedV" for scale in scales]
    totals = {name: {key: 0.0 for key in ("logit_err", "logit_base", "out_err", "out_base")} for name in variants}
    layer_rows = []
    processed = 0
    for sample in range(args.start, args.start + args.count):
        begin = sample * args.stride
        span = ids[begin : begin + 416].long()
        if len(span) < 416:
            break
        prefix, target, query = span[:256], span[256:384], span[384:416]
        offline_k, offline_v, _ = capture(target)
        full_k, full_v, full_h = capture(torch.cat([prefix, target, query]))
        full_x_k = full_k[:, :, :, 256:384, :].float()
        full_x_v = full_v[:, :, :, 256:384, :].float()
        cached_x_k = rotate(model, offline_k[:, :, :, :128, :], 256, device).float()
        cached_x_v = offline_v[:, :, :, :128, :].float()
        own_kv_all = torch.cat([offline_k[:, 0, :, :128, :], offline_v[:, 0, :, :128, :]], -1).float()
        pred_delta_k = predict_delta_k(predictors, own_kv_all)
        pred_x_k_by_scale = {}
        offline_target_k = offline_k[:, :, :, :128, :].float()
        for scale in scales:
            pred_offline_k = offline_target_k + scale * pred_delta_k.unsqueeze(1)
            pred_x_k_by_scale[scale] = rotate(model, pred_offline_k, 256, device).float()
        q_pos = torch.arange(384, 416)
        for layer_idx in range(cfg.num_hidden_layers):
            q = q_for_positions(model, full_h[layer_idx, 384:416].float(), q_pos, layer_idx, device).unsqueeze(0)
            k_full = repeat_kv(full_x_k[layer_idx], groups)
            k_cached = repeat_kv(cached_x_k[layer_idx], groups)
            v_full = repeat_kv(full_x_v[layer_idx], groups)
            v_cached = repeat_kv(cached_x_v[layer_idx], groups)
            logits_full = torch.matmul(q, k_full.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            logits_cached = torch.matmul(q, k_cached.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            p_full = torch.softmax(logits_full, dim=-1)
            p_cached = torch.softmax(logits_cached, dim=-1)
            out_full = torch.matmul(p_full, v_full)
            variant_tensors = {
                "cachedK_cachedV": (logits_cached, torch.matmul(p_cached, v_cached)),
                "fullK_cachedV": (logits_full, torch.matmul(p_full, v_cached)),
                "cachedK_fullV": (logits_cached, torch.matmul(p_cached, v_full)),
                "fullK_fullV": (logits_full, out_full),
            }
            for scale in scales:
                k_pred = repeat_kv(pred_x_k_by_scale[scale][layer_idx], groups)
                logits_pred = torch.matmul(q, k_pred.transpose(-1, -2)) / math.sqrt(q.shape[-1])
                p_pred = torch.softmax(logits_pred, dim=-1)
                variant_tensors[f"predK_s{scale:g}_cachedV"] = (logits_pred, torch.matmul(p_pred, v_cached))
            row = {"sample": sample, "layer": layer_idx}
            for name, (logits, out) in variant_tensors.items():
                logit_err = float((logits - logits_full).square().sum())
                logit_base = float(logits_full.square().sum())
                out_err = float((out - out_full).square().sum())
                out_base = float(out_full.square().sum())
                totals[name]["logit_err"] += logit_err
                totals[name]["logit_base"] += logit_base
                totals[name]["out_err"] += out_err
                totals[name]["out_base"] += out_base
                row[f"{name}_logit_rel"] = math.sqrt(logit_err / max(logit_base, 1e-30))
                row[f"{name}_out_rel"] = math.sqrt(out_err / max(out_base, 1e-30))
            layer_rows.append(row)
        processed += 1
        print(json.dumps({"sample": sample, "processed": processed}), flush=True)

    for handle in handles:
        handle.remove()
    summary = {}
    for name, total in totals.items():
        summary[name] = {
            "logit_rel_error": math.sqrt(total["logit_err"] / max(total["logit_base"], 1e-30)),
            "output_rel_error": math.sqrt(total["out_err"] / max(total["out_base"], 1e-30)),
        }
    by_layer = []
    for layer_idx in range(cfg.num_hidden_layers):
        rows = [r for r in layer_rows if r["layer"] == layer_idx]
        out = {"layer": layer_idx}
        for name in variants:
            out[f"{name}_logit_rel_mean"] = sum(r[f"{name}_logit_rel"] for r in rows) / max(len(rows), 1)
            out[f"{name}_out_rel_mean"] = sum(r[f"{name}_out_rel"] for r in rows) / max(len(rows), 1)
        by_layer.append(out)
    result = {
        "task": "functional predicted-K probe over future query attention to X",
        "samples": processed,
        "train_dir": args.train_dir,
        "train_count": len(train_paths),
        "eval_start": args.start,
        "eval_count_requested": args.count,
        "prefix_tokens": 256,
        "target_tokens": 128,
        "query_tokens": 32,
        "rank": args.rank,
        "alpha": args.alpha,
        "scales": scales,
        "train_k_predictor_mean_explained": sum(r["train_explained"] for r in train_rows) / len(train_rows),
        "summary": summary,
        "by_layer": by_layer,
        "notes": "PredictedK uses only offline target K/V through a train-fitted own_kv->DeltaK ridge predictor. Restricted to query attention over X only; Q is teacher-forced from full prefix+target+query run.",
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
