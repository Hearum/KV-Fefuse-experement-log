#!/usr/bin/env python3
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
    # h_in: [seq, hidden], returns [num_heads, q_len, head_dim]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--stride", type=int, default=416)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model-path", default="/home/hming/models/Qwen3-8B")
    args = parser.parse_args()

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
    totals = {
        name: {key: 0.0 for key in ("logit_err", "logit_base", "out_err", "out_base")}
        for name in ("cachedK_cachedV", "fullK_cachedV", "cachedK_fullV")
    }
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
        q_pos = torch.arange(384, 416)
        sample_layer = []
        for layer_idx in range(cfg.num_hidden_layers):
            q = q_for_positions(model, full_h[layer_idx, 384:416].float(), q_pos, layer_idx, device)
            q = q.unsqueeze(0)  # [1,QH,QL,D]
            k_full = repeat_kv(full_x_k[layer_idx], groups)
            k_cached = repeat_kv(cached_x_k[layer_idx], groups)
            v_full = repeat_kv(full_x_v[layer_idx], groups)
            v_cached = repeat_kv(cached_x_v[layer_idx], groups)
            logits_full = torch.matmul(q, k_full.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            logits_cached = torch.matmul(q, k_cached.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            p_full = torch.softmax(logits_full, dim=-1)
            p_cached = torch.softmax(logits_cached, dim=-1)
            out_full = torch.matmul(p_full, v_full)
            variants = {
                "cachedK_cachedV": (logits_cached, torch.matmul(p_cached, v_cached)),
                "fullK_cachedV": (logits_full, torch.matmul(p_full, v_cached)),
                "cachedK_fullV": (logits_cached, torch.matmul(p_cached, v_full)),
            }
            row = {"sample": sample, "layer": layer_idx}
            for name, (logits, out) in variants.items():
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
            sample_layer.append(row)
        layer_rows.extend(sample_layer)
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
        for name in totals:
            out[f"{name}_logit_rel_mean"] = sum(r[f"{name}_logit_rel"] for r in rows) / len(rows)
            out[f"{name}_out_rel_mean"] = sum(r[f"{name}_out_rel"] for r in rows) / len(rows)
        by_layer.append(out)
    result = {
        "task": "teacher-forced future-query attention over X functional K/V ablation",
        "samples": processed,
        "prefix_tokens": 256,
        "target_tokens": 128,
        "query_tokens": 32,
        "summary": summary,
        "by_layer": by_layer,
        "notes": "Restricted to query attention over X only. Q and full attention normalization are teacher-forced from full A+X+Q run.",
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
