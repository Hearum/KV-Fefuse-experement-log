#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
if not ROOT.exists():
    ROOT = Path("/home/hming/FusionRAG-pca-analysis")
sys.path.insert(0, str(ROOT))

from transformers import AutoConfig, AutoTokenizer

from ktransformers.models.custom_cache import StaticCache
from ktransformers.unified_process_cache import load_model
from ktransformers.util.utils import prepare_data, rotate_half


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build KVCOMM-like top-k anchor bias for setup-v2 examples.")
    p.add_argument("--model-path", default="/mnt/qjhs-sh-lab-01/models/Qwen3-32B")
    p.add_argument("--bge-model-path", default="/mnt/qjhs-sh-lab-01/models/bge-m3")
    p.add_argument("--cache-root", default="/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2")
    p.add_argument("--data-path", default=str(ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/data"))
    p.add_argument("--data-name", default="musique-v2.jsonl")
    p.add_argument("--source", choices=["raw", "preprocess"], default="preprocess")
    p.add_argument("--example", type=int, required=True, help="1-based example id")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-anchors", type=int, default=6)
    p.add_argument("--topk-anchors", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--matcher", choices=["prefix_len", "prefix_value", "hybrid"], default="hybrid")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-cache-len", type=int, default=32768)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--revert-rope", action="store_true", default=True)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return p.parse_args()


def rotate_key_offset(model, x: torch.Tensor, offset: int, device: str) -> torch.Tensor:
    if offset == 0:
        return x.detach().cpu()
    z = x.to(device)
    position_ids = torch.full((1, z.shape[-2]), int(offset), device=device)
    try:
        cos, sin = model.model.layers[0].self_attn.rotary_emb(z[0], position_ids)
    except Exception:
        cos, sin = model.model.rotary_emb(z[0], position_ids)
    cos = cos.unsqueeze(0).unsqueeze(2).to(z.device)
    sin = sin.unsqueeze(0).unsqueeze(2).to(z.device)
    return ((z * cos) + (rotate_half(z) * sin)).detach().cpu()


def make_anchor_orders(n_docs: int, num_anchors: int, seed: int) -> list[list[int]]:
    original = list(range(1, n_docs + 1))
    candidates = []
    if n_docs > 1:
        candidates.append(list(reversed(original)))
        for shift in range(1, min(n_docs, num_anchors + 2)):
            candidates.append(original[shift:] + original[:shift])
    rng = random.Random(seed)
    while len(candidates) < num_anchors * 5:
        order = original[:]
        rng.shuffle(order)
        candidates.append(order)
    seen = set()
    out = []
    for order in candidates:
        key = tuple(order)
        if key == tuple(original) or key in seen:
            continue
        seen.add(key)
        out.append(order)
        if len(out) >= num_anchors:
            break
    return out


def reset_cache(cache: StaticCache) -> None:
    for layer in range(len(cache.key_cache)):
        cache.past_tokens[layer] = 0


def full_doc_kv(model, cache: StaticCache, system: torch.Tensor, docs: list[torch.Tensor], order: list[int], device: str):
    reset_cache(cache)
    ids = torch.cat([system] + [docs[cid - 1] for cid in order]).unsqueeze(0).to(device)
    pos = torch.arange(ids.shape[1], device=device)
    with torch.no_grad():
        model(input_ids=ids, cache_position=pos, past_key_values=cache, use_cache=True, return_dict=False)
    start = int(system.shape[0])
    prefix_tokens = 0
    out = {}
    prefix_value_sum = None
    prefix_value_count = 0
    for rank, cid in enumerate(order, start=1):
        n = int(docs[cid - 1].shape[0])
        key = torch.stack([cache.key_cache[layer][:, :, start:start + n, :].detach().cpu() for layer in range(model.config.num_hidden_layers)])
        key = rotate_key_offset(model, key, -prefix_tokens, device)
        value = torch.stack([cache.value_cache[layer][:, :, start:start + n, :].detach().cpu() for layer in range(model.config.num_hidden_layers)])
        if prefix_value_sum is None:
            prefix_summary = torch.zeros((model.config.num_hidden_layers, model.config.num_key_value_heads, model.config.hidden_size // model.config.num_attention_heads), dtype=torch.float32)
        else:
            prefix_summary = prefix_value_sum / max(1, prefix_value_count)
        out[cid] = {
            "key": key,
            "value": value,
            "prefix_tokens": prefix_tokens,
            "rank": rank,
            "prefix_summary": prefix_summary.flatten(),
        }
        val_for_prefix = value.float().squeeze(1).mean(dim=2)
        prefix_value_sum = val_for_prefix if prefix_value_sum is None else prefix_value_sum + val_for_prefix
        prefix_value_count += 1
        start += n
        prefix_tokens += n
    return out


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    denom = float(af.norm() * bf.norm())
    if denom == 0.0:
        return 0.0
    return float(torch.dot(af, bf) / denom)


def matcher_score(args, cur, anchor, total_tokens: int) -> float:
    scores = []
    if args.matcher in {"prefix_len", "hybrid"}:
        diff = abs(cur["prefix_tokens"] - anchor["prefix_tokens"]) / max(1, total_tokens)
        scores.append(1.0 - diff)
    if args.matcher in {"prefix_value", "hybrid"}:
        scores.append(cosine(cur["prefix_summary"], anchor["prefix_summary"]))
    return sum(scores) / len(scores)


def softmax_weights(scores: list[float], temperature: float) -> list[float]:
    if not scores:
        return []
    scale = max(temperature, 1e-6)
    m = max(scores)
    vals = [math.exp((s - m) / scale) for s in scores]
    z = sum(vals)
    return [v / z for v in vals]


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    (out / "key").mkdir(parents=True, exist_ok=True)
    (out / "value").mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    _, tokens_data, _, _, _, raw_path, preprocess_path, _, _, _, _, _ = prepare_data(
        "Qwen3-32B", args.data_path.rstrip("/") + "/", args.data_name,
        args.cache_root.rstrip("/") + "/", tokenizer, args.topk, args.revert_rope,
        args.source == "preprocess", args.bge_model_path,
    )
    source_path = Path(preprocess_path if args.source == "preprocess" else raw_path)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    model = load_model("qwen3", args.model_path, config).to(args.device)
    model.eval()
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=args.max_cache_len, device=args.device, dtype=model.dtype, passage_len=args.max_cache_len)

    passages = tokens_data[args.example - 1]
    system = passages[0]
    docs = list(passages[1:-1])
    n_docs = len(docs)
    total_tokens = sum(int(d.shape[0]) for d in docs)
    original_order = list(range(1, n_docs + 1))
    anchor_orders = make_anchor_orders(n_docs, args.num_anchors, args.example * 9173 + args.num_anchors)
    current = full_doc_kv(model, cache, system, docs, original_order, args.device)
    anchors = []
    for anchor_idx, order in enumerate(anchor_orders):
        full = full_doc_kv(model, cache, system, docs, order, args.device)
        anchor = {"order": order, "chunks": {}}
        for cid, rec in full.items():
            src_k = torch.load(source_path / f"{args.example}_{cid}_key.pt", map_location="cpu", weights_only=True)
            src_v = torch.load(source_path / f"{args.example}_{cid}_value.pt", map_location="cpu", weights_only=True)
            anchor["chunks"][cid] = {
                "key_delta": rec["key"].float() - src_k.float(),
                "value_delta": rec["value"].float() - src_v.float(),
                "prefix_tokens": rec["prefix_tokens"],
                "prefix_summary": rec["prefix_summary"],
            }
        anchors.append(anchor)
        print(f"anchor={anchor_idx + 1}/{len(anchor_orders)} order={order}", flush=True)
        del full
        torch.cuda.empty_cache()

    save_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    meta = {"args": vars(args), "source_path": str(source_path), "anchor_orders": anchor_orders, "chunks": []}
    for cid in original_order:
        scored = []
        for idx, anchor in enumerate(anchors):
            score = matcher_score(args, current[cid], anchor["chunks"][cid], total_tokens)
            scored.append((score, idx))
        scored.sort(reverse=True)
        selected = scored[: max(1, args.topk_anchors)]
        weights = softmax_weights([s for s, _ in selected], args.temperature)
        key_bias = None
        value_bias = None
        selected_meta = []
        for weight, (score, idx) in zip(weights, selected):
            rec = anchors[idx]["chunks"][cid]
            kd = rec["key_delta"]
            vd = rec["value_delta"]
            key_bias = kd.mul(weight) if key_bias is None else key_bias.add(kd, alpha=weight)
            value_bias = vd.mul(weight) if value_bias is None else value_bias.add(vd, alpha=weight)
            selected_meta.append({"anchor_idx": idx, "score": score, "weight": weight, "prefix_tokens": rec["prefix_tokens"]})
        torch.save(key_bias.to(save_dtype), out / "key" / f"{args.example}_{cid}_key_bias.pt")
        torch.save(value_bias.to(save_dtype), out / "value" / f"{args.example}_{cid}_value_bias.pt")
        meta["chunks"].append({"chunk_id": cid, "current_prefix_tokens": current[cid]["prefix_tokens"], "selected": selected_meta})
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote kvcomm-like example={args.example} chunks={n_docs} out={out}")


if __name__ == "__main__":
    main()
