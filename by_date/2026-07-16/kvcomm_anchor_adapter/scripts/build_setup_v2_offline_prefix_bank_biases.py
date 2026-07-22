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
    p = argparse.ArgumentParser(description="Build deployable KVCOMM-like bias using offline prefix banks from other examples.")
    p.add_argument("--model-path", default="/mnt/qjhs-sh-lab-01/models/Qwen3-32B")
    p.add_argument("--bge-model-path", default="/mnt/qjhs-sh-lab-01/models/bge-m3")
    p.add_argument("--cache-root", default="/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2")
    p.add_argument("--data-path", default=str(ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/data"))
    p.add_argument("--data-name", default="musique-v2.jsonl")
    p.add_argument("--source", choices=["raw", "preprocess"], default="preprocess")
    p.add_argument("--example", type=int, required=True, help="1-based target example id")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-anchors", type=int, default=4)
    p.add_argument("--topk-anchors", type=int, default=2)
    p.add_argument(
        "--prefix-source",
        choices=["other_examples", "bge_neighbors", "mixed", "retrieval_traces"],
        default="other_examples",
    )
    p.add_argument("--bge-anchors", type=int, default=3, help="BGE anchor groups used by --prefix-source mixed")
    p.add_argument("--random-anchors", type=int, default=9, help="Random anchor groups used by --prefix-source mixed")
    p.add_argument("--random-seed", type=int, default=20260716)
    p.add_argument("--trace-anchors", type=int, default=12)
    p.add_argument("--chunk-ids", default="", help="Optional comma-separated 1-based target chunks for offline sharding")
    p.add_argument("--metadata-name", default="metadata.json")
    p.add_argument("--prefix-examples", default="1,2,3,4,5,6,7,8,9,10", help="comma-separated 1-based examples used only as offline prefix source")
    p.add_argument("--prefix-docs", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--matcher", choices=["prefix_len", "prefix_value", "hybrid"], default="hybrid")
    p.add_argument(
        "--selection-mode",
        choices=[
            "matcher", "oracle_value_l2", "oracle_kv_l2",
            "oracle_kv_lstsq", "oracle_kv_lstsq_layerwise",
        ],
        default="matcher",
        help="Oracle modes are analysis-only and use the true current-context Delta.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--oracle-ridge", type=float, default=1e-4)
    p.add_argument("--max-cache-len", type=int, default=32768)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--retrieval-topk", type=int, default=None, help="Optional wider BGE ranking depth without changing the KV cache path.")
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


def reset_cache(cache: StaticCache) -> None:
    for layer in range(len(cache.key_cache)):
        cache.past_tokens[layer] = 0


def source_paths(args, tokenizer):
    _, tokens_data, _, _, _, raw_path, preprocess_path, _, _, _, context_rank, corpus_lens = prepare_data(
        "Qwen3-32B", args.data_path.rstrip("/") + "/", args.data_name,
        args.cache_root.rstrip("/") + "/", tokenizer, args.topk, args.revert_rope,
        args.source == "preprocess", args.bge_model_path,
    )
    if args.retrieval_topk is not None and args.retrieval_topk != args.topk:
        _, _, _, _, _, _, _, _, _, _, context_rank, corpus_lens = prepare_data(
            "Qwen3-32B", args.data_path.rstrip("/") + "/", args.data_name,
            args.cache_root.rstrip("/") + "/", tokenizer, args.retrieval_topk, args.revert_rope,
            args.source == "preprocess", args.bge_model_path,
        )
    return tokens_data, Path(preprocess_path if args.source == "preprocess" else raw_path), context_rank, corpus_lens


def prefix_summary_from_full(model, cache, system, prefix_docs, target_doc, device):
    reset_cache(cache)
    pieces = [system] + prefix_docs + [target_doc]
    ids = torch.cat(pieces).unsqueeze(0).to(device)
    pos = torch.arange(ids.shape[1], device=device)
    with torch.no_grad():
        model(input_ids=ids, cache_position=pos, past_key_values=cache, use_cache=True, return_dict=False)
    prefix_tokens = sum(int(x.shape[0]) for x in prefix_docs)
    target_start = int(system.shape[0]) + prefix_tokens
    n = int(target_doc.shape[0])
    key = torch.stack([cache.key_cache[layer][:, :, target_start:target_start + n, :].detach().cpu() for layer in range(model.config.num_hidden_layers)])
    key = rotate_key_offset(model, key, -prefix_tokens, device)
    value = torch.stack([cache.value_cache[layer][:, :, target_start:target_start + n, :].detach().cpu() for layer in range(model.config.num_hidden_layers)])
    if prefix_docs:
        start = int(system.shape[0])
        end = target_start
        pv = torch.stack([cache.value_cache[layer][:, :, start:end, :].detach().cpu() for layer in range(model.config.num_hidden_layers)])
        prefix_summary = pv.float().squeeze(1).mean(dim=2).flatten()
    else:
        prefix_summary = torch.zeros((model.config.num_hidden_layers * model.config.num_key_value_heads * (model.config.hidden_size // model.config.num_attention_heads)), dtype=torch.float32)
    return {"key": key, "value": value, "prefix_tokens": prefix_tokens, "prefix_summary": prefix_summary}


def current_prefix_record(model, cache, system, docs, chunk_id, device):
    prefix_docs = docs[:chunk_id - 1]
    return prefix_summary_from_full(model, cache, system, prefix_docs, docs[chunk_id - 1], device)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    denom = float(af.norm() * bf.norm())
    if denom == 0.0:
        return 0.0
    return float(torch.dot(af, bf) / denom)


def relative_l2(candidate: torch.Tensor, target: torch.Tensor) -> float:
    target_norm = float(torch.linalg.vector_norm(target.float()))
    error_norm = float(torch.linalg.vector_norm(candidate.float() - target.float()))
    return error_norm / max(target_norm, 1e-12)


def layer_reconstruction_stats(candidate: torch.Tensor, target: torch.Tensor) -> list[dict]:
    candidate_layers = candidate.float().flatten(1)
    target_layers = target.float().flatten(1)
    target_norms = torch.linalg.vector_norm(target_layers, dim=1)
    error_norms = torch.linalg.vector_norm(candidate_layers - target_layers, dim=1)
    target_energy = target_norms.square()
    error_energy = error_norms.square()
    total_target_energy = float(target_energy.sum().clamp_min(1e-24))
    total_error_energy = float(error_energy.sum().clamp_min(1e-24))
    return [{
        "layer": layer,
        "target_l2": float(target_norms[layer]),
        "error_l2": float(error_norms[layer]),
        "relative_l2": float(error_norms[layer] / target_norms[layer].clamp_min(1e-12)),
        "target_energy_fraction": float(target_energy[layer] / total_target_energy),
        "error_energy_fraction": float(error_energy[layer] / total_error_energy),
    } for layer in range(target.shape[0])]


def oracle_kv_lstsq_coefficients(
    anchor_records: list[dict],
    oracle_key_delta: torch.Tensor,
    oracle_value_delta: torch.Tensor,
    ridge: float,
    device: str,
) -> torch.Tensor:
    n = len(anchor_records)
    gram = torch.zeros((n, n), dtype=torch.float32, device=device)
    rhs = torch.zeros(n, dtype=torch.float32, device=device)
    for field, target in (("key_delta", oracle_key_delta), ("value_delta", oracle_value_delta)):
        basis = torch.stack([rec[field].reshape(-1) for rec in anchor_records]).to(device)
        target_flat = target.reshape(-1).to(device)
        target_energy = torch.dot(target_flat, target_flat).clamp_min(1e-12)
        gram.add_(basis @ basis.T / target_energy)
        rhs.add_(basis @ target_flat / target_energy)
        del basis, target_flat
    gram.mul_(0.5)
    rhs.mul_(0.5)
    ridge_scale = max(float(torch.trace(gram)) / max(1, n), 1e-12) * ridge
    gram.add_(torch.eye(n, dtype=gram.dtype, device=device), alpha=ridge_scale)
    coefficients = torch.linalg.solve(gram, rhs)
    return coefficients.detach().cpu()


def oracle_kv_lstsq_layerwise_coefficients(
    anchor_records: list[dict],
    oracle_key_delta: torch.Tensor,
    oracle_value_delta: torch.Tensor,
    ridge: float,
    device: str,
) -> torch.Tensor:
    n = len(anchor_records)
    layers = oracle_key_delta.shape[0]
    gram = torch.zeros((layers, n, n), dtype=torch.float32, device=device)
    rhs = torch.zeros((layers, n), dtype=torch.float32, device=device)
    for field, target in (("key_delta", oracle_key_delta), ("value_delta", oracle_value_delta)):
        basis = torch.stack([rec[field] for rec in anchor_records]).to(device).flatten(2)
        target_flat = target.to(device).flatten(1)
        target_energy = torch.einsum("ld,ld->l", target_flat, target_flat).clamp_min(1e-12)
        gram.add_(torch.einsum("nld,mld->lnm", basis, basis) / target_energy[:, None, None])
        rhs.add_(torch.einsum("nld,ld->ln", basis, target_flat) / target_energy[:, None])
        del basis, target_flat
    gram.mul_(0.5)
    rhs.mul_(0.5)
    ridge_scale = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1) / max(1, n)
    ridge_scale.clamp_min_(1e-12).mul_(ridge)
    gram.add_(torch.diag_embed(ridge_scale[:, None].expand(-1, n)))
    coefficients = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
    return coefficients.detach().cpu()


def score(args, cur, anchor, max_prefix_tokens: int) -> float:
    scores = []
    if args.matcher in {"prefix_len", "hybrid"}:
        diff = abs(cur["prefix_tokens"] - anchor["prefix_tokens"]) / max(1, max_prefix_tokens)
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


def make_prefix_sets(tokens_data, target_example: int, prefix_examples: list[int], num_anchors: int, prefix_docs: int):
    rng = random.Random(target_example * 10007 + num_anchors)
    candidates = [ex for ex in prefix_examples if ex != target_example and 1 <= ex <= len(tokens_data)]
    if not candidates:
        raise ValueError("No prefix examples available after excluding target example")
    anchors = []
    for idx in range(num_anchors):
        ex = candidates[idx % len(candidates)]
        docs = list(tokens_data[ex - 1][1:-1])
        if not docs:
            continue
        if len(docs) <= prefix_docs:
            chosen = docs
        else:
            start = (idx * prefix_docs) % max(1, len(docs) - prefix_docs + 1)
            chosen = docs[start:start + prefix_docs]
        if idx >= len(candidates):
            chosen = chosen[:]
            rng.shuffle(chosen)
        anchors.append({"source_example": ex, "docs": chosen})
    return anchors


def doc_from_global(tokens_data, corpus_lens, global_idx: int):
    remaining = int(global_idx)
    for ex_idx, n_docs in enumerate(corpus_lens, start=1):
        n_docs = int(n_docs)
        if remaining < n_docs:
            return ex_idx, remaining + 1, tokens_data[ex_idx - 1][1 + remaining]
        remaining -= n_docs
    raise IndexError(f"global doc index out of range: {global_idx}")


def global_doc_index(corpus_lens, example: int, chunk_id: int) -> int:
    return sum(int(x) for x in corpus_lens[: example - 1]) + chunk_id - 1


def make_bge_neighbor_prefix_sets(tokens_data, context_rank, corpus_lens, target_example: int, chunk_id: int, num_anchors: int, prefix_docs: int):
    if context_rank is None or len(context_rank) == 0:
        raise ValueError("BGE neighbor prefix source requires preprocess=True context_rank")
    target_global = global_doc_index(corpus_lens, target_example, chunk_id)
    ranked = [int(x) for x in context_rank[target_global].tolist() if int(x) != target_global]
    if not ranked:
        raise ValueError(f"No BGE neighbors found for example={target_example} chunk={chunk_id}")
    anchors = []
    stride = max(1, prefix_docs)
    for anchor_idx in range(num_anchors):
        docs = []
        sources = []
        for gid in ranked[anchor_idx * stride: anchor_idx * stride + prefix_docs]:
            ex, cid, doc = doc_from_global(tokens_data, corpus_lens, gid)
            docs.append(doc)
            sources.append({"global_doc": gid, "example": ex, "chunk": cid})
        if docs:
            anchors.append({"source_example": -1, "source_type": "bge", "docs": docs, "sources": sources})
    if not anchors:
        raise ValueError(f"No BGE neighbor anchors built for example={target_example} chunk={chunk_id}")
    return anchors


def make_random_global_prefix_sets(
    tokens_data, corpus_lens, target_example: int, chunk_id: int,
    num_anchors: int, prefix_docs: int, seed: int,
):
    target_global = global_doc_index(corpus_lens, target_example, chunk_id)
    candidates = []
    for gid in range(sum(int(x) for x in corpus_lens)):
        ex, cid, _ = doc_from_global(tokens_data, corpus_lens, gid)
        if gid != target_global and ex != target_example:
            candidates.append((gid, ex, cid))
    if len(candidates) < prefix_docs:
        raise ValueError("Not enough cross-example documents for random anchors")
    rng = random.Random(seed + target_example * 10007 + chunk_id * 97)
    rng.shuffle(candidates)
    anchors = []
    cursor = 0
    for _ in range(num_anchors):
        if cursor + prefix_docs > len(candidates):
            rng.shuffle(candidates)
            cursor = 0
        selected = candidates[cursor:cursor + prefix_docs]
        cursor += prefix_docs
        docs = []
        sources = []
        for gid, ex, cid in selected:
            _, _, doc = doc_from_global(tokens_data, corpus_lens, gid)
            docs.append(doc)
            sources.append({"global_doc": gid, "example": ex, "chunk": cid})
        anchors.append({"source_example": -1, "source_type": "random", "docs": docs, "sources": sources})
    return anchors


def make_retrieval_trace_prefix_sets(
    tokens_data, context_rank, corpus_lens, target_example: int, chunk_id: int,
    num_anchors: int, prefix_docs: int,
):
    if context_rank is None or len(context_rank) == 0:
        raise ValueError("Retrieval-trace prefix source requires preprocess=True context_rank")
    target_global = global_doc_index(corpus_lens, target_example, chunk_id)
    ranked = [int(x) for x in context_rank[target_global].tolist() if int(x) != target_global]
    anchors = []
    seen = set()
    for gid in ranked:
        ex, cid, _ = doc_from_global(tokens_data, corpus_lens, gid)
        if ex == target_example or cid <= 1:
            continue
        all_prefix_docs = list(tokens_data[ex - 1][1:cid])
        if not all_prefix_docs:
            continue
        first_cid = max(1, cid - prefix_docs) if prefix_docs > 0 else 1
        selected_docs = all_prefix_docs[-prefix_docs:] if prefix_docs > 0 else all_prefix_docs
        source_ids = tuple((ex, source_cid) for source_cid in range(first_cid, cid))
        if source_ids in seen:
            continue
        seen.add(source_ids)
        sources = [{
            "global_doc": global_doc_index(corpus_lens, ex, source_cid),
            "example": ex,
            "chunk": source_cid,
        } for source_cid in range(first_cid, cid)]
        anchors.append({
            "source_example": ex,
            "source_type": "retrieval_trace",
            "docs": selected_docs,
            "sources": sources,
            "retrieval_neighbor": {"global_doc": gid, "example": ex, "chunk": cid},
        })
        if len(anchors) >= num_anchors:
            break
    if not anchors:
        raise ValueError(f"No retrieval-trace anchors built for example={target_example} chunk={chunk_id}")
    return anchors


def cached_prefix_summary(src_path: Path, sources: list[dict], config) -> torch.Tensor:
    values = []
    for source in sources:
        values.append(torch.load(
            src_path / f"{source['example']}_{source['chunk']}_value.pt",
            map_location="cpu", weights_only=True,
        ).float())
    if not values:
        width = config.num_hidden_layers * config.num_key_value_heads * (
            config.hidden_size // config.num_attention_heads
        )
        return torch.zeros(width, dtype=torch.float32)
    value = torch.cat(values, dim=-2)
    return value.squeeze(1).mean(dim=2).flatten()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    (out / "key").mkdir(parents=True, exist_ok=True)
    (out / "value").mkdir(parents=True, exist_ok=True)
    prefix_examples = [int(x) for x in args.prefix_examples.replace(";", ",").split(",") if x.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokens_data, src_path, context_rank, corpus_lens = source_paths(args, tokenizer)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    model = load_model("qwen3", args.model_path, config).to(args.device)
    model.eval()
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=args.max_cache_len, device=args.device, dtype=model.dtype, passage_len=args.max_cache_len)

    passages = tokens_data[args.example - 1]
    system = passages[0]
    docs = list(passages[1:-1])
    max_prefix_tokens = max(1, sum(int(d.shape[0]) for d in docs))
    save_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    meta = {"args": vars(args), "source_path": str(src_path), "prefixes": [], "chunks": []}
    selected_chunk_ids = {
        int(x) for x in args.chunk_ids.replace(";", ",").split(",") if x.strip()
    }

    for chunk_id, target_doc in enumerate(docs, start=1):
        if selected_chunk_ids and chunk_id not in selected_chunk_ids:
            continue
        if args.prefix_source == "retrieval_traces":
            offline_prefixes = make_retrieval_trace_prefix_sets(
                tokens_data, context_rank, corpus_lens, args.example, chunk_id,
                args.trace_anchors, args.prefix_docs,
            )
        elif args.prefix_source == "mixed":
            offline_prefixes = make_bge_neighbor_prefix_sets(
                tokens_data, context_rank, corpus_lens, args.example, chunk_id,
                args.bge_anchors, args.prefix_docs,
            )
            offline_prefixes.extend(make_random_global_prefix_sets(
                tokens_data, corpus_lens, args.example, chunk_id,
                args.random_anchors, args.prefix_docs, args.random_seed,
            ))
        elif args.prefix_source == "bge_neighbors":
            offline_prefixes = make_bge_neighbor_prefix_sets(
                tokens_data, context_rank, corpus_lens, args.example, chunk_id, args.num_anchors, args.prefix_docs
            )
        else:
            offline_prefixes = make_prefix_sets(tokens_data, args.example, prefix_examples, args.num_anchors, args.prefix_docs)
        current_sources = [
            {"example": args.example, "chunk": prefix_chunk_id}
            for prefix_chunk_id in range(1, chunk_id)
        ]
        cur = {
            "prefix_tokens": sum(int(d.shape[0]) for d in docs[:chunk_id - 1]),
            "prefix_summary": cached_prefix_summary(src_path, current_sources, config),
        }
        src_k = torch.load(src_path / f"{args.example}_{chunk_id}_key.pt", map_location="cpu", weights_only=True)
        src_v = torch.load(src_path / f"{args.example}_{chunk_id}_value.pt", map_location="cpu", weights_only=True)
        oracle_key_delta = None
        oracle_value_delta = None
        if args.selection_mode != "matcher":
            oracle = current_prefix_record(model, cache, system, docs, chunk_id, args.device)
            oracle_key_delta = oracle["key"].float() - src_k.float()
            oracle_value_delta = oracle["value"].float() - src_v.float()
        anchor_records = []
        for anchor_idx, pref in enumerate(offline_prefixes):
            rec = prefix_summary_from_full(model, cache, system, pref["docs"], target_doc, args.device)
            matcher_summary = (
                cached_prefix_summary(src_path, pref["sources"], config)
                if pref.get("sources") else rec["prefix_summary"]
            )
            anchor_records.append({
                "source_example": pref["source_example"],
                "source_type": pref.get("source_type", "other_examples"),
                "sources": pref.get("sources", []),
                "prefix_tokens": rec["prefix_tokens"],
                "prefix_summary": matcher_summary,
                "key_delta": rec["key"].float() - src_k.float(),
                "value_delta": rec["value"].float() - src_v.float(),
            })
        for rec in anchor_records:
            rec["matcher_score"] = score(args, cur, rec, max_prefix_tokens)
            if oracle_value_delta is not None:
                rec["oracle_value_rel_l2"] = relative_l2(rec["value_delta"], oracle_value_delta)
                rec["oracle_key_rel_l2"] = relative_l2(rec["key_delta"], oracle_key_delta)
        if args.selection_mode == "matcher":
            scored = sorted([(rec["matcher_score"], idx) for idx, rec in enumerate(anchor_records)], reverse=True)
        elif args.selection_mode == "oracle_value_l2":
            scored = sorted([(-rec["oracle_value_rel_l2"], idx) for idx, rec in enumerate(anchor_records)], reverse=True)
        elif args.selection_mode == "oracle_kv_l2":
            scored = sorted([(
                -0.5 * (rec["oracle_value_rel_l2"] + rec["oracle_key_rel_l2"]), idx
            ) for idx, rec in enumerate(anchor_records)], reverse=True)
        elif args.selection_mode == "oracle_kv_lstsq":
            coefficients = oracle_kv_lstsq_coefficients(
                anchor_records, oracle_key_delta, oracle_value_delta,
                args.oracle_ridge, args.device,
            )
            scored = [(
                -0.5 * (rec["oracle_value_rel_l2"] + rec["oracle_key_rel_l2"]), idx
            ) for idx, rec in enumerate(anchor_records)]
        else:
            coefficients = oracle_kv_lstsq_layerwise_coefficients(
                anchor_records, oracle_key_delta, oracle_value_delta,
                args.oracle_ridge, args.device,
            )
            scored = [(
                -0.5 * (rec["oracle_value_rel_l2"] + rec["oracle_key_rel_l2"]), idx
            ) for idx, rec in enumerate(anchor_records)]
        if args.selection_mode in {"oracle_kv_lstsq", "oracle_kv_lstsq_layerwise"}:
            selected = scored
            weights = coefficients if coefficients.ndim == 2 else coefficients.tolist()
        else:
            selected = scored[: max(1, args.topk_anchors)]
            weights = softmax_weights([s for s, _ in selected], args.temperature)
        key_bias = None
        value_bias = None
        selected_meta = []
        for selected_pos, (sc, idx) in enumerate(selected):
            rec = anchor_records[idx]
            if isinstance(weights, torch.Tensor):
                weight = weights[:, selected_pos].reshape(-1, 1, 1, 1, 1)
                weighted_key = rec["key_delta"].mul(weight)
                weighted_value = rec["value_delta"].mul(weight)
                weight_meta = {
                    "weight": None,
                    "weight_l2": float(torch.linalg.vector_norm(weight)),
                    "weight_max_abs": float(weight.abs().max()),
                }
            else:
                weight = weights[selected_pos]
                weighted_key = rec["key_delta"].mul(weight)
                weighted_value = rec["value_delta"].mul(weight)
                weight_meta = {"weight": weight}
            key_bias = weighted_key if key_bias is None else key_bias.add_(weighted_key)
            value_bias = weighted_value if value_bias is None else value_bias.add_(weighted_value)
            selected_meta.append({
                "source_example": rec["source_example"],
                "source_type": rec["source_type"],
                "score": sc,
                **weight_meta,
                "prefix_tokens": rec["prefix_tokens"],
                "sources": rec["sources"],
                "matcher_score": rec["matcher_score"],
                "oracle_value_rel_l2": rec.get("oracle_value_rel_l2"),
                "oracle_key_rel_l2": rec.get("oracle_key_rel_l2"),
            })
        reconstruction_meta = None
        if oracle_value_delta is not None:
            reconstruction_meta = {
                "value_rel_l2": relative_l2(value_bias, oracle_value_delta),
                "key_rel_l2": relative_l2(key_bias, oracle_key_delta),
                "value_by_layer": layer_reconstruction_stats(value_bias, oracle_value_delta),
                "key_by_layer": layer_reconstruction_stats(key_bias, oracle_key_delta),
                "coefficient_l2": float(torch.linalg.vector_norm(
                    weights if isinstance(weights, torch.Tensor) else torch.tensor(weights)
                )),
                "coefficient_max_abs": float(
                    weights.abs().max() if isinstance(weights, torch.Tensor)
                    else max(abs(float(x)) for x in weights)
                ),
            }
        torch.save(key_bias.to(save_dtype), out / "key" / f"{args.example}_{chunk_id}_key_bias.pt")
        torch.save(value_bias.to(save_dtype), out / "value" / f"{args.example}_{chunk_id}_value_bias.pt")
        candidates_meta = [{
            "anchor_id": idx,
            "source_type": rec["source_type"],
            "prefix_tokens": rec["prefix_tokens"],
            "matcher_score": rec["matcher_score"],
            "oracle_value_rel_l2": rec.get("oracle_value_rel_l2"),
            "oracle_key_rel_l2": rec.get("oracle_key_rel_l2"),
        } for idx, rec in enumerate(anchor_records)]
        meta["chunks"].append({
            "chunk_id": chunk_id,
            "current_prefix_tokens": cur["prefix_tokens"],
            "selected": selected_meta,
            "candidates": candidates_meta,
            "reconstruction": reconstruction_meta,
        })
        print(f"chunk={chunk_id}/{len(docs)} selected={selected_meta}", flush=True)
        del src_k, src_v, anchor_records, key_bias, value_bias
        if oracle_key_delta is not None:
            del oracle_key_delta, oracle_value_delta, oracle
        if args.selection_mode in {"oracle_kv_lstsq", "oracle_kv_lstsq_layerwise"}:
            del coefficients
        torch.cuda.empty_cache()
    if args.prefix_source in {"bge_neighbors", "mixed", "retrieval_traces"}:
        meta["prefixes"] = f"per_chunk_{args.prefix_source}"
    else:
        meta["prefixes"] = [{"source_example": p["source_example"], "num_docs": len(p["docs"]), "tokens": sum(int(d.shape[0]) for d in p["docs"])} for p in offline_prefixes]
    (out / args.metadata_name).write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote offline-prefix-bank example={args.example} chunks={len(meta['chunks'])} out={out}")


if __name__ == "__main__":
    main()
