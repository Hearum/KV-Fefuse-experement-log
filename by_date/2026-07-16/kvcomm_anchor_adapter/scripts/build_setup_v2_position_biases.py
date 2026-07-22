#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    p = argparse.ArgumentParser(description="Build setup-v2 per-chunk static KV bias from offline position anchors.")
    p.add_argument("--model-path", default="/mnt/qjhs-sh-lab-01/models/Qwen3-32B")
    p.add_argument("--bge-model-path", default="/mnt/qjhs-sh-lab-01/models/bge-m3")
    p.add_argument("--cache-root", default="/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2")
    p.add_argument("--data-path", default=str(ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/data"))
    p.add_argument("--data-name", default="musique-v2.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--source", choices=["raw", "preprocess"], default="preprocess")
    p.add_argument("--start", type=int, default=1, help="1-based inclusive example id")
    p.add_argument("--end", type=int, default=5, help="1-based inclusive example id")
    p.add_argument("--num-anchors", type=int, default=6)
    p.add_argument("--ridge", type=float, default=1e-3)
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
    orders: list[list[int]] = []
    candidates = []
    if n_docs > 1:
        candidates.append(list(reversed(original)))
        for shift in range(1, min(n_docs, num_anchors + 1)):
            candidates.append(original[shift:] + original[:shift])
    rng = random.Random(seed)
    attempts = 0
    while len(candidates) < num_anchors * 4 and attempts < num_anchors * 50:
        order = original[:]
        rng.shuffle(order)
        candidates.append(order)
        attempts += 1
    seen = set()
    for order in candidates:
        key = tuple(order)
        if key == tuple(original) or key in seen:
            continue
        seen.add(key)
        orders.append(order)
        if len(orders) >= num_anchors:
            break
    if not orders:
        raise ValueError("Need at least two document chunks to build position anchors")
    return orders


def feature(prefix_tokens: int, rank: int, n_docs: int, total_doc_tokens: int) -> torch.Tensor:
    denom_tokens = max(1, int(total_doc_tokens))
    denom_rank = max(1, int(n_docs) - 1)
    return torch.tensor([1.0, prefix_tokens / denom_tokens, (rank - 1) / denom_rank], dtype=torch.float64)


def reset_cache(cache: StaticCache) -> None:
    for layer in range(len(cache.key_cache)):
        cache.past_tokens[layer] = 0


def full_doc_kv(model, cache: StaticCache, system: torch.Tensor, docs: list[torch.Tensor], order: list[int], device: str):
    reset_cache(cache)
    pieces = [system] + [docs[cid - 1] for cid in order]
    ids = torch.cat(pieces).unsqueeze(0).to(device)
    pos = torch.arange(ids.shape[1], device=device)
    with torch.no_grad():
        model(input_ids=ids, cache_position=pos, past_key_values=cache, use_cache=True, return_dict=False)
    start = int(system.shape[0])
    out = {}
    prefix_tokens = 0
    for rank, cid in enumerate(order, start=1):
        n = int(docs[cid - 1].shape[0])
        key = torch.stack([cache.key_cache[layer][:, :, start:start + n, :].detach().cpu() for layer in range(model.config.num_hidden_layers)])
        key = rotate_key_offset(model, key, -prefix_tokens, device)
        value = torch.stack([cache.value_cache[layer][:, :, start:start + n, :].detach().cpu() for layer in range(model.config.num_hidden_layers)])
        out[cid] = {"key": key, "value": value, "prefix_tokens": prefix_tokens, "rank": rank}
        start += n
        prefix_tokens += n
    return out


def load_source(source_path: Path, example_id: int, chunk_id: int, kind: str) -> torch.Tensor:
    return torch.load(source_path / f"{example_id}_{chunk_id}_{kind}.pt", map_location="cpu", weights_only=True)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "key").mkdir(exist_ok=True)
    (out / "value").mkdir(exist_ok=True)

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
    cache = StaticCache(
        config=model.config, max_batch_size=1, max_cache_len=args.max_cache_len,
        device=args.device, dtype=model.dtype, passage_len=args.max_cache_len,
    )

    save_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    metadata = {
        "args": vars(args),
        "source_path": str(source_path),
        "examples": [],
        "fit": "per example/chunk elementwise ridge over [1, prefix_token_ratio, doc_rank_ratio]",
    }

    for example_id in range(args.start, args.end + 1):
        passages = tokens_data[example_id - 1]
        system = passages[0]
        docs = list(passages[1:-1])
        n_docs = len(docs)
        total_doc_tokens = sum(int(d.shape[0]) for d in docs)
        orders = make_anchor_orders(n_docs, args.num_anchors, seed=example_id * 1009 + args.num_anchors)
        xtx = {cid: torch.zeros((3, 3), dtype=torch.float64) for cid in range(1, n_docs + 1)}
        xty = {cid: {} for cid in range(1, n_docs + 1)}

        for anchor_idx, order in enumerate(orders):
            full = full_doc_kv(model, cache, system, docs, order, args.device)
            for cid in order:
                feat = feature(full[cid]["prefix_tokens"], full[cid]["rank"], n_docs, total_doc_tokens)
                xtx[cid] += torch.outer(feat, feat)
                for kind in ("key", "value"):
                    src = load_source(source_path, example_id, cid, kind).float()
                    delta = full[cid][kind].float() - src
                    if kind not in xty[cid]:
                        xty[cid][kind] = torch.zeros((3,) + tuple(delta.shape), dtype=torch.float32)
                    for j in range(3):
                        xty[cid][kind][j].add_(delta, alpha=float(feat[j]))
                    del src, delta
            del full
            torch.cuda.empty_cache()
            print(f"example={example_id} anchor={anchor_idx + 1}/{len(orders)} order={order}", flush=True)

        prefix = 0
        example_meta = {"example_id": example_id, "orders": orders, "chunks": []}
        for rank, cid in enumerate(range(1, n_docs + 1), start=1):
            cur_feat = feature(prefix, rank, n_docs, total_doc_tokens)
            inv = torch.linalg.pinv(xtx[cid] + torch.eye(3, dtype=torch.float64) * float(args.ridge))
            gamma = torch.matmul(cur_feat, inv).to(torch.float32)
            for kind in ("key", "value"):
                pred = torch.zeros_like(xty[cid][kind][0])
                for j in range(3):
                    pred.add_(xty[cid][kind][j], alpha=float(gamma[j]))
                subdir = "key" if kind == "key" else "value"
                torch.save(pred.to(save_dtype), out / subdir / f"{example_id}_{cid}_{kind}_bias.pt")
                del pred
            example_meta["chunks"].append({"chunk_id": cid, "tokens": int(docs[cid - 1].shape[0]), "prefix_tokens": prefix, "rank": rank})
            prefix += int(docs[cid - 1].shape[0])
        metadata["examples"].append(example_meta)
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"wrote example={example_id} chunks={n_docs} source={args.source} out={out}", flush=True)


if __name__ == "__main__":
    main()
