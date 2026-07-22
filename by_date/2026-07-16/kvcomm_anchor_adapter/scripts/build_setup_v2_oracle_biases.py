#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    p = argparse.ArgumentParser(description="Build exact full-minus-cache oracle KV biases for setup-v2 examples.")
    p.add_argument("--model-path", default="/mnt/qjhs-sh-lab-01/models/Qwen3-32B")
    p.add_argument("--bge-model-path", default="/mnt/qjhs-sh-lab-01/models/bge-m3")
    p.add_argument("--cache-root", default="/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2")
    p.add_argument("--data-path", default=str(ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/data"))
    p.add_argument("--data-name", default="musique-v2.jsonl")
    p.add_argument("--source", choices=["raw", "preprocess"], default="preprocess")
    p.add_argument("--example", type=int, required=True, help="1-based example id")
    p.add_argument("--output-dir", required=True)
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
    cache = StaticCache(
        config=model.config, max_batch_size=1, max_cache_len=args.max_cache_len,
        device=args.device, dtype=model.dtype, passage_len=args.max_cache_len,
    )

    passages = tokens_data[args.example - 1]
    system = passages[0]
    docs = list(passages[1:-1])
    ids = torch.cat([system] + docs).unsqueeze(0).to(args.device)
    pos = torch.arange(ids.shape[1], device=args.device)
    with torch.no_grad():
        model(input_ids=ids, cache_position=pos, past_key_values=cache, use_cache=True, return_dict=False)

    save_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    start = int(system.shape[0])
    prefix_tokens = 0
    meta = {"args": vars(args), "source_path": str(source_path), "chunks": []}
    for chunk_id, doc in enumerate(docs, start=1):
        n = int(doc.shape[0])
        full_k = torch.stack([cache.key_cache[layer][:, :, start:start + n, :].detach().cpu() for layer in range(model.config.num_hidden_layers)])
        full_k = rotate_key_offset(model, full_k, -prefix_tokens, args.device)
        full_v = torch.stack([cache.value_cache[layer][:, :, start:start + n, :].detach().cpu() for layer in range(model.config.num_hidden_layers)])
        src_k = torch.load(source_path / f"{args.example}_{chunk_id}_key.pt", map_location="cpu", weights_only=True)
        src_v = torch.load(source_path / f"{args.example}_{chunk_id}_value.pt", map_location="cpu", weights_only=True)
        key_bias = full_k.float() - src_k.float()
        value_bias = full_v.float() - src_v.float()
        torch.save(key_bias.to(save_dtype), out / "key" / f"{args.example}_{chunk_id}_key_bias.pt")
        torch.save(value_bias.to(save_dtype), out / "value" / f"{args.example}_{chunk_id}_value_bias.pt")
        meta["chunks"].append({
            "chunk_id": chunk_id,
            "tokens": n,
            "prefix_tokens": prefix_tokens,
            "key_bias_norm": float(key_bias.norm()),
            "value_bias_norm": float(value_bias.norm()),
        })
        start += n
        prefix_tokens += n
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote oracle example={args.example} chunks={len(docs)} out={out}")


if __name__ == "__main__":
    main()
