#!/usr/bin/env python3
"""Collect Qwen3-8B prefix+target DeltaKV records with full offline target KV.

This extends collect_qwen3_8b_wikitext_delta_fullprefix.py for the Scheme-B
probe. It stores all offline target K/V positions as an allowed sidecar so a
layer-parallel token_i update can attend to prefix + target(<i) without using
full-context hidden/KV as predictor inputs.
"""
import argparse
import json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="/home/hming/models/Qwen3-8B")
    args = parser.parse_args()

    device = "cuda:0"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"
    model, _ = load_model("qwen3", args.model_path, cfg, device, False)
    model.eval()
    cache = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=512,
        device=device,
        dtype=model.dtype,
        passage_len=512,
        model=model,
    )

    def capture(ids):
        for layer in range(len(cache.key_cache)):
            cache.past_tokens[layer] = 0
        ids = ids.unsqueeze(0).to(device)
        pos = torch.arange(ids.shape[1], device=device)
        with torch.no_grad():
            model(input_ids=ids, cache_position=pos, past_key_values=cache, use_cache=True, return_dict=False)
        keys = torch.stack([x[:, :, : ids.shape[1], :].detach().cpu().half() for x in cache.key_cache])
        values = torch.stack([x[:, :, : ids.shape[1], :].detach().cpu().half() for x in cache.value_cache])
        return keys, values

    def rotate(keys, delta):
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

    ids = torch.load(args.token_cache, map_location="cpu", weights_only=True)
    manifest = []
    for sample in range(args.start, args.start + args.count):
        begin = sample * args.stride
        span = ids[begin : begin + 384].long()
        if len(span) < 384:
            break
        prefix, target = span[:256], span[256:]
        offline_k, offline_v = capture(target)
        full_k, full_v = capture(torch.cat([prefix, target]))
        full_target_k = rotate(full_k[:, :, :, 256:, :], -256)
        full_target_v = full_v[:, :, :, 256:]
        delta_k = full_target_k.float() - offline_k.float()
        delta_v = full_target_v.float() - offline_v.float()
        idx = torch.linspace(0, 127, 8).long()

        artifact = {
            "model": "Qwen3-8B",
            "sample": sample,
            "token_start": begin,
            "prefix_tokens": 256,
            "target_tokens": 128,
            "sampled_positions": idx,
            "own_kv": torch.cat([offline_k[:, 0, :, idx], offline_v[:, 0, :, idx]], -1).half(),
            "own_target_kv_all": torch.cat([offline_k[:, 0], offline_v[:, 0]], -1).half(),
            "prefix_kv": torch.cat([full_k[:, 0, :, :256], full_v[:, 0, :, :256]], -1).half(),
            "delta_kv": torch.cat([delta_k[:, 0, :, idx], delta_v[:, 0, :, idx]], -1).half(),
            "metrics": {
                "k_original_gap": float(delta_k.square().sum().sqrt() / offline_k.float().square().sum().sqrt()),
                "v_original_gap": float(delta_v.square().sum().sqrt() / offline_v.float().square().sum().sqrt()),
            },
            "input_policy_note": "own_target_kv_all is offline target-only cache, not full-context target KV",
        }
        path = out / f"sample{sample:07d}.pt"
        torch.save(artifact, path)
        row = {"sample": sample, "path": path.name, "bytes": path.stat().st_size, **artifact["metrics"]}
        manifest.append(row)
        print(json.dumps(row), flush=True)

    (out / f"manifest_{args.start}_{args.start + args.count}.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
