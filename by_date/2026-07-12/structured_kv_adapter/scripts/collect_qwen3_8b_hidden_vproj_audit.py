#!/usr/bin/env python3
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
    captures = {}
    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            captures[layer_idx] = {
                "input": inputs[0].detach().cpu().half(),
                "output": output.detach().cpu().half(),
            }
        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_hook(layer_idx)))

    def capture(ids):
        captures.clear()
        for layer in range(len(cache.key_cache)):
            cache.past_tokens[layer] = 0
        ids = ids.unsqueeze(0).to(device)
        pos = torch.arange(ids.shape[1], device=device)
        with torch.no_grad():
            model(input_ids=ids, cache_position=pos, past_key_values=cache, use_cache=True, return_dict=False)
        values = torch.stack([x[:, :, : ids.shape[1], :].detach().cpu().half() for x in cache.value_cache])
        h_in = torch.stack([captures[i]["input"][0] for i in range(len(model.model.layers))])
        v_out = torch.stack([
            captures[i]["output"][0].reshape(ids.shape[1], model.config.num_key_value_heads, -1).permute(1, 0, 2)
            for i in range(len(model.model.layers))
        ]).half()
        return values, h_in, v_out

    ids = torch.load(args.token_cache, map_location="cpu", weights_only=True)
    manifest = []
    for sample in range(args.start, args.start + args.count):
        begin = sample * args.stride
        span = ids[begin : begin + 384].long()
        if len(span) < 384:
            break
        prefix, target = span[:256], span[256:]
        offline_v, offline_h, offline_vproj = capture(target)
        full_v, full_h, full_vproj = capture(torch.cat([prefix, target]))
        idx = torch.linspace(0, 127, 8).long()
        full_h_target = full_h[:, 256:, :]
        full_v_target = full_v[:, :, :, 256:, :]
        full_vproj_target = full_vproj[:, :, 256:, :]

        delta_h = full_h_target[:, idx, :].float() - offline_h[:, idx, :].float()
        delta_v = full_v_target[:, 0, :, idx, :].float() - offline_v[:, 0, :, idx, :].float()
        delta_vproj = full_vproj_target[:, :, idx, :].float() - offline_vproj[:, :, idx, :].float()
        vproj_cache_mismatch = float((delta_v - delta_vproj).square().sum().sqrt() / delta_v.square().sum().sqrt().clamp_min(1e-12))

        artifact = {
            "model": "Qwen3-8B",
            "sample": sample,
            "token_start": begin,
            "prefix_tokens": 256,
            "target_tokens": 128,
            "sampled_positions": idx,
            "own_h": offline_h[:, idx, :].half(),
            "delta_h": delta_h.half(),
            "delta_v": delta_v.half(),
            "metrics": {
                "h_original_gap": float(delta_h.square().sum().sqrt() / offline_h[:, idx, :].float().square().sum().sqrt()),
                "v_original_gap_sampled": float(delta_v.square().sum().sqrt() / offline_v[:, 0, :, idx, :].float().square().sum().sqrt()),
                "vproj_cache_mismatch": vproj_cache_mismatch,
            },
        }
        path = out / f"sample{sample:07d}.pt"
        torch.save(artifact, path)
        row = {"sample": sample, "path": path.name, "bytes": path.stat().st_size, **artifact["metrics"]}
        manifest.append(row)
        print(json.dumps(row), flush=True)

    for handle in handles:
        handle.remove()
    (out / f"manifest_{args.start}_{args.start + args.count}.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
