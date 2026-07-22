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


def first_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


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

    def make_v_hook(layer_idx):
        def hook(module, inputs, output):
            captures.setdefault(layer_idx, {})["v_input"] = inputs[0].detach().cpu().half()
            captures[layer_idx]["v_output"] = output.detach().cpu().half()
        return hook

    def make_attn_hook(layer_idx):
        def hook(module, inputs, output):
            captures.setdefault(layer_idx, {})["attn_output"] = first_tensor(output).detach().cpu().half()
        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(layer_idx)))
        handles.append(layer.self_attn.register_forward_hook(make_attn_hook(layer_idx)))

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
        h_in = torch.stack([captures[i]["v_input"][0] for i in range(len(model.model.layers))])
        v_out = torch.stack([
            captures[i]["v_output"][0].reshape(ids.shape[1], model.config.num_key_value_heads, -1).permute(1, 0, 2)
            for i in range(len(model.model.layers))
        ]).half()
        attn_out = torch.stack([captures[i]["attn_output"][0] for i in range(len(model.model.layers))])
        return keys, values, h_in, v_out, attn_out

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
        offline_k, offline_v, offline_h, offline_vproj, offline_attn = capture(target)
        full_k, full_v, full_h, full_vproj, full_attn = capture(torch.cat([prefix, target]))
        full_target_k = rotate(full_k[:, :, :, 256:, :], -256)
        idx = torch.linspace(0, 127, 8).long()
        delta_k = full_target_k[:, 0, :, idx, :].float() - offline_k[:, 0, :, idx, :].float()
        delta_v = full_v[:, 0, :, 256:, :][:, :, idx, :].float() - offline_v[:, 0, :, idx, :].float()
        delta_h = full_h[:, 256:, :][:, idx, :].float() - offline_h[:, idx, :].float()
        delta_attn = full_attn[:, 256:, :][:, idx, :].float() - offline_attn[:, idx, :].float()
        delta_vproj = full_vproj[:, :, 256:, :][:, :, idx, :].float() - offline_vproj[:, :, idx, :].float()
        vproj_cache_mismatch = float((delta_v - delta_vproj).square().sum().sqrt() / delta_v.square().sum().sqrt().clamp_min(1e-12))

        layer_k_gap = (delta_k.square().sum(dim=(1, 2, 3)).sqrt() / offline_k[:, 0, :, idx, :].float().square().sum(dim=(1, 2, 3)).sqrt().clamp_min(1e-12))
        layer_v_gap = (delta_v.square().sum(dim=(1, 2, 3)).sqrt() / offline_v[:, 0, :, idx, :].float().square().sum(dim=(1, 2, 3)).sqrt().clamp_min(1e-12))
        layer_h_gap = (delta_h.square().sum(dim=(1, 2)).sqrt() / offline_h[:, idx, :].float().square().sum(dim=(1, 2)).sqrt().clamp_min(1e-12))
        layer_attn_gap = (delta_attn.square().sum(dim=(1, 2)).sqrt() / offline_attn[:, idx, :].float().square().sum(dim=(1, 2)).sqrt().clamp_min(1e-12))

        artifact = {
            "model": "Qwen3-8B",
            "sample": sample,
            "token_start": begin,
            "prefix_tokens": 256,
            "target_tokens": 128,
            "sampled_positions": idx,
            "delta_k": delta_k.half(),
            "delta_v": delta_v.half(),
            "delta_h": delta_h.half(),
            "delta_attn": delta_attn.half(),
            "layer_k_gap": layer_k_gap.float(),
            "layer_v_gap": layer_v_gap.float(),
            "layer_h_gap": layer_h_gap.float(),
            "layer_attn_gap": layer_attn_gap.float(),
            "metrics": {
                "k_gap_sampled": float(delta_k.square().sum().sqrt() / offline_k[:, 0, :, idx, :].float().square().sum().sqrt()),
                "v_gap_sampled": float(delta_v.square().sum().sqrt() / offline_v[:, 0, :, idx, :].float().square().sum().sqrt()),
                "h_gap_sampled": float(delta_h.square().sum().sqrt() / offline_h[:, idx, :].float().square().sum().sqrt()),
                "attn_gap_sampled": float(delta_attn.square().sum().sqrt() / offline_attn[:, idx, :].float().square().sum().sqrt()),
                "vproj_cache_mismatch": vproj_cache_mismatch,
                "layer0_k_gap": float(layer_k_gap[0]),
                "layer0_v_gap": float(layer_v_gap[0]),
                "layer0_h_gap": float(layer_h_gap[0]),
                "layer0_attn_gap": float(layer_attn_gap[0]),
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
