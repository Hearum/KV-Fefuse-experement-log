#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
sys.path.insert(0, str(ROOT))
from ktransformers.models.custom_cache import StaticCache
from test_fusionrag_reflect_preprocess_exp import load_model


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--token-cache", default=str(ROOT / "MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/wikitext_qwen3_tokens_train_0_35.pt"))
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=2)
    p.add_argument("--output-dir", default=str(ROOT / "MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/wikitext_delta_smoke"))
    p.add_argument("--basis-checkpoint", default=str(ROOT / "MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/grouped_head_predictor_rank32.pt"))
    return p.parse_args()


def main():
    a = args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    model_path = "/mnt/qjhs-sh-lab-01/models/Qwen3-32B"; device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True); config._attn_implementation = "sdpa"
    model, _ = load_model("qwen3", model_path, config, device, False); model.eval()
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=512, device=device,
                        dtype=model.dtype, passage_len=512, model=model)
    checkpoint = torch.load(a.basis_checkpoint, map_location="cpu", weights_only=False)
    basis_model = checkpoint["basis"]

    def capture(ids):
        for layer in range(len(cache.key_cache)): cache.past_tokens[layer] = 0
        ids = ids.unsqueeze(0).to(device); positions = torch.arange(ids.shape[1], device=device)
        with torch.no_grad(): model(input_ids=ids, cache_position=positions, past_key_values=cache,
                                    use_cache=True, return_dict=False)
        keys = torch.stack([x[:, :, :ids.shape[1], :].detach().cpu().half() for x in cache.key_cache])
        values = torch.stack([x[:, :, :ids.shape[1], :].detach().cpu().half() for x in cache.value_cache])
        return keys, values

    def rotate(keys, delta):
        z = keys.to(device); positions = torch.full((1, z.shape[-2]), delta, device=device)
        try: cosine, sine = model.model.rotary_emb(z[0], positions)
        except AttributeError: cosine, sine = model.model.layers[0].self_attn.rotary_emb(z[0], positions)
        cosine = cosine.unsqueeze(0).unsqueeze(2); sine = sine.unsqueeze(0).unsqueeze(2); half = z.shape[-1] // 2
        return (z * cosine + torch.cat([-z[..., half:], z[..., :half]], -1) * sine).cpu()

    token_cache = torch.load(a.token_cache, map_location="cpu", weights_only=True)
    selected = []
    for sample_index in range(a.start, a.start + a.count):
        token_start = sample_index * a.stride
        span = token_cache[token_start:token_start + 384].long()
        if len(span) < 384: break
        selected.append(({"pair_id": sample_index, "token_start_in_slice": token_start,
                          "text_slice": [0.0, 0.35]}, span))

    manifest = []
    for row, ids in selected:
        prefix, target = ids[:256], ids[256:384]
        offline_k, offline_v = capture(target)
        full_k, full_v = capture(torch.cat([prefix, target]))
        aligned_full_k = rotate(full_k[:, :, :, 256:384], -256)
        target_full_v = full_v[:, :, :, 256:384]
        delta_k = aligned_full_k.float() - offline_k.float()
        delta_v = target_full_v.float() - offline_v.float()
        token_index = torch.linspace(0, 127, 8).long()
        own = torch.cat([offline_k[:, 0, :, token_index], offline_v[:, 0, :, token_index]], -1)
        prefix_k = full_k[:, 0, :, :256].float().mean(2)
        prefix_v = full_v[:, 0, :, :256].float().mean(2)
        prefix_features = torch.cat([prefix_k, prefix_v, prefix_k, prefix_v], -1).half()
        coefficients = []; metrics = {}
        for kind, delta in (("k", delta_k), ("v", delta_v)):
            sampled = delta[:, 0, :, token_index]
            mean = basis_model[kind]["mean"].float().unsqueeze(2)
            basis = basis_model[kind]["basis"].float()
            centered = sampled - mean
            coefficient = torch.einsum("lhtf,lhfr->lhtr", centered, basis)
            reconstruction = mean + torch.einsum("lhtr,lhfr->lhtf", coefficient, basis)
            coefficients.append(coefficient.half())
            error = float((sampled - reconstruction).square().sum()); energy = float(sampled.square().sum())
            metrics[f"{kind}_projection_remaining"] = (error / energy) ** 0.5
            metrics[f"{kind}_projection_explained_energy"] = 1 - error / energy
        artifact = {"pair_id": row["pair_id"], "token_start": row["token_start_in_slice"],
                    "text_slice": row["text_slice"], "prefix_tokens": 256, "target_tokens": 128,
                    "sampled_positions": token_index, "own_kv": own, "prefix_features": prefix_features,
                    "coefficients_kv": torch.cat(coefficients, -1), "metrics": metrics}
        path = out / f"pair{row['pair_id']:07d}.pt"; torch.save(artifact, path)
        manifest.append({"pair_id": row["pair_id"], "path": path.name, "bytes": path.stat().st_size, **metrics})
        print(json.dumps(manifest[-1]), flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
