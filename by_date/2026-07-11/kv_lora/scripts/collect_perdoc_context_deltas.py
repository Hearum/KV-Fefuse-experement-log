#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/raid/home/hming/FusionRAG-pca-analysis")
from transformers import AutoConfig, AutoTokenizer
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import load_kv_and_generate
from test_fusionrag_reflect_preprocess_exp import (
    PreprocessScope,
    RecallMethod,
    load_model,
    prepare_reflect_data,
)


def contexts(pool, count, seed):
    result = [()]
    rng = random.Random(seed)
    # Cover every prefix cardinality before randomizing additional orders/subsets.
    for size in range(1, len(pool) + 1):
        result.append(tuple(pool[:size]))
    seen = set(result)
    while len(result) < count:
        size = rng.randrange(0, len(pool) + 1)
        candidate = tuple(rng.sample(pool, size))
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return result[:count]


p = argparse.ArgumentParser()
p.add_argument("--example", type=int, default=0)
p.add_argument("--target", type=int, required=True)
p.add_argument("--contexts", type=int, default=50)
p.add_argument("--part", required=True)
a = p.parse_args()

torch.manual_seed(0)
torch.set_num_threads(8)
root = Path("/raid/home/hming/FusionRAG-pca-analysis")
out = root / "MOTIVATION_EXPERIMENTS/kv_lora/results/perdoc_context_deltas" / a.part
out.mkdir(parents=True, exist_ok=True)
model_path = "/mnt/qjhs-sh-lab-01/models/Qwen3-32B"
cache_root = Path("/raid/home/hming/fusionrag-reflect-qwen3-full-cache/Qwen3-32B/musique")
device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
config._attn_implementation = "sdpa"
model, _ = load_model("qwen3", model_path, config, device, False)
model.eval()
questions, system, _, _ = prepare_reflect_data(
    str(root / "data/result_reflect.json"), tokenizer,
    "/mnt/qjhs-sh-lab-01/models/bge-m3", "qwen3", topk=10,
    max_main_questions=a.example + 1, preprocess=False,
    recall_method=RecallMethod.BGE, preprocess_scope=PreprocessScope.GLOBAL,
)
data = questions[a.example]
sub = data["sub_questions"][0]
all_ids = list(range(1, min(10, len(data["doc_tensors"])) + 1))
if a.target not in all_ids:
    raise ValueError(f"target {a.target} not in {all_ids}")
pool = [x for x in all_ids if x != a.target]
ctxs = contexts(pool, a.contexts, seed=1000 * a.example + a.target)
target_doc = data["doc_tensors"][a.target - 1]
query = torch.tensor(tokenizer.encode(
    "<|im_end|>\n<|im_start|>user\nQuestion: " + sub["query"]
    + "\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: ",
    add_special_tokens=False,
))
cache = StaticCache(
    config=model.config, max_batch_size=1, max_cache_len=8192,
    device=device, dtype=model.dtype, passage_len=8192, model=model,
)
manifest = []
for context_index, prefix_ids in enumerate(ctxs):
    docs = [data["doc_tensors"][x - 1] for x in prefix_ids]
    passages = [system] + docs + [target_doc] + [query]
    target_start = len(system) + sum(len(x) for x in docs)
    target_tokens = len(target_doc)
    chunk_ids = [0] + list(prefix_ids) + [a.target]
    for source, cache_dir in (
        ("raw", cache_root / "kv_cache"),
        ("preprocess", cache_root / "preprocess_kv_cache_global_topk10_bge"),
    ):
        missing = [x for x in set(prefix_ids + (a.target,)) if not (cache_dir / f"{a.example}_{x}_key.pt").exists()]
        if missing:
            raise FileNotFoundError(f"{source} missing chunks {missing}")
        state = {}

        def snapshot(phase, past, selected, *_):
            label = "before" if phase == "before_reprocess" else "after"
            for kind, layers in (("k", past.key_cache), ("v", past.value_cache)):
                state[f"{label}_{kind}"] = torch.stack([
                    layer[:, :, target_start:target_start + target_tokens, :]
                    .detach().cpu().to(torch.float32)
                    for layer in layers
                ])
            if selected is not None:
                state["selected_target"] = sum(
                    target_start <= int(x) < target_start + target_tokens for x in selected
                )

        load_kv_and_generate(
            model, tokenizer, cache, passages, str(cache_dir), a.example,
            max_new_tokens=1, revert_rope=True, reprocess_method="FusionRAG",
            rate=1.0, preprocess=source == "preprocess", device=device,
            chunk_ids=chunk_ids, kv_snapshot_callback=snapshot,
        )
        if state.get("selected_target") != target_tokens:
            raise RuntimeError(f"selected target {state.get('selected_target')} != {target_tokens}")
        payload = {
            "delta_k": (state["after_k"] - state["before_k"]).to(torch.float16),
            "delta_v": (state["after_v"] - state["before_v"]).to(torch.float16),
            "base_k_norm2_layer": state["before_k"].square().flatten(1).sum(1),
            "base_v_norm2_layer": state["before_v"].square().flatten(1).sum(1),
            "meta": {
                "example": a.example, "target": a.target, "source": source,
                "context_index": context_index, "prefix_ids": list(prefix_ids),
                "prefix_tokens": target_start - len(system),
                "target_start": target_start, "target_tokens": target_tokens,
            },
        }
        path = out / f"context_{context_index:02d}_{source}.pt"
        torch.save(payload, path)
        manifest.append({**payload["meta"], "path": str(path.relative_to(root))})
        del state, payload
        print(json.dumps(manifest[-1]), flush=True)

(out / "manifest.json").write_text(json.dumps({
    "part": a.part, "example": a.example, "target": a.target,
    "contexts": len(ctxs), "context_definitions": [list(x) for x in ctxs],
    "runs": manifest,
}, indent=2) + "\n")
