#!/usr/bin/env python3
"""Small-model correctness checks for layer-parallel working-KV fusion."""

from __future__ import annotations

import os
import json
import sys
import tempfile
from pathlib import Path

import torch
from transformers import Qwen2Config

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import preserve_cache_past_tokens
import models.modeling_qwen3 as modeling_qwen3
from ktransformers.operators.sparse_attention import selected_query_block_attention
from models.modeling_qwen3 import Qwen2ForCausalLM


def build_model() -> tuple[Qwen2ForCausalLM, Qwen2Config]:
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=True,
    )
    config.architectures = ["Qwen2ForCausalLM"]
    return Qwen2ForCausalLM(config).eval(), config


def make_cache(config: Qwen2Config) -> StaticCache:
    return StaticCache(
        config=config,
        max_batch_size=1,
        max_cache_len=16,
        device="cpu",
        dtype=torch.float32,
    )


def clone_cache(src: StaticCache, config: Qwen2Config) -> StaticCache:
    dst = make_cache(config)
    for layer in range(config.num_hidden_layers):
        dst.key_cache[layer].copy_(src.key_cache[layer])
        dst.value_cache[layer].copy_(src.value_cache[layer])
        dst.past_tokens[layer] = src.past_tokens[layer]
    return dst


def selected(cache: StaticCache, positions: torch.Tensor):
    return [
        (
            cache.key_cache[layer].index_select(2, positions).clone(),
            cache.value_cache[layer].index_select(2, positions).clone(),
        )
        for layer in range(len(cache.key_cache))
    ]


def run_reprocess(model, base, config, positions, replacement, alpha, sparse, captured_attention=None):
    cache = clone_cache(base, config)
    base_snapshot = selected(base, positions)
    base_keys = [key.clone() for key, _ in base_snapshot]
    base_values = [value.clone() for _, value in base_snapshot]
    immutable_keys = [key.clone() for key in base_keys]
    immutable_values = [value.clone() for value in base_values]
    old_topk = os.environ.get("FUSIONRAG_SPARSE_BLOCK_TOPK")
    old_size = os.environ.get("FUSIONRAG_SPARSE_BLOCK_SIZE")
    original_sparse_attention = modeling_qwen3.selected_query_sparse_attention
    try:
        if sparse:
            os.environ["FUSIONRAG_SPARSE_BLOCK_TOPK"] = "2"
            os.environ["FUSIONRAG_SPARSE_BLOCK_SIZE"] = "2"
        kwargs = {}
        if alpha is not None:
            kwargs["reprocess_kv_alpha"] = alpha
            kwargs["reprocess_base_key_states"] = base_keys
            kwargs["reprocess_base_value_states"] = base_values
        if captured_attention is not None:
            def capture_sparse_attention(query, key, value, q_idx):
                captured_attention.append((key.clone(), value.clone(), q_idx.clone()))
                return original_sparse_attention(query, key, value, q_idx)
            modeling_qwen3.selected_query_sparse_attention = capture_sparse_attention
        with torch.no_grad():
            model(
                input_ids=replacement,
                cache_position=positions,
                past_key_values=cache,
                use_cache=True,
                return_dict=False,
                use_sparse_attention=sparse,
                **kwargs,
            )
    finally:
        modeling_qwen3.selected_query_sparse_attention = original_sparse_attention
        if old_topk is None:
            os.environ.pop("FUSIONRAG_SPARSE_BLOCK_TOPK", None)
        else:
            os.environ["FUSIONRAG_SPARSE_BLOCK_TOPK"] = old_topk
        if old_size is None:
            os.environ.pop("FUSIONRAG_SPARSE_BLOCK_SIZE", None)
        else:
            os.environ["FUSIONRAG_SPARSE_BLOCK_SIZE"] = old_size
    for actual, expected in zip(base_keys, immutable_keys):
        assert torch.equal(actual, expected), "immutable base key snapshot was modified"
    for actual, expected in zip(base_values, immutable_values):
        assert torch.equal(actual, expected), "immutable base value snapshot was modified"
    return cache


def assert_same(left, right, label):
    for layer, ((lk, lv), (rk, rv)) in enumerate(zip(left, right)):
        assert torch.equal(lk, rk), (label, "key", layer, (lk - rk).abs().max())
        assert torch.equal(lv, rv), (label, "value", layer, (lv - rv).abs().max())


def main() -> None:
    model, config = build_model()
    base = make_cache(config)
    with torch.no_grad():
        model(
            input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
            cache_position=torch.arange(8),
            past_key_values=base,
            use_cache=True,
            return_dict=False,
        )
    positions = torch.tensor([2, 5])
    replacement = torch.tensor([[19, 23]])
    base_before = selected(base, positions)

    # Overwrite-style reprocess must preserve logical lengths on both normal
    # completion and exceptions, without replacing the list object.
    past_tokens_object = base.past_tokens
    past_tokens_before = list(base.past_tokens)
    with preserve_cache_past_tokens(base):
        base.past_tokens[:] = [value + 3 for value in base.past_tokens]
    assert base.past_tokens is past_tokens_object
    assert base.past_tokens == past_tokens_before
    try:
        with preserve_cache_past_tokens(base):
            base.past_tokens[:] = [value + 7 for value in base.past_tokens]
            raise RuntimeError("injected reprocess failure")
    except RuntimeError as exc:
        assert str(exc) == "injected reprocess failure"
    assert base.past_tokens is past_tokens_object
    assert base.past_tokens == past_tokens_before

    dense_candidate = run_reprocess(model, base, config, positions, replacement, None, False)
    dense_alpha1 = run_reprocess(model, base, config, positions, replacement, 1.0, False)
    dense_alpha0 = run_reprocess(model, base, config, positions, replacement, 0.0, False)
    dense_quarter = run_reprocess(model, base, config, positions, replacement, 0.25, False)
    dense_key_only = run_reprocess(model, base, config, positions, replacement, (0.25, 0.0), False)
    dense_value_only = run_reprocess(model, base, config, positions, replacement, (0.0, 0.25), False)

    assert_same(selected(dense_alpha1, positions), selected(dense_candidate, positions), "alpha1")
    assert_same(selected(dense_alpha0, positions), base_before, "alpha0-dense")
    assert_same(selected(base, positions), base_before, "base-immutable")

    # Only layer 0 has an alpha-independent candidate. Deeper candidates depend
    # recursively on the previous layer's alpha-conditioned hidden state.
    base_k0, base_v0 = base_before[0]
    candidate_k0, candidate_v0 = selected(dense_candidate, positions)[0]
    quarter_k0, quarter_v0 = selected(dense_quarter, positions)[0]
    assert torch.allclose(quarter_k0, 0.75 * base_k0 + 0.25 * candidate_k0, atol=2e-6, rtol=2e-6)
    assert torch.allclose(quarter_v0, 0.75 * base_v0 + 0.25 * candidate_v0, atol=2e-6, rtol=2e-6)
    key_only_k0, key_only_v0 = selected(dense_key_only, positions)[0]
    value_only_k0, value_only_v0 = selected(dense_value_only, positions)[0]
    assert torch.allclose(key_only_k0, quarter_k0, atol=2e-6, rtol=2e-6)
    assert torch.equal(key_only_v0, base_v0)
    assert torch.equal(value_only_k0, base_k0)
    assert torch.allclose(value_only_v0, quarter_v0, atol=2e-6, rtol=2e-6)

    sparse_alpha0 = run_reprocess(model, base, config, positions, replacement, 0.0, True)
    captured_attention = []
    sparse_quarter = run_reprocess(
        model, base, config, positions, replacement, 0.25, True, captured_attention
    )
    assert_same(selected(sparse_alpha0, positions), base_before, "alpha0-sparse")
    for key, value in selected(sparse_quarter, positions):
        assert torch.isfinite(key).all()
        assert torch.isfinite(value).all()
    assert len(captured_attention) == config.num_hidden_layers
    groups = config.num_attention_heads // config.num_key_value_heads
    for layer, (attention_key, attention_value, q_idx) in enumerate(captured_attention):
        working_key, working_value = selected(sparse_quarter, positions)[layer]
        expected_key = working_key.repeat_interleave(groups, dim=1)
        expected_value = working_value.repeat_interleave(groups, dim=1)
        assert torch.equal(attention_key.index_select(2, positions), expected_key)
        assert torch.equal(attention_value.index_select(2, positions), expected_value)
        assert torch.equal(q_idx[0, 0], positions)

    # The current block is mandatory even when top-K=1.
    q = torch.tensor([[[[1.0, 0.0]]]])
    k = torch.tensor([[[[-2.0, 0.0], [10.0, 0.0], [8.0, 0.0], [0.0, 0.0]]]])
    v = torch.tensor([[[[1.0, 0.0], [7.0, 0.0], [3.0, 0.0], [11.0, 0.0]]]])
    q_idx = torch.tensor([[[3]]])
    current_only = selected_query_block_attention(q, k, v, q_idx, block_size=1, topk_blocks=1)
    assert torch.allclose(current_only, v[:, :, 3:4], atol=1e-6)

    # With two selected queries, j=3 routes to selected predecessor i=1 and
    # the analysis path records that dependency and the real KV support.
    q_pair = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    q_idx_pair = torch.tensor([[[1, 3]]])
    with tempfile.NamedTemporaryFile() as stats_file:
        os.environ["FUSIONRAG_SPARSE_ROUTER_STATS"] = stats_file.name
        try:
            predecessor = selected_query_block_attention(
                q_pair, k, v, q_idx_pair, block_size=1, topk_blocks=2
            )
        finally:
            os.environ.pop("FUSIONRAG_SPARSE_ROUTER_STATS", None)
        stats_file.seek(0)
        stats = json.loads(stats_file.read().decode("utf-8").strip())
    assert predecessor[0, 0, 1, 0] > 6.9
    assert stats["dependency_pairs"] == 1
    assert stats["dependency_covered"] == 1
    assert stats["dependency_coverage"] == 1.0
    assert stats["effective_kv_tokens"] > 0

    # Routing must use the supplied working K. Changing only predecessor i=1
    # from a weak to a strong key changes which historical value j retrieves.
    weak_k = k.clone()
    weak_k[:, :, 1] = torch.tensor([-10.0, 0.0])
    strong_k = weak_k.clone()
    strong_k[:, :, 1] = torch.tensor([20.0, 0.0])
    weak_route = selected_query_block_attention(q, weak_k, v, q_idx, block_size=1, topk_blocks=2)
    strong_route = selected_query_block_attention(q, strong_k, v, q_idx, block_size=1, topk_blocks=2)
    assert strong_route[0, 0, 0, 0] > weak_route[0, 0, 0, 0] + 3.0

    # A subsequent query call does not receive reprocess_kv_alpha and therefore
    # cannot re-blend document positions.
    query_cache = clone_cache(sparse_quarter, config)
    docs_before_query = selected(query_cache, positions)
    with torch.no_grad():
        model(
            input_ids=torch.tensor([[31, 37]]),
            cache_position=torch.tensor([8, 9]),
            past_key_values=query_cache,
            use_cache=True,
            return_dict=False,
            use_sparse_attention=False,
        )
    assert_same(selected(query_cache, positions), docs_before_query, "query-isolation")

    # Unsupported execution modes fail explicitly instead of silently falling
    # back to replacement or chunk-major semantics.
    original_backend = model.model._attn_implementation
    model.model._attn_implementation = "eager"
    try:
        run_reprocess(model, base, config, positions, replacement, 0.5, False)
        raise AssertionError("non-SDPA working-KV did not fail")
    except RuntimeError as exc:
        assert "only the SDPA backend" in str(exc)
    finally:
        model.model._attn_implementation = original_backend
    try:
        model(
            input_ids=replacement,
            cache_position=positions,
            past_key_values=clone_cache(base, config),
            use_cache=True,
            output_attentions=True,
            reprocess_kv_alpha=0.5,
            reprocess_base_key_states=[x for x, _ in base_before],
            reprocess_base_value_states=[x for _, x in base_before],
        )
        raise AssertionError("output_attentions working-KV did not fail")
    except RuntimeError as exc:
        assert "output_attentions=False" in str(exc)
    long_inputs = torch.zeros((1, 32769, config.hidden_size))
    try:
        model.model(
            inputs_embeds=long_inputs,
            cache_position=torch.arange(32769),
            past_key_values=clone_cache(base, config),
            use_cache=True,
            reprocess_kv_alpha=0.5,
        )
        raise AssertionError("long working-KV did not fail")
    except RuntimeError as exc:
        assert "one layer-parallel chunk" in str(exc)
    print(
        "PASS working-KV endpoints, independent K/V alpha, immutable base snapshots, pre-attention scatter, "
        "selected-predecessor read, working-K routing, dependency stats, query isolation, "
        "past-token restoration, and fail-fast guards"
    )


if __name__ == "__main__":
    main()
