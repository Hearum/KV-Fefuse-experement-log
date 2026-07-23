#!/usr/bin/env python3
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ktransformers.operators.linear_attention import (
    selected_query_normalized_linear_attention,
)
from linear_attention_reference import normalized_recurrent


def main():
    torch.manual_seed(123)
    batch, seq, q_heads, kv_heads, key_dim, value_dim = 1, 19, 8, 2, 5, 7
    q = torch.randn(batch, seq, q_heads, key_dim)
    k = torch.randn(batch, seq, kv_heads, key_dim)
    v = torch.randn(batch, seq, kv_heads, value_dim)

    q_ref = q.transpose(1, 2)
    k_ref = k.transpose(1, 2)
    v_ref = v.transpose(1, 2)
    positions = torch.tensor([0, 3, 7, 12, 18])
    selected_q = q_ref.index_select(2, positions)

    # Expand GQA KV for the per-query-head reference.
    k_expanded = k_ref.repeat_interleave(q_heads // kv_heads, dim=1)
    v_expanded = v_ref.repeat_interleave(q_heads // kv_heads, dim=1)
    expected_all, _ = normalized_recurrent(q, k_expanded.transpose(1, 2),
                                           v_expanded.transpose(1, 2))
    # The reference expects [B,T,H,D].
    expected = expected_all.transpose(1, 2).index_select(2, positions)

    actual = selected_query_normalized_linear_attention(
        selected_q, k_ref, v_ref, positions, chunk_size=4
    )
    error = (actual - expected).abs().max().item()
    if not torch.allclose(actual, expected, atol=3e-5, rtol=3e-5):
        raise AssertionError(f"GQA selected-query mismatch: {error}")
    print(f"PASS GQA selected-query reference max_error={error:.3e}")

    for chunk in (1, 3, 8, 32):
        actual = selected_query_normalized_linear_attention(
            selected_q, k_ref, v_ref, positions, chunk_size=chunk
        )
        error = (actual - expected).abs().max().item()
        if not torch.allclose(actual, expected, atol=3e-5, rtol=3e-5):
            raise AssertionError(f"chunk={chunk} mismatch: {error}")
        print(f"PASS chunk={chunk} max_error={error:.3e}")


if __name__ == "__main__":
    main()

