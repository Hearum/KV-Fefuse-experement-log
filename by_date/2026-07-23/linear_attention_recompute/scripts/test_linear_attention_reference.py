#!/usr/bin/env python3
"""Correctness tests for the Phase-0.5 normalized linear reference."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from linear_attention_reference import normalized_chunked, normalized_recurrent


def check_close(name, actual, expected, atol=2e-6, rtol=2e-6):
    error = (actual - expected).abs().max().item()
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"{name}: max_error={error}")
    print(f"PASS {name}: max_error={error:.3e}")


def explicit_prefix(q, k, v, initial_state=None):
    qf = torch.nn.functional.elu(q.float()) + 1.0
    kf = torch.nn.functional.elu(k.float()) + 1.0
    vf = v.float()
    scale = q.shape[-1] ** -0.5
    if initial_state is None:
        state = torch.zeros(
            q.shape[0], q.shape[2], q.shape[3], v.shape[3], dtype=torch.float32
        )
        z_state = torch.zeros(q.shape[0], q.shape[2], q.shape[3], dtype=torch.float32)
    else:
        state, z_state = (x.float().clone() for x in initial_state)
    outputs = []
    for t in range(q.shape[1]):
        state_t = state + torch.einsum("bthk,bthv->bhkv", kf[:, : t + 1], vf[:, : t + 1])
        z_t = z_state + kf[:, : t + 1].sum(dim=1)
        numerator = torch.einsum("bhk,bhkv->bhv", qf[:, t] * scale, state_t)
        denominator = (qf[:, t] * scale * z_t).sum(dim=-1, keepdim=True)
        outputs.append(numerator / (denominator + 1e-10))
    return torch.stack(outputs, dim=1), (state_t, z_t)


def main():
    torch.manual_seed(42)
    for length in (1, 3, 7, 16):
        q = torch.randn(2, length, 3, 5)
        k = torch.randn(2, length, 3, 5)
        v = torch.randn(2, length, 3, 4)
        initial = (
            torch.randn(2, 3, 5, 4),
            torch.randn(2, 3, 5),
        )
        expected, expected_state = explicit_prefix(q, k, v, initial)
        recurrent, recurrent_state = normalized_recurrent(
            q, k, v, initial_state=initial
        )
        check_close(f"recurrent explicit prefix T={length}", recurrent, expected)
        check_close(
            f"recurrent final S T={length}",
            recurrent_state[0],
            expected_state[0],
        )
        check_close(
            f"recurrent final z T={length}",
            recurrent_state[1],
            expected_state[1],
        )
        for chunk_size in (1, 2, 4, 8):
            chunked, chunked_state = normalized_chunked(
                q, k, v, chunk_size=chunk_size, initial_state=initial
            )
            check_close(
                f"chunked output T={length} C={chunk_size}",
                chunked,
                expected,
            )
            check_close(
                f"chunked final S T={length} C={chunk_size}",
                chunked_state[0],
                expected_state[0],
            )
            check_close(
                f"chunked final z T={length} C={chunk_size}",
                chunked_state[1],
                expected_state[1],
            )

    q = torch.randn(1, 9, 2, 4, dtype=torch.bfloat16)
    k = torch.randn(1, 9, 2, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 9, 2, 3, dtype=torch.bfloat16)
    recurrent, _ = normalized_recurrent(q, k, v)
    chunked, _ = normalized_chunked(q, k, v, chunk_size=3)
    check_close("bfloat16 recurrent/chunked", recurrent.float(), chunked.float(),
                atol=3e-2, rtol=3e-2)
    print("ALL PASS normalized linear reference")


if __name__ == "__main__":
    main()

