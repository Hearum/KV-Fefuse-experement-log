#!/usr/bin/env python3
"""Phase-0 semantic checks for linear/working-KV recomputation.

This is a small reference test. It intentionally uses explicit loops so that
future fast implementations can be compared against it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


DTYPE = torch.float64


def phi(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.elu(x) + 1.0


def state_from_tokens(k: torch.Tensor, v: torch.Tensor):
    p = phi(k)
    return p.transpose(-1, -2) @ v, p.sum(dim=-2)


def causal_states(k: torch.Tensor, v: torch.Tensor):
    states = []
    norm = []
    s = torch.zeros(k.shape[-1], v.shape[-1], dtype=k.dtype)
    z = torch.zeros(k.shape[-1], dtype=k.dtype)
    for i in range(k.shape[-2]):
        p = phi(k[i])
        s = s + torch.outer(p, v[i])
        z = z + p
        states.append(s.clone())
        norm.append(z.clone())
    return torch.stack(states), torch.stack(norm)


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor):
    error = (actual - expected).abs().max().item()
    if not torch.allclose(actual, expected, atol=1e-10, rtol=1e-10):
        raise AssertionError(f"{name} max_error={error}")
    print(f"PASS {name} max_error={error:.3e}")


@dataclass
class CacheMapping:
    mapping: dict[int, int]

    def map_positions(self, positions):
        return [self.mapping[p] for p in positions]


def test_block_additivity():
    torch.manual_seed(0)
    k = torch.randn(12, 4, dtype=DTYPE)
    v = torch.randn(12, 3, dtype=DTYPE)
    full_s, full_z = state_from_tokens(k, v)
    block_s = []
    block_z = []
    for start, end in ((0, 4), (4, 9), (9, 12)):
        s, z = state_from_tokens(k[start:end], v[start:end])
        block_s.append(s)
        block_z.append(z)
    assert_close("block state additivity", sum(block_s), full_s)
    assert_close("block normalizer additivity", sum(block_z), full_z)


def test_selected_rank_one_update():
    torch.manual_seed(1)
    k = torch.randn(10, 4, dtype=DTYPE)
    v = torch.randn(10, 3, dtype=DTYPE)
    k_new = torch.randn(4, dtype=DTYPE)
    v_new = torch.randn(3, dtype=DTYPE)
    index = 6
    old_s, old_z = state_from_tokens(k, v)
    updated_s = old_s - torch.outer(phi(k[index]), v[index])
    updated_s = updated_s + torch.outer(phi(k_new), v_new)
    updated_z = old_z - phi(k[index]) + phi(k_new)
    k[index] = k_new
    v[index] = v_new
    expected_s, expected_z = state_from_tokens(k, v)
    assert_close("rank-one state update", updated_s, expected_s)
    assert_close("rank-one normalizer update", updated_z, expected_z)


def test_causal_order():
    torch.manual_seed(2)
    k = torch.randn(8, 4, dtype=DTYPE)
    v = torch.randn(8, 3, dtype=DTYPE)
    states, norms = causal_states(k, v)
    for i in range(8):
        expected_s, expected_z = state_from_tokens(k[: i + 1], v[: i + 1])
        assert_close(f"causal state token {i}", states[i], expected_s)
        assert_close(f"causal norm token {i}", norms[i], expected_z)


def test_cache_position_mapping():
    # passages: system 0:2, doc1 2:5, missing doc2 5:8, doc3 8:11
    # loaded KV contains system, doc1, doc3; missing doc2 is appended at 8:11.
    mapping = CacheMapping({
        0: 0, 1: 1,
        2: 2, 3: 3, 4: 4,
        5: 8, 6: 9, 7: 10,
        8: 5, 9: 6, 10: 7,
    })
    selected = [5, 6, 7]
    mapped = mapping.map_positions(selected)
    if mapped != [8, 9, 10]:
        raise AssertionError(f"wrong cache mapping: {mapped}")
    if set(mapped) & set([5, 6, 7]):
        raise AssertionError(f"missing document overwrites loaded document: {mapped}")
    print("PASS cache position mapping avoids overwrite")


def test_blend_endpoints():
    torch.manual_seed(3)
    base = torch.randn(5, 4, dtype=DTYPE)
    candidate = torch.randn(5, 4, dtype=DTYPE)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        working = base + alpha * (candidate - base)
        assert_close(f"blend alpha={alpha}", working,
                     (1.0 - alpha) * base + alpha * candidate)


def main():
    test_block_additivity()
    test_selected_rank_one_update()
    test_causal_order()
    test_cache_position_mapping()
    test_blend_endpoints()
    print("ALL PASS phase-0 linear state semantics")


if __name__ == "__main__":
    main()

