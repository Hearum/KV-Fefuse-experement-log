"""Reference normalized causal linear attention for Phase 0.5.

Shapes are batch-first and head-aware:
q, k: [B, T, H, Dk]
v:    [B, T, H, Dv]
state S: [B, H, Dk, Dv]
state z: [B, H, Dk]

The implementation is intentionally explicit and is not a production kernel.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


State = Tuple[torch.Tensor, torch.Tensor]


def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.elu(x) + 1.0


def _check_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape [B, T, H, D]")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, v must agree on [B, T, H]")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k feature dimensions must match")


def _prepare_state(
    q: torch.Tensor,
    v: torch.Tensor,
    initial_state: Optional[State],
) -> State:
    batch, _, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if initial_state is None:
        state = torch.zeros(
            batch, heads, key_dim, value_dim, device=q.device, dtype=torch.float32
        )
        z_state = torch.zeros(
            batch, heads, key_dim, device=q.device, dtype=torch.float32
        )
        return state, z_state

    state, z_state = initial_state
    expected_state = (batch, heads, key_dim, value_dim)
    expected_z = (batch, heads, key_dim)
    if tuple(state.shape) != expected_state or tuple(z_state.shape) != expected_z:
        raise ValueError(
            f"initial_state shapes must be {expected_state} and {expected_z}, "
            f"got {tuple(state.shape)} and {tuple(z_state.shape)}"
        )
    return state.to(device=q.device, dtype=torch.float32).clone(), z_state.to(
        device=q.device, dtype=torch.float32
    ).clone()


def normalized_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: Optional[float] = None,
    initial_state: Optional[State] = None,
    eps: float = 1e-10,
    output_final_state: bool = True,
) -> Tuple[torch.Tensor, Optional[State]]:
    """Reference token-loop implementation of normalized causal attention."""
    _check_inputs(q, k, v)
    dtype = q.dtype
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    qf, kf, vf = (x.to(torch.float32) for x in (q, k, v))
    qf = elu_plus_one(qf)
    kf = elu_plus_one(kf)
    state, z_state = _prepare_state(q, v, initial_state)
    outputs = []
    for t in range(q.shape[1]):
        state = state + torch.einsum("bhk,bhv->bhkv", kf[:, t], vf[:, t])
        z_state = z_state + kf[:, t]
        numerator = torch.einsum("bhk,bhkv->bhv", qf[:, t] * scale, state)
        denominator = (qf[:, t] * scale * z_state).sum(dim=-1, keepdim=True)
        outputs.append(numerator / (denominator + eps))
    output = torch.stack(outputs, dim=1).to(dtype)
    final_state = (state, z_state) if output_final_state else None
    return output, final_state


def normalized_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    chunk_size: int = 64,
    scale: Optional[float] = None,
    initial_state: Optional[State] = None,
    eps: float = 1e-10,
    output_final_state: bool = True,
) -> Tuple[torch.Tensor, Optional[State]]:
    """Chunked reference with inter-chunk state and intra-chunk causality."""
    _check_inputs(q, k, v)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    dtype = q.dtype
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    qf, kf, vf = (x.to(torch.float32) for x in (q, k, v))
    qf = elu_plus_one(qf)
    kf = elu_plus_one(kf)
    state, z_state = _prepare_state(q, v, initial_state)
    outputs = []
    for start in range(0, q.shape[1], chunk_size):
        end = min(start + chunk_size, q.shape[1])
        q_chunk = qf[:, start:end] * scale
        k_chunk = kf[:, start:end]
        v_chunk = vf[:, start:end]
        inter = torch.einsum("bchk,bhkv->bchv", q_chunk, state)
        causal_scores = torch.einsum("bchk,bshk->bchs", q_chunk, k_chunk)
        length = end - start
        lower = torch.tril(
            torch.ones(length, length, device=q.device, dtype=torch.bool)
        )
        causal_scores = causal_scores.masked_fill(~lower[None, :, None, :], 0.0)
        intra = torch.einsum("bchs,bshv->bchv", causal_scores, v_chunk)
        k_cumulative = k_chunk.cumsum(dim=1) + z_state[:, None]
        denominator = (q_chunk * k_cumulative).sum(dim=-1, keepdim=True)
        outputs.append((inter + intra) / (denominator + eps))
        state = state + torch.einsum("bchk,bchv->bhkv", k_chunk, v_chunk)
        z_state = z_state + k_chunk.sum(dim=1)
    output = torch.cat(outputs, dim=1).to(dtype)
    final_state = (state, z_state) if output_final_state else None
    return output, final_state

