"""Self-contained selected-query normalized causal linear attention prototype.

Layout is [B, H, T, D]. K/V keep H_kv heads; Q heads are grouped over them.
The feature map is phi(x)=ELU(x)+1 and the causal state is
S=sum(phi(k)^T v), z=sum(phi(k)).  ``key_start`` and ``prefix_s/z`` allow a
+ caller to evaluate a suffix against a precomputed prefix state.
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU/reference-only environments
    triton = None
    tl = None


def feature_map(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.elu(x.float()) + 1.0


def reference_attention(q, k, v, query_positions=None, *, key_start=0,
                        prefix_s=None, prefix_z=None, scale=None, eps=1e-10):
    """Float32 reference; output has Q's dtype and shape [B,Hq,Q,Dv]."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/v must be [B,H,T,D]")
    b, hq, nq, d = q.shape
    bk, hkv, nk, dk = k.shape
    if b != bk or k.shape[:2] != v.shape[:2] or d != dk:
        raise ValueError("incompatible q/k/v shapes")
    if hq % hkv:
        raise ValueError("q_heads must be divisible by kv_heads")
    if query_positions is None:
        pos = torch.arange(nq, device=q.device, dtype=torch.long) + key_start
        pos = pos.unsqueeze(0).expand(b, -1)
    else:
        pos = query_positions.to(q.device, torch.long)
        if pos.ndim == 1 and pos.numel() == nq:
            pos = pos.unsqueeze(0).expand(b, -1)
        elif pos.ndim != 2 or tuple(pos.shape) != (b, nq):
            raise ValueError("query_positions must be [Q] or [B,Q]")
    if pos.numel() and (pos.min() < key_start or pos.max() >= key_start + nk):
        raise ValueError("query_positions must refer to the supplied K/V span")
    scale = d ** -0.5 if scale is None else scale
    s = torch.zeros(b, hkv, d, v.shape[-1], device=q.device, dtype=torch.float32)
    z = torch.zeros(b, hkv, d, device=q.device, dtype=torch.float32)
    if prefix_s is not None:
        s = prefix_s.to(q.device, torch.float32).clone()
        z = prefix_z.to(q.device, torch.float32).clone()
    out = torch.empty(b, hq, nq, v.shape[-1], device=q.device, dtype=torch.float32)
    groups = hq // hkv
    qphi = feature_map(q) * scale
    kphi = feature_map(k)
    for qi in range(nq):
        ends = pos[:, qi] - key_start + 1
        ss = s.clone()
        zz = z.clone()
        for bi in range(b):
            end = int(ends[bi].item())
            ss[bi] += torch.einsum("hnd,hnv->hdv", kphi[bi, :, :end], v[bi, :, :end].float())
            zz[bi] += kphi[bi, :, :end].sum(1)
        ss = ss.repeat_interleave(groups, 1)
        zz = zz.repeat_interleave(groups, 1)
        out[:, :, qi] = torch.einsum("bhd,bhdv->bhv", qphi[:, :, qi], ss) / (
            torch.einsum("bhd,bhd->bh", qphi[:, :, qi], zz)[..., None] + eps)
    return out.to(q.dtype)


if triton is not None:
    @triton.jit
    def _selected_kernel(q, k, v, out, prefix_s, prefix_z, positions,
                         hq, hkv, nk, nq, d_model, value_dim, key_start, scale, eps,
                         sqb, sqh, sqn, sqd, skb, skh, skn, skd,
                         svb, svh, svn, svv, sob, soh, son, sov,
                         spsb, spsh, spsd, spsv, szb, szh, szd,
                         BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_N: tl.constexpr,
                         HAS_PREFIX: tl.constexpr):
        pid = tl.program_id(0)
        bidx = tl.program_id(0)
        qidx = tl.program_id(1)
        qhead = tl.program_id(2)
        kvhead = qhead // (hq // hkv)
        d = tl.arange(0, BLOCK_D)
        vv = tl.arange(0, BLOCK_V)
        qptr = q + bidx * sqb + qhead * sqh + qidx * sqn + d * sqd
        qq = tl.where(d < BLOCK_D, tl.load(qptr, mask=d < BLOCK_D, other=0.0).to(tl.float32), 0.0)
        qq = tl.where(qq >= 0, qq + 1, tl.exp(qq)) * scale
        ss = tl.zeros((BLOCK_D, BLOCK_V), tl.float32)
        zz = tl.zeros((BLOCK_D,), tl.float32)
        if HAS_PREFIX:
            sp = prefix_s + bidx * spsb + kvhead * spsh + d[:, None] * spsd + vv[None, :] * spsv
            zp = prefix_z + bidx * szb + kvhead * szh + d * szd
            ss += tl.load(sp, mask=(d[:, None] < d_model) & (vv[None, :] < value_dim), other=0.0)
            zz += tl.load(zp, mask=d < BLOCK_D, other=0.0)
        qpos = tl.load(positions + bidx * nq + qidx)
        for start in range(0, nk, BLOCK_N):
            n = start + tl.arange(0, BLOCK_N)
            valid = (n < nk) & (key_start + n <= qpos)
            kp = k + bidx * skb + kvhead * skh + n[None, :] * skn + d[:, None] * skd
            vp = v + bidx * svb + kvhead * svh + n[:, None] * svn + vv[None, :] * svv
            kval = tl.load(kp, mask=(d[:, None] < d_model) & (n[None, :] < nk), other=0.0).to(tl.float32)
            kval = tl.where(kval >= 0, kval + 1, tl.exp(kval))
            kval = tl.where(valid[None, :], kval, 0.0)
            vval = tl.load(vp, mask=(n[:, None] < nk) & (vv[None, :] < value_dim), other=0.0).to(tl.float32)
            vval = tl.where(valid[:, None], vval, 0.0)
            if BLOCK_D >= 16 and BLOCK_V >= 16:
                ss += tl.dot(kval, vval)
            else:
                # Contract the scan dimension explicitly for small feature
                # or value dimensions.
                ss += tl.sum(kval[:, :, None] * vval[None, :, :], axis=1)
            zz += tl.sum(kval, axis=1)
        num = tl.sum(qq[:, None] * ss, axis=0)
        den = tl.sum(qq * zz, axis=0) + eps
        op = out + bidx * sob + qhead * soh + qidx * son + vv * sov
        tl.store(op, num / den, mask=vv < value_dim)


def triton_attention(q, k, v, query_positions=None, *, key_start=0,
                     prefix_s=None, prefix_z=None, scale=None, eps=1e-10):
    """Triton selected-query kernel. Falls back only when Triton is unavailable."""
    if triton is None or not q.is_cuda:
        return reference_attention(q, k, v, query_positions, key_start=key_start,
                                   prefix_s=prefix_s, prefix_z=prefix_z, scale=scale, eps=eps)
    if not all(x.is_contiguous() for x in (q, k, v)):
        q, k, v = [x.contiguous() for x in (q, k, v)]
    b, hq, nq, d = q.shape
    if d > 128 or v.shape[-1] > 128 or hq % k.shape[1]:
        raise ValueError("prototype supports head/value dimensions <=128 and GQA")
    if query_positions is None:
        pos = torch.arange(nq, device=q.device, dtype=torch.long) + key_start
    else:
        pos = query_positions.to(q.device, torch.long).contiguous()
        if pos.ndim == 1 and pos.numel() != nq:
            raise ValueError("query_positions [Q] must match q.shape[2]")
        if pos.ndim == 2 and tuple(pos.shape) != (b, nq):
            raise ValueError("query_positions [B,Q] must match batch and q length")
        if pos.ndim not in (1, 2):
            raise ValueError("query_positions must be [Q] or [B,Q]")
    if pos.ndim == 1:
        pos = pos.unsqueeze(0).expand(b, -1).contiguous()
    if prefix_s is None:
        prefix_s = torch.zeros(b, k.shape[1], d, v.shape[-1], device=q.device, dtype=torch.float32)
        prefix_z = torch.zeros(b, k.shape[1], d, device=q.device, dtype=torch.float32)
        has_prefix = False
    else:
        prefix_s, prefix_z = prefix_s.contiguous().float(), prefix_z.contiguous().float()
        has_prefix = True
    out = torch.empty(b, hq, nq, v.shape[-1], device=q.device, dtype=torch.float32)
    scale = d ** -0.5 if scale is None else scale
    _selected_kernel[(b, nq, hq)](
        q, k, v, out, prefix_s, prefix_z, pos, hq, k.shape[1], k.shape[2], nq, d, v.shape[-1], key_start,
        scale, eps, *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        *prefix_s.stride(), *prefix_z.stride(), BLOCK_D=triton.next_power_of_2(d),
        BLOCK_V=triton.next_power_of_2(v.shape[-1]), BLOCK_N=128, HAS_PREFIX=has_prefix,
    )
    return out.to(q.dtype)


def linear_attention(*args, implementation="auto", **kwargs):
    if implementation == "reference" or (implementation == "auto" and not args[0].is_cuda):
        return reference_attention(*args, **kwargs)
    try:
        return triton_attention(*args, **kwargs)
    except Exception as exc:
        if implementation == "triton":
            raise
        print(f"[selected_query] auto fallback to reference: {type(exc).__name__}: {exc}")
        return reference_attention(*args, **kwargs)


def selected_query_attention(q, k, v, query_positions, *, key_start=0,
                             prefix_s=None, prefix_z=None, scale=None,
                             eps=1e-10, implementation="auto"):
    """Qwen3 document-reprocess API; layouts are [B,H,Q,D] and [B,Hkv,S,D]."""
    return linear_attention(
        q, k, v, query_positions, key_start=key_start,
        prefix_s=prefix_s, prefix_z=prefix_z, scale=scale, eps=eps,
        implementation=implementation,
    )
