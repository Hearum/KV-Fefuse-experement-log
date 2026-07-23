"""Tiled Triton selected-query block kernel for queries sharing a block."""
import torch
import triton
import triton.language as tl


@triton.jit
def _tiled_kernel(q, k, v, ps, pz, pos, out, hq, hkv, nk, d, dv, key_start, scale, eps,
                  sqb, sqh, sqm, sqd, skb, skh, skn, skd, svb, svh, svn, svv,
                  spb, sph, spd, spv, szb, szh, szd, sob, soh, som, sov,
                  spb_pos, spm_pos, m_count, GROUP: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr,
                  BV: tl.constexpr, BN: tl.constexpr):
    b = tl.program_id(0)
    mt = tl.program_id(1)
    kvh = tl.program_id(2)
    m = mt * BM + tl.arange(0, BM)
    dd = tl.arange(0, BD)
    vv = tl.arange(0, BV)
    valid_m = m < m_count
    qpos = tl.load(pos + b * spb_pos + m * spm_pos, mask=valid_m, other=0)
    # One prefix state is shared by all rows in this tile.
    s = tl.load(ps + b * spb + kvh * sph + dd[:, None] * spd + vv[None, :] * spv,
                mask=(dd[:, None] < d) & (vv[None, :] < dv), other=0.0)
    z = tl.load(pz + b * szb + kvh * szh + dd * szd, mask=dd < d, other=0.0)
    for g in range(GROUP):
        qptr = q + b * sqb + (kvh * GROUP + g) * sqh + m[:, None] * sqm + dd[None, :] * sqd
        qq = tl.load(qptr, mask=valid_m[:, None] & (dd[None, :] < d), other=0.0).to(tl.float32)
        qq = tl.where(qq >= 0, qq + 1, tl.exp(qq)) * scale
        num = tl.dot(qq, s)
        den = tl.sum(qq * z[None, :], axis=1) + eps
        for start in range(0, BN, BN):
            n = start + tl.arange(0, BN)
            absolute_n = key_start + n
            valid_n = (n < nk) & (absolute_n <= qpos[:, None])
            kp = k + b * skb + kvh * skh + n[None, :] * skn + dd[:, None] * skd
            vp = v + b * svb + kvh * svh + n[:, None] * svn + vv[None, :] * svv
            kval = tl.load(kp, mask=(n[None, :] < nk) & (dd[:, None] < d), other=0.0).to(tl.float32)
            kval = tl.where(kval >= 0, kval + 1, tl.exp(kval))
            vval = tl.load(vp, mask=(n[:, None] < nk) & (vv[None, :] < dv), other=0.0).to(tl.float32)
            scores = tl.dot(qq, kval)
            scores = tl.where(valid_n, scores, 0.0)
            num += tl.dot(scores, vval)
            den += tl.sum(scores, axis=1)
        qptr = out + b * sob + (kvh * GROUP + g) * soh + m[:, None] * som + vv[None, :] * sov
        tl.store(qptr, num / den[:, None], mask=valid_m[:, None] & (vv[None, :] < dv))


def tiled_block_attention(q, k, v, positions, *, prefix_s, prefix_z,
                          key_start, scale=None, eps=1e-10, block_size=32):
    """Compute a contiguous group of queries against one block with tiled Q GEMM."""
    if not q.is_cuda:
        raise RuntimeError("Triton CUDA required")
    q, k, v = [x.contiguous() for x in (q, k, v)]
    ps, pz, positions = prefix_s.contiguous().float(), prefix_z.contiguous().float(), positions.contiguous().long()
    b, hq, m, d = q.shape
    hkv, nk = k.shape[1], k.shape[2]
    if hq % hkv or positions.shape != (b, m):
        raise ValueError("invalid tiled GQA shape")
    dv = v.shape[-1]; group = hq // hkv
    out = torch.empty(b, hq, m, dv, device=q.device, dtype=torch.float32)
    _tiled_kernel[(b, triton.cdiv(m, 32), hkv)](
        q, k, v, ps, pz, positions, out, hq, hkv, nk, d, dv, key_start,
        d ** -0.5 if scale is None else scale, eps, *positions.stride(),
        *q.stride(), *k.stride(), *v.stride(), *ps.stride(), *pz.stride(),
        *out.stride(), m_count=m, GROUP=group, BM=32, BD=triton.next_power_of_2(d),
        BV=triton.next_power_of_2(dv), BN=max(16, triton.next_power_of_2(nk)),
    )
    return out.to(q.dtype)
