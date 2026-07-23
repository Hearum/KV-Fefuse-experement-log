"""Block-summary selected-query adapter for Qwen3 document reprocess."""
import torch

from selected_query_kernel import selected_query_attention


def build_block_prefix_states(k, v, block_size=64):
    """Return state immediately before every block: S [B,Nb,Hkv,D,Dv], z [B,Nb,Hkv,D]."""
    if k.ndim != 4 or v.ndim != 4 or k.shape[:3] != v.shape[:3]:
        raise ValueError("k/v must be [B,Hkv,S,D] and [B,Hkv,S,Dv]")
    b, h, s, d = k.shape
    nb = (s + block_size - 1) // block_size
    ps = torch.zeros(b, nb, h, d, v.shape[-1], device=k.device, dtype=torch.float32)
    pz = torch.zeros(b, nb, h, d, device=k.device, dtype=torch.float32)
    state = torch.zeros(b, h, d, v.shape[-1], device=k.device, dtype=torch.float32)
    norm = torch.zeros(b, h, d, device=k.device, dtype=torch.float32)
    phi = torch.nn.functional.elu(k.float()) + 1
    for block in range(nb):
        ps[:, block] = state
        pz[:, block] = norm
        start, end = block * block_size, min((block + 1) * block_size, s)
        state = state + torch.einsum("bhsd,bhsv->bhdv", phi[:, :, start:end], v[:, :, start:end].float())
        norm = norm + phi[:, :, start:end].sum(2)
    return ps, pz


def block_summary_selected_attention(q, k, v, query_positions, *, block_size=64,
                                     block_prefix_s, block_prefix_z,
                                     block_k=None, block_v=None,
                                     implementation="auto", eps=1e-10):
    """Evaluate selected queries using only their local block plus prior summaries."""
    b, hq, nq, _ = q.shape
    hkv, s = k.shape[1], k.shape[2]
    if query_positions.ndim == 1:
        query_positions = query_positions.unsqueeze(0).expand(b, -1)
    if tuple(query_positions.shape) != (b, nq):
        raise ValueError("query_positions must be [Q] or [B,Q]")
    out = torch.empty(b, hq, nq, v.shape[-1], device=q.device, dtype=q.dtype)
    for bi in range(b):
        groups = {}
        for qi, pos in enumerate(query_positions[bi].tolist()):
            groups.setdefault(int(pos) // block_size, []).append(qi)
        for block, indices in groups.items():
            start, end = block * block_size, min((block + 1) * block_size, s)
            idx = torch.tensor(indices, device=q.device, dtype=torch.long)
            if indices == list(range(indices[0], indices[0] + len(indices))):
                q_part = q[bi:bi + 1, :, indices[0]:indices[-1] + 1]
                pos_part = query_positions[bi, indices[0]:indices[-1] + 1]
            else:
                q_part = q[bi:bi + 1].index_select(2, idx)
                pos_part = query_positions[bi].index_select(0, idx)
            kb = (block_k[block] if block_k is not None else k[bi:bi + 1, :, start:end].contiguous())
            vb = (block_v[block] if block_v is not None else v[bi:bi + 1, :, start:end].contiguous())
            part = selected_query_attention(
                q_part, kb, vb,
                pos_part, key_start=start,
                prefix_s=block_prefix_s[bi:bi + 1, block],
                prefix_z=block_prefix_z[bi:bi + 1, block],
                eps=eps, implementation=implementation,
            )
            out[bi:bi + 1, :, idx] = part
    return out


def block_summary_selected_attention_matmul(q, k, v, query_positions, *, block_size=64,
                                            block_prefix_s, block_prefix_z, eps=1e-10):
    """Vectorized local-block path; never materializes a full [Q,S] matrix."""
    b, hq, nq, d = q.shape
    hkv = k.shape[1]
    if query_positions.ndim == 1:
        query_positions = query_positions.unsqueeze(0).expand(b, -1)
    out = torch.empty(b, hq, nq, v.shape[-1], device=q.device, dtype=q.dtype)
    group = hq // hkv
    qf = (torch.nn.functional.elu(q.float()) + 1) * (d ** -0.5)
    kf = torch.nn.functional.elu(k.float()) + 1
    for bi in range(b):
        groups = {}
        for qi, pos in enumerate(query_positions[bi].tolist()):
            groups.setdefault(int(pos) // block_size, []).append(qi)
        for block, indices in groups.items():
            start, end = block * block_size, min((block + 1) * block_size, k.shape[2])
            idx = torch.tensor(indices, device=q.device, dtype=torch.long)
            qpart = qf[bi:bi + 1].index_select(2, idx)
            pospart = query_positions[bi].index_select(0, idx)
            kp = kf[bi:bi + 1, :, start:end]
            vp = v[bi:bi + 1, :, start:end].float()
            kp = kp[:, torch.arange(hq, device=q.device) // group]
            scores = torch.matmul(qpart, kp.transpose(-1, -2))
            causal = torch.arange(start, end, device=q.device)[None, None, :] <= pospart[None, :, None]
            scores = scores.masked_fill(~causal[:, None], 0.0)
            ps = block_prefix_s[bi:bi + 1, block][:, torch.arange(hq, device=q.device) // group]
            pz = block_prefix_z[bi:bi + 1, block][:, torch.arange(hq, device=q.device) // group]
            numerator = torch.matmul(qpart, ps) + torch.matmul(scores, vp[:, torch.arange(hq, device=q.device) // group])
            denominator = (qpart * pz.unsqueeze(2)).sum(-1, keepdim=True) + scores.sum(-1, keepdim=True)
            out[bi:bi + 1, :, idx] = (numerator / (denominator + eps)).to(out.dtype)
    return out.to(q.dtype)
