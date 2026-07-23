import argparse
import time
import torch
import torch.nn.functional as F

from block_summary_selected import (build_block_prefix_states, block_summary_selected_attention,
                                    block_summary_selected_attention_matmul,
                                    block_summary_selected_attention_batched,
                                    block_summary_selected_attention_fused)


def measure(fn, warmup, iters):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters, torch.cuda.max_memory_allocated() / 2**20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", nargs="+", type=int, default=[1024, 4096, 8192])
    p.add_argument("--queries", nargs="+", type=int, default=[1, 16, 64, 256])
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--distribution", choices=["final", "clustered", "uniform", "random"], default="final")
    p.add_argument("--query-chunk", type=int, default=32)
    a = p.parse_args()
    print(f"B=1 Hq=32 Hkv=8 D=Dv=128 FP16; selected distribution={a.distribution}")
    print("length,Q,sdpa_ms,triton_block_ms,matmul_block_ms,batched_gqa_ms,fused_triton_ms,sdpa_MiB,triton_MiB,matmul_MiB,batched_gqa_MiB,fused_triton_MiB")
    for s in a.lengths:
        k = torch.randn(1, 8, s, 128, device="cuda", dtype=torch.float16) * .1
        v = torch.randn(1, 8, s, 128, device="cuda", dtype=torch.float16)
        kq, vq = k.repeat_interleave(4, 1), v.repeat_interleave(4, 1)
        ps, pz = build_block_prefix_states(k, v, a.block_size)
        block_k = [k[:, :, start:min(start + a.block_size, s)].contiguous()
                   for start in range(0, s, a.block_size)]
        block_v = [v[:, :, start:min(start + a.block_size, s)].contiguous()
                   for start in range(0, s, a.block_size)]
        nb = len(block_k)
        block_kt = torch.zeros(1, nb, 8, a.block_size, 128, device="cuda", dtype=k.dtype)
        block_vt = torch.zeros(1, nb, 8, a.block_size, 128, device="cuda", dtype=v.dtype)
        for block, (kb, vb) in enumerate(zip(block_k, block_v)):
            block_kt[:, block, :, :kb.shape[2]] = kb
            block_vt[:, block, :, :vb.shape[2]] = vb
        for nq in a.queries:
            if a.distribution == "final":
                qpos = torch.arange(s - nq, s, device="cuda")
            elif a.distribution == "clustered":
                low = max(0, s - 4 * a.block_size)
                qpos = torch.randperm(s - low, device="cuda")[:nq].sort().values + low
            elif a.distribution == "uniform":
                qpos = torch.linspace(0, s - 1, nq, device="cuda").round().long().unique()
                if qpos.numel() < nq:
                    qpos = torch.arange(nq, device="cuda")
            else:
                qpos = torch.randperm(s, device="cuda")[:nq].sort().values
            q = torch.randn(1, 32, nq, 128, device="cuda", dtype=torch.float16) * .1
            mask = torch.arange(s, device="cuda")[None, None, :] <= qpos[None, :, None]
            sdpa = lambda: F.scaled_dot_product_attention(q, kq, vq, attn_mask=mask)
            linear = lambda: block_summary_selected_attention(q, k, v, qpos, block_size=a.block_size, block_prefix_s=ps, block_prefix_z=pz, block_k=block_k, block_v=block_v, implementation="triton")
            matmul = lambda: block_summary_selected_attention_matmul(q, k, v, qpos, block_size=a.block_size, block_prefix_s=ps, block_prefix_z=pz)
            batched = lambda: block_summary_selected_attention_batched(q, k, v, qpos, block_size=a.block_size, block_prefix_s=ps, block_prefix_z=pz, block_k=block_kt, block_v=block_vt, query_chunk=a.query_chunk)
            fused = lambda: block_summary_selected_attention_fused(q, k, v, qpos, block_size=a.block_size, block_prefix_s=ps, block_prefix_z=pz, block_k=block_kt, block_v=block_vt)
            sm, sp = measure(sdpa, a.warmup, a.iters)
            lm, lp = measure(linear, a.warmup, a.iters)
            mm, mp = measure(matmul, a.warmup, a.iters)
            bm, bp = measure(batched, a.warmup, a.iters)
            fm, fp = measure(fused, a.warmup, a.iters)
            print(f"{s},{nq},{sm:.3f},{lm:.3f},{mm:.3f},{bm:.3f},{fm:.3f},{sp:.1f},{lp:.1f},{mp:.1f},{bp:.1f},{fp:.1f}")


if __name__ == "__main__": main()
