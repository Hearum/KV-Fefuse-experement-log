import argparse
import time
import torch
import torch.nn.functional as F

from block_summary_selected import build_block_prefix_states, block_summary_selected_attention, block_summary_selected_attention_matmul


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
    a = p.parse_args()
    print("B=1 Hq=32 Hkv=8 D=Dv=128 FP16; selected queries are the final contiguous Q positions")
    print("length,Q,sdpa_ms,triton_block_ms,matmul_block_ms,sdpa_MiB,triton_MiB,matmul_MiB")
    for s in a.lengths:
        k = torch.randn(1, 8, s, 128, device="cuda", dtype=torch.float16) * .1
        v = torch.randn(1, 8, s, 128, device="cuda", dtype=torch.float16)
        kq, vq = k.repeat_interleave(4, 1), v.repeat_interleave(4, 1)
        ps, pz = build_block_prefix_states(k, v, a.block_size)
        block_k = [k[:, :, start:min(start + a.block_size, s)].contiguous()
                   for start in range(0, s, a.block_size)]
        block_v = [v[:, :, start:min(start + a.block_size, s)].contiguous()
                   for start in range(0, s, a.block_size)]
        for nq in a.queries:
            qpos = torch.arange(s - nq, s, device="cuda")
            q = torch.randn(1, 32, nq, 128, device="cuda", dtype=torch.float16) * .1
            mask = torch.arange(s, device="cuda")[None, None, :] <= qpos[None, :, None]
            sdpa = lambda: F.scaled_dot_product_attention(q, kq, vq, attn_mask=mask)
            linear = lambda: block_summary_selected_attention(q, k, v, qpos, block_size=a.block_size, block_prefix_s=ps, block_prefix_z=pz, block_k=block_k, block_v=block_v, implementation="triton")
            matmul = lambda: block_summary_selected_attention_matmul(q, k, v, qpos, block_size=a.block_size, block_prefix_s=ps, block_prefix_z=pz)
            sm, sp = measure(sdpa, a.warmup, a.iters)
            lm, lp = measure(linear, a.warmup, a.iters)
            mm, mp = measure(matmul, a.warmup, a.iters)
            print(f"{s},{nq},{sm:.3f},{lm:.3f},{mm:.3f},{sp:.1f},{lp:.1f},{mp:.1f}")


if __name__ == "__main__": main()
