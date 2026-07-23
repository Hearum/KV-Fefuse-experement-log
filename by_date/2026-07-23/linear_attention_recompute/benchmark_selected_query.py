"""Qwen3 selected-document prefill benchmark."""
import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from selected_query_kernel import selected_query_attention


def samples(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        vals.append((time.perf_counter() - t0) * 1000)
    p95_index = max(0, int(.95 * len(vals)) - 1)
    return statistics.mean(vals), statistics.median(vals), sorted(vals)[p95_index], torch.cuda.max_memory_allocated() / 2**20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", nargs="+", type=int, default=[1024, 4096, 8192])
    p.add_argument("--queries", nargs="+", type=int, default=[1, 16, 64, 256])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    print("Qwen3 selected document: B=1 Hq=32 Hkv=8 D=Dv=128 FP16; absolute positions")
    print("length,Q,sdpa_mean,sdpa_p50,sdpa_p95,linear_mean,linear_p50,linear_p95,sdpa_MiB,linear_MiB")
    for length in args.lengths:
        k = torch.randn(1, 8, length, 128, device="cuda", dtype=torch.float16) * .1
        v = torch.randn(1, 8, length, 128, device="cuda", dtype=torch.float16)
        kq, vq = k.repeat_interleave(4, 1), v.repeat_interleave(4, 1)
        for nq in args.queries:
            if nq > length:
                continue
            qpos = torch.linspace(0, length - 1, nq, device="cuda").round().long()
            q = torch.randn(1, 32, nq, 128, device="cuda", dtype=torch.float16) * .1
            mask = torch.arange(length, device="cuda")[None, None, :] <= qpos[None, :, None]
            sdpa = lambda: F.scaled_dot_product_attention(q, kq, vq, attn_mask=mask)
            linear = lambda: selected_query_attention(q, k, v, qpos, implementation="triton")
            sm, s50, s95, sp = samples(sdpa, args.warmup, args.iters)
            lm, l50, l95, lp = samples(linear, args.warmup, args.iters)
            print(f"{length},{nq},{sm:.3f},{s50:.3f},{s95:.3f},{lm:.3f},{l50:.3f},{l95:.3f},{sp:.1f},{lp:.1f}")


if __name__ == "__main__":
    main()
