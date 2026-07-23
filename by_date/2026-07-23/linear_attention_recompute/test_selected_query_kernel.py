"""Correctness gate for the local Qwen3 selected-query operator."""
import argparse
import torch

from selected_query_kernel import selected_query_attention, reference_attention


def run(device, implementation, dtype):
    torch.manual_seed(17)
    b, hq, hkv, d, dv, s = 2, 8, 2, 16, 12, 23
    q = torch.randn(b, hq, 5, d, device=device, dtype=dtype)
    k = torch.randn(b, hkv, s, d, device=device, dtype=dtype)
    v = torch.randn(b, hkv, s, dv, device=device, dtype=dtype)
    positions = torch.tensor([[0, 3, 7, 12, 22], [1, 4, 8, 15, 21]], device=device)
    ref = reference_attention(q, k, v, positions, key_start=0)
    got = selected_query_attention(q, k, v, positions, implementation=implementation)
    err = (got.float() - ref.float()).abs().max().item()
    print(f"positions[B,Q] implementation={implementation} max_error={err:.6g}")
    limit = 1e-5 if q.dtype == torch.float32 else 1e-2
    assert err <= limit

    prefix = 7
    phi = torch.nn.functional.elu(k[:, :, :prefix].float()) + 1
    ps = torch.einsum("bhnd,bhnv->bhdv", phi, v[:, :, :prefix].float())
    pz = phi.sum(2)
    suffix_q = q[:, :, 1:4]
    suffix_pos = torch.tensor([9, 15, 22], device=device)
    ref = reference_attention(suffix_q, k[:, :, prefix:], v[:, :, prefix:], suffix_pos,
                              key_start=prefix, prefix_s=ps, prefix_z=pz)
    got = selected_query_attention(suffix_q, k[:, :, prefix:], v[:, :, prefix:], suffix_pos,
                                   key_start=prefix, prefix_s=ps, prefix_z=pz,
                                   implementation=implementation)
    err = (got.float() - ref.float()).abs().max().item()
    print(f"prefix+noncontiguous implementation={implementation} max_error={err:.6g}")
    assert err <= limit


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--implementation", default="auto", choices=["auto", "reference", "triton"])
    p.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    args = p.parse_args()
    run(torch.device(args.device), args.implementation, getattr(torch, args.dtype))
