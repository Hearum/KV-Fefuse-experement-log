# Qwen3 selected-query Linear Attention reprocess

本目录实现 document reprocess 使用的 selected-query 接口，不修改主 pipeline 或共享 operator。

接口：`q [B,Hq,Q,D]`、`k/v [B,Hkv,S,D]`、绝对 `query_positions [Q]` 或 `[B,Q]`，可选 `prefix_s [B,Hkv,D,Dv]` 与 `prefix_z [B,Hkv,D]`，输出 `[B,Hq,Q,Dv]`。默认 `phi(x)=ELU(x)+1`，float32 state/accumulator，支持 GQA、非连续位置和 causal prefix。

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python test_selected_query_kernel.py --device cpu --implementation reference
PYTHONPATH=/tmp/hming-fla-venv/lib/python3.10/site-packages:/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/lib/python3.10/site-packages LIBRARY_PATH=/tmp/hming-triton-lib CUDA_VISIBLE_DEVICES=1 /tmp/hming-fla-venv/bin/python test_selected_query_kernel.py --device cuda --implementation reference
```

当前 reference 和 Triton 均覆盖 Qwen3 GQA、B>1、prefix 与 `[B,Q]` positions。此前多 head/batch 错误的根因是 `BLOCK_V` 大于实际 `Dv` 时 output store 越界，已改为 `vv < value_dim`；修复后 FP16 GPU max error 为 9.77e-4，prefix 为 4.88e-4。FP32 Triton 仍需单独优化 exp/dot 数值误差，不能把该误差写成 reference 等价。

融合边界：Q/K/V projection、Q/K RMSNorm、RoPE、cache_position、cache 写回和 o_proj 由 Qwen3 外层负责；本接口只做 selected document attention read。后续 question/query prefill 继续走 dense SDPA。

## Block-summary selected mode

新增 `block_summary_selected.py`。`build_block_prefix_states` 为每个 block 保存 block 之前的 `S/z`，selected query 只把所在 block 的 K/V 交给 Triton，并通过 `prefix_s/prefix_z` 读取历史 summary；不再扫描完整 S。该模式支持离线保存的 BF16 summary 作为输入，summary 的计算仍使用 float32 accumulator。

复测（H20、FP16、B=1、Hq=32、Hkv=8、D=Dv=128，selected query 为末尾连续 token）：

| S | Q | SDPA ms | block-linear ms |
|---:|---:|---:|---:|
| 1024 | 1 | 0.090 | 0.402 |
| 1024 | 16 | 0.089 | 0.497 |
| 4096 | 1 | 0.299 | 0.352 |
| 4096 | 16 | 0.303 | 0.475 |
| 8192 | 1 | 0.571 | 0.334 |
| 8192 | 16 | 0.591 | 0.460 |

block summary correctness（S=23、block=8、GQA、FP16）max error 为 9.77e-4。Q=64/256 时当前 Python 分组和 32-head output launch overhead 仍使 linear 慢于 SDPA，后续需要 fused multi-query block output。

优化后，block adapter 预打包 contiguous K/V blocks，并对连续 selected query 使用 slice fast path，避免重复 `index_select` 和 contiguous copy。S=4096,Q=1/16 为 0.267/0.429 ms；S=8192,Q=1/16 为 0.253/0.390 ms。

新增 `block_summary_selected_attention_matmul`：对当前 block 内的多个 query 一次构造局部 `[Q, block_size]` 计算，不构造完整 `[Q,S]` 矩阵。S=8192 时，Q=64 为 0.809 ms、Q=256 为 2.628 ms，相比逐 query Triton 的 0.957/3.739 ms 更快；局部 matmul correctness max error=0。由于临时局部矩阵和 PyTorch matmul workspace，显存高于 Triton path，当前仍作为 Q 较大时的候选 backend。

新增 batched GQA block backend：将 selected query 按 block id 批量 gather，按 Hkv=8 做状态/局部计算，避免分散 query 的 Python block loop，也不 repeat KV 到 Hq。H20、S=8192、block=64、FP16 的 uniform positions（ms，SDPA / 逐 block Triton / local matmul / batched GQA）为：Q=64，0.637 / 13.923 / 33.365 / 1.524；Q=256，0.871 / 28.587 / 73.291 / 5.204。末尾连续 positions 则为 Q=64，0.626 / 0.987 / 0.910 / 1.561；Q=256，0.877 / 3.780 / 2.873 / 5.664。因此当前应按 query 分布选择 backend：连续尾部使用 Triton/local matmul，分散位置使用 batched GQA；后者已大幅消除 Python launch 放大，但仍有 gather 和 float32 workspace 开销。

FLA 对照：v0.3 的 `chunk_simple_gla` 是可复用的 chunk state-scan，`chunk_linear_attn` 主要是在其上加 normalization；其 forward output kernel 的 head 维度仍按 Q/K/V 同头数设计，不能直接提供本项目需要的 Hq=32、Hkv=8、绝对位置 selected-query GQA。故本目录复用了 FLA 的 state-scan 思路，另写了 grouped/batched output 适配层，而没有简单 `repeat_kv`。

最终 selected-query 路径已加入 `triton_block_attention`：单次 `(B,Q,Hq)` launch，kernel 内根据 absolute query position 计算 block id，直接读取 `[B,Nb,Hkv,L,D]` block K/V 与 `[B,Nb,Hkv,D,Dv]` prefix state。PyTorch batched gather/einsum 仅保留作语义对照，不作为最终 backend。H20、S=8192、FP16、block=64 的 fused Triton（ms）为：连续尾部 Q=1/16/64/256：0.165/0.179/0.315/1.077；uniform positions：0.165/0.170/0.337/1.133。
