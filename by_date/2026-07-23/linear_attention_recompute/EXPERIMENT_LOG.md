# 实验日志

## 2026-07-24：selected-query API Phase 0

- 阅读 `DESIGN_UPDATE_20260723.md`、`PLAN.md`、`RESEARCH_NOTES.md`、`REQUIREMENTS.md`；保持 selected document-only、question dense MHA、global absolute position、GQA 和 prefix state 语义。
- 新增 `selected_query_kernel.py`，独立于 `ktransformers/operators/linear_attention.py`；提供 reference、Triton/auto/triton 显式接口、`[Q]`/`[B,Q]` positions、prefix S/z。
- 新增 `test_selected_query_kernel.py` 和 `benchmark_selected_query.py`。
- reference CPU correctness：GQA、`[B,Q]`、prefix、非连续 positions max_error=0。
- 发现旧 selected Triton prototype 存在隐蔽问题：`Hq>1`/`B>1` 的 launch 结果错误，且小 Dv store mask 会越界；已修复 store mask，并对未通过的多头/多 batch Triton 形状显式报错，防止 auto fallback 被误报为 Triton 正确性。
- 当前状态：reference API 可作为 Qwen3 接入基线；native multi-head selected Triton 仍需单独融合 kernel 后才能进入性能 benchmark。
- 修复 selected Triton 的真实根因：输出 store 使用 `mask=vv < BLOCK_V`，当 Dv 不是 BLOCK_V 时越界写出 tensor，污染相邻 head/batch；改为 `mask=vv < value_dim`。恢复单次 `(B,Q,Hq)` launch 后，B=2/Hq=8/Hkv=2 FP16：普通 positions max_error=9.77e-4，prefix+noncontiguous max_error=4.88e-4。
- 当前 Triton FP32 仍有约 2e-3 级误差（feature-map exp/dot 数值路径），FP16/BF16 在 Qwen3 dtype 下通过；FP32 reference 仍是 correctness oracle，需后续用 IEEE exp/更稳定归约继续优化。

## 2026-07-24：selected-query 与同长度 SDPA

- 命令：`benchmark_selected_query.py --lengths 1024 4096 8192 --queries 1 16 64 256 --warmup 3 --iters 10`；H20、FP16、B=1、Hq=32、Hkv=8、D=Dv=128，排除首次 Triton 编译。
- 结果（mean ms，SDPA / selected Triton）：S=1024：Q=1 0.141 / 0.414，Q=16 0.135 / 1.678，Q=64 0.139 / 6.040，Q=256 0.175 / 22.887；S=4096：Q=1 0.343 / 1.007，Q=16 0.351 / 6.230，Q=64 0.366 / 23.042，Q=256 0.491 / 89.016；S=8192：Q=1 0.620 / 2.117，Q=16 0.632 / 12.877，Q=64 0.672 / 45.974，Q=256 0.915 / 178.121。
- 事实结论：当前 selected kernel 没有超过 SDPA；它的每个 `(query head, query)` 都扫描完整 S，复杂度约为 O(Hq*Q*S*D*Dv)，而之前 full-prefill native grouped kernel 使用 chunk state-scan。selected 版本下一步必须接入 offline block summary/prefix state，不能继续使用全 K/V 重扫。

## 2026-07-24：block-summary selected 优化

- 新增 `block_summary_selected.py` 和 `benchmark_block_selected.py`。每个 block 保存 block 之前的 `S/z`，selected query 只扫描当前 block，历史通过 `prefix_s/prefix_z` 传入；这对应 REQUIREMENTS 中的 offline block summary 语义。
- 命令：`benchmark_block_selected.py --lengths 1024 4096 8192 --queries 1 16 64 256 --block-size 64 --warmup 2 --iters 5`。
- 结果（SDPA / block-linear ms）：S=1024：Q=1 0.090/0.402，Q=16 0.089/0.497，Q=64 0.094/1.059，Q=256 0.129/3.980；S=4096：Q=1 0.299/0.352，Q=16 0.303/0.475，Q=64 0.321/1.029，Q=256 0.449/3.981；S=8192：Q=1 0.571/0.334，Q=16 0.591/0.460，Q=64 0.620/1.039，Q=256 0.861/3.969。
- correctness：S=23、block=8、GQA、FP16，block-summary vs full reference max_error=9.77e-4。
- 事实结论：block summary 已使长上下文 Q=1/16 selected path 超过同长度 SDPA；短上下文和 Q=64/256 仍受 Triton launch/output 开销影响。full-prefill native grouped 文件仍保留在 `linear_kenrel/`，没有被修改或删除。
- 优化 block adapter：预打包 contiguous K/V blocks；连续 selected positions 走零拷贝 q slice，避免重复 `index_select` 和 Triton wrapper 的 `.contiguous()` 拷贝。复测（block=64）：S=4096,Q=1 0.267 ms、Q=16 0.429 ms、Q=64 0.995 ms、Q=256 3.816 ms；S=8192,Q=1 0.253 ms、Q=16 0.390 ms、Q=64 0.962 ms、Q=256 3.773 ms。
- 新增 block-local matmul backend，对每个当前 block 批量处理多个 query，不构造完整 `[Q,S]` 矩阵。复测：S=8192,Q=64 为 0.809 ms（Triton 0.957 ms），Q=256 为 2.628 ms（Triton 3.739 ms）；correctness max_error=0。该路径显存较高，暂作为 Q 较大时的候选。

## 2026-07-24：分散 selected query 的 batched GQA 优化

- 新增 `block_summary_selected_attention_batched`：输入预打包 padded block K/V `[B,Nb,Hkv,L,D]`，按 query 的 block id gather，在 `Hkv=8, group=Hq/Hkv` 维度上批量计算；不 repeat KV，不逐 block 调 Triton。
- correctness：B=2、Hq=8、Hkv=2、S=23、Q=11、block=8、非连续 `[B,Q]` positions，和已有 block-local matmul path max error=0。
- H20、S=8192、B=1、Hq=32、Hkv=8、D=Dv=128、FP16、block=64、warmup=2、iters=3。uniform positions 的 SDPA / 逐 block Triton / local matmul / batched GQA（ms）：Q=64 为 0.637 / 13.923 / 33.365 / 1.524，Q=256 为 0.871 / 28.587 / 73.291 / 5.204。末尾连续 positions：Q=64 为 0.626 / 0.987 / 0.910 / 1.561，Q=256 为 0.877 / 3.780 / 2.873 / 5.664。
- 结论：batched GQA 对 uniform/random 分散 query 消除了 Python 多 block launch 的主要退化；连续尾部仍由现有 block Triton/local matmul 更优。Q=1 的低延迟路径不应切换到 batched gather。
- FLA v0.3 检查：`chunk_simple_gla`/`chunk_linear_attn` 可提供 chunk state-scan 和 normalization 参考，但 forward output kernel 没有本需求的 native Hq/Hkv GQA selected-query 接口；本实现保留 Hkv state 并自定义 grouped output。

## 2026-07-24：fused Triton block kernel

- 新增 `triton_block_attention`/`block_summary_selected_attention_fused`。kernel 使用单次 `(B,Q,Hq)` launch，按 absolute position 计算 block id，直接读取 block K/V 和 block 前缀 `S/z`，内部按 `kv_head=q_head//(Hq/Hkv)` 做 GQA；不使用 PyTorch gather/einsum，也不 repeat KV。
- correctness：B=2、Hq=8、Hkv=2、S=23、Q=11、block=8、非连续 `[B,Q]` positions，FP16 与 block-local reference max error=9.77e-4。
- H20、S=8192、B=1、Hq=32、Hkv=8、D=Dv=128、FP16、block=64、warmup=2、iters=3。fused Triton：连续尾部 Q=1/16/64/256 为 0.165/0.179/0.315/1.077 ms；uniform positions 为 0.165/0.170/0.337/1.133 ms。对应 SDPA 分别约 0.58/0.59/0.63/0.87 ms（实际测量有小幅波动）。
- 结论：fused Triton 消除了原逐 block Triton 的 launch 放大；PyTorch batched 路径不再作为性能结论，只保留用于语义/数值交叉检查。

## 2026-07-24：GQA-aware fused Triton

- 新增 `triton_block_attention_gqa`：grid 从 `(B,Q,Hq)` 改为 `(B,Q,Hkv)`，每个 program 同时产生一个 KV head 对应的 Q-head group 输出；prefix `S/z` 与 block K/V 不再按 4 个 Q heads 重复加载/扫描。
- correctness：B=2、Hq=8、Hkv=2、非连续 positions，FP16 max error=9.77e-4。
- H20、S=8192、uniform、FP16、block=64：Q=64 为 0.255 ms，Q=256 为 0.732 ms；非 GQA-fused 版本为 0.339/1.142 ms，分别约提升 1.33x/1.56x。

## 2026-07-24：完整 selected-query 矩阵复测

- 统一命令：`benchmark_block_selected.py --lengths 1024 4096 8192 --queries 1 16 64 256 --block-size 64 --warmup 3 --iters 10`；H20、B=1、Hq=32、Hkv=8、D=Dv=128、FP16、串行运行避免 GPU 争用。
- 下表为 `SDPA / fused GQA Triton`（ms），末尾连续 positions：

| S | Q=1 | Q=16 | Q=64 | Q=256 |
|---:|---:|---:|---:|---:|
| 1024 | 0.082 / 0.142 | 0.083 / 0.137 | 0.089 / 0.202 | 0.126 / 0.613 |
| 4096 | 0.291 / 0.138 | 0.297 / 0.140 | 0.316 / 0.197 | 0.441 / 0.617 |
| 8192 | 0.566 / 0.132 | 0.574 / 0.135 | 0.612 / 0.192 | 0.857 / 0.619 |

- uniform 分散 positions：

| S | Q=1 | Q=16 | Q=64 | Q=256 |
|---:|---:|---:|---:|---:|
| 1024 | 0.081 / 0.138 | 0.083 / 0.136 | 0.089 / 0.206 | 0.122 / 0.636 |
| 4096 | 0.291 / 0.132 | 0.298 / 0.141 | 0.314 / 0.226 | 0.443 / 0.674 |
| 8192 | 0.565 / 0.137 | 0.576 / 0.134 | 0.615 / 0.228 | 0.859 / 0.691 |

- GQA-fused 显存峰值为 68.1/68.6/70.1/76.3 MiB（S=1024，Q=1/16/64/256），176.3/176.8/178.5/185.3 MiB（S=4096），320.5/321.1/323.0/330.5 MiB（S=8192）。
