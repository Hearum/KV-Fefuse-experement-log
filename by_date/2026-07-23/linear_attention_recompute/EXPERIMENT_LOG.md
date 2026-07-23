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
