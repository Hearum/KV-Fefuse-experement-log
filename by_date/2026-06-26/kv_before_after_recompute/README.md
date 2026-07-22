# 重算前后 KV 差异分析

对比口径：`full_context KV` 作为重算后/full-attention 目标；`raw` 和 `preprocess_top10` 作为重算前 KV。

- examples: 1-20
- chunk: 1

| source | kind | relative L2 | cosine | token rel L2 mean | token rel L2 p90 |
|---|---|---:|---:|---:|---:|
| raw | key | 0.1690 | 0.9815 | 0.1563 | 0.2285 |
| raw | value | 0.2904 | 0.9500 | 0.2551 | 0.4301 |
| preprocess_top10 | key | 0.1221 | 0.9900 | 0.1108 | 0.1598 |
| preprocess_top10 | value | 0.2230 | 0.9699 | 0.1924 | 0.3158 |

## 初步结论

- preprocess top-10 相比 raw 更接近 full-context KV，key/value 都有改善。
- value 的绝对差距通常比 key 更大；这意味着重算收益不只是改善检索注意力分数，也会直接影响后续信息聚合内容。
- 差异具有明显层分布，可以看 `kv_before_after_layer_relative_l2.png`；token 位置分布见 `kv_before_after_position_profile.png`。

## 现象拆解

1. `preprocess_top10` 明显把 KV 拉向 full-context KV。
   - key relative L2: raw `0.1690` -> preprocess_top10 `0.1221`，下降约 `27.8%`。
   - value relative L2: raw `0.2904` -> preprocess_top10 `0.2230`，下降约 `23.2%`。
   - cosine 也同步提高：key `0.9815` -> `0.9900`，value `0.9500` -> `0.9699`。

2. 差距主要不是均匀分布在所有层，而是集中在中后层。
   - raw key 差距最大的层：layer 17, 16, 20, 18, 22, 14。
   - raw value 差距最大的层：layer 18, 15, 17, 14, 20, 16。
   - preprocess_top10 之后，最大残差仍然集中在类似层：key 的 layer 16/17/22/20，value 的 layer 18/15/20/17。

3. value 比 key 更需要重算校正。
   - raw value 的 relative L2 `0.2904` 明显高于 raw key 的 `0.1690`。
   - preprocess_top10 后 value 仍有 `0.2230` 的 residual，key 则降到 `0.1221`。
   - 这说明重算的收益不只是让 QK attention score 更准，也会改变后续读出的 value 内容。

4. token 位置上，chunk 开头的差距最大。
   - preprocess_top10 的 value 在第 0 个 token decile 的 mean relative L2 为 `0.3315`，明显高于后续 decile。
   - 中后段 token 多数在 `0.15-0.22` 附近。
   - 一个可能解释是：chunk 开头 token 的表示更依赖前文上下文接入方式，isolated/preprocess KV 和 full-context KV 的上下文边界差异在这里最明显。

## 对“重算”机制的解释

重算本质上是在修正一部分 token 的 KV 表示，使它们从 isolated/preprocess 的局部上下文表示，移动到 full-context 的文档上下文表示。当前结果说明：

- preprocess 已经完成了一部分这个移动，所以它比 raw 更接近 full-context。
- 但 preprocess 没完全消除差距，尤其 value 和中后层仍有明显 residual。
- 如果要省掉在线重算，不能只保存 selected token index；真正要省的是把这些 token 的 corrected KV 也提前算好或预测好。
