# Reflect Online Recompute K/V Delta

## 实验设置

- 数据：`./data/result_reflect.json`
- 样本：前 `200` 个可测试 sub-question
- 模型：`Qwen2.5-7B-Instruct`
- cache_mode：`fresh_raw`
- cache：`/raid/home/hming/fusionrag-reflect-full-cache`，`preprocess_kv_cache_global_topk10_bge`
- 方法：FusionRAG selection，`rate=0.15`，`topk=10`，`revert_rope=True`
- 统计对象：只比较 selected document tokens 在 online recompute 前后的 K/V cache。

## Cache 一致性检查

本实验原计划直接复用 `/raid/home/hming/fusionrag-reflect-full-cache` 中的 preprocess cache，比较真实 cache reuse 状态下的 online recompute 前后 K/V。实际 smoke test 发现该 cache 与当前 `data/result_reflect.json` 的 tokenization 不一致：例如 example 1 的 system chunk 长度匹配为 792，但当前数据重建的第 1 个 document 是 50 tokens，而 cache 中 chunk 1 是 90 tokens；后续 chunk 也整体错位。同时该 cache 到 example 34 出现缺失 chunk。因此不能直接用现有 cache 做 200-sample before/after，否则 selected token 的 token id 与 KV 位置会错位。

为保证本轮统计不犯错，正式结果改用 `cache_mode=fresh_raw`：对当前 `data/result_reflect.json` 的每个 sub-question 现场构建 raw chunk KV，再执行 FusionRAG query-based selection 和 selected-token online recompute，统计 selected document tokens 的 K/V 更新幅度。这个结果反映 online recompute 对 K/V 的相对改写强度，但不是 preprocess-cache 口径；如果后续需要完全对齐 preprocess-cache，需要先重新生成一份与当前数据一致的 preprocess KV cache。

## 汇总

| metric | key | value | value/key |
|---|---:|---:|---:|
| relative_l2 mean | 0.169465 | 0.864218 | 5.099688 |
| diff_norm mean | 2798.414482 | 1858.228246 | 0.664029 |
| token_rel_mean mean | 0.168900 | 0.916209 | 5.424564 |
| token_rel_p50 mean | 0.167510 | 0.798067 | 4.764284 |
| token_rel_p90 mean | 0.186904 | 1.574080 | 8.421856 |
| token_rel_p99 mean | 0.203388 | 2.111102 | 10.379676 |
| cosine mean | 0.986164 | 0.658391 | - |

## 样本规模

- doc_len mean/p50/p90: `2227.7` / `2156.0` / `2789.1`
- selected_count mean/p50/p90: `214.9` / `204.0` / `299.0`

## 结论

- selected token 上，value 的 relative L2 均值是 `0.8642`，key 是 `0.1695`，value/key 约 `5.10x`。
- value cosine 均值 `0.6584`，明显低于 key cosine `0.9862`，说明 value 不只是幅度变化，方向也变得更多。
- 这支持之前的机制观察：online recompute 对 V 的表示改写更强；但结合 strict K/V 写回消融，K 的变化虽然数值小，仍可能通过 attention 路由产生高杠杆影响，因此 K/V 应配套看。

## 文件

- `per_sample_kv_delta.csv`：逐 sub-question 标量指标
- `per_sample_kv_delta.jsonl`：逐 sub-question 完整指标和问题文本
- `layer_relative_l2.csv`：逐层 K/V relative L2
- `summary.json`：完整汇总
