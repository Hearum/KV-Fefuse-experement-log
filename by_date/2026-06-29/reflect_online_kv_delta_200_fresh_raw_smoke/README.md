# Reflect Online Recompute K/V Delta

## 实验设置

- 数据：`./data/result_reflect.json`
- 样本：前 `2` 个可测试 sub-question
- 模型：`Qwen2.5-7B-Instruct`
- cache_mode：`fresh_raw`
- cache：`/raid/home/hming/fusionrag-reflect-full-cache`，`preprocess_kv_cache_global_topk10_bge`
- 方法：FusionRAG selection，`rate=0.15`，`topk=10`，`revert_rope=True`
- 统计对象：只比较 selected document tokens 在 online recompute 前后的 K/V cache。

## 汇总

| metric | key | value | value/key |
|---|---:|---:|---:|
| relative_l2 mean | 0.166125 | 0.815131 | 4.906727 |
| diff_norm mean | 2639.019165 | 1692.760925 | 0.641436 |
| token_rel_mean mean | 0.165577 | 0.850436 | 5.136199 |
| token_rel_p50 mean | 0.163999 | 0.701288 | 4.276178 |
| token_rel_p90 mean | 0.182881 | 1.498741 | 8.195192 |
| token_rel_p99 mean | 0.203525 | 2.171809 | 10.670972 |
| cosine mean | 0.986734 | 0.696490 | - |

## 样本规模

- doc_len mean/p50/p90: `2096.0` / `2096.0` / `2103.2`
- selected_count mean/p50/p90: `195.0` / `195.0` / `195.8`

## 结论

- selected token 上，value 的 relative L2 均值是 `0.8151`，key 是 `0.1661`，value/key 约 `4.91x`。
- value cosine 均值 `0.6965`，明显低于 key cosine `0.9867`，说明 value 不只是幅度变化，方向也变得更多。
- 这支持之前的机制观察：online recompute 对 V 的表示改写更强；但结合 strict K/V 写回消融，K 的变化虽然数值小，仍可能通过 attention 路由产生高杠杆影响，因此 K/V 应配套看。

## 文件

- `per_sample_kv_delta.csv`：逐 sub-question 标量指标
- `per_sample_kv_delta.jsonl`：逐 sub-question 完整指标和问题文本
- `layer_relative_l2.csv`：逐层 K/V relative L2
- `summary.json`：完整汇总
