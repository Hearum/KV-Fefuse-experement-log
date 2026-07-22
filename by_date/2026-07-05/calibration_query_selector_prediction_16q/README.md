# Calibration Query Selector Prediction (16q)

## 目的

用已有 16-query selection cache 验证：如果 offline/calibration 阶段已经看过同一文档的多个 query，能否用这些历史 query 的 selected-token frequency 预测 held-out query 的 selector 输出。

这不是最终 pipeline accuracy 实验，也没有调用模型；它是判断“pure offline stable ranking / multi-query calibration”是否有潜力的诊断实验。

## 设置

- 数据：`MOTIVATION_EXPERIMENTS/query_selection_frequency_stability/query_selection_frequency_stability.json` 指向的 20 examples, passages=10, 每个 example 16 queries。
- selector：`DraftModel(raw)` 与 `Target-QK(preprocess KV)`。
- 做法：每次留出 1 个 query，用其他 15 个 query 的 selected positions 做 frequency ranking，预测 held-out selected positions。
- `same_size`：预测集合大小等于 held-out selected set 大小。
- `rate_budget`：预测集合大小按当前 rate 的 doc budget 取 top tokens。

## 汇总结果

| selector | rate | mode | n | avg recall | median recall | avg precision | avg Jaccard | avg target size |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| DraftModel(raw) | 0.1 | rate_budget | 320 | 84.20% | 84.66% | 84.13% | 73.62% | 735.4 |
| DraftModel(raw) | 0.1 | same_size | 320 | 84.15% | 84.66% | 84.15% | 73.60% | 735.4 |
| DraftModel(raw) | 0.2 | rate_budget | 320 | 86.25% | 87.08% | 86.23% | 76.55% | 1471.2 |
| DraftModel(raw) | 0.2 | same_size | 320 | 86.24% | 87.08% | 86.24% | 76.55% | 1471.2 |
| DraftModel(raw) | 0.3 | rate_budget | 320 | 87.96% | 88.91% | 87.96% | 79.10% | 2207.1 |
| DraftModel(raw) | 0.3 | same_size | 320 | 87.96% | 88.91% | 87.96% | 79.10% | 2207.1 |
| DraftModel(raw) | 0.5 | rate_budget | 320 | 90.96% | 91.82% | 90.96% | 83.76% | 3678.8 |
| DraftModel(raw) | 0.5 | same_size | 320 | 90.96% | 91.82% | 90.96% | 83.76% | 3678.8 |
| Target-QK(preprocess KV) | 0.1 | rate_budget | 320 | 85.43% | 86.04% | 85.43% | 75.11% | 735.4 |
| Target-QK(preprocess KV) | 0.1 | same_size | 320 | 85.43% | 86.04% | 85.43% | 75.11% | 735.4 |
| Target-QK(preprocess KV) | 0.2 | rate_budget | 320 | 87.21% | 88.39% | 87.21% | 77.74% | 1471.2 |
| Target-QK(preprocess KV) | 0.2 | same_size | 320 | 87.21% | 88.39% | 87.21% | 77.74% | 1471.2 |
| Target-QK(preprocess KV) | 0.3 | rate_budget | 320 | 89.23% | 89.92% | 89.23% | 80.87% | 2207.1 |
| Target-QK(preprocess KV) | 0.3 | same_size | 320 | 89.23% | 89.92% | 89.23% | 80.87% | 2207.1 |
| Target-QK(preprocess KV) | 0.5 | rate_budget | 320 | 92.62% | 92.97% | 92.62% | 86.42% | 3678.8 |
| Target-QK(preprocess KV) | 0.5 | same_size | 320 | 92.62% | 92.97% | 92.62% | 86.42% | 3678.8 |

## 结论

- 与 native-query residual 预测不同，16-query calibration 对完整 selector selected set 的预测 recall 很高，说明 selector 输出确实包含明显 stable component。
- `Target-QK(preprocess KV)` 的可预测性通常高于 `DraftModel(raw)`，和之前 stable-set convergence 结论一致。
- 这支持长期任务 A/E：offline 不应只保存单一 fixed set，而应保存完整 ranking 或多候选 sets；online 可以做轻量 routing/补偿。
- 但这还没有证明 residual 部分也同样可预测；下一步需要在同样 16/32-query 设置下引入 offline base set，统计 residual 的 held-out recall。

## 产物

- `loo_detail.csv`：每个 held-out query 的预测结果。
- `loo_summary.csv`：按 selector/rate/mode 汇总。
