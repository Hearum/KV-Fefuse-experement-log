# Full-Attention Query Anchor Stability: Multi-Rate Analysis

## 设置

复用 `MOTIVATION_EXPERIMENTS/full_attention_query_anchor_stability/` 中保存的 full-attention query-to-doc distributions，不重新 forward。

对每个 query 按 doc-token attention score 从高到低分别取 top 15%、25%、30%、50%、80%，重新统计集合重合度和 token 频率分布。

## Aggregate

| rate | pairwise Jaccard | related-related | related-unrelated | all-query intersection/top-k | never selected | always selected | selected >=50% queries | selected >=80% queries |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 0.7452 | 0.7579 | 0.7343 | 0.7223 | 0.6494 | 0.1081 | 0.1336 | 0.1202 |
| 0.25 | 0.7854 | 0.7949 | 0.7769 | 0.7548 | 0.4930 | 0.1885 | 0.2304 | 0.2089 |
| 0.30 | 0.8111 | 0.8218 | 0.8032 | 0.7729 | 0.4434 | 0.2317 | 0.2811 | 0.2570 |
| 0.50 | 0.8860 | 0.8892 | 0.8797 | 0.8469 | 0.2847 | 0.4233 | 0.4854 | 0.4581 |
| 0.80 | 0.9503 | 0.9499 | 0.9471 | 0.9223 | 0.0915 | 0.7377 | 0.7945 | 0.7711 |

## 文件

- `aggregate_summary_by_rate.csv`
- `example_summary_by_rate.csv`
- `frequency_hist_by_rate.csv`
- `stable_intersection_curve_by_rate.csv`
- `figures/stability_metrics_vs_rate.png`
