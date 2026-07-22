# Full-Attention Query Anchor Stability

## 实验设置

- 固定每个 example 的 `system + all unique docs` token 序列。
- 对同一固定文档序列，更换 related query 和 unrelated query。
- 对每个 query 跑 full attention，统计 query tokens 对 doc tokens 的 all-layer mean attention 分布。
- 每个 query 取 top `0.15` doc tokens 作为热点集合。

## Aggregate

| metric | value |
|---|---:|
| jaccard_all_mean | 0.7594 |
| jaccard_related_related_mean | 0.7588 |
| jaccard_unrelated_unrelated_mean | 0.7798 |
| jaccard_related_unrelated_mean | 0.7442 |
| all_query_intersection_ratio | 0.7829 |
| related_intersection_ratio | 0.8629 |
| unrelated_intersection_ratio | 0.8286 |
| never_selected_ratio | 0.7848 |
| always_selected_ratio | 0.1170 |
| selected_ge_50pct_ratio | 0.1418 |
| selected_ge_80pct_ratio | 0.1255 |

## Files

- `example_summary.csv`
- `query_detail.csv`
- `stable_intersection_curve.csv`
- `example_XXX_attention_distributions.npz`
- `figures/example_XXX_frequency_hist.png`
- `figures/aggregate_pairwise_jaccard.png`
- `figures/stable_intersection_convergence.png`
