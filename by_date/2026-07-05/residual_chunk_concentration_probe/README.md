# Residual Chunk Concentration Probe

## 目的

验证 `online_draft_selected - offline_fixed_set` residual tokens 是否集中在少数 chunk。如果集中，则可以考虑只在 offline fixed set 覆盖不足的 chunk 上调用 DraftModel，从而降低 online DraftModel selection 的范围。

## 数据

- online set：`MOTIVATION_EXPERIMENTS/qwen3_online_draft_trace_rate015/selected_indices`。
- offline base：`draft_smart_frequency_global` 与 `draft_smart_mean_score_global`。
- residual 定义：完整 `online_draft_selected_doc_tokens - offline_base_selected_doc_tokens`。
- chunk 映射：根据 online trace 中的 `system_len`、`passages_len`、`chunk_ids` 将 token absolute position 映射到 document chunk。

## 汇总结果

| offline base | n | avg residual | avg chunks | avg nonempty chunks | top1 chunk frac | top2 chunk frac | top3 chunk frac | chunks for 90% residual | normalized entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| freq | 250 | 117.39 | 12.26 | 8.03 | 35.27% | 59.25% | 75.39% | 5.25 | 0.66 |
| mean | 250 | 127.82 | 12.26 | 9.92 | 32.14% | 54.78% | 70.61% | 6.04 | 0.73 |

## 解读

- residual 并不只落在 1-2 个 chunk 上；平均有多个 chunk 非空，覆盖 90% residual 通常需要多个 chunk。
- top-1/top-2/top-3 chunk 确实覆盖了一部分 residual，但不够集中到可以无损地只跑极少数 chunk。
- 这条线仍有价值，但更适合作为 chunk gate / coarse filter，而不是直接替代 token-level selector。
- 下一步如果要做 chunk-gated DraftModel，应先做 oracle gate：只在能覆盖 70/80/90% residual 的最少 chunk 上跑 selection，估算 accuracy/overlap 上界，再尝试轻量预测这些 chunk。

## 产物

- `summary.csv`：整体统计。
- `per_query_chunk_concentration.csv`：每个 query 的 residual chunk concentration 指标。
- `per_chunk_residual_counts.csv`：每个 query/chunk 的 residual count。
- `top1_chunk_frac_hist_*.png`：top-1 chunk 覆盖 residual 比例分布。
- `chunks_for_90pct_hist_*.png`：覆盖 90% residual 需要的 chunk 数分布。
