# Residual Stability From Online Draft Trace

## 目的

验证 `online Draft selected set - offline fixed set` 得到的 residual tokens 是否具有稳定性。如果 residual 可以由历史/calibration queries 稳定预测，就有机会把当前 `online Draft residual` 的 DraftModel online selection 开销进一步去掉。

## 数据与定义

- residual 口径：这里的 residual 是完整 `online_draft_selected_doc_tokens - offline_base_selected_doc_tokens`，用于判断 online draft 没被 offline fixed set 覆盖的全部 query-dependent 部分；它不等于 residual pipeline 中额外补的 0.05 budget。
- online set：`MOTIVATION_EXPERIMENTS/qwen3_online_draft_trace_rate015/selected_indices`，共 250 个 native sub-question traces。
- offline base set：`draft_smart_frequency_global` 和 `draft_smart_mean_score_global`，来自 residual 实验保存的 `offline_fixed_selected_indices`。
- residual set：对同一个 example/sub-question，定义为 `online_draft_selected_doc_tokens - offline_base_selected_doc_tokens`。
- 当前分析只使用已有 native sub-questions；多数 example 只有 1-2 个 query，因此这是第一版弱证据分析，不等价于 16/32-query calibration 收敛实验。

## 汇总结果

| offline base | queries | examples | examples >=2q | online covered by base | avg residual size | median residual size | tokens never residual | tokens always residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| freq | 250 | 135 | 106 | 52.48% | 117.39 | 109.00 | 87.78% | 0.83% |
| mean | 250 | 135 | 106 | 48.04% | 127.82 | 121.00 | 86.50% | 1.04% |

解释：`online covered by base` 是 offline fixed set 覆盖 online draft selected tokens 的比例；`tokens never/always residual` 只在同一 example 有至少 2 个 native query 的子集上统计。

## Leave-One-Query-Out Residual Prediction

对每个至少有 2 个 query 的 example，留出 1 个 query，用其他 query 的 residual frequency 排名来预测 held-out query 的 residual tokens。

| offline base | prediction mode | n heldout | avg recall | median recall | avg precision | avg Jaccard | median Jaccard |
|---|---|---:|---:|---:|---:|---:|---:|
| freq | same_size | 221 | 10.96% | 8.33% | 12.17% | 6.48% | 4.47% |
| freq | rate005 | 221 | 8.58% | 6.19% | 14.09% | 5.98% | 4.06% |
| mean | same_size | 221 | 12.40% | 10.11% | 13.58% | 7.27% | 5.62% |
| mean | rate005 | 221 | 9.27% | 7.23% | 16.48% | 6.63% | 4.84% |

`same_size` 表示预测集合大小等于 held-out residual size；`rate005` 表示预测 5% doc tokens，模拟当前 residual 0.05 的预算。

## 初步结论

- 按本实验的完整 selected_abs 口径，offline base 对 online draft selected set 的覆盖率约 48%-52%；如果采用前序 overlap 文件的过滤口径，覆盖率约 55%-61%。两者交集数量一致，差异主要来自 online selected token denominator 的定义。因此 residual 不是少量噪声，而是 online draft 质量的重要来源。
- 在 native sub-question 这个弱设置下，residual token 呈现一定稳定性，但 leave-one-query-out 的 residual recall 仍然有限；这说明直接用少量历史 query 的 residual frequency 替代 online DraftModel 还不够稳。
- 当前证据更支持“两阶段方案”：offline fixed set 负责稳定主体，online 侧需要 query-conditioned residual；但 residual selector 不一定必须是完整 DraftModel，可以继续尝试 chunk gate、轻量 predictor 或多 candidate set routing。
- 由于 native query 数太少，下一步必须补一个专门的 calibration experiment：每个 example 至少 16/32 个 generated queries，保存完整 online draft ranking/score，再做 residual frequency 收敛曲线和 held-out recall。

## 产物

- `per_query_residual.csv`：每个 query 的 online/base/residual 大小和覆盖率。
- `frequency_histogram.csv`：residual frequency histogram。
- `leave_one_query_out_prediction.csv`：每个 held-out query 的预测结果。
- `leave_one_query_out_summary.csv`：LOO 汇总。
- `residual_frequency_hist_freq.png`, `residual_frequency_hist_mean.png`：频率分布图。
