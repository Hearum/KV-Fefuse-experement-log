# Selection Rank vs Online KV Delta

## 这个实验在问什么

FusionRAG 会先用当前 query 对 document tokens 计算 `importance_score`，然后按分数从高到低排序，取 top `rate=0.15` 的 tokens 做 online recompute。这个实验检查：在这些被选中的 tokens 内部，排序越靠前的 token，重算前后的 K/V 变化比例是否越大。

如果答案是“越靠前变化越大、越靠后变化越小”，那么可以考虑在 selected tokens 内部继续跳过后排 token，节省重算。反过来，如果后排 token 也经常变化很大，那么简单按 rank 二次截断就不可靠。

## 排序和指标定义

- `importance_score`：FusionRAG selection 给每个 document token 的原始重要性分数。分数越高，越优先被选。
- `rank`：按 `importance_score` 从高到低排序后的名次。`rank=1` 是最高分 token。
- `rank_pct`：归一化排名位置，0 表示最靠前，1 表示最靠后。
- `early rank`：为了相关性计算使用的变量，等于 `1 - rank_pct`。数值越大表示越靠前。它不是新的 selection 分数，只是排序位置。
- `key delta`：selected token 的 K cache 重算前后相对变化，`||K_after - K_before|| / ||K_before||`。
- `value delta`：selected token 的 V cache 重算前后相对变化，`||V_after - V_before|| / ||V_before||`。
- `Pearson`：数值线性相关。容易被少数极端 score 拉高。
- `Spearman`：排序相关，更适合判断“排名靠前是否通常变化更大”。

## 实验设置

- cache_mode: `fresh_raw`
- sub-question: `200`
- selected token rows: `42983`
- method: FusionRAG selection, `rate=0.15`

## 相关性结果

| x | y | Pearson | Spearman | 解读 |
|---|---|---:|---:|---|
| early rank | key delta | 0.0013 | -0.0220 | 排序位置和 K 变化几乎无关。 |
| early rank | value delta | 0.0609 | 0.0323 | 排序位置和 V 变化只有极弱关系。 |
| importance score | key delta | 0.3577 | -0.0075 | 原始分数和 K 变化有一定线性关系，但排序关系几乎没有。 |
| importance score | value delta | 0.2218 | 0.0680 | 原始分数和 V 变化弱相关，排序关系也很弱。 |

重点看 Spearman：如果 selection 排名能稳定预测变化幅度，Spearman 应该明显为正。但这里 value delta 只有 `0.0323 / 0.0680`，非常弱。

## Rank 分桶结果

每一行表示 selected tokens 内部的一个 rank 区间。`0.0-0.1` 是最靠前 10%，`0.9-1.0` 是最靠后 10%。

| rank pct | count | score mean | key mean | key p90 | value mean | value p90 |
|---|---:|---:|---:|---:|---:|---:|
| 0.0-0.1 | 4294 | 127.5780 | 0.1863 | 0.2072 | 1.1660 | 1.5613 |
| 0.1-0.2 | 4311 | 17.0341 | 0.1739 | 0.1900 | 0.9156 | 1.7491 |
| 0.2-0.3 | 4283 | 12.2789 | 0.1724 | 0.1881 | 0.8400 | 1.6026 |
| 0.3-0.4 | 4304 | 9.8502 | 0.1728 | 0.1895 | 0.8215 | 1.5207 |
| 0.4-0.5 | 4249 | 8.3824 | 0.1742 | 0.1916 | 0.8374 | 1.4888 |
| 0.5-0.6 | 4350 | 7.3862 | 0.1754 | 0.1930 | 0.8687 | 1.5313 |
| 0.6-0.7 | 4292 | 6.6282 | 0.1763 | 0.1936 | 0.8749 | 1.5482 |
| 0.7-0.8 | 4295 | 6.0371 | 0.1772 | 0.1954 | 0.9097 | 1.5815 |
| 0.8-0.9 | 4299 | 5.5421 | 0.1784 | 0.1959 | 0.9233 | 1.5796 |
| 0.9-1.0 | 4306 | 5.1500 | 0.1793 | 0.1969 | 0.9291 | 1.5679 |

## 结论

1. 最靠前 10% 的 token 的确变化更大，尤其是 value delta mean 为 `1.1660`。这说明最高分 token 中有一批确实被 online recompute 大幅改写。
2. 但从 10% 到 100% 不存在单调下降。value delta 在中间降到约 `0.82-0.87` 后，后排又回升到 `0.91-0.93`。最靠后的 10% selected tokens 仍然有很大的 V 变化。
3. key delta 更平，除了最前 10% 略高，后面基本在 `0.172-0.179` 附近。rank 对 K 变化幅度也不是一个稳定预测器。
4. 因此，FusionRAG 的 importance rank 主要是在回答“哪些 token 对 query attention/selection 重要”，不是在回答“哪些 token 重算后 KV 会变化大”。
5. 如果想省重算，不能简单地在 selected tokens 内部按 rank 再砍后半段。更合理的方向是额外学习或构造一个 `delta / benefit predictor`，判断 selected token 重算后是否真的产生大 KV 变化或输出收益。

## 文件

- `token_rank_delta.csv`: 每个 selected token 的 rank、score、K/V delta。
- `rank_bin_summary.csv`: rank percentile 分桶统计。
- `summary.json`: 相关性和分桶完整结果。

## K/V 变化分布分位数

这组统计不再按 rank 分桶，而是直接看所有 selected token 的 K/V 相对变化分布。分位数越高，表示变化更大的 token 区间。

图文件：

- `kv_delta_percentiles_p10_p90.png`：Key 和 Value 分开画，避免量纲差异压扁 Key 曲线。
- `kv_delta_percentiles_p10_p90_log_compare.png`：Key/Value 放在同一张 log-y 图里，方便看整体比例差异。
- `kv_delta_percentiles_p10_p90.csv`：对应原始数值。

| Percentile | Key rel-delta | Value rel-delta | Value/Key |
|---:|---:|---:|---:|
| P10 | 0.1589 | 0.4095 | 2.58x |
| P20 | 0.1646 | 0.4894 | 2.97x |
| P30 | 0.1688 | 0.5648 | 3.35x |
| P40 | 0.1724 | 0.6519 | 3.78x |
| P50 | 0.1760 | 0.7795 | 4.43x |
| P60 | 0.1796 | 0.9755 | 5.43x |
| P70 | 0.1838 | 1.1253 | 6.12x |
| P80 | 0.1887 | 1.3596 | 7.20x |
| P90 | 0.1956 | 1.5746 | 8.05x |

直接结论：Key 的变化分布很窄，P10 到 P90 只从 0.1589 增加到 0.1956；Value 的变化分布明显更宽，P10 到 P90 从 0.4095 增加到 1.5746。Value/Key 的比例随分位数升高从 2.58x 增加到 8.05x，说明重算带来的大幅变化主要集中在 Value，尤其是高变化尾部。
