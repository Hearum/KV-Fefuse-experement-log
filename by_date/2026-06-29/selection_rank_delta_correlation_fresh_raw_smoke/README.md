# Selection Rank vs Online KV Delta

## 实验设置

- cache_mode: `fresh_raw`
- sub-question: `3`
- method: FusionRAG selection, `rate=0.15`
- 统计对象：被 FusionRAG importance 排序选中的 selected document tokens。rank 越小表示 importance 越高。

## 相关性

| x | y | Pearson | Spearman |
|---|---|---:|---:|
| early rank vs key delta | - | 0.101276 | 0.037500 |
| early rank vs value delta | - | 0.146969 | 0.103342 |
| importance score vs key delta | - | 0.491547 | 0.017581 |
| importance score vs value delta | - | 0.266615 | 0.088173 |

## Rank Percentile 分桶

| rank pct | count | score mean | key delta mean | value delta mean | value delta p90 |
|---|---:|---:|---:|---:|---:|
| 0.0-0.1 | 61 | 122.963115 | 0.187925 | 1.125287 | 1.438711 |
| 0.1-0.2 | 60 | 19.907292 | 0.173055 | 0.846486 | 1.397192 |
| 0.2-0.3 | 61 | 13.451844 | 0.167811 | 0.918268 | 1.757150 |
| 0.3-0.4 | 60 | 10.752083 | 0.170154 | 0.829385 | 1.573781 |
| 0.4-0.5 | 61 | 9.075307 | 0.167766 | 0.787761 | 1.617254 |
| 0.5-0.6 | 61 | 7.780225 | 0.168615 | 0.797044 | 1.391176 |
| 0.6-0.7 | 60 | 6.753125 | 0.172973 | 0.736840 | 1.425946 |
| 0.7-0.8 | 61 | 5.959529 | 0.175590 | 0.894455 | 1.518548 |
| 0.8-0.9 | 60 | 5.237240 | 0.173600 | 0.867299 | 1.552757 |
| 0.9-1.0 | 606 | 20.727929 | 0.173237 | 0.853754 | 1.519510 |

## 文件

- `token_rank_delta.csv`: 每个 selected token 的 rank、score、K/V delta。
- `rank_bin_summary.csv`: rank percentile 分桶统计。
- `summary.json`: 相关性和分桶完整结果。
