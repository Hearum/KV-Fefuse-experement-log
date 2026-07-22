# 四种 rate=0.15 selector 的 doc-token 重合度（修正版）

本表只统计 document tokens；online trace 中的 query tokens 已过滤掉。

对齐样本数：250。

## 后三个方法相对 online_draft

| method | Jaccard | online_draft covered by method | method covered by online_draft | avg online doc tokens | avg method doc tokens | avg intersection |
|---|---:|---:|---:|---:|---:|---:|
| offline_hybrid70 | 0.3313 | 0.4890 | 0.5008 | 213.5 | 208.7 | 104.4 |
| draft_max_score | 0.3238 | 0.4807 | 0.4922 | 213.5 | 208.7 | 102.7 |
| draft_top2_mean | 0.3272 | 0.4846 | 0.4962 | 213.5 | 208.7 | 103.5 |

## 两两重合度

| A | B | Jaccard | A covered by B | B covered by A | avg intersection |
|---|---|---:|---:|---:|---:|
| online_draft | offline_hybrid70 | 0.3313 | 0.4890 | 0.5008 | 104.4 |
| online_draft | draft_max_score | 0.3238 | 0.4807 | 0.4922 | 102.7 |
| online_draft | draft_top2_mean | 0.3272 | 0.4846 | 0.4962 | 103.5 |
| offline_hybrid70 | draft_max_score | 0.7518 | 0.8565 | 0.8565 | 179.0 |
| offline_hybrid70 | draft_top2_mean | 0.8030 | 0.8895 | 0.8895 | 185.9 |
| draft_max_score | draft_top2_mean | 0.9063 | 0.9505 | 0.9505 | 198.5 |
