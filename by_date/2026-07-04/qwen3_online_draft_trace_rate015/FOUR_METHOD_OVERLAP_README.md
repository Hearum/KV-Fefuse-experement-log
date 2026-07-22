# 四种 rate=0.15 selector 的 token 重合度

对齐样本数：250 个 `(example_id, sub_q_idx)`。

定义：Jaccard = intersection / union；A covered by B = intersection / A。

## 两两重合度

| A | B | Jaccard | A covered by B | B covered by A | avg A tokens | avg B tokens | avg intersection |
|---|---|---:|---:|---:|---:|---:|---:|
| online_draft | offline_hybrid70 | 0.3004 | 0.4248 | 0.5008 | 244.2 | 208.7 | 104.4 |
| online_draft | draft_max_score | 0.2937 | 0.4176 | 0.4922 | 244.2 | 208.7 | 102.7 |
| online_draft | draft_top2_mean | 0.2968 | 0.4210 | 0.4962 | 244.2 | 208.7 | 103.5 |
| offline_hybrid70 | draft_max_score | 0.7518 | 0.8565 | 0.8565 | 208.7 | 208.7 | 179.0 |
| offline_hybrid70 | draft_top2_mean | 0.8030 | 0.8895 | 0.8895 | 208.7 | 208.7 | 185.9 |
| draft_max_score | draft_top2_mean | 0.9063 | 0.9505 | 0.9505 | 208.7 | 208.7 | 198.5 |

## 后三个方法相对 online_draft 的重合度

| method | Jaccard vs online_draft | online_draft covered by method | method covered by online_draft | avg online_draft tokens | avg method tokens | avg intersection |
|---|---:|---:|---:|---:|---:|---:|
| offline_hybrid70 | 0.3004 | 0.4248 | 0.5008 | 244.2 | 208.7 | 104.4 |
| draft_max_score | 0.2937 | 0.4176 | 0.4922 | 244.2 | 208.7 | 102.7 |
| draft_top2_mean | 0.2968 | 0.4210 | 0.4962 | 244.2 | 208.7 | 103.5 |
