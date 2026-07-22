# Draft Smart Global vs Online Draft Overlap

只统计 document tokens，过滤 online trace 中的 query tokens。

| method | Jaccard | online covered by fixed | fixed covered by online | avg online tokens | avg fixed tokens | avg intersection |
|---|---:|---:|---:|---:|---:|---:|
| draft_smart_frequency_global | 0.4429 | 0.6051 | 0.6290 | 213.5 | 221.9 | 126.8 |
| draft_smart_mean_score_global | 0.3872 | 0.5537 | 0.5708 | 213.5 | 219.9 | 116.3 |
| draft_smart_max_score_global | 0.3831 | 0.5578 | 0.5582 | 213.5 | 223.5 | 117.4 |
| draft_smart_top2_mean_global | 0.3870 | 0.5599 | 0.5641 | 213.5 | 222.7 | 117.7 |
