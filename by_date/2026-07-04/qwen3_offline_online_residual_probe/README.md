# Offline Smart Set 的 Online Draft Residual 诊断

目标：统计 offline smart fixed set 覆盖 online draft 后，剩余 query-specific residual token 的规模和分布。

| method | online covered | residual avg | residual p50 | residual p90 | residual ratio | nonempty chunks | max chunk frac | chunk entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| draft_smart_frequency_global | 0.6051 | 86.7 | 78 | 140 | 0.3949 | 7.0 | 0.4152 | 0.6268 |
| draft_smart_mean_score_global | 0.5537 | 97.2 | 91 | 148 | 0.4463 | 8.9 | 0.3763 | 0.7039 |

解释：residual = online draft 选中但 offline fixed set 没覆盖的 doc tokens。chunk entropy 越高，说明 residual 越分散；max chunk frac 越高，说明 residual 越集中于某个 chunk。
