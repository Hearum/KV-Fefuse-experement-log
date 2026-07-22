# Query Attention Mass vs Selected Token Rank

## 实验目的

验证 FusionRAG importance 排名前若干比例 token 是否已经覆盖大部分 query attention mass，用于解释 `v_only` 随 rate 增大平均收益很弱的现象。

## 实验设置

- score 文件数: 20
- query 数: 640
- passage 数: 10 passages + system prompt。
- KV: preprocess top-10 KV cache。
- score 定义: 对每个 query，用 preprocess KV 的 context K 计算 query-context QK softmax attention，再对 layer/head/query token 求和；排序和 mass 统计时排除 system tokens。
- rate: [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]

## 结果汇总

| top rate | selected tokens mean | cumulative mass mean | mass p10 | mass p50 | mass p90 | incremental mass mean | incremental mass/token mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 367.4 | 0.3663 | 0.2725 | 0.3654 | 0.4557 | 0.3663 | 0.00100053 |
| 0.10 | 735.4 | 0.4632 | 0.3825 | 0.4623 | 0.5384 | 0.0969 | 0.00027297 |
| 0.15 | 1103.2 | 0.5331 | 0.4614 | 0.5320 | 0.5991 | 0.0699 | 0.00019659 |
| 0.20 | 1471.2 | 0.5898 | 0.5248 | 0.5893 | 0.6480 | 0.0568 | 0.00015961 |
| 0.30 | 2207.1 | 0.6812 | 0.6274 | 0.6799 | 0.7281 | 0.0914 | 0.00012849 |
| 0.50 | 3678.8 | 0.8155 | 0.7825 | 0.8146 | 0.8463 | 0.1343 | 0.00009443 |
| 0.80 | 5886.0 | 0.9491 | 0.9386 | 0.9493 | 0.9589 | 0.1336 | 0.00006274 |

## 初步解读

- 如果低 rate 已覆盖较高 cumulative mass，说明新增 token 虽然数量多，但 query 实际读取较少，这可以解释 `v_only` 提高 rate 后平均 F1/EM 不稳定。
- `incremental mass/token` 随 rate 下降越快，说明 importance 排序后半段 token 的边际注意力贡献越低。
- 这组实验只分析 query attention/importance mass，不直接等价于 answer accuracy；它用于解释 rate 的边际收益。

## 输出文件

- `attention_mass_by_query.csv`: 每个 example/query 的明细。
- `attention_mass_summary.json`: 汇总统计。
- `attention_mass_cumulative_by_rate.png`: 累计 mass 曲线。
- `attention_mass_marginal_by_rate.png`: 边际 mass 曲线。
