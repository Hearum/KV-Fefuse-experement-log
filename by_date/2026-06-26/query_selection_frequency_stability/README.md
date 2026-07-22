# Query Selection Frequency and Stable-Set Convergence

分析对象：固定同一组文档时，不同 query 触发的 selected token 集合。

- 当前缓存：每个 example 16 个 query。
- frequency histogram：统计每个 document token 在 16 个 query 中被选中 0..16 次。
- convergence：随机采样 2/4/8/16 个 query 做交集，统计 stable intersection ratio。
- 32-query 需要额外生成更多 query score cache，当前结果先覆盖到 16-query。

## Passages=10 Summary

| selector | rate | never selected | selected 15/16 | selected 16/16 | polarized 0 or 16 |
|---|---:|---:|---:|---:|---:|
| DraftModel(raw) | 0.1 | 0.8061 | 0.0059 | 0.0568 | 0.8629 |
| DraftModel(raw) | 0.2 | 0.6451 | 0.0124 | 0.1198 | 0.7649 |
| DraftModel(raw) | 0.3 | 0.5107 | 0.0190 | 0.1880 | 0.6987 |
| DraftModel(raw) | 0.5 | 0.2975 | 0.0305 | 0.3503 | 0.6478 |
| Target-QK(preprocess KV) | 0.1 | 0.8100 | 0.0050 | 0.0612 | 0.8712 |
| Target-QK(preprocess KV) | 0.2 | 0.6516 | 0.0099 | 0.1286 | 0.7801 |
| Target-QK(preprocess KV) | 0.3 | 0.5230 | 0.0146 | 0.2052 | 0.7282 |
| Target-QK(preprocess KV) | 0.5 | 0.3172 | 0.0212 | 0.3838 | 0.7010 |

## Figures

- `token_frequency_hist_passages10_draftmodelraw.png`
- `token_frequency_hist_passages10_target-qkpreprocess_kv.png`
- `stable_set_convergence_passages10.png`

## Passages=10 Stable-Set Convergence

数值表示随机采样若干 query 做交集后，交集大小占最小 selected set 的比例。当前已有缓存最多 16 个 query。

| selector | rate | 2 queries | 4 queries | 8 queries | 16 queries |
|---|---:|---:|---:|---:|---:|
| DraftModel(raw) | 0.1 | 0.7953 | 0.6798 | 0.6130 | 0.5686 |
| DraftModel(raw) | 0.2 | 0.8203 | 0.7136 | 0.6464 | 0.5993 |
| DraftModel(raw) | 0.3 | 0.8427 | 0.7427 | 0.6753 | 0.6268 |
| DraftModel(raw) | 0.5 | 0.8852 | 0.8048 | 0.7460 | 0.7006 |
| Target-QK(preprocess KV) | 0.1 | 0.8101 | 0.7066 | 0.6493 | 0.6127 |
| Target-QK(preprocess KV) | 0.2 | 0.8342 | 0.7376 | 0.6805 | 0.6429 |
| Target-QK(preprocess KV) | 0.3 | 0.8609 | 0.7751 | 0.7212 | 0.6842 |
| Target-QK(preprocess KV) | 0.5 | 0.9056 | 0.8423 | 0.7991 | 0.7676 |

初步观察：随着 query 数从 2 增加到 16，交集比例下降但没有塌到 0，而是保持在较高水平。Target-QK(preprocess KV) 在所有 rate 上都比 DraftModel(raw) 更稳定，支持“文档存在天然 stable update set”的判断。
