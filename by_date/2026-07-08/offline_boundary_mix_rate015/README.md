# Offline Boundary Mix Rate 0.15

目的：验证在总重算比例仍为 0.15 的条件下，用 chunk/document 边界 token 替换一部分纯 offline draft-smart token，是否能提升纯 offline 方法的 QA accuracy。

构造方式：

- 输入 score cache 与 `draft_smart_*_global` 相同：`MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_draft_rate015_full/combined/score_cache_npz`。
- 总预算固定为 `int(0.15 * doc_len)`。
- base 排序使用 `draft_smart_frequency_global` 或 `draft_smart_mean_score_global` 的同口径排序。
- 边界排序按 token 到每个 retrieved passage 的开头/结尾距离排序，距离越小越优先；同距离时用 calibration mean score 排序。
- `boundary0p02/0p03/0p05` 分别表示从总 15% 预算中拿出 2%/3%/5% doc tokens 给边界集合，其余预算保留 base offline 排序。

方法名：

- `draft_smart_freq_boundary0p02_global`
- `draft_smart_mean_boundary0p02_global`
- `draft_smart_freq_boundary0p03_global`
- `draft_smart_mean_boundary0p03_global`
- `draft_smart_freq_boundary0p05_global`
- `draft_smart_mean_boundary0p05_global`
## Accuracy 结果

| method | status | segments | Main Acc | Sub Acc | F1 | EM |
|---|---|---:|---:|---:|---:|---:|
| draft_smart_freq_boundary0p02_global | complete | 8/8 | 97/135 (71.85%) | 203/250 (81.20%) | 0.1987 | 0.0120 |
| draft_smart_mean_boundary0p02_global | complete | 8/8 | 96/135 (71.11%) | 202/250 (80.80%) | 0.1864 | 0.0120 |
| draft_smart_freq_boundary0p03_global | complete | 8/8 | 95/135 (70.37%) | 199/250 (79.60%) | 0.1994 | 0.0120 |
| draft_smart_mean_boundary0p03_global | complete | 8/8 | 96/135 (71.11%) | 200/250 (80.00%) | 0.2068 | 0.0120 |
| draft_smart_freq_boundary0p05_global | complete | 8/8 | 94/135 (69.63%) | 199/250 (79.60%) | 0.1969 | 0.0120 |
| draft_smart_mean_boundary0p05_global | complete | 8/8 | 94/135 (69.63%) | 200/250 (80.00%) | 0.2054 | 0.0240 |

参考纯 offline baseline：`draft_smart_frequency_global` = 94/135 Main, 203/250 Sub；`draft_smart_mean_score_global` = 95/135 Main, 204/250 Sub。

## 关键记录

这组实验的正向改善主要发生在 `draft_smart_frequency_global` 这条纯 offline baseline 上：

- 原始 `draft_smart_frequency_global`：Main 94/135 (69.63%)，Sub 203/250 (81.20%)。
- `draft_smart_freq_boundary0p02_global`：总重算比例仍为 0.15，其中 13% 使用 frequency offline ranking，2% 替换为靠近 retrieved passage 边界的 token；结果为 Main 97/135 (71.85%)，Sub 203/250 (81.20%)。
- 因此该改法在 Main Acc 上提升 +3，Sub Acc 保持不变。

在 `draft_smart_mean_score_global` 上，2% boundary 的改善不稳定：Main 从 95/135 到 96/135，但 Sub 从 204/250 降到 202/250。因此当前可记录的有效 observation 是：boundary-near token 可以作为 frequency-based pure offline set 的小比例补偿信号，但不能大比例替代原 offline ranking。

