# Doc-generated Offline Selector Diagnostics

本诊断复用已经跑完的 `offline10_docgen_draft005`、旧版 `offline10_draft005` 和 `pure_online_draft_rate010` 的 selected token 文件，不重新跑前向。

## 对比对象

- `docgen_offline10`：每个文档块生成 32 个问题后，用 chunk-local DraftModel 聚合得到的 offline 10% fixed set。
- `old_offline10`：之前的 calibration-query draft smart fixed set。
- `online_draft10`：真实 online DraftModel 在当前真实 query 下选出的 10% token。

## 集合重合度摘要

共比较 `250` 个 example/sub-question。

| 指标 | 数值 |
|---|---:|
| docgen 平均 token 数 | 137.67 |
| old offline 平均 token 数 | 148.04 |
| online Draft10 平均 token 数 | 142.18 |
| docgen vs online Jaccard | 0.2325 |
| old vs online Jaccard | 0.3832 |
| docgen 覆盖 online Draft10 | 0.3626 |
| old offline 覆盖 online Draft10 | 0.5328 |
| online Draft10 覆盖 docgen | 0.3745 |
| online Draft10 覆盖 old offline | 0.5743 |
| docgen vs old Jaccard | 0.2955 |

## 位置分布摘要

位置值归一化到 `[0, 1]`。`global_pos` 是整个拼接文档区域内的位置；`chunk_pos` 是 token 在所属 chunk 内的位置。

详表见：

- `selector_overlap_detail.csv`
- `selector_overlap_summary.csv`
- `selector_position_detail.csv`
- `selector_position_summary.csv`
- `selector_position_bins.csv`

## 初步结论

1. docgen offline10 与 online Draft10 的重合度低于旧 offline10，这和 accuracy 结果一起说明：docgen 的收益不是来自更好拟合 online Draft selector。
2. 后续应检查 docgen 选中的 token 是否更偏向 chunk 内 anchor/实体/boundary 位置，以及是否更覆盖 answer-support 信息。
3. 如果要继续提升，下一步更合理的是把 docgen set 与旧 offline set 做 union/rerank，而不是只替换旧 offline set。
