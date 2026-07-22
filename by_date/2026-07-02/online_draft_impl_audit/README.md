# Online Draft Selector Implementation Audit

## 背景

`full_accuracy_offline_selector_reflect_summary` 里补入 `online_draft_rate015` 后，完整 248 subquestion 结果明显高于 `online_qk_rate015`：

| method | Main Acc | Sub Acc | F1 | EM |
|---|---:|---:|---:|---:|
| online_qk_rate015 | 62.22% | 75.40% | 0.4897 | 0.2137 |
| online_draft_old_dense_custom_fulltable | 73.33% | 84.68% | 0.5347 | 0.2177 |

由于这个提升幅度较大，本目录用于记录实现检查和对照实验。

## 关键实现检查

1. `test_fusionrag_reflect_preprocess_exp.py` 原本允许 `--reprocess_method DraftModel`，但 `ktransformers/util/utils.py::load_kv_and_generate` 没有真正的 `DraftModel` selection 分支，直接跑会 `NotImplementedError`。
2. 已补入 `DraftModel` selection 分支：使用 Qwen2.5-3B draft model 的 query-to-document attention，经后半层 RRF 聚合后调用 `smart_query_selection` 选 token。
3. 当前主实现保留默认行为：`reprocess_method != FusionRAG` 时 `use_sparse_attention=False`。同时新增环境变量 `FUSIONRAG_FORCE_SPARSE_ATTENTION=1`，用于强制走 sparse recompute/query 路径，便于和 FusionRAG-QK 口径对齐。
4. 检查未发现答案泄漏：draft selector 输入为 `system + retrieved docs + question prompt`，不包含 ground truth answer；judge 只在生成后调用。

## 小段对照: examples [0,25)

同一批 38 个 subquestion，对齐 `(Main Question, Sub Question)` 后：

| method | Main Acc | Sub Acc | F1 | EM | rows |
|---|---:|---:|---:|---:|---:|
| online_draft_old_fulltable_dense_custom | 16/21 (76.19%) | 33/38 (86.84%) | 0.5318 | 0.2368 | 38 |
| online_draft_profile_dense_seg000_025 | 18/21 (85.71%) | 35/38 (92.11%) | 0.5424 | 0.2368 | 38 |
| online_draft_profile_sparse_seg000_025 | 19/21 (90.48%) | 36/38 (94.74%) | 0.5449 | 0.2368 | 38 |
| online_qk_fulltable_seg000_025 | 14/21 (66.67%) | 29/38 (76.32%) | 0.4766 | 0.2368 | 38 |

观察：小段上 draft 的优势不是由 dense path 单独造成；强制 sparse 后没有下降，反而略高。

## 完整对照: 248 subquestion

完整样本对齐检查：

```text
rate1_full_attention raw=250 uniq=248 missing_vs_qk=0 extra_vs_qk=0
online_qk_rate015 raw=250 uniq=248 missing_vs_qk=0 extra_vs_qk=0
online_draft_old_dense_custom_fulltable raw=250 uniq=248 missing_vs_qk=0 extra_vs_qk=0
online_draft_profile_sparse_full raw=250 uniq=248 missing_vs_qk=0 extra_vs_qk=0
```

按 `(Main Question, Sub Question)` 去重后的完整结果：

| method | Main Acc | Sub Acc | F1 | EM | raw rows | uniq rows |
|---|---:|---:|---:|---:|---:|---:|
| rate1_full_attention | 105/135 (77.78%) | 218/248 (87.90%) | 0.5692 | 0.2298 | 250 | 248 |
| online_qk_rate015 | 84/135 (62.22%) | 187/248 (75.40%) | 0.4897 | 0.2137 | 250 | 248 |
| online_draft_old_dense_custom_fulltable | 99/135 (73.33%) | 210/248 (84.68%) | 0.5347 | 0.2177 | 250 | 248 |
| online_draft_profile_sparse_full | 100/135 (74.07%) | 206/248 (83.06%) | 0.5410 | 0.2137 | 250 | 248 |

## 当前判断

1. `online_draft_rate015` 的高质量基本可信：即使强制使用 sparse recompute/query 路径，完整结果仍明显高于 online QK。
2. 旧完整表中的 `online_draft_rate015` 不应被解释为“只换 selector 的严格 FusionRAG-QK 对照”，因为旧行默认使用 `DraftModel` 分支的 dense path。
3. 更干净的表述应使用 `online_draft_profile_sparse_full` 作为公平 online draft selector 对照：Main Acc 74.07%，Sub Acc 83.06%，F1 0.5410。
4. 这个结果支持之前的方向：draft model 是更强的 online selector，但在线 selection 开销较大，因此更适合作为 offline stable set calibration 的教师信号。

## 复现开关

当前 `ktransformers/util/utils.py` 保留了两个实现维度：

```bash
# selection 变体
export FUSIONRAG_DRAFT_SELECTION_VARIANT=profile   # 默认；对齐 profile/attention-mass 脚本
export FUSIONRAG_DRAFT_SELECTION_VARIANT=compact   # 复现旧完整表那版临时 compact selection

# recompute/query attention 路径
export FUSIONRAG_FORCE_SPARSE_ATTENTION=1          # 强制 sparse；更公平地对齐 FusionRAG-QK
unset FUSIONRAG_FORCE_SPARSE_ATTENTION             # 默认 DraftModel dense path
```

## 相关目录

- 旧完整 online draft 结果：
  `MOTIVATION_EXPERIMENTS/full_accuracy_online_draft_reflect`
- 小段 dense/sparse 对照：
  `MOTIVATION_EXPERIMENTS/online_draft_impl_audit/sparse_vs_dense_seg000_025`
- 完整 profile smart + forced sparse 结果：
  `MOTIVATION_EXPERIMENTS/online_draft_impl_audit/full_profile_smart_sparse_rate015`
- 完整 sparse summary json：
  `MOTIVATION_EXPERIMENTS/online_draft_impl_audit/full_profile_smart_sparse_rate015_summary.json`
