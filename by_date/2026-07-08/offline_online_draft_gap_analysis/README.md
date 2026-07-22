# Offline vs Online DraftModel 漏选 Token 分析

## 实验目的
按既定顺序先分析第一个问题：独立 online DraftModel selector 选中的 token 中，old/unrelated-control offline draft set 漏掉的部分到底有什么结构。这里不考虑 residual、不跑 QA，只分析集合差异。

## 数据来源与口径
- `A = offline old/unrelated draft set`。rate=0.10/0.15 使用 `draft_smart_frequency_global`；rate=0.30 使用已有 rate-sweep 的 `draft_frequency_per_chunk`，因为 old smart global 没有保存 0.30。
- `B = independent online DraftModel selection`。rate=0.10 使用 `pure_online_draft_rate010`；rate=0.15 使用 `qwen3_online_draft_trace_rate015`；rate=0.30 使用本目录补跑的 `pure_online_draft_rate030`。
- `missing = B - A`，即 online Draft 选中但 offline 没选中的 token。
- online Draft target 均为 DraftModel attention score + RRF(k=18) + smart selection，不是裸 top-k。

## 复现命令
```bash
cd /raid/home/hming/FusionRAG-pca-analysis
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python /tmp/analyze_online_draft_missing_tokens.py
```

如果 `pure_online_draft_rate030` 不存在，先补跑：
```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/selector_aware_draft_model/trace_pure_online_draft_selector.py \
  --output_dir MOTIVATION_EXPERIMENTS/offline_vs_online_draft_selection_norm/pure_online_draft_rate030/selected_indices \
  --rate 0.30 --start_sample 0 --end_sample 200 --device cuda:0
```

## 总体集合差异
| rate | n | offline size | online size | hit | missing | online recall | offline precision | missing / online |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 250 | 148.0 | 142.2 | 74.7 | 67.4 | 0.5257 | 0.5049 | 0.4743 |
| 0.15 | 250 | 221.9 | 244.2 | 126.8 | 117.4 | 0.5192 | 0.5713 | 0.4808 |
| 0.30 | 250 | 422.8 | 427.4 | 253.0 | 174.4 | 0.5920 | 0.5984 | 0.4080 |

## Missing token 类型分布

### rate=0.10
| category | frac | count |
|---|---:|---:|
| word | 0.3422 | 5769 |
| capitalized | 0.2707 | 4563 |
| short_word | 0.1576 | 2656 |
| punct_symbol | 0.1152 | 1942 |
| number | 0.0921 | 1553 |
| whitespace | 0.0222 | 375 |

### rate=0.15
| category | frac | count |
|---|---:|---:|
| word | 0.2574 | 7554 |
| capitalized | 0.1882 | 5524 |
| short_word | 0.1195 | 3507 |
| punct_symbol | 0.0779 | 2285 |
| number | 0.0737 | 2162 |
| whitespace | 0.0221 | 649 |

### rate=0.30
| category | frac | count |
|---|---:|---:|
| word | 0.3489 | 15215 |
| capitalized | 0.2153 | 9386 |
| short_word | 0.1587 | 6920 |
| number | 0.1386 | 6045 |
| punct_symbol | 0.0699 | 3050 |
| whitespace | 0.0685 | 2987 |

## Missing token 全局位置分布

### rate=0.10
| global position bin | frac | count |
|---|---:|---:|
| 00-10% | 0.3032 | 5111 |
| 10-20% | 0.0650 | 1096 |
| 20-30% | 0.0405 | 682 |
| 30-40% | 0.0369 | 622 |
| 40-50% | 0.0250 | 421 |
| 50-60% | 0.0188 | 317 |
| 60-70% | 0.0230 | 388 |
| 70-80% | 0.0300 | 505 |
| 80-90% | 0.1011 | 1704 |
| 90-100% | 0.3566 | 6012 |

### rate=0.15
| global position bin | frac | count |
|---|---:|---:|
| 00-10% | 0.1975 | 5795 |
| 10-20% | 0.0533 | 1565 |
| 20-30% | 0.0356 | 1045 |
| 30-40% | 0.0306 | 899 |
| 40-50% | 0.0238 | 697 |
| 50-60% | 0.0188 | 552 |
| 60-70% | 0.0210 | 615 |
| 70-80% | 0.0228 | 670 |
| 80-90% | 0.0748 | 2196 |
| 90-100% | 0.2606 | 7647 |

### rate=0.30
| global position bin | frac | count |
|---|---:|---:|
| 00-10% | 0.2352 | 10257 |
| 10-20% | 0.0742 | 3237 |
| 20-30% | 0.0601 | 2619 |
| 30-40% | 0.0536 | 2338 |
| 40-50% | 0.0464 | 2023 |
| 50-60% | 0.0436 | 1900 |
| 60-70% | 0.0481 | 2098 |
| 70-80% | 0.0635 | 2768 |
| 80-90% | 0.1082 | 4718 |
| 90-100% | 0.2671 | 11645 |

## Missing token 所在 chunk rank 分布

### rate=0.10
| chunk rank bin | frac | count |
|---|---:|---:|
| 00-10% | 0.2923 | 4928 |
| 10-20% | 0.0705 | 1188 |
| 20-30% | 0.0424 | 715 |
| 30-40% | 0.0342 | 576 |
| 40-50% | 0.0246 | 415 |
| 50-60% | 0.0254 | 429 |
| 60-70% | 0.0147 | 247 |
| 70-80% | 0.0297 | 501 |
| 80-90% | 0.0506 | 853 |
| 90-100% | 0.4156 | 7006 |

### rate=0.15
| chunk rank bin | frac | count |
|---|---:|---:|
| 00-10% | 0.1936 | 5681 |
| 10-20% | 0.0567 | 1664 |
| 20-30% | 0.0347 | 1018 |
| 30-40% | 0.0284 | 834 |
| 40-50% | 0.0223 | 653 |
| 50-60% | 0.0234 | 687 |
| 60-70% | 0.0148 | 434 |
| 70-80% | 0.0239 | 702 |
| 80-90% | 0.0401 | 1177 |
| 90-100% | 0.3009 | 8831 |

### rate=0.30
| chunk rank bin | frac | count |
|---|---:|---:|
| 00-10% | 0.2399 | 10462 |
| 10-20% | 0.0728 | 3175 |
| 20-30% | 0.0571 | 2491 |
| 30-40% | 0.0513 | 2236 |
| 40-50% | 0.0456 | 1989 |
| 50-60% | 0.0458 | 1998 |
| 60-70% | 0.0468 | 2040 |
| 70-80% | 0.0566 | 2467 |
| 80-90% | 0.0769 | 3355 |
| 90-100% | 0.3071 | 13390 |

## 当前结论
1. old/unrelated offline draft set 能覆盖 online Draft 的一半左右，但仍有 40% 左右 online Draft token 没被 offline 覆盖；这个 missing 部分是后续 cheap rerank 或轻量 online selector 要解决的对象。
2. 从分布上看，missing token 不是单纯随机噪声；后续应优先检查它是否集中在特定全局位置、chunk rank 或 token 类型。如果集中，可以尝试位置/类型启发式；如果不集中，则需要 query-conditioned rerank。
3. 下一步实验建议：构造 `offline candidate pool@30/50 + cheap rerank@15`，用本文件的 missing 分布指导 rerank 特征。

## 输出文件
- `missing_summary_by_rate.csv`：总体覆盖和漏选比例。
- `missing_token_detail.csv`：每个 missing token 的位置、chunk、token 文本和粗类型。
- `missing_category_distribution.csv`：missing token 类型分布。
- `missing_global_position_distribution.csv`：missing token 在拼接文档中的全局位置分布。
- `missing_chunk_rank_distribution.csv`：missing token 所在召回 chunk rank 分布。
- `per_example_overlap_metrics.csv`：每个 sub-question 的 overlap 指标。
## 实验二：Offline Candidate Pool + Cheap Rerank

### 目的
验证 offline 多存候选池后，是否能用无需模型前向的 cheap rule 从 30%/50% 候选池中选回 15%，从而更接近 online Draft@15。

### 方法
- online target: independent online Draft@15。
- candidate pool: draft/qk/hybrid/docgen 的 30% 或 50% offline 候选池。
- rerank@15: 从候选池中选与 online target 相同数量 token。`oracle_upper_bound` 表示候选池本身最多能覆盖多少 online target，不是可部署方法。其他 rerank 包括 random、global edge、chunk edge、tail/head、combined edge。

### 关键结果
| pool | pool rate | rerank | Jaccard | online recall | precision |
|---|---:|---|---:|---:|---:|
| old_unrelated_offline15 | 0.15 | none | 0.3965 | 0.5248 | 0.6290 |
| draft_freq_pool30 | 0.30 | oracle_upper_bound | 0.2798 | 0.5927 | 0.3448 |
| hybrid70_pool30 | 0.30 | oracle_upper_bound | 0.2645 | 0.5672 | 0.3300 |
| docgen_rank_pool30 | 0.30 | oracle_upper_bound | 0.2566 | 0.5534 | 0.3218 |
| qk_freq_pool30 | 0.30 | oracle_upper_bound | 0.2371 | 0.5196 | 0.3024 |
| draft_freq_pool50 | 0.50 | oracle_upper_bound | 0.2292 | 0.7244 | 0.2507 |
| hybrid70_pool50 | 0.50 | oracle_upper_bound | 0.2247 | 0.7128 | 0.2468 |
| docgen_rank_pool50 | 0.50 | oracle_upper_bound | 0.2094 | 0.6726 | 0.2327 |
| qk_freq_pool50 | 0.50 | oracle_upper_bound | 0.2077 | 0.6682 | 0.2313 |
| draft_freq_pool30 | 0.30 | chunk_edge | 0.3105 | 0.4708 | 0.4708 |
| hybrid70_pool30 | 0.30 | chunk_edge | 0.2999 | 0.4587 | 0.4587 |
| draft_freq_pool50 | 0.50 | chunk_edge | 0.2954 | 0.4532 | 0.4532 |
| draft_freq_pool30 | 0.30 | combined_edge | 0.2932 | 0.4517 | 0.4517 |
| hybrid70_pool50 | 0.50 | chunk_edge | 0.2925 | 0.4498 | 0.4498 |
| draft_freq_pool50 | 0.50 | combined_edge | 0.2915 | 0.4479 | 0.4479 |
| hybrid70_pool50 | 0.50 | combined_edge | 0.2847 | 0.4398 | 0.4398 |
| qk_freq_pool50 | 0.50 | chunk_edge | 0.2829 | 0.4385 | 0.4385 |
| draft_freq_pool30 | 0.30 | chunk_rank_edge | 0.2811 | 0.4369 | 0.4369 |
| docgen_rank_pool30 | 0.30 | chunk_edge | 0.2808 | 0.4351 | 0.4351 |
| qk_freq_pool30 | 0.30 | chunk_edge | 0.2782 | 0.4328 | 0.4328 |
| hybrid70_pool30 | 0.30 | combined_edge | 0.2756 | 0.4307 | 0.4307 |

### 结论
1. 候选池上界明显高于 offline15 baseline，说明“多存候选池”确实包含更多 online Draft token。
2. 但当前无需模型的 edge/tail/head 等 cheap rerank 没有稳定超过 offline15 baseline，说明仅靠位置启发式不足以替代 online Draft。
3. 如果继续走 candidate-pool 路线，需要 query-conditioned rerank，例如 query embedding/token summary 打分、shallow draft score、或轻量 predictor；单纯位置 bias 不够。

输出：`candidate_pool_rerank_summary.csv`, `candidate_pool_rerank_detail.csv`。

## 实验三：Selection Overlap 与 QA 效果关系

### 目的
把 offline 方法和 online Draft 的 selection overlap 与已有 QA accuracy 放到一起，判断“越像 online Draft”是否必然带来更好 QA。

### 关键观察
- old/unrelated offline10 最像 online Draft10，recall 约 0.533；docgen10 只约 0.363。
- 但 QA 上 `offline10_docgen_draft005` 与 `offline10_draft005` 接近，`old70/docgen30 + Draft5` 甚至更高。
- 因此“像 online Draft”不是唯一有效轴；docgen 可能选到另一类 answer-useful token，而不是复刻 online Draft selector。
- 纯 offline docgen15 明显弱，说明 docgen 本身不能替代 online residual；它更像适合作为 offline base 的补充信号。

输出：`qa_overlap_relation_table.csv`。


## 可视化：online Draft 漏选 token 与文档边界

新增图：`figures/online_draft_missing_tokens_boundary_distribution.png`。

这张图把 `missing = online Draft selection - offline old/unrelated draft set` 的 token 按位置展开：
- 左上：missing token 在拼接后的 RAG 文档序列中的全局位置分布；
- 右上：missing token 在各自 retrieved chunk 内部的相对位置分布；
- 下方：rate=0.15 时，missing token 同时按 chunk 召回顺序和 chunk 内位置统计的二维热力图。

红色阴影表示文档或 chunk 的前 10% / 后 10% 边界区域。图中可以看到，query-conditioned gap 并不是均匀分布，而是明显向全局文档首尾、chunk 首尾以及较早/较晚召回 chunk 偏斜。因此，“query-conditioned”描述 offline 无法覆盖 online Draft 的来源；“boundary concentrated”描述这些未覆盖 token 的位置结构。
