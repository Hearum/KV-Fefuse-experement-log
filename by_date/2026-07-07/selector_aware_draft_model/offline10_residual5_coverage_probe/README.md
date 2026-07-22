# offline10 + residual5 覆盖率验证

## 目的

验证当前 `offline fixed 10% + online residual 5%` 这条路线里，前面的 offline 10% 是否已经覆盖了 full DraftModel selector 的主要 token。如果 offline10 已经覆盖大部分，那么继续优化 residual5 的意义会较小；反之，如果 offline10 仍漏掉大量 full Draft 认为重要的 token，则 residual5 的挑选质量就是关键优化空间。

## 数据与设置

- 数据：MuSiQue 200 examples，对应 250 个 sub-questions。
- offline set：`full_accuracy_offline_hybrid70_rate_sweep_reflect/rate_0p1/offline_fixed_selected_indices`，即 `hybrid_draft70_qk30_score_per_chunk` 的 offline fixed 10%。
- online target：`qwen3_online_draft_trace_rate015/selected_indices`，即完整 Qwen2.5-3B DraftModel online selector 的 top15%。
- 指标均在 doc token absolute index 集合上计算。

定义：

```text
A = offline fixed 10% token set
B = full DraftModel online top15% token set
missing = B - A
```

核心指标：

```text
offline10_cover_full_draft15 = |A ∩ B| / |B|
offline10_precision_vs_full_draft15 = |A ∩ B| / |A|
remaining_frac = |B - A| / |B|
```

另外估算 residual 5% 的理论上限：由于当前 trace 没保存 full DraftModel 的完整排序，只保存了 top15 集合，因此不能精确复原“排除 A 后的 full Draft top5”。这里用 residual budget 约等于 offline10 token 数的一半，估计如果 residual5 能完美命中 `B-A`，最终最多能覆盖多少 full Draft top15。

## 结果

| 指标 | 数值 |
|---|---:|
| 样本数 | 250 sub-questions |
| offline10 平均 token 数 | 137.67 |
| full Draft top15 平均 token 数 | 244.16 |
| offline10 覆盖 full Draft top15 | 0.3352 |
| offline10 precision vs full Draft top15 | 0.5993 |
| full Draft top15 中 offline10 未覆盖比例 | 0.6648 |
| residual5 预算估计 | 68.86 tokens |
| 理想 residual5 对 missing set 的覆盖上限 | 0.4232 |
| offline10 + 理想 residual5 最终覆盖 full Draft top15 上限 | 0.6149 |

分位数：

| 指标 | p10 | p50 | p90 |
|---|---:|---:|---:|
| offline10 覆盖 full Draft top15 | 0.2820 | 0.3379 | 0.3927 |
| offline10 + 理想 residual5 覆盖上限 | 0.5525 | 0.6194 | 0.6749 |

## 结论

1. offline10 并没有覆盖 full DraftModel top15 的大部分，平均只覆盖约 33.5%。
2. offline10 的 precision 较高，约 59.9%，说明 offline set 选到的 token 里有不少确实和 full Draft 重合；但它的 recall 明显不足。
3. full Draft top15 中约 66.5% token 仍在 offline10 之外，因此 residual5 的挑选质量确实是主要优化空间，而不是一个已经被 offline10 吃掉的边际问题。
4. 在当前预算下，即使 residual5 完美命中 offline10 漏掉的 full Draft token，最终覆盖 full Draft top15 的理论上限也只有约 61.5%。这说明 `offline10 + residual5` 的上限受 token budget 限制，不能期望完全复刻 online full Draft top15。

## 后续需要补充

当前 full Draft trace 只保存了 top15 集合，没有保存完整分数排序。为了精确比较 `HS residual5`、`middle residual5`、`layer4 residual5` 的边际质量，后续需要在 residual pipeline 中额外保存：

- offline fixed set A；
- residual selected set C；
- selector 的完整或至少 top15/top20 排序；
- C 对 `B-A` 的覆盖率。

这样才能回答：不同 residual selector 在“offline10 未覆盖的剩余重要 token”上到底谁挑得更好。

## 2026-07-07 修正：改用真正 residual 实验里的 offline10 set

上面第一版 `offline10_cover_full_draft15=33.5%` 使用的是：

```text
full_accuracy_offline_hybrid70_rate_sweep_reflect/rate_0p1/offline_fixed_selected_indices
method = hybrid_draft70_qk30_score_per_chunk
```

这不是 `offline10 + residual5` 主实验里的 offline set，因此不能作为当前 residual 路线的最终结论。

重新使用真正 residual 实验中的 offline set：

```text
MOTIVATION_EXPERIMENTS/qwen3_fair_budget_offline_vs_residual/offline10_draft005/seg_*/offline_fixed_selected_indices
method = draft_smart_frequency_global
rate = 0.10
```

对齐独立的 online full Draft top15 trace：

```text
MOTIVATION_EXPERIMENTS/qwen3_online_draft_trace_rate015/selected_indices
method = DraftModel
rate = 0.15
```

修正结果如下：

| 指标 | 数值 |
|---|---:|
| 样本数 | 250 sub-questions |
| offline10 平均 token 数 | 148.04 |
| online full Draft top15 平均 token 数 | 244.16 |
| 交集平均 token 数 | 97.30 |
| Jaccard | 0.3422 |
| offline10 覆盖 online Draft top15，即 recall | 0.4029 |
| offline10 被 online Draft top15 覆盖，即 precision | 0.7256 |
| online Draft top15 中 offline10 未覆盖比例 | 0.5971 |
| recall p10 / p50 / p90 | 0.2902 / 0.3991 / 0.5223 |

解释：真正 `draft_smart_frequency_global` 的 offline10 比刚才误用的 hybrid70 rate0.10 更接近 online Draft：recall 从 33.5% 提高到 40.3%，precision 从 59.9% 提高到 72.6%。但它仍然只覆盖 online Draft top15 的约 40%，还有约 60% 的 online Draft top15 token 在 offline10 外面。

另外，如果拿 `offline10_draft005` 同一次运行里保存的 `selected/example*_DraftModel_rate0p1.json` 作为 B，会得到：

| 指标 | 数值 |
|---|---:|
| offline10 覆盖 same-run selected | 0.5731 |
| same-run selected 覆盖 offline10 | 1.0000 |

这个结果不能解释为 offline10 与 full Draft target 的自然重合度，因为 same-run selected 本身包含 offline fixed set，`same-run selected 覆盖 offline10 = 1.0` 是实现构造导致的。

因此后续真正要比较 residual selector 的质量，应该明确使用：

```text
A = offline10 draft_smart_frequency_global
B = independent online full Draft top15
C = residual selector 额外补的 token
residual5_cover_remaining = |C ∩ (B - A)| / |B - A|
```

## 2026-07-07 严格同 budget 对比：offline10 vs pure online Draft top10

用户指出：判断 `offline10` 本身质量时，比较对象必须也是 `online Draft top10`，不能拿 top15 比。前面 top15 对比只适合讨论 residual 的目标空间，不适合评价 offline10 是否准确。

为避免完整 Qwen3 pipeline OOM，并避免 `offline10_draft005/selected/*.json` 中包含 offline fixed set 的实现污染，新增轻量 trace 脚本：

```text
MOTIVATION_EXPERIMENTS/selector_aware_draft_model/trace_pure_online_draft_selector.py
```

该脚本只加载：

- Qwen3 tokenizer，用于复用原 pipeline 的 tokenization 和 absolute index；
- Qwen2.5-3B DraftModel，用于计算 DraftModel selector score；

不加载 Qwen3-32B 主模型，不做生成。输出目录：

```text
MOTIVATION_EXPERIMENTS/selector_aware_draft_model/pure_online_draft_rate010/selected_indices
```

对比设置：

```text
A = offline10 draft_smart_frequency_global, rate=0.10
B = pure online DraftModel, rate=0.10
```

结果：

| 指标 | 数值 |
|---|---:|
| 样本数 | 250 sub-questions |
| offline10 平均 token 数 | 148.04 |
| pure online Draft10 平均 token 数 | 142.18 |
| 平均交集 token 数 | 74.75 |
| Jaccard | 0.3832 |
| offline10 覆盖 online Draft10 | 0.5328 |
| online Draft10 覆盖 offline10 | 0.5743 |
| online Draft10 中 offline10 未覆盖比例 | 0.4672 |

分位数：

| 指标 | p10 | p50 | p90 |
|---|---:|---:|---:|
| Jaccard | 0.1905 | 0.3847 | 0.5882 |
| offline10 覆盖 online Draft10 | 0.3371 | 0.5410 | 0.7009 |
| online Draft10 覆盖 offline10 | 0.2747 | 0.5751 | 0.9002 |

结论：在相同 10% budget 下，offline10 与 pure online Draft10 的重合度不是很低，但也远没有接近一致。offline10 平均能覆盖 online Draft10 的约 53.3%，仍有约 46.7% 的 online Draft10 token 不在 offline10 中。这支持当前思路：offline10 负责 query-agnostic/stable 部分，后续 residual 5% 的目标就是补 online query-conditioned selector 认为重要但 offline10 没覆盖的部分。
