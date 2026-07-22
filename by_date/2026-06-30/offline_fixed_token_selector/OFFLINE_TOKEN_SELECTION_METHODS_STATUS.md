# Offline Token Selection Methods Status

## 当前问题

目标是在不使用真实 online query 的情况下，offline 阶段提前选出质量足够高的 recompute token set，降低 online selection 开销。

已有完整 accuracy 结果显示：

| method | Main Acc | Sub Acc | F1 | selection |
|---|---:|---:|---:|---:|
| online QK rate=0.15 | 62.22% | 75.40% | 0.4897 | 0.1032s |
| online draft rate=0.15 | 73.33% | 84.68% | 0.5347 | 0.1606s |
| offline draft frequency | 62.96% | 73.79% | 0.4909 | 0 |
| offline draft mean | 62.22% | 73.79% | 0.4916 | 0 |
| offline QK frequency | 58.52% | 72.18% | 0.4696 | 0 |
| offline full-attn frequency | 56.30% | 68.95% | 0.4373 | 0 |

结论：online draft selector 的质量显著高于 online QK，但 selection 仍要在线跑 draft model，开销更高。offline draft stable set 是当前最强的纯 offline 方案。QK fixed set 虽然更像 online QK，但 accuracy 更差；full-attention calibration 更差。

## 新增候选方法

### 1. QK + Draft Hybrid

目录：

`MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_hybrid_qk_draft_rate015_full`

方法：

- `hybrid_draft70_qk30_score_per_chunk`
- `hybrid_draft50_qk50_score_per_chunk`
- `hybrid_draft70_then_qk30_per_chunk`
- `hybrid_qk_draft_intersection_fill_draft_per_chunk`
- `hybrid_qk_draft_rrf_per_chunk`

动机：

draft selector 的 attention mass 和 accuracy 更好，QK selector 与原 FusionRAG online 行为更接近。hybrid 用于测试能否保留 draft 的质量，同时补充 QK 的覆盖。

### 2. Pure Lexical / Structure Offline Set

目录：

`MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_lexical_rate015_full`

方法：

- `lexical_entity_number_per_chunk`
- `lexical_idf_entity_per_chunk`
- `lexical_idf_entity_boundary_per_chunk`

动机：

完全不做 query/draft/QK 前向，只利用文档内实体、数字、罕见 token、边界和句首先验。它是最便宜的 offline selector，但 overlap 显示它与 draft/QK 差异很大，因此更适合作为下界或补充项。

### 3. Draft + Lexical Fill

目录：

`MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_draft_lexical_rate015_full`

方法：

- `draft_score_lexical_tiebreak_per_chunk`
- `draft90_lexical10_per_chunk`
- `draft85_lexical15_per_chunk`
- `draft80_lexical20_per_chunk`

动机：

保留 draft stable core，用少量 lexical/entity token 填充，希望补到 draft 没覆盖的答案实体。

## CPU Overlap 结果

输出：

`MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/offline_set_overlap_analysis/README.md`

重点结果：

| candidate | reference | Jaccard | coverage reference by candidate |
|---|---|---:|---:|
| draft_score_lexical_tiebreak | draft_freq | 0.9764 | 0.9874 |
| hybrid_draft70_qk30 | draft_freq | 0.9244 | 0.9590 |
| hybrid_intersection_fill | draft_freq | 0.8976 | 0.9437 |
| draft90_lexical10 | draft_freq | 0.8235 | 0.9032 |
| hybrid_draft50_qk50 | qk_freq | 0.6763 | 0.8003 |

解释：

- `draft_score_lexical_tiebreak` 几乎是 draft frequency 的轻微重排，风险最低。
- `hybrid_draft70_qk30` 和 `hybrid_intersection_fill` 仍高度覆盖 draft，同时引入一部分 QK 信息。
- `draft90_lexical10` 是比较温和的 lexical 补充。
- 纯 lexical 和 draft/QK overlap 很低，单独 accuracy 可能不会高，但可以检验“文档实体先验”是否有独立价值。

## Accuracy 待跑队列

等待脚本：

`launch_offline_selector_candidates_when_free.sh`

结果目录：

`MOTIVATION_EXPERIMENTS/full_accuracy_offline_selector_candidates_reflect`

当前机器状态：

8 张 GPU 被 root 的 `sglang::scheduler_TP*` 服务占用，每张约 91GB，当前不能直接跑 Qwen accuracy，否则会 OOM。

等待脚本已经后台挂起；只有检测到 8 张 GPU 显存使用都低于 20GB 后才会启动以下 8 组：

1. `draft_score_lexical_tiebreak`
2. `hybrid_draft70_qk30_score`
3. `hybrid_intersection_fill_draft`
4. `draft90_lexical10`
5. `draft80_lexical20`
6. `hybrid_draft50_qk50_score`
7. `lexical_entity_number`
8. `lexical_idf_entity_boundary`

## 2026-07-01 完整 Accuracy 结果

输出：

`MOTIVATION_EXPERIMENTS/full_accuracy_offline_selector_candidates_reflect_summary/README.md`

所有候选均已跑满 248 个 subquestion，没有 killed/traceback。

| method | Main Acc | Sub Acc | F1 | EM | prefill(s) | selection(s) |
|---|---:|---:|---:|---:|---:|---:|
| online QK rate=0.15 | 62.22% | 75.40% | 0.4897 | 0.2137 | 0.2311 | 0.1032 |
| online draft rate=0.15 | 73.33% | 84.68% | 0.5347 | 0.2177 | 0.3659 | 0.1606 |
| offline draft frequency | 62.96% | 73.79% | 0.4909 | 0.1935 | 0.1338 | 0 |
| offline draft mean | 62.22% | 73.79% | 0.4916 | 0.1895 | 0.1352 | 0 |
| hybrid draft70/qk30 score | 65.19% | 75.00% | 0.4872 | 0.1815 | 0.1721 | 0 |
| hybrid intersection-fill-draft | 63.70% | 74.60% | 0.4935 | 0.1855 | 0.1707 | 0 |
| draft score + lexical tiebreak | 62.96% | 73.39% | 0.4943 | 0.1935 | 0.1769 | 0 |
| draft90 lexical10 | 60.74% | 72.98% | 0.4924 | 0.1895 | 0.1656 | 0 |
| draft80 lexical20 | 61.48% | 73.79% | 0.4809 | 0.1935 | 0.1772 | 0 |
| hybrid draft50/qk50 score | 61.48% | 73.39% | 0.4794 | 0.1774 | 0.1723 | 0 |
| lexical entity number | 60.00% | 73.39% | 0.4755 | 0.2016 | 0.1722 | 0 |
| lexical idf entity boundary | 55.56% | 68.95% | 0.4392 | 0.1492 | 0.1684 | 0 |

关键观察：

1. online draft rate=0.15 的质量明显高于 online QK：Main Acc 73.33% vs 62.22%，Sub Acc 84.68% vs 75.40%，说明 draft model selector 本身更强。
2. online draft 的代价也更高：selection 0.1606s，高于 online QK 的 0.1032s；prefill 0.3659s，也明显更重。因此它更适合作为 offline calibration 信号，而不是直接在线使用。
3. `hybrid_draft70_qk30_score` 是当前 Main Acc 最高的 offline fixed selector：65.19%，高于 online QK 的 62.22%，并且 selection 开销为 0。
4. `hybrid_intersection_fill_draft` 的 Sub Acc 74.60%，F1 0.4935，也略高于 offline draft，接近 online QK。
5. 纯 lexical 方法没有崩，但不能作为主 selector；`lexical_entity_number` Sub Acc 73.39%，说明文档内实体/数字先验有一定信号，但不如 draft/QK hybrid 稳。
6. `lexical_idf_entity_boundary` 明显退化，说明简单 IDF+边界先验会选到过多结构性或罕见但无用 token。
7. 最强方向变成：用 online draft 证明“更强 selector 存在”，再把 draft/QK 信号迁移到 offline fixed set，减少 online selection。

## 当前判断

最有希望的方向不是纯 QK，也不是 full-attention calibration，而是：

```text
offline draft stable set as the main anchor
+ small auxiliary signals from QK or lexical/entity prior
```

当前完整 accuracy 已经支持更具体的说法：

```text
offline draft stable set as the main anchor
+ calibrated QK score as a secondary offline signal
```

其中 `hybrid_draft70_qk30_score` 目前最值得作为候选方法继续扩展 rate sweep 和更多数据集验证。

## 2026-07-02 Hybrid Draft70/QK30 Rate Sweep

输出：

`MOTIVATION_EXPERIMENTS/full_accuracy_offline_hybrid70_rate_sweep_reflect_summary/README.md`

方法固定为：

```text
hybrid_draft70_qk30_score_per_chunk
```

含义：

```text
offline draft calibration score * 0.7
+ offline QK calibration score * 0.3
```

不同 rate 下，仍然是完全 offline fixed set，online selection 开销为 0。

| rate | Main Acc | Sub Acc | F1 | EM | rows |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 59.26% | 71.77% | 0.4663 | 0.1855 | 248 |
| 0.10 | 56.30% | 70.16% | 0.4719 | 0.1855 | 248 |
| 0.15 | 65.19% | 75.00% | 0.4872 | 0.1815 | 248 |
| 0.20 | 62.22% | 75.40% | 0.4979 | 0.2016 | 248 |
| 0.30 | 68.89% | 79.03% | 0.5127 | 0.2016 | 248 |
| 0.40 | 71.85% | 81.45% | 0.5399 | 0.2177 | 248 |
| 0.50 | 77.04% | 85.08% | 0.5475 | 0.2218 | 248 |

对照：

| method | Main Acc | Sub Acc | F1 | EM |
|---|---:|---:|---:|---:|
| rate1 full attention | 77.78% | 87.90% | 0.5692 | 0.2298 |
| online draft rate=0.15 | 73.33% | 84.68% | 0.5347 | 0.2177 |
| online QK rate=0.15 | 62.22% | 75.40% | 0.4897 | 0.2137 |
| offline draft frequency rate=0.15 | 62.96% | 73.79% | 0.4909 | 0.1935 |

关键观察：

1. rate=0.15 已经超过 offline draft 和 online QK 的 Main Acc，同时 selection=0。
2. rate=0.30 出现明显质量跃升，Sub Acc 达到 79.03%，说明 offline fixed anchors 增加后确实能持续补齐有用 token。
3. rate=0.50 基本接近 full attention：Main Acc 77.04% vs 77.78%，Sub Acc 85.08% vs 87.90%。
4. 这说明该方法不是只在一个固定 rate 上偶然有效，而是存在比较清楚的 rate-quality scaling。
5. 当前表里的 prefill 时间受 GPU 共享负载影响较大，不适合做严格速度比较；accuracy 和 selection=0 是可信的。若要写系统性能，需要之后单独做 clean profile。

当前更强的 motivation 表述：

```text
Document KV update contains query-stable anchors.
These anchors can be estimated offline by calibrating a draft selector and a QK selector on unrelated queries.
At inference time, online token selection can be skipped entirely.
Increasing the offline anchor budget smoothly recovers the quality of full attention.
```
