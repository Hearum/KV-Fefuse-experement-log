# Offline Doc-Generated Query Calibration 实验计划

## 目标

当前最强的 fair-budget residual baseline 是 `offline10_draft005`：

- offline：`draft_smart_frequency_global` 选 10% token；
- online：full DraftModel residual 再从剩余 token 中补选 5%；
- 总重算预算：15%；
- 现有结果：Main Acc 97/135 (71.85%)，Sub Acc 208/250 (83.20%)。

但现有 offline10 的构造 query 来自 `control_other_example_*`，也就是其他样本的问题。它更像 query-agnostic anchor probing，而不是针对当前文档块自身语义生成的 calibration。

本实验要验证：如果离线阶段只给当前文档块内容，让 GLM 为每个文档块生成一批不泄漏真实测试问题的 calibration queries，再用这些问题构造 offline fixed set，是否能提高 `offline10 + online residual5` 的最终效果。

## 核心假设

对一个文档块来说，由该文档块自身内容生成的问题，会比无关 example 的问题更好地激活文档内部稳定重要 token。

如果这个假设成立，新的 offline10 fixed set 应该：

1. 更接近 online DraftModel 对真实 query 选出的 token；
2. 让 online residual 5% 需要补救的空间变小；
3. 在保持总重算预算 15% 不变的情况下，提高 QA accuracy。

## 实验边界

为了避免数据泄漏：

- 生成 query 时只输入当前文档块文本；
- 不输入真实 evaluation question；
- 不输入 gold answer；
- 不输入同一个 example 的推理链或答案；
- 生成的问题只作为 offline calibration probes，不参与最终回答。

## 新方法名称

建议命名：

- fixed set 方法名：`docgen_draft_smart_frequency_global`
- 10% offline fixed set：`docgen_draft_smart_global_rate010`
- fair-budget pipeline：`offline10_docgen_draft005`

其中 `docgen` 表示 document-generated calibration queries。

## 实验流程

### Step 1. 文档块级 query 生成

对每个 evaluation example 的 retrieved top-k 文档块分别生成 calibration queries。

默认配置：

- dataset：当前 qwen3/musique reflect pipeline 使用的数据；
- retrieved docs：与现有 `offline10_draft005` 完全一致；
- topk：10；
- 每个文档块生成问题数：32；
- 生成模型：项目中已有 GLM API；
- 输入粒度：单个文档块文本；
- 输出格式：JSONL，包含 `example_id`, `doc_id`, `query_id`, `question`, `source_doc_hash`, `prompt_version`, `glm_model`。

这一步是一次性离线预处理资产，不绑定某一次 pipeline run。后续构造 10%/15%/20% fixed set、计算 overlap、跑 QA accuracy，都直接复用生成好的 query JSONL，不重复调用 GLM。

生成 prompt 使用 `glm_docgen_prompt_current.md`。prompt 只要求 LLM 基于给定文本生成 32 个多样化自然问题，不暴露 calibration、token selection 或真实评测任务语境。


### Query Cache 设计

生成好的问题统一作为可复用缓存保存：

- `generated_queries/docgen_queries.jsonl`：逐条问题记录；
- `generated_queries/docgen_query_cache.json`：以 `source_doc_hash + prompt_version + glm_model` 为 key 的缓存索引；
- `generated_queries/docgen_query_manifest.csv`：每个文档块的生成状态、问题数、去重后问题数、失败/重试次数；
- `generated_queries/sample_queries.md`：抽样查看生成质量。

缓存规则：

1. 如果同一个文档块的 `source_doc_hash`、`prompt_version`、`glm_model` 已存在且问题数满足 32，则直接复用；
2. 如果不足 32 或 JSON 解析失败，则只补跑该文档块；
3. 如果修改 prompt，则提升 `prompt_version`，保留旧版本结果，不覆盖；
4. query 生成和后续 selector score cache 分离，避免因为调 rate 或 selector 聚合方式而重复请求 GLM。

### Step 2. 计算 calibration score cache

对每个 example，固定它的 retrieved document sequence。

对 Step 1 生成的 doc-generated calibration queries 运行 DraftModel selector score 计算，保存完整 score/rank cache。

要求与现有 online DraftModel 语义对齐：

- `preprocess=True`；
- `topk=10`；
- DraftModel 使用现有 full DraftModel 路径；
- layer selection 对齐 `dispatch_glm_overnight.sh`：`DRAFT_LAYER_SEL=rrf`, `RRF_K=18`；
- selection 后处理仍使用现有 `smart_query_selection`，不是裸 top-k。

缓存文件需要保留完整分数，便于后续不重跑前向就 sweep rate：

- `scores`: shape = `(num_calibration_queries, doc_token_count)`；
- `queries`: calibration query 文本；
- `labels`: `docgen_exampleXXX_docYY_qZZ`；
- `starts/context_lengths/system_len/context_len`；
- 每个 query 的 smart selected set 或 rank。

### Step 3. 构造 offline fixed set

先复用现有 `draft_smart_frequency_global` 思路，只替换 calibration query 来源。

对每个 calibration query：

1. 根据 DraftModel score 运行 smart selection；
2. 得到该 query 下的 selected token set；
3. 统计每个 doc token 被多少个 doc-generated queries 选中，得到 `freq(token)`；
4. 用 `freq desc, mean_score desc, token_index asc` 排序；
5. 按目标 rate 取 fixed set。

主实验先取 rate=0.10，对齐 `offline10_draft005`。

同时必须保存完整排序，方便后续直接导出 0.05/0.15/0.20/0.30，不重新跑 DraftModel 前向。

### Step 4. Pipeline 评测

主评测只替换 offline fixed set，其他保持与 `offline10_draft005` 完全一致：

- offline fixed set：`docgen_draft_smart_frequency_global`, rate=0.10；
- online residual：full DraftModel residual 0.05；
- total update budget：15%；
- preprocess：沿用当前 pipeline；
- model/dataset/cache/topk/recall/reprocess 参数：对齐 `qwen3_fair_budget_offline_vs_residual`；
- 评测指标：Main Acc, Sub Acc, F1, EM, selection time, wall time。

主对照组：

| 方法 | offline set | online residual | 总预算 | 目的 |
|---|---|---|---:|---|
| `offline10_draft005` | unrelated-control `draft_smart_frequency_global` 10% | full Draft 5% | 15% | 当前 baseline |
| `offline10_docgen_draft005` | doc-generated `draft_smart_frequency_global` 10% | full Draft 5% | 15% | 本实验主方法 |
| `online_draft_rate015` | none | full Draft 15% | 15% | online selector 上限参考 |
| `offline15_docgen_only` | doc-generated fixed set 15% | none | 15% | 观察纯 offline 是否提升 |
| `offline15_draft_smart_only` | unrelated-control fixed set 15% | none | 15% | 纯 offline baseline |

## 需要记录的中间分析

除了最终 accuracy，还要记录 selector 质量：

1. `docgen offline10` vs `pure online Draft10` 的 Jaccard / recall / precision；
2. `docgen offline10` vs 当前 `unrelated-control offline10` 的 Jaccard；
3. `docgen offline10 + online residual5` 覆盖 `online Draft15` 的比例；
4. 每个文档块 generated queries 的去重率、平均长度和失败率；
5. 生成 query 是否过于模板化或集中在段首实体。

## 预期结果解读

可能结果 A：`offline10_docgen_draft005` 明显高于 `offline10_draft005`。

说明 doc-specific calibration queries 能更好构造 offline stable set，是一个值得继续做的方向。

可能结果 B：selector overlap 提高，但 QA accuracy 不提高。

说明 token set 更接近 online Draft 不一定等价于更好回答，需要看这些 token 是否真的影响 online update 后的生成。

可能结果 C：效果无提升或下降。

说明 generated questions 可能引入偏置，或者文档块内生成的问题与真实 RAG query 分布仍不匹配，需要尝试 question generation prompt、query 数、per-doc/per-global 聚合方式或混合 unrelated/docgen。

## 当前状态日志

- 2026-07-07：建立实验文件夹和计划文档。尚未开始生成 query 和跑 pipeline。

- 2026-07-07：根据反馈修正方案：query generation 作为可复用离线资产缓存；prompt 简化为普通文档问题生成，不暴露 calibration/token-selection 语境。当前 prompt 文件：`glm_docgen_prompt_v2.md` / `glm_docgen_prompt_current.md`。

## 2026-07-07 实验结果：doc-generated offline10 + online Draft residual5

### 实验完成状态

- 文档块问题生成完成：共 `3408` 个 document chunk，每个 chunk 生成 `32` 个问题，总计 `109056` 条 generated queries。
- Query 缓存路径：`generated_queries/docgen_queries.jsonl`
- Query manifest：`generated_queries/docgen_query_manifest.csv`
- 示例问题：`generated_queries/sample_queries.md`
- Chunk-local DraftModel score cache 已完成，输出目录：`docgen_chunk_draft_smart_full/`
- 已生成可复用固定集合：`rate_0p1`、`rate_0p15`、`rate_0p2`。
- RAG accuracy pipeline 已跑完 8 个 shard，无 Traceback/KeyError。

### 方法定义

本轮方法命名为 `offline10_docgen_draft005`。

- Offline fixed set：对每个 document chunk 预生成 32 个问题；用 DraftModel 对 `system + single chunk + generated question` 做 chunk-local attention scoring；对 32 个问题的 token 选择结果按 `frequency desc + mean_score desc` 聚合，取 chunk 内 top 10% token 作为 offline fixed set。
- Online residual：沿用现有 `offline10_draft005` 的流程，在 online 阶段再用 DraftModel 从剩余 token 中补充 5%。
- 总重算预算：10% offline + 5% online residual，总计 15%。
- 推理时仍使用 preprocess KV cache；只替换 offline fixed set 的构造方式。

### Accuracy 结果

| 方法 | Offline 构造 | Online residual | 总预算 | Main Acc | Sub Acc | F1 | EM | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `offline10_draft005` | 旧版 calibration-query draft smart fixed set | DraftModel 5% | 15% | 97/135 (71.85%) | 208/250 (83.20%) | 0.1955 | 0.0200 | 之前主表 baseline |
| `offline10_docgen_draft005` | 每个 chunk 生成 32 个问题后构造 fixed set | DraftModel 5% | 15% | 99/135 (73.33%) | 207/250 (82.80%) | 0.2033 | 0.0080 | 本轮新实验 |

结论：doc-generated offline set 在 Main Acc 和 F1 上略优于旧版 calibration-query fixed set，但 Sub Acc 和 EM 没有同步提升。因此它不是稳定碾压，但说明“按文档块自身生成问题来构造 offline set”是有价值的方向。

### 和 pure online Draft10 的 selector overlap

对比对象：`pure_online_draft_rate010`，即真实 online DraftModel 在 rate=10% 下直接选择的 token 集合。

| Offline 方法 | Offline 平均 token 数 | Online Draft10 平均 token 数 | 平均交集 | Jaccard | Offline 覆盖 Online Draft10 | Online Draft10 覆盖 Offline |
|---|---:|---:|---:|---:|---:|---:|
| 旧版 `draft_smart_frequency_global` offline10 | - | - | - | 38.32% | 53.28% | 57.43% |
| 新版 `docgen_draft_smart_frequency_global` offline10 | 137.67 | 142.18 | 53.43 | 23.25% | 36.26% | 37.45% |

结论：doc-generated offline set 与 pure online Draft10 的 token 重合度反而更低，但 QA 主指标略有提升。这说明它不只是“更像 online DraftModel”，而可能选到了一类更偏 chunk-intrinsic 的稳定 token。后续需要进一步看这些 token 的位置分布、attention mass、以及 answer-support sentence 覆盖率。

### 结果文件

- Accuracy summary: `qwen3_docgen_fair_budget/accuracy_summary_docgen.csv`
- Accuracy JSON: `qwen3_docgen_fair_budget/accuracy_summary_docgen.json`
- Selector overlap: `qwen3_docgen_fair_budget/docgen_offline10_vs_pure_online_draft10_overlap.csv`
- Pipeline selected indices: `qwen3_docgen_fair_budget/offline10_docgen_draft005/seg_*/offline_fixed_selected_indices/`
- Chunk fixed sets: `docgen_chunk_draft_smart_full/rate_0p1/chunk_fixed_sets_npz/`

### 后续建议

1. 先不要只看 overlap，因为 docgen set 与 online Draft10 overlap 低但 QA 并不差，说明评价 offline set 不能只用“像不像 online Draft”。
2. 下一步应该比较 docgen offline token 的 attention mass、token 位置分布、以及 answer-support sentence 覆盖率。
3. 如果继续优化 docgen prompt，可以重点让问题覆盖实体关系、因果、时间、地点、定义和比较，但仍保持只基于 chunk 内容生成，避免引入真实 query。

## 2026-07-07 追加实验：old70 + docgen30 hybrid offline10

### 实验目的

前一轮诊断显示：`docgen_offline10` 与 `online Draft10` 的 token overlap 低于旧版 `old_offline10`，但 QA 主指标略好。因此进一步测试 docgen 是否适合作为旧 offline fixed set 的补充，而不是直接替换旧方法。

### 方法定义

方法名：`offline10_hybrid_old70_docgen30_draft005`

- Offline fixed set 总预算仍为 10%。
- 每个 chunk 先保留旧版 `draft_smart_frequency_global` offline10 结果中的前 70%。
- 剩余 30% 预算用 `docgen_draft_smart_frequency_global` 的完整 rank 补足。
- Online residual 仍为 DraftModel 5%。
- 总重算预算仍为 15%。

实现注意：第一次生成 hybrid fixed set 时错误地只处理了 `chunk00...chunk09`，而部分 example 有 `chunk10+`，导致 KeyError。失败目录已备份为 `offline10_hybrid_old70_docgen30_draft005_failed_missing_chunk_*`；最终有效结果使用动态 chunk key 重新生成，覆盖 135 个有效 example、2098 个 chunk，无缺失旧 offline chunk。

### Accuracy 对比

| 方法 | Offline 构造 | Online residual | 总预算 | Main Acc | Sub Acc | F1 | EM | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `offline10_draft005` | 旧版 calibration-query draft smart fixed set | DraftModel 5% | 15% | 97/135 (71.85%) | 208/250 (83.20%) | 0.1955 | 0.0200 | 之前主表 baseline |
| `offline10_docgen_draft005` | 每个 chunk 生成 32 个问题后构造 fixed set | DraftModel 5% | 15% | 99/135 (73.33%) | 207/250 (82.80%) | 0.2033 | 0.0080 | docgen 直接替换 offline10 |
| `offline10_hybrid_old70_docgen30_draft005` | 旧 offline10 保留 70%，docgen rank 补 30% | DraftModel 5% | 15% | 100/135 (74.07%) | 209/248 (84.27%) | 0.1800 | 0.0161 | 本轮 hybrid；最终按 8 个 CSV 去重汇总 |

### 当前结论

1. Hybrid 在 Main Acc 与 Sub Acc 上是这三组里最高的，说明 docgen token 作为补充信号有一定价值。
2. Hybrid 的 F1 低于 docgen-only 和旧 baseline，说明它提高了 judge correctness，但生成答案的字符串匹配质量没有同步提高。
3. 这组结果支持继续研究“多个 offline token source 的融合/rerank”，而不是只寻找单一最强 offline set。
4. 下一步更合理的实验是扫 old/docgen 混合比例，例如 old90/docgen10、old50/docgen50，以及用 answer-support/attention mass 做 rerank，而不是只用固定 70/30。

### 结果文件

- Fixed set: `hybrid_old70_docgen30_rate0p1/chunk_fixed_sets_npz/`
- Accuracy JSON: `qwen3_docgen_fair_budget/accuracy_summary_hybrid_old70_docgen30.json`
- Accuracy CSV: `qwen3_docgen_fair_budget/accuracy_summary_hybrid_old70_docgen30.csv`
- Pipeline outputs: `qwen3_docgen_fair_budget/offline10_hybrid_old70_docgen30_draft005/`

## 2026-07-07 主结果表：Qwen3-32B / MuSiQue / topk=10 / preprocess KV

这一节维护当前可继续开发的主表。后续如果在任意方法基础上继续改，都优先把结果追加到这里，避免散落在不同实验目录里。

### Accuracy 主表

| 方法名 | 重算预算 | Selector / fixed set 组成 | Main Acc | Sub Acc | F1 | EM | 结果路径 | 启动方式 |
|---|---:|---|---:|---:|---:|---:|---|---|
| `online_draft_rate015` | online 15% | 真实 query 下在线调用 full DraftModel 选 15% token | 99/135 (73.33%) | 209/248 (84.27%) | 0.1905 | 0.0081 | `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/summary.csv` | `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/launch_qwen3_rate015_segments.sh` |
| `online_draft_rate050` | online 50% | 真实 query 下在线调用 full DraftModel 选 50% token | 108/135 (80.00%) | 219/248 (88.31%) | 0.2091 | 0.0161 | `MOTIVATION_EXPERIMENTS/qwen3_hybrid70_online_baselines/summary.csv` | 同目录历史 run；如复跑需按该 summary 对应参数重新发起 online Draft rate=0.50 |
| `offline10_draft005` | offline 10% + online 5% | 旧版 calibration-query `draft_smart_frequency_global` fixed set 10%，online full DraftModel residual 5% | 97/135 (71.85%) | 208/250 (83.20%) | 0.1955 | 0.0200 | `MOTIVATION_EXPERIMENTS/qwen3_fair_budget_offline_vs_residual/accuracy_summary.csv` | 历史 fair-budget residual pipeline；同参数可参考 docgen 启动脚本替换 `offline_fixed_set_dir/method` |
| `offline10_docgen_draft005` | offline 10% + online 5% | 每个 chunk 用 GLM 生成 32 个问题，chunk-local DraftModel 聚合成 offline10，online full DraftModel residual 5% | 99/135 (73.33%) | 207/250 (82.80%) | 0.2033 | 0.0080 | `MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/qwen3_docgen_fair_budget/accuracy_summary_docgen.csv` | `MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/launch_docgen_offline10_draft005_accuracy.sh` |
| `offline10_hybrid_old70_docgen30_draft005` | offline 10% + online 5% | offline10 中保留旧 fixed set 70%，用 docgen rank 补 30%，online full DraftModel residual 5% | 100/135 (74.07%) | 209/248 (84.27%) | 0.1800 | 0.0161 | `MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/qwen3_docgen_fair_budget/accuracy_summary_hybrid_old70_docgen30.csv` | `MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/launch_hybrid_old70_docgen30_offline10_draft005_accuracy.sh` |
| `offline20_only` | offline 20% | 纯 offline fixed set 20%，无 online residual | 91/135 (67.41%) | 198/250 (79.20%) | 0.2069 | 0.0120 | `MOTIVATION_EXPERIMENTS/qwen3_fair_budget_offline_vs_residual/accuracy_summary.csv` | 历史 fair-budget residual pipeline 的 `offline20_only` run |

### 方法名解释

- `online_draft_rate015`：不使用 offline fixed set；在线时直接用 full DraftModel 根据真实 query 选择 15% 文档 token 重算。它是当前 15% 预算下的强 online selector 基准，但需要在线跑 DraftModel selection。
- `online_draft_rate050`：同上，但 rate=50%。它不是 fair-budget 15% 对照，主要作为更高重算预算下的上界参考。
- `offline10_draft005`：旧 fair-budget residual baseline。offline 10% 来自之前的 calibration-query `draft_smart_frequency_global` fixed set；online 5% 仍用 full DraftModel 从剩余 token 里补。
- `offline10_docgen_draft005`：本轮 docgen 主方法。offline 阶段只给每个文档块内容，让 GLM 生成 32 个多样化问题；再用这些问题运行 chunk-local DraftModel selector，按 `frequency desc + mean_score desc` 取每个 chunk 的 10%。在线 residual 不变。
- `offline10_hybrid_old70_docgen30_draft005`：本轮组合方法。由于 docgen 和旧 offline set 的重合度较低但 QA 有轻微收益，所以用旧 offline set 作为主干，保留其 70%，再用 docgen rank 补足剩余 30%。总 offline 仍是 10%，不是额外增加预算。
- `offline20_only`：纯 offline 20% fixed set，无 online residual。这个结果说明当前纯 offline set 即使预算更大，accuracy 仍明显弱于带 online Draft residual 的方法。

### 关键资产与路径

- Doc-generated query 缓存：`MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/generated_queries/docgen_queries.jsonl`
- Query manifest：`MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/generated_queries/docgen_query_manifest.csv`
- Query 示例：`MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/generated_queries/sample_queries.md`
- Docgen chunk score/fixed set：`MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/docgen_chunk_draft_smart_full/`
- Docgen offline10 fixed set：`MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/docgen_chunk_draft_smart_full/rate_0p1/chunk_fixed_sets_npz/`
- Hybrid old70+docgen30 fixed set：`MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/hybrid_old70_docgen30_rate0p1/chunk_fixed_sets_npz/`
- Docgen selector diagnostics：`MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/qwen3_docgen_fair_budget/selector_diagnostics/`

### 启动与复现说明

生成 docgen queries：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/generate_docgen_queries.py
```

生成 docgen chunk-local fixed sets：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
bash MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/launch_docgen_chunk_fixed_sets.sh
```

跑 `offline10_docgen_draft005` accuracy：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
bash MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/launch_docgen_offline10_draft005_accuracy.sh
```

跑 `offline10_hybrid_old70_docgen30_draft005` accuracy：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
bash MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/launch_hybrid_old70_docgen30_offline10_draft005_accuracy.sh
```

跑 `online_draft_rate015` / rate=0.15 相关在线基准：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
bash MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/launch_qwen3_rate015_segments.sh
```

注意：不同历史表中 `Sub Acc` 的分母有 `248` 和 `250` 两种，原因是不同 run 的 judge/去重口径略有差异。主表保留各 summary 文件中的原始分母，不强行改写。


## 2026-07-07 追加启动：hybrid ratio 消融与 docgen pure offline

### 启动脚本

- 脚本：`launch_docgen_ablation_full.sh`
- 位置：`MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/launch_docgen_ablation_full.sh`
- 启动方式：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
./MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/launch_docgen_ablation_full.sh
```

### Run name 含义

| Run name | Offline fixed set | Online residual | 总预算 | 目的 |
|---|---|---:|---:|---|
| `offline10_hybrid_old90_docgen10_draft005` | 旧 offline10 保留 90%，docgen rank 补 10% | DraftModel 5% | 15% | 检查少量 docgen 补充是否优于 old70/docgen30 |
| `offline10_hybrid_old50_docgen50_draft005` | 旧 offline10 保留 50%，docgen rank 补 50% | DraftModel 5% | 15% | 检查更激进 docgen 融合是否有效 |
| `offline15_docgen_only` | docgen draft-smart 直接选 15% | 0% | 15% | 检查不调用 online Draft residual 的纯 offline docgen 效果 |

### 当前状态

- 2026-07-07 23:21：三组实验已并行拉起。
- GPU 分配：0-2 跑 `old90/docgen10`；3-5 跑 `old50/docgen50`；6-7 跑 `offline15_docgen_only`。
- 注意：此前 `old90/docgen10` 和 `old50/docgen50` 有一次中止的 partial 输出，已在重新启动前备份为 `*_partial_aborted_*`，不计入正式结果。

## 2026-07-07 实验结果：hybrid ratio 消融与 docgen pure offline

### 主表追加

| 方法 | Offline 构造 | Online residual | 总预算 | Main Acc | Sub Acc | F1 | EM | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `offline10_draft005` | 旧版 calibration-query draft smart fixed set | DraftModel 5% | 15% | 97/135 (71.85%) | 208/250 (83.20%) | 0.1955 | 0.0200 | 旧 baseline |
| `offline10_docgen_draft005` | docgen draft-smart offline10 | DraftModel 5% | 15% | 99/135 (73.33%) | 207/250 (82.80%) | 0.2033 | 0.0080 | docgen 直接替换有轻微收益 |
| `offline10_hybrid_old90_docgen10_draft005` | 旧 offline10 保留 90%，docgen rank 补 10% | DraftModel 5% | 15% | 95/135 (70.37%) | 205/250 (82.00%) | 0.1928 | 0.0200 | 少量 docgen 补充反而下降 |
| `offline10_hybrid_old70_docgen30_draft005` | 旧 offline10 保留 70%，docgen rank 补 30% | DraftModel 5% | 15% | 100/135 (74.07%) | 209/250 (83.60%) | 0.1837 | 0.0160 | 当前 Main/Sub accuracy 最好 |
| `offline10_hybrid_old50_docgen50_draft005` | 旧 offline10 保留 50%，docgen rank 补 50% | DraftModel 5% | 15% | 95/135 (70.37%) | 203/250 (81.20%) | 0.2050 | 0.0160 | docgen 比例过高会伤 accuracy |
| `offline15_docgen_only` | docgen draft-smart offline15 | 0% | 15% | 87/135 (64.44%) | 192/250 (76.80%) | 0.1699 | 0.0080 | 纯 offline docgen 不能替代 online residual |

### 结果解读

1. `old70/docgen30 + online Draft5` 是目前这组里最好的 accuracy 组合，说明 docgen token 适合作为旧 offline set 的中等比例补充。
2. `old90/docgen10` 和 `old50/docgen50` 都比 old70/docgen30 差，说明融合比例不是越少或越多越好，存在一个较窄的有效区间。
3. `offline15_docgen_only` 明显弱于 `offline10_docgen_draft005`，说明 online Draft residual 选出的 5% token 仍然很关键，不能被离线 docgen 直接替代。
4. F1 与 judge accuracy 不完全一致：`old50/docgen50` 的 F1 最高，但 Main/Sub accuracy 低；如果论文主打 answer correctness，优先看 Main/Sub accuracy。

### 结果文件

- 汇总表：`qwen3_docgen_fair_budget/accuracy_summary_docgen_ablation_main_table.csv`
- 单独 summary：
  - `qwen3_docgen_fair_budget/accuracy_summary_offline10_hybrid_old90_docgen10_draft005.json`
  - `qwen3_docgen_fair_budget/accuracy_summary_offline10_hybrid_old50_docgen50_draft005.json`
  - `qwen3_docgen_fair_budget/accuracy_summary_offline15_docgen_only.json`
  - `qwen3_docgen_fair_budget/accuracy_summary_offline10_hybrid_old70_docgen30_draft005.json`
  - `qwen3_docgen_fair_budget/accuracy_summary_offline10_docgen_draft005.json`

### 对应启动方式

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
./MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/launch_docgen_ablation_full.sh
```

该脚本并行启动三组：`old90/docgen10`、`old50/docgen50`、`offline15_docgen_only`。历史 `old70/docgen30` 使用 `launch_hybrid_old70_docgen30_offline10_draft005_accuracy.sh` 单独启动。

## 2026-07-08 实验结果：docgen mean-score offline15

### 方法定义

方法名：`offline15_docgen_mean_score_only`

- 对齐旧版 `draft_smart_mean_score_global` 的思想。
- 但 calibration query 来源从旧 calibration/其他 query 换成每个 document chunk 自己生成的 32 个 docgen query。
- 对每个 chunk，读取已缓存的 docgen DraftModel score，计算 token-wise mean score，按 mean score 从高到低取 top 15%。
- 纯 offline 15%，不使用 online Draft residual。

### Accuracy

| 方法 | Offline 构造 | Online residual | 总预算 | Main Acc | Sub Acc | F1 | EM |
|---|---|---:|---:|---:|---:|---:|---:|
| `offline15_docgen_only` | docgen smart frequency offline15 | 0% | 15% | 87/135 (64.44%) | 192/250 (76.80%) | 0.1699 | 0.0080 |
| `offline15_docgen_mean_score_only` | docgen mean-score offline15 | 0% | 15% | 84/135 (62.22%) | 188/250 (75.20%) | 0.1212 | 0.0000 |

### 结论

在 docgen query 设置下，mean-score 聚合不如 frequency/smart selection 聚合。也就是说，单纯平均 token score 容易选到“普遍高分但不一定关键”的 token；docgen query 更适合用“多问题下反复被选中”的 frequency/stable-set 信号。

### 结果文件

- Fixed set: `docgen_chunk_draft_mean_score_full/rate_0p15/chunk_fixed_sets_npz/`
- Summary: `qwen3_docgen_fair_budget/accuracy_summary_offline15_docgen_mean_score_only.json`
- Pipeline output: `qwen3_docgen_fair_budget/offline15_docgen_mean_score_only/`
- Launch script: `launch_offline15_docgen_mean_score.sh`
