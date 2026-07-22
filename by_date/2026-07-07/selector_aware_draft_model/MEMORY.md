# Selector-Aware Draft Model Memory

这个文件专门记录“低成本 online draft selector / 小 draft model 蒸馏”这条主线。后续所有实验、定义、结果都追加到这里，避免和 offline fixed-set、PCA、KV 更新等其他 motivation 实验混在一起。

## 当前研究目标

FusionRAG 里 online DraftModel selector 的质量较强，但需要额外跑一次 draft model prefill 来给文档 token 打分，selection 开销明显。当前目标不是继续优化 preprocess KV，而是研究能不能构造一个更低成本的 draft selector，在 FusionRAG pipeline 中替代 full DraftModel selector 的 online 选 token 步骤，同时尽量保持 answer accuracy。

目标 selector 应满足：

- online 阶段根据当前 query 和已召回文档给 doc token 排序；
- 输出的是完整 token ranking / score distribution，而不是只服务某一个固定 rate；
- 推理时可以自由选择 top-5%、top-15%、top-30% 等不同重算比例；
- 需要和已有 baseline 在相同数据、相同 rate、相同 pipeline 下比较 accuracy 和 selection latency。

## 已纠正的概念：Middle Draft Model

之前我把 `middle draft model` 理解得不够准确。仓库里历史实验的 `middle` 来自 draft model layer 消融，不是一个训练出来的新模型，也不是一段 middle layers。

代码位置：`ktransformers/util/utils.py`，函数 `_fusionrag_compute_draft_doc_attention_scores`。

当前实现定义：

- 默认 full Draft selector：如果不显式指定 `score_layers`，收集 `range(num_layers // 2, num_layers)`，即 draft model 后半层 attention score，并用 RRF 融合。
- `first4`：收集第 0-3 层。
- `first8`：收集第 0-7 层。
- `first12`：收集第 0-11 层。
- `4,8,12`：只收集指定层 4、8、12 的 score。
- `8,12,16`：只收集指定层 8、12、16 的 score。
- `last`：只收集最后一层。
- `middle`：`collect_layers = {num_layers // 2}`，只收集正中间单层。

当前 draft model 是 `/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`，配置为 36 层，所以：

- default full Draft selector = 第 18-35 层融合；
- `middle` = 第 18 层单层；
- `first12` = 第 0-11 层融合。

旧实验入口通过环境变量/参数控制：

- `FUSIONRAG_DRAFT_SCORE_LAYERS=middle|first12|4,8,12|...`
- 显式指定 score layers 时，默认 `FUSIONRAG_DRAFT_STOP_AFTER_SCORE_LAYERS=1`，会在收集到最高指定层后提前停止 forward。

## 历史 layer 消融结果

来源：`MOTIVATION_EXPERIMENTS/low_cost_draft_selector_probe/README.md`。

Stage 1：20-example selector alignment，对齐对象是 default full Draft selector。

| method | token recall vs default | token Jaccard | residual recall | score forward time(s) |
|---|---:|---:|---:|---:|
| default | 100.00% | 100.00% | 100.00% | 0.1452 |
| first4 | 47.89% | 31.94% | 54.52% | 0.0210 |
| first8 | 63.99% | 47.72% | 60.36% | 0.0366 |
| first12 | 66.56% | 50.48% | 60.08% | 0.0538 |
| layer4_8_12 | 65.10% | 48.71% | 56.40% | 0.0536 |
| layer8_12_16 | 66.59% | 50.34% | 57.18% | 0.0689 |
| middle | 72.21% | 57.04% | 65.62% | 0.0764 |
| last | 56.61% | 39.80% | 58.09% | 0.1401 |

Stage 2：完整 pipeline，配置为 offline fixed set 10% + partial Draft residual 5%，总更新预算 15%。

| method | Main Acc | Sub Acc | residual select(s) | prompt eval(s) |
|---|---:|---:|---:|---:|
| layer4 residual5 | 92/135 (68.15%) | 198/250 (79.20%) | 0.0235 | 0.8817 |
| first12 residual5 | 92/135 (68.15%) | 197/250 (78.80%) | 0.0524 | 0.9139 |
| middle residual5 | 95/135 (70.37%) | 202/250 (80.80%) | 0.0743 | 0.9328 |
| full Draft residual5 | 97/135 (71.85%) | 208/250 (83.20%) | 0.1406 | 1.0007 |

结论：`middle` 是旧消融里最好的 partial Draft 折中，但它不是训练模型；它只是 Qwen2.5-3B draft model 第 18 层单层 attention selector。质量仍低于 full Draft residual。

## 训练目标约束

新的小 draft model / distilled selector 训练不能把 loss 写死为某个 selection percentage，例如固定 top-15%、固定 top-5% mass、固定 BCE positive set。原因是 FusionRAG 后续可能需要 sweep rate，训练出来的模型必须提供可排序的连续 score，而不是只拟合某个固定阈值。

允许的训练目标方向：

- distribution-level KD：对所有 doc token 的 teacher importance distribution 做 KL / JS / cross entropy；
- listwise ranking KD：让 student 的整体排序接近 teacher，不绑定固定 top-k；
- pairwise ranking：抽样 token pair 学习相对大小，也不绑定固定 rate；
- causal LM KD：如果目标是蒸馏真正的小 draft LM，可以蒸馏 teacher logits/hidden states，但最终仍需验证其 attention/token ranking 能否接近 full Draft selector。

不应作为主训练目标：

- 只针对 top-15% 构造 hard label BCE；
- 只最大化 top-15% attention mass；
- 只为 residual 5% 写死一个 selector head。

这些可以作为评估指标或后验分析，但不能作为主训练 loss。

## 必须维护的 Baseline

后续任何新方法都要和下面这些行放在同一张主表里比较：

| baseline | 含义 |
|---|---|
| online_draft_rate015 | 完整 Qwen2.5-3B DraftModel online selector，rate=0.15 |
| offline10_full_draft_residual005 | offline fixed 10% + full Draft residual 5% |
| offline10_middle_residual005 | offline fixed 10% + middle layer residual 5% |
| offline10_layer4_residual005 | offline fixed 10% + layer4/first4 residual 5% |
| offline20_only | 纯 offline fixed set 20%，用于检查 residual 方法是否只是因为总 token 预算更大 |
| HS hidden_head | 训练 selector head 的路线，是参考结果，不等同于 draft LM 蒸馏 |
| CD causal distill 4-layer | 之前做过的 causal LM KD 路线，目前 selector 对齐弱于 native layer4，不能直接当成功结果 |

## 后续记录格式

每次实验追加：

1. 日期和 git commit；
2. 方法定义，尤其是 selector score 从哪里来；
3. 训练数据、验证数据、teacher 信号；
4. loss 公式，明确是否 rate-agnostic；
5. 训练配置和 checkpoint；
6. selector alignment 指标，至少包含 R@5/R@15/R@30、Jaccard@15、score cosine；
7. 真实 FusionRAG accuracy，至少 rate=0.15，必要时再 sweep rate；
8. selection latency / prompt eval latency；
9. 结论和下一步。

## 2026-07-07 记录

- 已确认 `middle` 的真实含义：Qwen2.5-3B draft model 36 层中的第 18 层单层 attention selector，不是训练模型，也不是 middle layers block。
- 已确认新的训练主线必须避免固定 selection rate 的 loss。后续训练脚本中如存在 top-k mass / hard top-15 BCE，只能作为旧尝试记录，不继续作为主线方案。
- 下一步应先整理/冻结现有 CD、HS、partial-layer 三类结果到统一主表，再设计 rate-agnostic 的蒸馏训练方案。

## 2026-07-07 实验启动：first12 attn-prob KL pilot

### 目的

验证旧 ablation 中成本较低但质量不足的 `first12` selector，是否可以通过 rate-agnostic 的 full-distribution KL 蒸馏拉近 full Draft selector。

### 方法定义

- Student：Qwen2.5-3B-Instruct embedding + 前 12 层，使用第 12 层 query-to-doc attention probability 作为 doc token score。
- 初始化：原生 Qwen2.5-3B-Instruct 前 12 层权重。
- Teacher：缓存的 full DraftModel selector score，即 Qwen2.5-3B 后半层 attention/RRF 产生的 teacher_scores。
- Loss：对完整 doc-token score distribution 做 masked KL，temperature=2.0。
- 重要约束：训练 loss 不包含固定 top-k/top-rate hard label；top-5/10/15/30 只作为 validation 指标。

### 启动配置

- 训练缓存：`MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/teacher_cache_wikitext103_train_500k`
- 训练样本：前 100k pair
- Wiki validation：`teacher_cache_wikitext103_val_50k` 前 5k
- MuSiQue validation：`native_initialized_layer4_distill/teacher_cache_musique_val` 全量 685
- GPU：0,1,2,3
- epoch：3
- batch size：1/process
- lr：1e-6
- 输出目录：`MOTIVATION_EXPERIMENTS/selector_aware_draft_model/checkpoints/first12_attnprob_kl_100k_e3`
- 日志：`MOTIVATION_EXPERIMENTS/selector_aware_draft_model/logs/first12_attnprob_kl_100k_e3.log`
- 启动脚本：`MOTIVATION_EXPERIMENTS/selector_aware_draft_model/launch_first12_attnprob_kl_pilot.sh`

### 待观察

重点看 MuSiQue R@15/J@15 是否超过未训练 first12 的趋势，并和 `middle residual5`、`full Draft residual5` 保持同表比较。若 Wiki 提升但 MuSiQue 不动，说明继续扩大 WikiText 训练意义有限，应换 RAG-domain calibration 或更接近 online query 的构造。

### epoch 0 baseline

训练开始前已输出未微调 first12 baseline：

| split | KL | R@5 | J@5 | R@10 | J@10 | R@15 | J@15 | R@30 | J@30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Wiki val | 0.0079 | 0.6168 | 0.4569 | 0.6235 | 0.4597 | 0.6378 | 0.4734 | 0.6834 | 0.5223 |
| MuSiQue val | 0.0101 | 0.5862 | 0.4332 | 0.5728 | 0.4093 | 0.5903 | 0.4237 | 0.6600 | 0.4957 |

当前进度：epoch 1/3 训练中。该表是 selector alignment 指标，不是最终 RAG accuracy。

### epoch 1 validation 更新

first12 attn-prob KL pilot 已完成 epoch 1，训练仍在继续 epoch 2/3。

| epoch | train loss | Wiki KL | Wiki R@15 | Wiki J@15 | MuSiQue KL | MuSiQue R@15 | MuSiQue J@15 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0079 | 0.6378 | 0.4734 | 0.0101 | 0.5903 | 0.4237 |
| 1 | 0.0078 | 0.0078 | 0.6500 | 0.4870 | 0.0101 | 0.5999 | 0.4331 |

观察：epoch 1 对 Wiki 和 MuSiQue selector alignment 都有小幅提升，说明 rate-agnostic KL 对 first12 attention selector 有正向信号；但提升幅度目前还不够大，是否能超过 `middle`/HS 或转化为 FusionRAG accuracy 需要等 epoch 2/3 和后续真实 pipeline 验证。

### 2026-07-07 停止 first12 attn-prob KL pilot

按用户要求停止当前训练，切换到新的训练路线。停止时已完成 epoch 2，epoch 3 未完成。

| epoch | train loss | Wiki R@15 | Wiki J@15 | MuSiQue R@15 | MuSiQue J@15 | MuSiQue KL |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.6378 | 0.4734 | 0.5903 | 0.4237 | 0.0101 |
| 1 | 0.0078 | 0.6500 | 0.4870 | 0.5999 | 0.4331 | 0.0101 |
| 2 | 0.0078 | 0.6500 | 0.4871 | 0.5986 | 0.4318 | 0.0101 |

结论：first12 selector-score KL 有轻微信号，但 epoch 1 后基本饱和，MuSiQue epoch 2 略回落；同时 first12 不够轻量，因此不作为主线继续扩大。后续切换到 `prefill_logits_draft_distill`，训练真正的小 draft LM，再从其 attention 中导出 token importance。

## 2026-07-07 offline10 是否已经覆盖 residual 目标的验证

新增覆盖率验证：`MOTIVATION_EXPERIMENTS/selector_aware_draft_model/offline10_residual5_coverage_probe/README.md`。

结论：offline fixed 10% 并没有覆盖 full DraftModel top15 的大部分。对 250 个 MuSiQue sub-questions，offline10 平均只覆盖 full Draft top15 的 33.5%，full Draft top15 中仍有 66.5% 在 offline10 之外。offline10 的 precision 约 59.9%，说明它选到的 token 质量尚可，但 recall 不足。

因此当前路线的核心问题确实是“剩下 5% residual 怎么挑得更好”，而不是 offline10 已经把 HS/full Draft 要选的 token 基本吃完。按当前 token budget，理想 residual5 最多把最终覆盖率推到约 61.5%，这也说明 offline10+residual5 有预算上限。后续 residual selector 实验必须记录 residual selected set，并报告 `residual5_cover_remaining = |C ∩ (B-A)| / |B-A|`。

### 修正：offline10 覆盖率使用了错误 set

前一段 `offline10 cover full Draft top15 = 33.5%` 使用的是 `hybrid_draft70_qk30_score_per_chunk rate=0.10`，不是 `offline10 + residual5` 主实验里的 `draft_smart_frequency_global rate=0.10`。这个数只能说明 hybrid70-rate0.10 的覆盖情况，不能作为 residual 主线结论。

修正后，使用 `qwen3_fair_budget_offline_vs_residual/offline10_draft005/seg_*/offline_fixed_selected_indices`：

- offline method: `draft_smart_frequency_global`, rate=0.10
- target: independent online full DraftModel top15 trace
- n=250 sub-questions
- offline10 平均 token 数: 148.04
- online Draft top15 平均 token 数: 244.16
- 平均交集: 97.30
- Jaccard: 0.3422
- offline10 recall vs online Draft top15: 40.29%
- offline10 precision vs online Draft top15: 72.56%
- online Draft top15 未被 offline10 覆盖: 59.71%

结论更新：`draft_smart_frequency_global` 的 offline10 本身比误用的 hybrid70 更准，但仍只能覆盖 online full Draft top15 的约 40%。因此 residual 5% 依然有明确优化空间。以后必须用 `residual5_cover_remaining = |C ∩ (B-A)| / |B-A|` 来评价剩下 5% 是否挑得好。

### 严格同 budget 对比：offline10 vs pure online Draft top10

已补充轻量 trace 脚本，只加载 Qwen3 tokenizer + Qwen2.5-3B DraftModel，不加载 Qwen3-32B：

`MOTIVATION_EXPERIMENTS/selector_aware_draft_model/trace_pure_online_draft_selector.py`

输出：

`MOTIVATION_EXPERIMENTS/selector_aware_draft_model/pure_online_draft_rate010/selected_indices`

同 budget 对比：

- A: `offline10 draft_smart_frequency_global`, rate=0.10
- B: `pure online DraftModel`, rate=0.10
- n=250 sub-questions

结果：offline10 平均 148.04 tokens，online Draft10 平均 142.18 tokens，平均交集 74.75 tokens，Jaccard=38.32%，offline10 覆盖 online Draft10=53.28%，online Draft10 覆盖 offline10=57.43%。

结论：offline10 并非很差，但和 query-conditioned online Draft10 仍有明显差距。约 46.7% 的 online Draft10 token 没被 offline10 覆盖，因此 residual 5% 有明确目标空间。
