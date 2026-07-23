# Offline/Residual Selector Long-Term Research Tracker

本文档用于维护 FusionRAG selector / recompute token 方向的长期 motivation 实验。主线问题是：能否在不完整依赖 online DraftModel selection 的情况下，获得接近 online draft 或 full recompute 的质量，同时降低 online 开销。

关联主表：

```text
MOTIVATION_EXPERIMENTS/qwen3_offline_improvement_probe_summary.md
```

维护规则：

- 同一模型/数据集配置的新实验结果，必须追加到上面主表，而不是只写在子实验文件夹里。
- 每个新实验必须记录：模型、数据集、topk、runtime KV 类型、selector 方法、rate、是否调用 DraftModel、是否使用真实 query、是否泄漏 test query、结果 CSV 路径、启动脚本、Main Acc/Sub Acc/F1/EM、关键 profile 时间。
- 如果某轮实验分母不同，例如 `248` vs `250` sub-questions，保留原始分母，并在 note 中说明。
- 子实验目录可以保存详细日志、trace 和中间数据，但主表必须提供可横向比较的摘要。

## 当前已知背景

在 Qwen3-32B / MuSiQue / Reflect pipeline 上已有结果显示：

- `full_rate1`: Main 105/135，Sub 215/250。
- `online_draft_rate015`: Main 99/135，Sub 209/248。
- `offline_hybrid70_rate015`: Main 93/135，Sub 196/248。
- `draft_smart_mean_score_global`: Main 95/135，Sub 204/250。
- `draft_smart_frequency_global`: Main 94/135，Sub 203/250。
- `offline frequency + online Draft residual 0.05`: Main 103/135，Sub 217/250。

解释：纯 offline fixed set 有提升，但仍低于 online draft。加入 5% online Draft residual 后质量明显提升，说明 offline set 可以覆盖稳定主体 token，但仍存在 query-specific residual token。

重要 caveat：`online Draft residual` 仍然完整调用 DraftModel selection，所以它证明的是质量路径，而不是已经证明 online selection 开销被省掉。下一步重点是把 residual selection 的 online 成本降下来，或完全 offline 化。

## Long-Term Task A: 纯 offline 方法逼近 online DraftModel 质量

目标：不在 online 阶段调用 DraftModel，只依赖 offline 预处理出的 fixed/ranked token sets，尽量接近 `online_draft_rate015` 的质量。

核心问题：

- 是否存在对每个文档块稳定有效的 token ranking？
- fixed set 的失败主要来自 token 覆盖不足、chunk 选择错误，还是排序/聚合方式错误？
- 是否需要 per-document、per-chunk、per-position、per-query-cluster 多粒度 fixed sets？

已有证据：

- 早期 per-chunk draft score aggregation 效果一般，低于 hybrid70。
- smart offline draft global aggregation 比旧 offline 略好，但仍低于 online draft。
- online draft trace 聚合成单一 fixed set 的 leaky oracle 反而较差，说明简单固定集合不能完整复现 query-specific selection。

建议实验：

1. 保存每个文档块的 offline token ranking，而不是只保存 top rate set。
2. 复用同一套 ranking 做 rate sweep：0.05, 0.10, 0.15, 0.20, 0.30, 0.50。
3. 对比 ranking 来源：raw KV QK、preprocess KV QK、draft attention、full attention calibration、lexical/entity/boundary heuristic、hybrid score。
4. 对比聚合粒度：per-chunk、per-document global、position-normalized、chunk-order-robust。
5. 对比部署方式：固定 top-r、固定 top-n、按 chunk 长度自适应、按 chunk role/boundary 自适应。

需要保存的中间数据：

- 每个 document chunk 的 token list、global token index、chunk-local token index。
- 每种 offline 方法得到的完整 ranking score，而不是只保存 selected set。
- 每个 rate 下 materialized selected set，可以从 ranking 后处理生成。
- 每个 query 的 online draft selected set，用于覆盖率和 error case 分析。

成功标准：

- 不调用 online DraftModel，rate=0.15 时接近或超过 `online_draft_rate015` 的 Main/Sub Acc。
- 或者在更高 offline rate 下，例如 0.30，接近 full_rate1，但 online 开销显著低于 full recompute。

## Long-Term Task B: 轻量 predictor 替代 online DraftModel

目标：保留 query-conditioned 能力，但不运行完整 DraftModel selection。用一个轻量 predictor 预测 residual tokens 或从多个 offline candidate sets 中选择。

候选形式：

- 小 MLP / linear probe：输入 query embedding、token offline summary、chunk summary，输出 token score。
- Bi-encoder / dot product：query embedding 与 token/document summary 做轻量打分。
- Learning-to-rank：用 online draft selected token 作为 pseudo label，训练 token ranking。
- Chunk-level gate + token-level ranking：先预测哪些 chunk 需要补 residual，再只在这些 chunk 上挑 token。

训练标签：

- online DraftModel selected set。
- full attention high-mass token set。
- online recompute 后 KV delta 较大的 token。
- answer correctness sensitive token，如果后续能构造 leave-one-out 或 mask ablation。

最小实验：

1. 用现有 calibration queries 生成 token labels：selected / not selected，或 rank score。
2. 为每个 token 构造 offline features：position、chunk id、token type、offline draft frequency、offline QK score、preprocess/raw KV score、entity/boundary flags。
3. query features 先用简单 embedding，不必立刻训练大模型。
4. 用 held-out examples 测 ranking 覆盖率：Jaccard、online selected recall、top-rate selected mass。
5. 通过 pipeline 跑 accuracy，和 online draft / offline smart / full rate1 对比。

关键约束：

- 不能用 test query 构造对应 example 的 offline set，否则只作为 leaky oracle 诊断。
- 必须区分 calibration split 和 evaluation split。

## Long-Term Task C: 只在 offline fixed set 覆盖不足的 chunk 上调用 DraftModel

目标：不是完全去掉 DraftModel，而是减少 DraftModel selection 的作用范围。只对 offline fixed set 可能覆盖不足的 chunk 调用 online DraftModel 或更重的 selector。

假设：

- 有些 chunk 的 stable token set 覆盖率高，不需要 online residual。
- query-specific residual 可能集中在少数 chunk；如果能先定位这些 chunk，就能减少 DraftModel scoring 范围。

实验设计：

1. 对每个 query，计算 online draft selected set 与 offline fixed set 的 residual token 分布。
2. 统计 residual 落在哪些 chunk：non-empty chunk 数、top chunk residual 占比、entropy、是否和 retrieval rank 相关。
3. 设计 chunk gate：按 retrieval rank、query-document similarity、offline coverage confidence、chunk entropy 等选择少数 chunk。
4. 只在 gated chunks 上运行 DraftModel residual selection，其他 chunk 只用 offline set。
5. 对比 full online draft residual：accuracy、selection time、prompt eval time。

需要保存：

- 每个 query 的 per-chunk residual count。
- 每个 chunk 的 offline coverage confidence。
- chunk gate 后的 selected chunks。
- gate miss 的 tokens 和对应错误样本。

## Long-Term Task D: Query embedding 与 offline token summary 轻量打分

目标：online 阶段只做 embedding-level 或 vector-level 打分，不跑 DraftModel forward。

可行实现：

- offline 为每个 token 保存 summary vector，例如 raw/preprocess KV 的低维投影、token hidden state PCA、chunk-local statistics。
- online 只计算 query embedding，与 token summary 点积或小 MLP 打分。
- ranking 后取 top residual tokens 或直接取 top rate tokens。

实验设计：

1. 先不训练，做 simple score：query embedding dot token summary。
2. 比较不同 summary：raw key、preprocess key、value mean、token hidden state、PCA-reduced vector。
3. 用 online draft selected token 作为目标，评估 recall/Jaccard。
4. 若 recall 有信号，再跑 pipeline accuracy。

风险：

- 如果 token summary 和 query-specific draft selection 弱相关，这条线可能只能作为 chunk gate，而不是 token-level selector。

## Long-Term Task E: 多个 offline candidate sets + online lightweight routing

目标：offline 不只保存一个 fixed set，而是保存多个候选 token sets；online 只做轻量 routing，选择一个或组合几个 sets。

候选 set 来源：

- 不同 calibration query cluster。
- 不同 document role：early docs / late docs / entity-heavy / boundary-heavy。
- 不同 selector：draft-frequency、draft-mean、QK、full-attention-anchor、lexical/entity。
- 不同 rate 或不同 precision/recall 偏好的 sets。

online routing：

- query embedding 到 cluster。
- query 与 document/chunk summary 相似度。
- retrieve-time similarity / reranker score。
- 很小的 classifier 选择 candidate set id。

实验设计：

1. 构造 K 个 candidate sets，不在线调用 DraftModel。
2. oracle routing：对每个 query 选择与 online draft overlap 最高的 candidate set，得到上界。
3. lightweight routing：用 query embedding/metadata 预测 candidate set。
4. 对比 single fixed set、online draft、full rate1。

成功标准：

- oracle routing 明显高于 single fixed set，说明多候选集合有潜力。
- lightweight routing 接近 oracle routing，说明可以低成本部署。

## Long-Term Task F: residual token 是否能由历史/calibration query 稳定预测

目标：验证 online residual token 是否具有稳定性。如果 residual 也稳定，就有机会离线预存；如果 residual 高度 query-specific，就需要 predictor/routing。

已有相关实验：

- `qwen3_offline_online_residual_probe`：smart offline set 对 online draft 的覆盖率约 55%-60%，residual 平均 86-97 tokens，分布在多个 chunk。
- 这说明 residual 不是极小噪声，但还没有回答 residual 是否能由历史 query 收敛预测。

最小实验设计：

1. 对每个 document/example，收集 M 个 calibration queries 的 online draft selected set。
2. 固定一个 offline base set，例如 `draft_smart_frequency_global`。
3. 对每个 query 计算 residual set：`online_draft_selected - offline_base_selected`。
4. 统计每个 token 成为 residual 的频率，画 histogram：0 次、1 次、...、M 次。
5. 随着 calibration query 数增加，构造 residual stable set，看它对 held-out query residual 的 recall/Jaccard 是否收敛。
6. 做 related query 与 unrelated query 的分组比较，检查 residual 稳定性是否只在相关 query 内成立。
7. 从 residual frequency ranking 生成 offline residual candidate set，替代 online residual 0.05 跑 pipeline。

需要保存：

- 每个 query 的 full online draft ranking/score，不只保存 selected set。
- 每个 query 的 residual set。
- 每个 token 的 residual frequency、mean score、max score。
- 不同 calibration query 数下的 stable residual set。
- held-out query 的 coverage/Jaccard/accuracy。

判断标准：

- 如果少数 token residual frequency 很高，并且 held-out recall 随 calibration query 数稳定上升，则 residual 可离线化。
- 如果 frequency 分布平坦，且 held-out recall 不收敛，则 residual 强 query-specific，需要轻量 predictor 或 routing。

## Execution Log

- 2026-07-05：完成 layer4-only offline smart fixed set 完整实验，目录：MOTIVATION_EXPERIMENTS/qwen3_layer4_draft_smart_global_rate015。结果：layer4_draft_smart_frequency_global 为 Main 87/135、Sub 190/250；layer4_draft_smart_mean_score_global 为 Main 84/135、Sub 190/250。结论：单层 layer4 可以低成本给 residual 近似信号，但作为纯 offline smart ranking 明显不够，低于 full Draft smart frequency 的 94/135、203/250。
- 2026-07-05：完成 fair-budget shallow 对照：offline draft_smart_frequency_global 0.10 + shallow Draft layer4 residual 0.05，目录：MOTIVATION_EXPERIMENTS/qwen3_offline10_plus_shallow_layer4_residual_rate015。结果 Main 92/135、Sub 198/250、residual select time 0.0235s。结论：同为 15% 总预算时，该方法低于 offline10 + full Draft residual5 的 97/135、208/250，也低于 pure offline15 的 94/135、203/250；layer4 residual 省时但质量不足。
- 2026-07-05：完成 fair-budget 对照，目录：`MOTIVATION_EXPERIMENTS/qwen3_fair_budget_offline_vs_residual`。结果：total 15% 下 `offline10 + Draft residual5` 为 97/135、208/250，高于 `offline15 only` 的 94/135、203/250；total 20% 下 `offline20 only` 为 91/135、198/250，明显低于 `offline15 + Draft residual5` 的 103/135、217/250。结论：residual 收益不是单纯来自更多重算 token，而是来自 query-conditioned selection。
- 2026-07-05：完成完整 pipeline：`offline draft_smart_frequency_global 0.15 + shallow Draft layer4 residual 0.05`，目录：`MOTIVATION_EXPERIMENTS/qwen3_offline_freq_plus_shallow_layer4_residual_rate015`。结果 Main 96/135、Sub 204/250，residual select time 0.0238s。结论：layer4 residual 显著省 selection 开销，但质量远低于 full Draft residual 103/135、217/250；单层浅层 predictor 不够，需要 first-k-layer aggregation、稍深层或训练式 predictor。
- 2026-07-05：完成 shallow DraftModel selector smoke + 20-example alignment，目录：`MOTIVATION_EXPERIMENTS/shallow_draft_selector_probe`。结论：layer4 是当前最有希望的浅层 predictor，20-example 上 token recall vs default DraftModel 为 62.1%，chunk recall 99.5%，score-forward time 0.024s vs default 0.145s；小样本答案 14/18 main、26/32 sub，低于 default 15/18、29/32 但明显好于 layer0-2。下一步应跑 `offline fixed 0.15 + layer4 residual 0.05` 的完整 accuracy。
- 2026-07-05：完成 chunk gate heuristic 诊断，目录：`MOTIVATION_EXPERIMENTS/chunk_gate_heuristic_probe`。结论：oracle top-5 chunks 可覆盖 86%-91% residual，但可部署静态启发式 top-5 只覆盖约 25%-34%；因此 chunk gate 有上界潜力，但需要 query-conditioned chunk predictor，不能只靠长度、检索顺序或 offline base 密度。
- 2026-07-05：完成 residual chunk concentration 诊断，目录：`MOTIVATION_EXPERIMENTS/residual_chunk_concentration_probe`。结论：residual 不是只集中在 1-2 个 chunk；freq base 下平均 8.0 个 chunk 非空，覆盖 90% residual 约需 5.3 个 chunk；mean base 下平均 9.9 个 chunk 非空，覆盖 90% residual 约需 6.0 个 chunk。因此 chunk gate 可作为 coarse filter，但不能直接无损替代 token-level selector。
- 2026-07-05：完成 16-query calibration selector prediction 诊断，目录：`MOTIVATION_EXPERIMENTS/calibration_query_selector_prediction_16q`。结论：用 15 个 calibration queries 的 selected-token frequency 预测 held-out query，DraftModel(raw) recall 约 84.2%-91.0%，Target-QK(preprocess KV) recall 约 85.4%-92.6%；说明完整 selector 输出有强 stable component，支持保存 offline ranking / candidate sets。
- 2026-07-05：完成 Task F 的第一版 native-query residual stability 分析，目录：`MOTIVATION_EXPERIMENTS/residual_stability_from_online_draft_trace`。结论：用已有 native sub-questions 做 leave-one-query-out，历史 residual frequency 对 held-out residual 的 recall 约 8.6%-12.4%，说明只靠少量原生 query 直接 offline 化 residual 还不够；下一步需要 16/32 calibration queries 保存完整 online draft ranking。
- 2026-07-05：创建本 tracker。当前主线从“纯 offline fixed set”扩展为两阶段结构：offline stable set + low-cost query-conditioned residual。后续同模型/同数据集实验必须追加到 `qwen3_offline_improvement_probe_summary.md` 的 unified main table。

## Offline10 + Residual5 Selector 对照主表

记录时间：2026-07-06。

目的：比较同样围绕 `offline fixed 10% + online residual 5%` 的不同 online residual selector。这个表作为后续接入训练式 4-layer distilled selector 的直接 baseline。

固定设置：

- dataset：MuSiQue reflect / `data/result_reflect.json`
- backbone：Qwen3-32B
- offline fixed set：`draft_smart_frequency_global`
- offline rate：0.10
- residual rate：0.05
- topk：10
- judge：GLM-5.2

| 类别 | 方法 | token budget | Main Acc | Sub Acc | F1 | EM | online/residual selection time | 来源 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| online selector | `online_draft_rate015` | online Draft 15% | 99/135 (73.33%) | 209/248 (84.27%) | 0.1905 | 0.0081 | 0.1417s | `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/summary.csv` |
| hybrid fair budget | `offline10_draft005` | offline 10% + full Draft residual 5% | 97/135 (71.85%) | 208/250 (83.20%) | 0.1955 | 0.0200 | 0.1406s | `MOTIVATION_EXPERIMENTS/qwen3_fair_budget_offline_vs_residual/accuracy_summary.csv` |
| partial layer ablation | `offline10_middle_residual005` | offline 10% + middle-layer Draft residual 5% | 95/135 (70.37%) | 202/250 (80.80%) | 0.2044 | 0.0160 | 0.0743s | `MOTIVATION_EXPERIMENTS/qwen3_offline10_plus_partial_draft_residual_rate015/accuracy_summary.csv` |
| partial layer ablation | `offline10_first12_residual005` | offline 10% + first-12-layer Draft residual 5% | 92/135 (68.15%) | 197/250 (78.80%) | 0.2072 | 0.0120 | 0.0524s | `MOTIVATION_EXPERIMENTS/qwen3_offline10_plus_partial_draft_residual_rate015/accuracy_summary.csv` |
| shallow ablation | `offline10_layer4_residual005` | offline 10% + layer4 Draft residual 5% | 92/135 (68.15%) | 198/250 (79.20%) | 0.2021 | 0.0240 | 0.0235s | `MOTIVATION_EXPERIMENTS/qwen3_offline10_plus_shallow_layer4_residual_rate015/README.md` |
| trained residual | `offline10_distilled005` | offline 10% + WikiText distilled 4-layer residual 5% | 92/135 (68.15%) | 195/248 (78.63%) | 0.2081 | 0.0202 | 0.0728s all / 0.0046s steady | `MOTIVATION_EXPERIMENTS/qwen3_offline10_plus_distilled_residual_rate015/accuracy_summary.csv`; timing from `MOTIVATION_EXPERIMENTS/qwen3_layer4_selector_gap_to_full_draft/summary.csv` |

当前结论：

- `offline10_draft005` 是 total 15% budget 下最强 residual baseline，质量接近 `online_draft_rate015`，但 selection time 仍约 0.14s。
- partial Draft 能降低 selection time，但 `middle` 仍掉 2/6 个 Main/Sub，`first12` 和 `layer4` 掉得更多。
- WikiText-103 蒸馏得到的 4-layer selector 第一版没有带来 accuracy 提升：Main Acc 与未微调 `layer4` 持平，Sub Acc 略低。
- 重新拆分 timing 后，`offline10_distilled005` 的 steady-state selector 很快，约 0.0046s；此前 0.0768s 主要被每个 segment 第一次加载 distilled checkpoint 的 cold-start 污染。

## Layer4 Selector 与 Full DraftModel 的 Token Gap

记录时间：2026-07-06。

目标：用 full DraftModel selector 作为 teacher，比较原生 layer4 selector 和 WikiText 蒸馏 4-layer selector 在同一批 Qwen3/MuSiQue reflect 样本上的选 token 差距。

统计口径：

- `final`：offline fixed 10% + online residual 5% 的最终 15% selected set。
- `residual`：从 `final` 中扣掉共同 offline fixed 10% 后，在线补选出来的 5% token。
- `teacher recall`：候选 selector 覆盖 full DraftModel selected token 的比例。

| method | items | final Jaccard | final teacher recall | residual Jaccard | residual teacher recall | selection all(s) | selection steady(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `native_layer4` | 250 | 0.6591 | 0.7912 | 0.3522 | 0.5157 | 0.0235 | 0.0200 |
| `distilled_layer4` | 250 | 0.6561 | 0.7907 | 0.3411 | 0.5060 | 0.0728 | 0.0046 |

结果文件：

- 汇总表：`MOTIVATION_EXPERIMENTS/qwen3_layer4_selector_gap_to_full_draft/summary.csv`
- 原生 layer4 明细：`MOTIVATION_EXPERIMENTS/qwen3_layer4_selector_gap_to_full_draft/native_layer4_vs_full_draft_detail.csv`
- 微调 layer4 明细：`MOTIVATION_EXPERIMENTS/qwen3_layer4_selector_gap_to_full_draft/distilled_layer4_vs_full_draft_detail.csv`

当前结论：

- 原生 layer4 和微调 layer4 在最终 15% selected set 上都能覆盖 full DraftModel 约 79% 的 token，差距很小。
- 只看真正在线补选的 residual 5%，二者和 full DraftModel 的重合都不高；原生 layer4 的 residual teacher recall 为 51.57%，微调 layer4 为 50.60%。
- WikiText 蒸馏 4-layer selector 当前更像是降低 steady-state selection 开销，而不是提升对 full DraftModel token 分布的拟合质量。
