# Low-Cost Draft Selector 研究计划

## 背景

当前 Qwen3/MuSiQue reflect 实验里，online DraftModel selector 是最强的 query-conditioned selector 之一，但 selection 开销约 0.14s。单层 layer4 selector 可以把 selection 开销降到约 0.024s，但完整实验质量明显下降。因此下一步不是继续押单层 layer4，而是寻找更公平的低成本 Draft selector：用 draft model 的部分层或稀疏层组合逼近 full Draft selector。

## 核心问题

能不能在明显低于 full Draft selector 开销的情况下，保留足够的 token ranking / selected set 质量，使 accuracy 接近 online Draft 或 offline + full Draft residual？

## 候选方法

默认 full Draft selector：使用 draft model 后半层 attention score 做 RRF 融合，是当前强 baseline。

低成本候选：

- `first4`：只跑前 4 层并融合这些层。
- `first8`：只跑前 8 层。
- `first12`：只跑前 12 层。
- `4,8,12`：跑到第 12 层，只收集第 4/8/12 层 score。
- `8,12,16`：跑到第 16 层，只收集第 8/12/16 层 score。
- `middle`：只收集中间层；成本接近半个 draft forward。
- `last`：只收集最后一层；成本接近完整 draft forward，用作质量上界对照。

## 实验阶段

### Stage 1: 20-example selector alignment

数据：MuSiQue reflect，`start_sample=0,end_sample=20`。

对每个候选 selector 保存 selected token trace，并和 default full Draft selector 比较：

- token recall vs default selected set。
- token Jaccard vs default selected set。
- chunk recall / chunk Jaccard。
- 对 offline base residual token 的 recall。
- selection time / score forward time。
- 小样本 answer accuracy 只作为 sanity check，不作为最终结论。

判断标准：

- 如果某个候选 token recall 接近或超过 75%，且 score time 明显低于 default，则进入 Stage 2。
- 如果 recall 很低，即使小样本 accuracy 偶然好，也不作为主推方法。

### Stage 2: 完整 pipeline accuracy

对 Stage 1 最有希望的 1-2 个候选跑完整 200-example pipeline，优先测试：

- `offline10 + candidate residual5`：公平 15% 总预算。
- 如果效果接近 full Draft residual，再测试 `offline15 + candidate residual5`。

主要对照：

- `online_draft_rate015`: 99/135, 209/248。
- `offline10 + full Draft residual5`: 97/135, 208/250。
- `offline10 + layer4 residual5`: 92/135, 198/250。
- `offline15 + full Draft residual5`: 103/135, 217/250。

## 预期结论类型

如果 `first8` 或 `4,8,12` 接近 full Draft residual，则可以形成一个系统优化点：partial draft forward approximates full draft token selection。

如果所有 partial selector 都明显差，则说明 online Draft 的有效信号主要来自更深层语义表示，低成本路线需要训练式 predictor，而不是手工选层。

## 执行日志

- 2026-07-05：创建计划。下一步启动 Stage 1 20-example selector alignment。
- 2026-07-05：完成 Stage 1 20-example alignment。结果：`middle` token recall 72.21%、score time 0.0764s；`first12` token recall 66.56%、score time 0.0538s，小样本 answer sanity 为 16/18 main、29/32 sub。已启动 Stage 2 完整实验：`offline10 + first12 residual5` 与 `offline10 + middle residual5`。
- 2026-07-05：完成 Stage 2 完整 pipeline。`offline10 + first12 residual5` 得到 Main 92/135、Sub 197/250、selection 0.0524s；`offline10 + middle residual5` 得到 Main 95/135、Sub 202/250、selection 0.0743s。结论：middle 是目前 partial Draft 最优折中，质量介于 layer4 和 full Draft residual 之间，但仍低于 full Draft residual 的 97/135、208/250。first12 没有明显优于 layer4。
