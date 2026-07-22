# KV/Hidden Surrogate Update 探究

本实验方向研究：是否可以不用主模型完整重算 document token，而是利用缓存 KV、离线 token 评分或轻量 attention 近似，恢复 full recompute 后的 hidden/KV，从而接近 FusionRAG 重算效果但降低计算量。

当前结论先写在 `PLAN.md`：这不是换 selector 的问题，而是替代主模型 block 前向的问题。第一阶段不直接跑端到端 accuracy，先做小样本机制探针，验证这种 surrogate update 是否能接近 full recompute 的 hidden/KV。

## 当前状态

- 已建立计划文档：`PLAN.md`
- 已建立日志：`EXPERIMENT_LOG.md`
- 当前没有启动新 GPU 实验；qjy000 正在跑 `qwen3_musique_v2_hidden_state_gap` 的完整数据集统计，本实验先不抢卡。

## 核心问题

1. 如果提前有 token 重要性评分，能否把它转成每个 token 的近似 attention/readout 分布？
2. 对 token `i`，能否用 `<i` 的 cached `V^{l-1}` 加权和构造一个便宜的 hidden correction？
3. 更新 token `i+1` 时，是否需要使用 token `i` 已更新后的 KV/hidden，还是各 token 可独立更新？
4. K/V 更新应该直接使用 `Wk/Wv(new_hidden)`，还是 blend：`beta * old + (1-beta) * projected(new_hidden)`？
5. 这个过程的计算量是否真的小于 full recompute？如果需要完整 QK attention、MLP 或逐层串行全 token 更新，就失去意义。

## 文档入口

- 详细计划：`PLAN.md`
- 按真实 Qwen block 修正的设计：`ACTUAL_MODEL_OPS.md`
- 首个 probe 规格：`FIRST_PROBE_SPEC.md`
- 实验日志：`EXPERIMENT_LOG.md`



## 真实模型口径

本方向后续以 `ACTUAL_MODEL_OPS.md` 为准：Value 加权和只是 attention readout，必须经过 `o_proj`、residual、post-attention RMSNorm、MLP/或其近似；K/V 来自每层 attention 前的 `RMSNorm(input hidden)` 后投影，并且 Key 还要经过 RoPE。后续 probe 不再使用“V 平均直接等于 hidden”的简化口径。


## 2026-07-16 首个 Smoke 结果

在 qjy003 GPU0 上跑了 `probe_late_layer_value_readout.py` 的 1-example smoke：

- examples：1
- chunks：24
- sampled tokens：每 chunk 16，总计每个配置 384 tokens
- layers：56-58
- top-m：16, 32
- score modes：uniform, recency, value_norm

核心结果：最朴素的 cheap sparse Value readout 信号很弱。

| 指标 | 最好配置 | 数值 |
|---|---|---:|
| `o_proj(z_hat)` vs `Delta x^{l+1}` cosine | layer 57, top_m 32, uniform | 0.0181 |
| hidden delta 标量拟合 R² | layer 56, top_m 32, uniform | 0.0025 |
| readout vs `Delta V` cosine | layer 56, top_m 16, value_norm | -0.0014 |
| Value delta 标量拟合 R² | layer 58, top_m 16, recency | 0.0084 |

解释：

1. 直接用固定分布从 cached Value 做 sparse readout，再过真实 `o_proj`，几乎不能解释 full recompute 的后层 hidden delta。
2. 对 Value delta 也只有极弱信号，R² 不到 1%。
3. 这说明“已知 token 重要性分数 + 简单 Value 加权”不足以恢复 KV/hidden；如果继续这条路线，必须引入更强的条件信息，例如 query/current-prefix dependent score、真实 Q/K 的低成本近似、或小 Adapter 学习 correction。
4. 这个结果不否定 surrogate update，但否定了最简单的 score-only readout 版本。

输出文件：

- `results/smoke_readout_1/results/probe_readout_predictability_summary.csv`
- `results/smoke_readout_1/results/probe_readout_predictability_rows.csv`
- `results/smoke_readout_1/results/probe_readout_predictability.json`
- `results/smoke_readout_1/run.log`

## 2026-07-16 Offline Score vs Online True Score

> **状态标记：本节实验设计有误，暂不作为有效结论使用。** 用户已指出该对比口径需要重新指导和定义；下方结果仅保留为错误尝试记录，不再用于判断 offline score 与 online true score 的真实差距。

用户问题：offline 得到的 attention/token score 和 online 真实算出来的 FusionRAG attention score 差距有多大？由于两者原始量级不同，本实验不直接比较绝对值，而是比较归一化后的相关性和 Top-k 排序一致性。

实验口径：

- 数据集：`musique-v2`，先跑 5 examples smoke。
- offline score：`setup_standard_v2_cross_dataset/score_cache_full_3b_20260715/musique-v2`，Qwen2.5-3B draft model，用 8 个 control query 得到的 document-token score。
- online true score：Qwen3-32B 在真实 `system + docs + query` 中，FusionRAG selection forward 产生的 `importance_cache[-1]`，只取 document span。
- cache：只读 setup-v2 shared preprocess KV，`/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2/data/musique-v2-preprocess-10-revert_rope-True/Qwen3-32B`。
- 归一化：报告 `z-score/min-max Pearson`、`Spearman rank`、`Top-k overlap/Jaccard`、`NDCG@k`。原始 mean/std 只作为量级诊断。

启动命令：

```bash
cd /home/hming/FusionRAG-pca-analysis
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kv_hidden_surrogate_update/scripts/compare_offline_online_scores.py \
  --dataset musique-v2 \
  --max-examples 5 \
  --start-example 0 \
  --device cuda:0 \
  --output-dir MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kv_hidden_surrogate_update/results/offline_online_score_gap_smoke5
```

关键结果：

| offline 聚合 | Pearson(z) | Spearman | rate=0.15 Top-k overlap | rate=0.15 Jaccard | rate=0.15 NDCG |
|---|---:|---:|---:|---:|---:|
| `offline3b_mean` | 0.2783 | 0.2746 | 0.3697 | 0.2269 | 0.4832 |
| `offline3b_top2_mean` | 0.2790 | 0.2549 | 0.3560 | 0.2167 | 0.4681 |
| `offline3b_freq_top15` | 0.2656 | 0.2666 | 0.3707 | 0.2277 | 0.4389 |

量级诊断：online score 原始 mean/std 在 5 个样本上约为 `2.09-3.79 / 3.67-7.36`，offline mean score 原始 mean/std 约为 `0.0071-0.0074 / 0.0206-0.0235`。因此原始 score 不能直接相减或直接看 L2；必须看归一化和排序。

解释：

1. offline score 与 online true score 有弱到中等一致性，不是随机噪声。以 rate=0.15 为例，随机 Top-k overlap 期望约 15%，实际约 36%-37%。
2. 但一致性远不到可以替代 online attention 的程度：Spearman 只有约 0.25-0.27，Jaccard 只有约 0.22。
3. `mean`、`top2_mean`、`freq_top15` 三种 offline 聚合差别不大，说明瓶颈不主要是简单聚合方式，而是 offline control-query/draft-model score 与真实 query/main-model score 的分布本身不一致。
4. 这支持一个中间判断：offline score 可以作为 cheap prior 或 coarse selector，但如果要恢复 full recompute 行为，仍需要 online 条件信息或轻量校正器，而不能直接把 offline score 当作真实 attention 分布。

输出文件：

- `results/offline_online_score_gap_smoke5/summary_metrics.csv`
- `results/offline_online_score_gap_smoke5/per_example_method_metrics.csv`
- `results/offline_online_score_gap_smoke5/score_scale_diagnostics.csv`
- `results/offline_online_score_gap_smoke5/metadata.json`
- `results/offline_online_score_gap_smoke5/run.log`

