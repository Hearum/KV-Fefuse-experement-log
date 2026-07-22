# KV-LoRA 下一阶段目标：真实 Reprocess Delta 的结构与可压缩性

## 1. 核心对象

本阶段唯一主对象是 FusionRAG 实际写回 document cache 的变化：

```text
Delta_reprocess = KV_after_reprocess - KV_before_reprocess
```

其中：

- `KV_before_reprocess`：将同一个 example 所有 document chunk 的 raw 或 preprocess KV 装入多-document cache 后、selector 和重算开始前的 KV。
- `KV_after_reprocess`：在同一 cache 中执行 FusionRAG reprocess 后、decode 前的 KV。
- 只统计 document token；system/query token 不纳入 Delta。
- K/V 分开统计；Value 为主，Key 仅使用 RoPE-aligned 口径。

后置 query 的作用只限于 FusionRAG selector 决定 support。它不是 rate=1 全部 document token 重算时 document vector Delta 的直接条件变量，因此不把“不同后置 query 的 full KV”作为主实验。

## 2. 研究问题

1. raw 和 preprocess 起点下，重算实际改写了多少 K/V？
2. `Delta_reprocess` 是否随 rate 呈现稳定的 layer/head/token 结构？
3. 低 rate 所选 token 是否覆盖 full-rate Delta 的主要能量？
4. Value Delta 是否具有可压缩结构，可用低 rank 近似？
5. 多 example 的 Value Delta 是否存在可复用的 per-layer basis？

## 3. 实验矩阵

固定模型 Qwen3-32B，保持真实 FusionRAG `load_kv_and_generate` 路径和完整 sub-question document chunk 列表。

| 维度 | 设置 |
|---|---|
| KV 起点 | raw `kv_cache`；preprocess `preprocess_kv_cache_global_topk10_bge` |
| rate | 0.05、0.15、0.30、1.00 |
| 样本 | 先 example 0 sanity；通过后 example 0--4，每个第一个 sub-question |
| 方法 | `FusionRAG`，真实 selector，不使用 synthetic token selection |
| 对齐 | chunk id、document offset、RoPE/revert_rope 配置固定并记录 |

`rate=1.00` 是 full-reprocess reference，而不是后置-query比较。每个低 rate 结果都与相同起点的 rate=1 Delta 比较。

## 4. Phase A：对齐与支持集 sanity

对每个起点、每个 rate 保存：

- selected token absolute index、chunk id、chunk-local offset；
- `KV_before`、`KV_after` 的 shape/offset manifest；
- selected/unselected 的 Delta L2。

验收标准：

- unselected token 的直接写回 Delta 为 0；
- rate=1 选中全部 document token；
- raw/preprocess 的 chunk 对齐和 document token 数一致。

## 5. Phase B：重算 Delta 的分布

对 K（RoPE-aligned）和 V 分别输出：

- layer L2、relative L2、累计能量；
- layer × KV-head energy heatmap；
- token energy、top-k coverage、chunk-level coverage；
- 元素近零比例、token/head/block 结构化稀疏；
- raw/preprocess 的幅度比例。

关键问题：Value 是否稳定偏后层？selector 是否覆盖高 Delta-energy token？preprocess 是否只是降低幅度，还是改变结构？

## 6. Phase C：Rate-Distortion

以同一起点的 rate=1 `Delta_reprocess` 为 target。对低 rate 的实际更新，以及 target 的 rank-r SVD 重建，分别评估：

```text
r = 2, 4, 8, 16, 32, 64, ...
```

逐层输出 relative L2 error、cosine、explained variance；重点 Value。必须区分：

- selector support 缺失造成的误差；
- 已选 token 内 Delta 的低秩近似误差。

这两种误差不能混为“低秩失败”。

## 7. Phase D：共享 Basis（只在 Phase C 有可压缩性后）

汇集多个 example 的 rate=1 Value `Delta_reprocess`，按 layer 构造 basis：

```text
Delta_value(layer, example) ≈ B_layer c_example
```

检验：

- leave-one-example-out basis reconstruction；
- basis rank 与 explained variance；
- layer basis 相似性、head basis 相似性；
- raw/preprocess 是否共享 basis。

只有 leave-one-example-out 重建优于 per-example 零/均值 baseline，才允许称“shared basis”。

## 8. Phase E：轻量 Predictor（最后执行）

仅在 shared basis 通过后。输入 selector/query 的低维特征、prefix/document position、retrieval score；预测 `c_example`，使用 Ridge/小 MLP。评价相对 L2、cosine、explained variance、R2。禁止直接预测完整 KV。

## 9. 输出与记

每轮必须追加 `EXPERIMENT_LOG.md`，README 汇总结论。目录规范：

- `scripts/`：snapshot、统计、作图脚本；
- `results/`：manifest、CSV、JSON；
- `figures/`：layer/head heatmap、rate-distortion curve；
- `NEXT_STAGE_GOAL.md`：本计划。

## 10. 当前第一步

实现一个可复用 snapshot runner，覆盖 raw/preprocess × rate，并保存 before/after、support 与按 chunk 的 offset manifest；先完成 example 0 的 8 组（2 起点 × 4 rate），再进行任何 shared basis 或 predictor 结论。
