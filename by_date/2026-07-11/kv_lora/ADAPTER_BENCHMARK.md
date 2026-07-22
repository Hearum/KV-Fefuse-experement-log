# FusionRAG KV Adapter Canonical Benchmark

## 1. 唯一系统目标

把 FusionRAG 对 document token 的 Transformer recompute 替换为缓存 KV 上的轻量 Adapter：

```text
KV_target = KV_cache_before + Adapter(current_RAG_context, KV_cache_before)
```

主实验只使用当前真实 FusionRAG 顺序和 cache/reprocess 路径，不使用 `query -> documents` 反事实 prompt。

## 2. Canonical 状态定义

固定一个真实 example/sub-question 的 system、完整 document chunk 列表、chunk 顺序和 token offset。对 cache source `s ∈ {raw, preprocess}`：

```text
B_s = 所有 chunk KV 装入当前 RAG multi-document cache 后、reprocess 前的 document KV
T_s = 在完全相同 cache/source/offset 下，rate=1 重算全部 document token 后的 document KV
Delta_s = T_s - B_s
```

- `B_raw` 与 `B_preprocess` 分开，禁止混合统计。
- `T_s` 使用真实 `load_kv_and_generate` rate=1 reprocess 输出，不用另一个 dense-forward 实现替代。
- 只统计 document token；system/query/decode token排除。
- Key 必须在最终 multi-document positions 上 RoPE-aligned；Value 直接比较。

Adapter 的唯一监督目标是 `Delta_s`。

## 3. Query 与 Context 的职责

当前部署是 `system + documents + query`。因果掩码决定 rate=1 document Delta vector 不依赖后置 query。当前 RAG context 对 Delta vector 的有效条件包括：

- 当前 document 前面的 system/document tokens；
- document 顺序、chunk offset和绝对/相对位置；
- raw/preprocess cache source；
- layer、head、token/chunk身份。

Query 仍可用于现有 selector 决定 update support：

```text
support = Selector(query, documents)
update  = support * Adapter(context, cached_KV)
```

因此研究拆成两个正交问题：selector 是否选对 support；Adapter 是否能在给定 support 上恢复 Delta。主线先解决 Adapter，不把 query-prefix 反事实结果当成系统证据。

## 4. 三个必须同时报告的指标

禁止再单独报告容易误解的“100% gap”。每个结果同时给：

```text
Original gap       g = ||T-B|| / ||B||
Delta recovery     r = ||Delta-Delta_hat|| / ||Delta||
Final KV error     e = ||T-(B+Delta_hat)|| / ||B|| = g*r
```

并补充 cosine、explained variance。K/V、raw/preprocess、layer/head/token分开。

## 5. 基准方法

1. `No update`: `Delta_hat=0`。
2. `Full Transformer`: `Delta_hat=Delta`，oracle上界。
3. `Mean template`: 训练 examples 的 per-layer mean Delta。
4. `Rank-r shared basis`: `mean + B_r c`，`r=1/2/4/8/16/32`。
5. `Actual FusionRAG rate=0.05/0.15/0.30`: 用真实 selector support和真实 recompute Delta，作为稀疏更新参考。
6. `Light predictor`: 只有 held-out shared basis 成立后，才用 Ridge/小 MLP 预测 coefficient。

## 6. 数据划分与验证顺序

### Stage A：单例管线正确性

example 0，raw/preprocess × rate 1，确认 shape、offset、RoPE、all-token support、unselected行为和三个指标。

### Stage B：跨 example 的 Adapter target

先取 5 examples，每个第一个 sub-question：4 train/1 held-out，轮换 leave-one-example-out。随后扩展到至少20 examples固定train/test split。

主问题：同一 per-layer mean/basis能否重建未见 example/document context 的 Delta？

### Stage C：Rate-Distortion

在 held-out examples 上报告 rank-r 的 `g/r/e` 曲线。必须区分：

- full support 内的 basis compression error；
- selector support 缺失造成的 support error。

### Stage D：Predictor

只有 rank-r basis 在 held-out examples 上显著优于 mean template，才训练 context feature -> coefficient。输入优先使用 prefix/document position、cache source、layer/head、document embedding；query只用于selector support。

## 7. 成功判据

一个 Adapter 路线值得继续，至少满足：

- held-out final KV error `e` 显著低于 no-update original gap `g`；
- rank-r 在 held-out 上显著优于 mean template，而非只改善训练集；
- Value 与 RoPE-aligned Key 分别有效；
- 多数层有效，不由少数层平均掩盖；
- 后续 logits/attention/answer质量验证没有明显退化。

## 8. 现有实验如何归档

- Phase 3真实 raw/preprocess reprocess snapshot：属于 canonical pipeline sanity，可保留。
- docs-before-query rate=1 query不变性：作为因果机制说明，可保留，但不是 Adapter basis 主结果。
- query-prefix Phase 6/7：属于反事实探索，只放附录；不能支持当前 FusionRAG Adapter。
- 下一轮不继续扩 query-prefix，而执行 canonical Stage A/B。

## 9. 下一次实际实验

实现统一 benchmark runner，输出 raw/preprocess、K/V 的 `g/r/e`。先完成 example 0 rate=1，再运行5-example leave-one-example-out mean/rank-r Adapter重建；所有表格和图统一写回 README/EXPERIMENT_LOG。

## 11. 可行性优先的实验规划

在设计 Predictor 前，先回答两个问题：Delta 是否具有足够的内部低秩性；跨 context 的 Basis 是否能重建未见 example。

### 11.1 单个 Delta 的内部 Rank-Distortion

当前真实 Pipeline 下定义 `B=cache装载后KV`、`T=rate1全部document token重算后KV`、`Delta=T-B`。逐层分别将 RoPE-aligned K 和 V reshape 为 `[document_tokens, kv_heads*head_dim]`，测试 rank `1/2/4/8/16/32/64/128`。raw/preprocess 分开。

每个结果同时输出：`g=||Delta||/||B||`、`r=||Delta-Delta_hat||/||Delta||`、`e=||Delta-Delta_hat||/||B||=g*r`、cosine和explained variance。

### 11.2 跨 Context Shared Basis

单例可压缩后，收集至少20个真实 examples 的 rate1 Delta，先做 leave-one-example-out oracle projection。只有 shared Basis 在 held-out example 上显著优于 mean template，才进入 coefficient predictor。

### 11.3 Predictor 输入消融决定架构

依次比较 position-only、target cached KV、cached prefix KV、target+cached prefix KV、target+fully-updated prefix KV。最后两者的差距决定架构：若 cached prefix 接近 updated prefix，采用一次性并行 Adapter；若 updated prefix 明显更好，采用 chunk串行、chunk内token并行 Adapter。暂不做 token级串行。

### 11.4 候选 Adapter

并行形式：每层/每head共享 Basis，所有 token 并行预测 coefficient。串行形式：按 document chunk 更新轻量 prefix state，每个 chunk 内一次输出全部 token Delta。Value/Key 使用独立 Basis 与 predictor。

## 12. 样本规模要求

- 单example仅用于shape/offset/实现sanity，禁止作为最终低秩或可预测性结论。
- 所有rank-distortion、局部低秩、shared basis和predictability主结论至少使用50个有效examples，报告mean/std/median/p10/p90以及逐layer分布。
- 50-example阶段通过后再scaling到更多examples或其他dataset/model。
- Qwen3-32B现有cache可复用：`fusionrag-reflect-qwen3-full-cache`下raw 3418 chunks、preprocess 2233 chunks，不重新生成重复cache。


## 13. Per-Document Basis 主实验（下一阶段）

### 13.1 假设

每个 document/chunk 可以与 cached KV 一起离线维护自己的更新 Basis：

```text
Delta_s(target_doc | prefix_context)
  ≈ Mean_s,target + Coef(prefix_context) @ Basis_s,target
```

其中 raw/preprocess、RoPE-aligned K/Value、layer/head 独立。Basis 可以是 document-specific；在线 Adapter 只预测低维 coefficient。后置 query 不参与 document Delta。

### 13.2 Context 构造

固定 target document 的 token 内容和 cache 文件，只改变它之前的 document 集合、顺序与长度。每个 target 至少生成 50 个确定性 context，覆盖：

- 空 prefix、单 document prefix、多 document prefix；
- 相同 prefix 集合的不同顺序；
- 短/中/长 prefix；
- target 位于不同绝对 token offset；
- 相关与较弱相关的前序 chunks。

Prompt 始终为 `system + prefix_docs + target_doc + query`，target 必须位于 query 前。rate=1 重算 prefix 与 target 的全部 token，但指标只截取 target document。每个 context 验证 target offset、token 数和 selected support。

### 13.3 严格数据划分

每个 target 使用 40 train contexts 和 10 held-out contexts。禁止在 held-out Delta 上重新拟合 Basis。至少选择 5 个 cache 完整、长度不同的 target documents；主结论因此至少包含 250 个 context runs，而每个 per-document 结论有 50 个样本。

### 13.4 对照方法

1. No update。
2. Per-document mean Delta template。
3. Per-document shared Basis，rank=1/2/4/8/16/32。
4. Global Basis：由其他 target documents 训练，测试未见 target。
5. Global Basis + document-specific residual Basis。
6. In-context oracle SVD，仅作为不可部署的压缩上界。

Shared Basis 使用训练 contexts 联合拟合；held-out 阶段允许 oracle projection coefficient，先隔离表示能力。只有该上界有效后才训练 coefficient predictor。

### 13.5 两种 Basis 粒度

- Feature basis：把训练 Delta 拼为 `[contexts * target_tokens, head_dim]` 或 `[contexts * target_tokens, kv_heads*head_dim]`；每个 context/token 有独立 coefficient。
- Context-template basis：把每个完整 target Delta flatten 为一个向量，矩阵为 `[contexts, target_tokens*feature]`；每个 context 只产生 r 个 coefficient。

前者对应一次输出每个 token coefficient；后者对应一次输出整篇 document 的 r 个 coefficient。必须同时测试，直接回答 Adapter 应“一次性输出整篇 gap”还是“为每个 token 输出 coefficient”。

### 13.6 判据与后续 Predictor

报告 original gap、Delta recovery、final KV error、explained energy、cosine，并提供 mean/std/median/p10/p90与逐层结果。优先判断：

- per-document Basis 是否明显优于 global Basis；
- context-template 小 rank 是否足够；
- 若只有 token-wise feature Basis 有效，是否能用 prefix summary 并行预测全部 token coefficient；
- cached-prefix 与 oracle updated-prefix predictor 的差距是否要求 chunk 串行。

preprocess full-cache gap 异常必须先审计生成配置；raw/preprocess 不混合训练或统计。
