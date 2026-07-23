# 设计修正：Linear residual / KV blending reprocess

日期：2026-07-23

## 设计意图修正

原始背景文档把 Candidate D 写成“用纯 Linear Attention 替换 selected-token MHA”。这不是当前最想验证的主方案。

更符合当前设计意图的目标是：

> 保留已有 cache 或局部 MHA 重算作为稳定基线，用 Linear Attention 估计前序 document 的跨文档补充信息，再进行受控的 KV 或 attention-output 融合。

纯 Linear reprocess 仍然保留，但作为下界/诊断对照，不作为唯一候选。

## 推荐的融合记号

先明确三个量：

- KV_base：已有 raw/preprocess cache，或局部 MHA 计算得到的稳定基线；
- KV_linear：使用 Linear Attention 读取跨文档 prefix 后得到的重算 KV；
- KV_mha_full/local：标准 MHA full-prefix 或 local-prefix 重算结果，用于 teacher/对照。

建议主实验先使用：

KV_blend = gamma * KV_base + (1 - gamma) * KV_linear

这里 gamma 是 base KV 的权重。若采用用户提出的形式：

KV_blend = gamma * KV_mha_local + (1 - gamma) * KV_compute

必须在实验表中明确 KV_compute 是 Linear 结果还是其他重算结果，不能只写 compute。

更适合解释为“在已有 cache 上添加 Linear delta”的形式是：

KV_blend = KV_base + lambda * (KV_linear - KV_base)

其中 lambda=0 等价于不重算，lambda=1 等价于纯 Linear。两种参数化可以互相转换，但报告时必须固定一种定义。

## 更推荐的 hidden-state 语义

不能只在每层 forward 完成后独立混合 K/V，然后继续沿用没有对应来源的 hidden state。这样会产生“KV 已融合，但下一层 hidden state 不知道融合结果”的语义断裂。

优先级如下：

### 方案 A：在 attention memory/KV 写回前融合

当前层由原模型生成新的 q/k/v 后，将 cached KV 与 current recomputed KV 融合，再让当前 token 执行 attention：

KV_used = gamma * KV_base + (1 - gamma) * KV_linear

attention output 使用 KV_used，之后照常执行 output projection、residual、normalization 和 MLP，得到下一层 hidden state。

这是最容易保持逐层递归一致的方案。下一层的 Q/K/V 自然来自已经使用融合 KV 的 hidden state。

### 方案 B：融合 attention output

如果同时有 local MHA output 和 Linear output，可定义：

o_blend = gamma * o_local_mha + (1 - gamma) * o_linear

然后继续原始 output projection、residual、MLP。这个方案比事后融合最终 KV 更容易定义 hidden-state 传播，但需要同时计算两条 attention read 路径。

### 方案 C：事后直接融合 K/V

可以作为简单 ablation，但必须标为 post-hoc KV blend。若之后没有用融合后的 KV 重新计算 attention output 和下一层 hidden state，则不能把它解释为完整的逐层 Linear reprocess。

## 当前代码中已有的相关机制

当前仓库已有两个相关接口，需要区分其语义：

1. FUSIONRAG_KV_ADAPTER_ALPHA

在 Qwen3 attention 中，current recomputed key/value 写回 cache 前会执行类似：

KV_used = (1 - alpha) * KV_cached + alpha * KV_current

这已经接近本方向的 KV blending，可以作为实现入口或 sanity baseline。

2. FUSIONRAG_REPROCESS_KV_BLEND_BETA

utils.py 中已有旧 KV 与 current KV 的 blend 逻辑。其 beta 定义是旧 KV 权重：

KV_blend = beta * KV_original + (1 - beta) * KV_current

在 strict reprocess 路径中，blend 发生在 query prefill 前；非-strict 路径需要特别检查 blend 发生时机，因为如果 query logits 已经算完，blend 不会影响当前答案，只会影响后续 cache。

Linear 方向不能直接复用这些参数而不记录语义，必须在 profile metadata 中保存 base、linear、mha-local 的定义和 gamma/beta 的实际含义。

## token i 是否影响 token j

因果性只允许前面的 token 影响后面的 token：

- i < j：i 可能影响 j；
- i > j：i 不应影响 j。

当前代码把排序后的 selected positions 一次传入模型，并在 attention 前通过 cache_position 将新的 K/V 写入对应位置。之后使用 causal mask。因此，在实现和 mask 正确的前提下，后面的 selected token j 可能看到前面的 selected token i 的更新 K/V。

这意味着不能未经验证地把 selected tokens 当成完全独立重算。必须增加一个显式实验开关或 trace，区分：

- independent：每个 selected token 都读取同一个原始 cache；
- causal-sequential：后面的 selected token 可以看到前面 selected token 的融合更新。

两种语义必须分别报告，否则 Linear 与 MHA 的差异无法归因。

## 需要加入的 hidden/KV trace

对于少量真实 examples，每层保存：

- KV_base；
- KV_mha_local 或 KV_mha_full；
- KV_linear；
- KV_blend；
- attention output；
- post-attention hidden；
- post-MLP hidden；
- 下一层 K/V。

至少验证：

1. gamma=1 时完全退化为 base；
2. gamma=0 时退化为 pure Linear；
3. blended KV 是否真正参与当前层 attention；
4. blended attention output 是否生成下一层 hidden；
5. selected token i 对后续 selected token j 的影响是否存在；
6. 未选 token 的 KV 是否保持不变。

## 当前实验方向

第一阶段不应直接做“纯 Linear 是否替代 MHA”的单一结论，而应保留以下矩阵：

- cache/base；
- pure MHA reprocess；
- pure Linear reprocess；
- base + Linear KV blend；
- local MHA + Linear blend；
- 必要时 attention-output blend。

gamma/lambda 扫描应先小规模进行，例如 0、0.25、0.5、0.75、1.0。最终 plan 需要先确定 base 的具体定义，以及融合发生在 K/V 写回前、attention output，还是 post-hoc。

