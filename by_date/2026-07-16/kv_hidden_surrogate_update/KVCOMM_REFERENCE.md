# KVCOMM 对 FusionRAG KV 更新的启发

## 阅读对象

- Paper: `KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based Multi-agent Systems`
- Code: `https://github.com/FastMAS/KVCOMM` / paper 中链接 `https://github.com/HankYe/KVCOMM`
- 阅读时间：2026-07-16
- 当前 commit：`c168540` 起，本文档后续提交新 commit。

## KVCOMM 的核心方法

KVCOMM 解决的是 multi-agent 中“同一段共享文本出现在不同 prefix/context 下，KV cache 不能直接复用”的问题。它的关键不是训练 predictor，而是把跨 context KV 复用建模成：

```text
KV_under_context ≈ base_KV + context_offset
```

其中：

- `base_KV`：共享文本单独计算得到的 KV。
- `context_offset`：该文本放到某个 prefix/context 后，真实 KV 相对 `base_KV` 的偏移。
- anchor pool：保存历史样本的 base KV、真实 context 下 offset、邻近 prefix segment offset。
- online 使用：新请求来了以后，根据 embedding/长度匹配相似 anchor，用 anchor offset 插值近似新请求 offset。
- Key 必须做 RoPE 位置对齐；Value 可以直接加 offset。
- 如果找不到可靠 anchor，就 fallback 到 dense prefill，并把新样本加入 anchor pool。

它的优势是 training-free、online adaptive，不要求预先训练大 predictor。

## 和 FusionRAG 当前问题的对应关系

FusionRAG 的 document KV 问题可以映射成 KVCOMM 的形式：

```text
base_KV      = raw KV 或 preprocess KV
context_KV   = system + doc1 + doc2 + ... + docn + query/full prompt 下 document 的真实 KV
Delta_KV     = context_KV - base_KV
```

我们之前一直在问：能不能不用 Transformer recompute，而是给 cached document KV 加一个 adapter/bias。KVCOMM 给出的答案不是“训练一个一次性 predictor”，而是：

```text
new_Delta(doc, prefix) ≈ sum_i weight_i(doc, prefix, anchor_i) * Delta_anchor_i
KV_hat = base_KV + new_Delta
```

这和我们已经观察到的几点一致：

1. Delta 主要由 prefix/context 诱导，不一定由 query 本身决定。
2. Raw Key 的大变化有 RoPE 成分，因此 Key 必须先做 RoPE-aligned Delta。
3. Value Delta 更适合作为第一阶段目标，因为无 RoPE，且之前统计里后层更稳定。
4. 如果每个 document/chunk 自己维护若干低秩方向或 offset template，本质上就是 document-local anchor pool。

## 不能直接照搬的地方

KVCOMM 的场景是 multi-agent placeholder/prefix segment，它通常有明确的“同一共享文本”或相似 placeholder，并且可以在 online 中把失败样本加入 anchor pool。FusionRAG 的难点不同：

- RAG document 很多，offline 时不一定知道未来会和哪些 doc 组合成 prefix。
- 我们的真实 prompt 是 `system + doc1 + ... + docn + query`，document 的 context offset 受到前序 docs、doc 顺序、总长度、system prompt、preprocess KV 定义影响。
- 如果每个 doc 都保存大量 full-rank offset anchor，内存可能接近重新存多份 KV。
- 如果 anchor matching 需要完整 token-level embedding 和 full-rank offset 插值，可能省不了足够计算。

因此不能把 KVCOMM 直接当作最终方案，但它非常适合作为下一阶段的机制实验路线。

## 建议的新实验路线

### Phase 1：验证 KVCOMM 假设是否在 FusionRAG 成立

目标：验证相似 context/prefix 下的 document Delta 是否可迁移。

固定一个 target document/chunk `x`，构造不同前缀：

```text
system + A + B + C + x + query
system + A' + B' + C' + x + query
system + random docs + x + query
```

对每个 prefix 计算：

```text
Delta_K = RoPE_aligned(full_K - base_K)
Delta_V = full_V - base_V
```

然后测试：

- prefix embedding 距离 vs Delta 距离的 Spearman/Pearson。
- anchor nearest-neighbor Delta 是否能重建 held-out prefix 的 Delta。
- mean anchor、top-k weighted anchor、random anchor 三者对比。
- Raw base vs preprocess base 分开做。

评价指标：

- relative L2 to full KV；
- cosine to full Delta；
- per-layer/per-head error；
- rate=0.15 selector token 子集 vs all-token 差异；
- 后续再接端到端 accuracy。

### Phase 2：从 full-rank anchor 过渡到低秩 anchor

如果 full-rank anchor offset 有效果，再压缩：

```text
Delta_anchor ≈ B_doc,layer,head @ c_anchor
new_Delta ≈ B_doc,layer,head @ weighted(c_anchor)
```

需要比较：

- full-rank anchor offset；
- rank 4/8/16/32 offset；
- per-doc basis；
- per-layer basis；
- global basis。

这一阶段直接回答“每个 doc 是否可以自己维护低秩方向”。

### Phase 3：online policy

推理时不能用真实 Delta。可用输入只能是：

- target doc/chunk embedding；
- 前序 docs 的 embedding/长度/order；
- preprocess 召回文档集合；
- selector/offline score；
- position features。

先不训练神经网络，测试 KVCOMM-style training-free 权重：

```text
weight_i = softmax(-distance(prefix_feature, anchor_prefix_feature) / tau)
Delta_hat = sum_i weight_i * Delta_anchor_i
```

只有当这种方法有效，再考虑 ridge/MLP predictor。

## 推荐优先级

我建议下一步不要继续先训练 predictor，而是先做 KVCOMM-style anchor experiment：

1. 先在 `musique-v2` 取 20-50 个样本。
2. 对每个 target chunk 构造/收集多个不同 prefix 的 full KV 与 base KV。
3. 用 `train prefixes -> held-out prefixes` 测试 anchor offset 重建。
4. 先只看 Value；Key 只看 RoPE-aligned Key。
5. 如果 `top-k weighted anchor` 明显优于 `mean offset/random anchor/base no update`，再做低秩压缩和端到端。

## 当前判断

KVCOMM 给我们提供了一条比“直接训练 predictor”更稳的中间路线：

```text
offline/preprocess KV + anchor offset interpolation + low-rank compression
```

它最适合验证“Delta 是否是可迁移的 context offset”。如果这个假设不成立，训练 predictor 大概率也会因为样本量不足和分布变化而失败；如果成立，则 predictor 可以从“预测完整 KV/Delta”降级成“预测 anchor 权重或低秩系数”，训练难度会低很多。

