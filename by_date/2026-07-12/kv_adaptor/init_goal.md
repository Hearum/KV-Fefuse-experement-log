**可以做，而且这是一个合理的研究方向。**但需要把它准确表述为：

> 用一个轻量的、上下文条件化的 KV Adapter，直接预测 FusionRAG 在线重算产生的
> (\Delta K,\Delta V)，而不是字面上只激活两个普通线性层 (W_K',W_V')。

FusionRAG 当前做法是：根据 query 选择一部分 document token，再把这些 token、对应 position IDs 和已经拼接好的缓存 KV 输入模型，逐层执行稀疏 prefill。虽然只更新少量 token，但这些 token仍然要经过 attention、输出投影、MLP 和逐层 hidden-state propagation。([arXiv][1])

## 1. 你真正想替代的对象

对于第 (\ell) 层、第 (t) 个 document token，设离线缓存为：

[
K_{\ell,t}^{\mathrm{cache}},\qquad
V_{\ell,t}^{\mathrm{cache}}
]

FusionRAG 在线重算后的结果为：

[
K_{\ell,t}^{\mathrm{rep}},\qquad
V_{\ell,t}^{\mathrm{rep}}
]

真正需要学习的是：

[
\Delta K_{\ell,t}
=================

## K_{\ell,t}^{\mathrm{rep}}

K_{\ell,t}^{\mathrm{cache}}
]

[
\Delta V_{\ell,t}
=================

## V_{\ell,t}^{\mathrm{rep}}

V_{\ell,t}^{\mathrm{cache}}
]

在线阶段直接写回：

[
\widehat K_{\ell,t}
===================

K_{\ell,t}^{\mathrm{cache}}
+
\widehat{\Delta K}_{\ell,t}
]

[
\widehat V_{\ell,t}
===================

V_{\ell,t}^{\mathrm{cache}}
+
\widehat{\Delta V}_{\ell,t}
]

这就把“重新跑 Transformer block”变成了“直接编辑缓存”。

---

## 2. 为什么普通的 (W_K')、(W_V') 不够

原始模型中，以 attention 前的归一化状态 (\bar h_{\ell,t}) 为例：

[
K_{\ell,t}=W_K^{(\ell)}\bar h_{\ell,t},
\qquad
V_{\ell,t}=W_V^{(\ell)}\bar h_{\ell,t}
]

因此：

[
\Delta K_{\ell,t}
=================

W_K^{(\ell)}
\left(
\bar h_{\ell,t}^{\mathrm{rep}}
------------------------------

\bar h_{\ell,t}^{\mathrm{cache}}
\right)
]

[
\Delta V_{\ell,t}
=================

W_V^{(\ell)}
\left(
\bar h_{\ell,t}^{\mathrm{rep}}
------------------------------

\bar h_{\ell,t}^{\mathrm{cache}}
\right)
]

也就是说，KV 的更新本质上来自：

[
\Delta \bar h_{\ell,t}
======================

## \bar h_{\ell,t}^{\mathrm{rep}}

\bar h_{\ell,t}^{\mathrm{cache}}
]

而这个 hidden-state delta 是逐层 attention 和 MLP 产生的。

假设你只定义：

[
\widehat{\Delta K}_{\ell,t}
===========================

z_{\ell,t}W_K'^{(\ell)}
]

[
\widehat{\Delta V}_{\ell,t}
===========================

z_{\ell,t}W_V'^{(\ell)}
]

其中 (z_{\ell,t}) 只是该 token 原来的 KV 或原来的 hidden state，那么它有一个根本问题：

> 同一个 document token 放在不同 document 组合、不同顺序下，输入 (z_{\ell,t}) 相同，因此预测的更新也相同。

但真实的更新显然不同。例如：

```text
system + docA + doc1
system + docB + doc1
system + docA + docB + doc1
```

这三种情况下，doc1 能看到的 causal prefix 不同，因此正确的 KV delta 不同。

所以：

[
\boxed{
\text{固定的 }W_K',W_V'
+\text{单个 token 的旧状态}
\quad\text{不足以预测真实更新}
}
]

它最多学到一个“平均上下文化修正”。

---

## 3. Query 应该扮演什么角色

在你的输入顺序中：

```text
system + doc1 + doc2 + ... + docn + query
```

由于 causal mask，document token 看不到位于其后的 query。因此，严格意义上：

[
\Delta KV_{\mathrm{doc}}
\not\leftarrow \mathrm{query}
]

对 doc (i) 来说，其真实更新取决于：

[
\mathrm{system}
+
\mathrm{doc}*1+\cdots+\mathrm{doc}*{i-1}
+
\mathrm{doc}_i\text{ 的局部前缀}
]

而不是 query。

FusionRAG 中 query 的作用是**决定哪些 token 值得重算**。论文使用 query 最后一层的 query matrix 与 document 最后一层 key matrix 计算重要性分数，从而选择关键 token。([arXiv][1])

因此更合理的分工是：

[
\text{Query}
\longrightarrow
\text{选择更新哪些 token}
]

[
\text{System + 前序 documents}
\longrightarrow
\text{决定这些 token 应该如何更新}
]

除非你主动设计一种 query-conditioned approximate cache，使 document KV 专门服务当前问题，否则不要把 query 直接作为真实 (\Delta KV) 的因果输入。

---

## 4. 更合理的 Adapter 结构

你的 (W_K')、(W_V') 应该是一个上下文条件化更新器的输出头，而不是完整模块。

可以设计成：

[
c_{\ell,t}
==========

\operatorname{ContextAgg}*{\ell}
\left(
z*{\ell,t},
\mathcal C_{<t}
\right)
]

其中：

* (z_{\ell,t})：该 token 的离线状态；
* (\mathcal C_{<t})：system prompt 和前序 documents 的缓存表示；
* (c_{\ell,t})：当前在线组合产生的轻量上下文摘要。

然后：

[
u_{\ell,t}
==========

\operatorname{Adapter}*{\ell}
\left(
z*{\ell,t},c_{\ell,t},
p_t,d_t
\right)
]

[
\widehat{\Delta K}_{\ell,t}
===========================

B_K^{(\ell)}A_K^{(\ell)}u_{\ell,t}
]

[
\widehat{\Delta V}_{\ell,t}
===========================

B_V^{(\ell)}A_V^{(\ell)}u_{\ell,t}
]

这里 (A,B) 可以采用低秩结构，例如 rank 32、64 或 128。

整体结构是：

```text
Cached token state ───────────────┐
                                  ├─ Lightweight Adapter ─ ΔK, ΔV
Prefix-document context summary ──┤
Position / document order ────────┘
```

## 5. 上下文摘要不能太重

假如 Adapter 仍然让每个 selected token 在每一层对所有前序 token 做一次完整 cross-attention：

[
O(|S|Nd)
]

那么你虽然跳过了大 MLP，却仍保留了长上下文 attention，长序列下收益会受限。

更适合 FusionRAG 的方式是先构建 document-level summary：

[
g_{\ell,i}
==========

\operatorname{Pool}
\left(
K_{\ell,\mathrm{doc}*i},
V*{\ell,\mathrm{doc}_i}
\right)
]

对 doc (i) 的 selected token，只聚合：

[
c_{\ell,t}
==========

\operatorname{LightAttn}
\left(
z_{\ell,t},
g_{\ell,\mathrm{system}},
g_{\ell,1},\ldots,g_{\ell,i-1}
\right)
]

复杂度从 token-to-token 的长上下文交互，降低到 token-to-document interaction：

[
O(|S|n_{\mathrm{doc}}r)
]

而不是：

[
O(|S|Nr)
]

FusionRAG 的 preprocess KV 已经提前注入过相似文档的信息，因此在线 residual delta 可能比 raw KV 的 delta 更小、更结构化，也更适合这种粗粒度上下文摘要。FusionRAG 的离线 preprocessing 本身就是通过相似 chunk 的 preliminary cross-attention 来降低在线 KV deviation。([arXiv][1])

---

## 6. Adapter 输入不能只有 KV

直接从缓存的 (K,V) 反推出原始 hidden state 通常不可靠，因为 GQA 模型中的 KV projection 是降维映射，信息不可逆。

因此建议离线额外缓存一个很小的 adapter latent：

[
z_{\ell,t}=P_{\ell}\bar h_{\ell,t},
\qquad
z_{\ell,t}\in\mathbb R^r
]

例如 (r=64) 或 (128)。

离线保存：

```text
K cache
V cache
small adapter latent z
```

在线输入：

```text
z + prefix document summaries + position/order
```

再预测 (\Delta K,\Delta V)。

这比尝试从 K/V 重构 hidden state更干净。若 (r=128)，额外缓存开销通常只是原始 KV cache 的一个较小比例。

---

## 7. 并行预测还是逐层预测

有两条路线。

### 路线 A：每层独立、一次性预测

[
(\widehat{\Delta K}*{1:L},
\widehat{\Delta V}*{1:L})
=========================

F_\theta
\left(
KV^{\mathrm{cache}}_{1:L},
\mathcal C
\right)
]

所有层并行预测，不再执行原始的 28 层串行传播。

优点是延迟最低。缺点是不同层预测的 KV 不一定来自一条真实 Transformer hidden-state trajectory。

但从系统角度，decode 阶段只读取每层 KV，并不强制这些 KV 必须由一个显式 hidden state 生成。因此，只要最终 attention 和生成结果正确，严格的 hidden-state consistency 未必是必要条件。这是非常值得验证的路线。

### 路线 B：轻量串行状态

维护一个小状态 (s_{\ell,t})：

[
s_{\ell+1,t}
============

f_\theta
\left(
s_{\ell,t},c_{\ell,t},z_{\ell,t}
\right)
]

然后：

[
\Delta K_{\ell,t}=A_K^{(\ell)}s_{\ell,t}
]

[
\Delta V_{\ell,t}=A_V^{(\ell)}s_{\ell,t}
]

它仍然有 28 层串行依赖，但每层只是小维度 Adapter，而不是完整 Transformer block。准确率可能高于完全独立预测。

建议两者都做：独立 predictor 是效率上限，轻量串行 predictor 是质量上限。

---

## 8. 训练目标

Teacher 应该直接使用 FusionRAG 当前真实 reprocess 的输出。

对于每个真实 RAG example：

1. 加载 raw cache 或 preprocess cache；
2. 按实际召回顺序组装 documents；
3. 使用 FusionRAG 原始 sparse reprocess 得到 teacher KV；
4. 构造真实 delta；
5. 训练 Adapter 预测 delta。

基础损失：

[
\mathcal L_{\mathrm{KV}}
========================

\sum_{\ell,t\in S}
\frac{
\left|
\widehat{\Delta K}_{\ell,t}
---------------------------

\Delta K_{\ell,t}
\right|*2^2
}{
\left|\Delta K*{\ell,t}\right|*2^2+\epsilon
}
+
\lambda_V
\frac{
\left|
\widehat{\Delta V}*{\ell,t}
---------------------------

\Delta V_{\ell,t}
\right|*2^2
}{
\left|\Delta V*{\ell,t}\right|_2^2+\epsilon
}
]

但只做 KV-MSE 不够，最好加：

[
\mathcal L
==========

\mathcal L_{\mathrm{KV}}
+
\lambda_a\mathcal L_{\mathrm{attention}}
+
\lambda_l\mathcal L_{\mathrm{logit}}
+
\lambda_o\mathcal L_{\mathrm{output}}
]

因为 KV 数值存在一定的功能等价空间，数值距离最小不一定对应生成效果最好。

另外，**raw cache 和 preprocess cache 必须分别处理**：

[
\Delta_{\mathrm{raw}}
=====================

## KV_{\mathrm{rep}}

KV_{\mathrm{raw}}
]

[
\Delta_{\mathrm{pre}}
=====================

## KV_{\mathrm{rep}}

KV_{\mathrm{pre}}
]

这两种 delta 的尺度、方向和可预测性可能完全不同。可以训练两个 Adapter，也可以加入 cache-type embedding，但不能把二者直接混成同一个无条件目标。

---

## 9. 最大风险：更新可能需要传播到下游 token

最近关于可编辑 KV cache 的研究发现，模型可能在 prefill 时把某些中间结论写入后续 delimiter 或 aggregator token；只修改信息源 token 自身的 KV，有时会因为后续缓存仍保留旧“结论”而失效。([arXiv][2])

对应到你的场景，风险是：

```text
重要 token 的 ΔKV 被正确预测
    ↓
但受它影响的后续 aggregator token 没有更新
    ↓
decode 仍然读取旧的聚合表示
```

所以 Adapter 的预测对象可能不能只覆盖 FusionRAG 最初选择的 token，还需要：

* 同时选择少量下游 aggregator/delimiter token；
* 或者让 Adapter 直接预测这些 token 的更新；
* 或者通过 logit distillation 让模型自行学会把必要影响写入选中的 KV。

这也是为什么最终验收不能只看 KV-MSE。

---

## 10. 我对这个方向的判断

最不可能成功的形式是：

[
\boxed{
\Delta K=h^{\mathrm{old}}W_K',
\qquad
\Delta V=h^{\mathrm{old}}W_V'
}
]

因为没有当前 RAG 组合的上下文输入。

更有希望的形式是：

[
\boxed{
(\Delta K,\Delta V)
===================

\operatorname{LowRankAdapter}
\left(
\text{cached token latent},
\text{preceding-doc summaries},
\text{position/order}
\right)
}
]

其中 query 只用于生成 selected-token mask。

已经有工作表明，可以通过轻量训练适配器缓解独立 cache 之间的上下文断裂。例如 KV Packet 将文档缓存视为不可变 packet，并通过自监督蒸馏训练轻量 soft-token adapters，尝试在不重算文档 KV 的情况下修复上下文 discontinuity；它与直接预测逐 token (\Delta KV) 不同，但说明“训练轻量模块替代在线 recomputation”这一大方向是可行的。([arXiv][3])

所以你的核心创新可以定义为：

> **将 FusionRAG 的在线 sparse reprocessing，从逐层执行 Transformer，转化为对 selected document KV 施加由当前 causal RAG context 决定的轻量结构化更新。**

第一步不应该立刻训练完整 Adapter，而应先验证：

1. (\Delta_{\mathrm{pre}}) 在不同 layer、token、RAG 组合下是否低秩；
2. 旧 KV 单独能预测多少，加入 preceding-document summary 后能提升多少；
3. 各层独立预测是否足够，还是必须有跨层串行状态；
4. KV 误差下降是否真正对应 attention、logit 和最终生成质量恢复。

只有第二项显著成立，(W_K'/W_V') 这条路线才真正具备可行性。

[1]: https://arxiv.org/html/2601.12904v1 "From Prefix Cache to Fusion RAG Cache: Accelerating LLM Inference in Retrieval-Augmented Generation"
[2]: https://arxiv.org/pdf/2606.17107 "Models Take Notes at Prefill: KV Cache Can Be Editable and Composable"
[3]: https://arxiv.org/abs/2604.13226 "[2604.13226] KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs"
