对，我们现在固定采用**后者假设**：

> 某个 document token 的 KV 更新不仅取决于它自己的 cached KV，还取决于当前 RAG 排列中位于它之前的 system prompt 和 documents。

因此目标不是简单的逐 token 线性映射，而是一个**跨文档、因果条件化的 Delta-KV Predictor**。Selection 与它完全分离：训练时对所有 token 做 dense supervision；在线时已有 selector 只决定哪些预测结果被写回。

---

# 1. 核心训练目标

为避免 (L) 同时表示层数和序列长度，下面用：

* (N)：Transformer 层数；
* (T)：document token 总长度；
* (d_{kv})：每层 K 或 V 的维度。

对于第 (\ell) 层：

[
K_{\mathrm{cache}}^{(\ell)},
V_{\mathrm{cache}}^{(\ell)}
\in\mathbb R^{T\times d_{kv}}
]

完整上下文 prefill 得到：

[
K_{\mathrm{full}}^{(\ell)},
V_{\mathrm{full}}^{(\ell)}
\in\mathbb R^{T\times d_{kv}}
]

监督信号是：

[
\Delta K^{(\ell)}
=================

## K_{\mathrm{full}}^{(\ell)}

K_{\mathrm{cache}}^{(\ell)}
]

[
\Delta V^{(\ell)}
=================

## V_{\mathrm{full}}^{(\ell)}

V_{\mathrm{cache}}^{(\ell)}
]

训练模型：

[
F_\theta:
\left{
K_{\mathrm{cache}}^{(\ell)},
V_{\mathrm{cache}}^{(\ell)}
\right}*{\ell=1}^{N}
\rightarrow
\left{
\widehat{\Delta K}^{(\ell)},
\widehat{\Delta V}^{(\ell)}
\right}*{\ell=1}^{N}
]

最终更新为：

[
\widehat K^{(\ell)}
===================

K_{\mathrm{cache}}^{(\ell)}
+
\widehat{\Delta K}^{(\ell)}
]

[
\widehat V^{(\ell)}
===================

V_{\mathrm{cache}}^{(\ell)}
+
\widehat{\Delta V}^{(\ell)}
]

对于每一层，形式上就是：

[
(T,2d_{kv})
\rightarrow
(T,2d_{kv})
]

或者两个输出头：

[
(T,2d_{kv})
\rightarrow
(T,d_{kv})_{\Delta K}
]

[
(T,2d_{kv})
\rightarrow
(T,d_{kv})_{\Delta V}
]

这里的 (W_K')、(W_V') 是最后两个输出矩阵。

---

# 2. 因果关系

在线输入顺序为：

[
\text{system}
+\mathrm{doc}_1+\mathrm{doc}_2+\cdots+\mathrm{doc}_n
+\text{query}
]

因此，对 doc (i) 中的 token，其 KV 更新只能依赖：

[
\text{system}
+\mathrm{doc}*1+\cdots+\mathrm{doc}*{i-1}
+\mathrm{doc}_i\text{ 的局部前缀}
]

不能依赖：

* 后面的 documents；
* 最后的 query。

所以 Predictor 必须使用 causal context：

[
\widehat{\Delta KV}_{i,t}
=========================

F_\theta
\left(
KV_{i,t}^{\mathrm{cache}},
KV_{\mathrm{system}}^{\mathrm{cache}},
KV_{1:i-1}^{\mathrm{cache}},
KV_{i,<t}^{\mathrm{cache}}
\right)
]

不过 chunk cache 本身已经包含了 doc 内部局部前缀的信息，因此模型主要需要补充的是：

[
\boxed{\text{system 和前序 documents 带来的额外上下文化偏置}}
]

---

# 3. 训练数据

第一阶段使用纯文本长文档，不需要 QA，不需要 query，不需要 selection。

比较适合的文本类型包括：

* Wikipedia 长页面；
* PG-19 等长篇书籍；
* arXiv 或论文全文；
* 长新闻、网页正文；
* 通用预训练语料中的长文档。

## 3.1 构造多个 document chunk

从一个长文本或相关文本集合中采样：

[
d_1,d_2,\ldots,d_n
]

建议初始范围：

* 每个 chunk：128–512 tokens；
* 每组：4–16 个 chunks；
* 总 document 长度：2K–8K tokens。

训练数据需要混合几种组合。

### 连续 chunks

学习真实长文档中的上下文传播：

```text
section 1 → section 2 → section 3 → section 4
```

### 同文档非连续 chunks

模拟 retrieval 后的非连续组合：

```text
section 1 → section 5 → section 3 → section 8
```

### 顺序扰动

同一组 chunks 使用多个排列：

```text
docA + docB + docC
docC + docA + docB
docB + docC + docA
```

顺序扰动很重要，因为它能让同一个 token 在不同 causal prefix 下产生不同的 delta。

### 相关多文档

从同一主题、实体或事件下采样多个文本块。

### 无关文档控制组

随机加入部分弱相关或无关 chunks，防止 Predictor 对所有上下文都产生过强更新。

---

# 4. Teacher 和 Cache 的生成

对于每一个训练组合：

[
C=
\text{system}
+d_1+d_2+\cdots+d_n
]

运行一次冻结的原始 LLM full prefill，保存全部 document token、全部层的：

[
KV_{\mathrm{full}}
]

然后构造两种 cache。

## 4.1 Raw cache

每个 chunk 独立执行 prefill：

[
KV_i^{\mathrm{raw}}=F(d_i)
]

目标：

[
\Delta KV_i^{\mathrm{raw}}
==========================

## KV_i^{\mathrm{full}}

KV_i^{\mathrm{raw}}
]

## 4.2 Preprocess cache

按照 FusionRAG preprocess 流程提前生成：

[
KV_i^{\mathrm{pre}}
===================

F_{\mathrm{pre}}(d_i,\mathcal N_i)
]

目标：

[
\Delta KV_i^{\mathrm{pre}}
==========================

## KV_i^{\mathrm{full}}

KV_i^{\mathrm{pre}}
]

这两个任务不能直接混为一谈。建议第一版训练两个独立 Predictor：

[
F_{\theta_{\mathrm{raw}}}
]

[
F_{\theta_{\mathrm{pre}}}
]

原因是 raw delta 与 preprocess delta 的：

* 大小；
* 方向；
* 秩；
* layer 分布；
* 上下文敏感性

都可能不同。

后续确认二者结构相似后，再考虑共享 backbone，加 cache-type embedding。

---

# 5. 必须先处理位置变化

这是训练数据生成中很关键的一点。

独立缓存 doc 时，token 可能处于局部位置：

[
0,1,\ldots,T_i-1
]

在线拼接后，它处于全局位置：

[
o_i,o_i+1,\ldots,o_i+T_i-1
]

由于 RoPE，K 会随 position 改变。如果直接相减：

[
K_{\mathrm{full}}-K_{\mathrm{raw}}
]

其中会同时包含：

1. 确定性的 position relocation；
2. 真正的上下文更新。

这会污染监督信号。

因此应该先把缓存 K 映射到在线全局位置：

[
\widetilde K_{\mathrm{cache}}
=============================

\operatorname{RoPEShift}
\left(
K_{\mathrm{cache}},
p_{\mathrm{local}}\rightarrow p_{\mathrm{global}}
\right)
]

然后定义：

[
\Delta K
========

## K_{\mathrm{full}}

\widetilde K_{\mathrm{cache}}
]

V 不受 RoPE 位置旋转影响，但仍需做 token alignment。

另一种方案是直接监督 pre-RoPE K，但实现上通常需要改模型 hook。第一版更适合先做 RoPE relocation，再计算 delta。

---

# 6. Predictor 模型结构

不能只使用：

[
\Delta KV_t=X_tW'
]

因为它看不到其他 documents。

推荐结构是：

```text
所有 document cached KV
          │
          ▼
低维 token projection
          │
          ▼
每个 document 压缩成少量 summary tokens
          │
          ▼
Causal cross-document mixer
          │
          ▼
每个 token 读取前序 document summaries
          │
          ▼
W_K' 和 W_V' 输出 ΔK、ΔV
```

## 6.1 输入表示

对第 (\ell) 层：

[
X^{(\ell)}
==========

\left[
K_{\mathrm{cache}}^{(\ell)};
V_{\mathrm{cache}}^{(\ell)}
\right]
\in\mathbb R^{T\times2d_{kv}}
]

先降维：

[
Z^{(\ell)}
==========

X^{(\ell)}A_{\mathrm{in}}
\in\mathbb R^{T\times r}
]

其中：

[
r\ll 2d_{kv}
]

例如：

[
r\in{64,128,256}
]

同时加入：

* layer embedding；
* document index embedding；
* token-within-document position；
* global position offset；
* cache type embedding，可选。

## 6.2 Document 压缩

不要只用单个 mean vector，容易损失太多信息。建议每个 document 压缩为 (m) 个 summary tokens：

[
G_i^{(\ell)}
============

\operatorname{Pool}_m
\left(
Z_i^{(\ell)}
\right)
\in\mathbb R^{m\times r}
]

例如：

[
m\in{4,8,16}
]

可以使用：

* learned attention pooling；
* Perceiver-style latent pooling；
* 简单分段 pooling，作为 baseline。

## 6.3 跨文档因果 Mixer

将 summaries 按在线顺序排列：

[
G_{\mathrm{sys}},G_1,G_2,\ldots,G_n
]

执行一个很小的 causal mixer：

[
\widetilde G
============

M_{\mathrm{causal}}(G)
]

可以选择：

* 1–2 层小 Transformer；
* linear attention；
* SSM；
* causal MLP-mixer。

关键约束是 doc (i) 只能读取：

[
G_{\mathrm{sys}},G_1,\ldots,G_i
]

不能读取后续 documents。

## 6.4 Token 条件化

doc (i) 中的 token (t) 读取前序 summary：

[
C_{i,t}^{(\ell)}
================

\operatorname{CrossAttn}
\left(
Z_{i,t}^{(\ell)},
\widetilde G_{\le i}^{(\ell)}
\right)
]

然后融合 token-local 和 context 信息：

[
U_{i,t}^{(\ell)}
================

\operatorname{MLP}
\left(
Z_{i,t}^{(\ell)},
C_{i,t}^{(\ell)},
Z_{i,t}^{(\ell)}\odot C_{i,t}^{(\ell)}
\right)
]

最后：

[
\widehat{\Delta K}_{i,t}^{(\ell)}
=================================

U_{i,t}^{(\ell)}W_{K,\ell}'
]

[
\widehat{\Delta V}_{i,t}^{(\ell)}
=================================

U_{i,t}^{(\ell)}W_{V,\ell}'
]

因此，(W_K')、(W_V') 的确只负责输出 bias；真正的跨文档条件信息由前面的轻量 mixer 产生。

---

# 7. 层间参数设计

第一版建议：

* 所有层共享 context mixer；
* 使用 layer embedding 区分不同层；
* 每层有独立的 (W_{K,\ell}')、(W_{V,\ell}') 输出头。

即：

[
U^{(\ell)}
==========

M_\theta(X^{(\ell)},e_\ell)
]

[
\Delta K^{(\ell)}
=================

U^{(\ell)}W_{K,\ell}'
]

[
\Delta V^{(\ell)}
=================

U^{(\ell)}W_{V,\ell}'
]

这样比每层训练完整独立网络更省参数，也允许不同层输出不同的更新分布。

第一版暂时不做跨层递归：

[
U^{(\ell+1)}\leftarrow U^{(\ell)}
]

因为这会重新引入层间串行依赖。先验证各层能否从各自 cached KV 独立预测 delta。

---

# 8. 第一阶段损失：Dense Delta Alignment

训练时对所有 document token、所有层进行监督，不使用 selection。

由于不同层的 K/V 尺度不同，先统计每层标准差：

[
\sigma_{K,\ell},\qquad\sigma_{V,\ell}
]

使用标准化 Huber 或 MSE：

[
\mathcal L_{\Delta K}
=====================

\sum_{\ell,t}
w_\ell
\operatorname{Huber}
\left(
\frac{
\widehat{\Delta K}_{\ell,t}
---------------------------

\Delta K_{\ell,t}
}{
\sigma_{K,\ell}+\epsilon
}
\right)
]

[
\mathcal L_{\Delta V}
=====================

\sum_{\ell,t}
w_\ell
\operatorname{Huber}
\left(
\frac{
\widehat{\Delta V}_{\ell,t}
---------------------------

\Delta V_{\ell,t}
}{
\sigma_{V,\ell}+\epsilon
}
\right)
]

基础损失：

[
\mathcal L_{\mathrm{dense}}
===========================

\mathcal L_{\Delta K}
+
\lambda_V\mathcal L_{\Delta V}
]

不建议只用相对误差：

[
\frac{|\widehat\Delta-\Delta|}{|\Delta|}
]

因为 delta 很小的 token 会产生异常大的梯度。

---

# 9. 最终 KV 对齐损失

除 delta 外，还应直接约束更新后的 KV：

[
\widehat K
==========

K_{\mathrm{cache}}
+
\widehat{\Delta K}
]

[
\widehat V
==========

V_{\mathrm{cache}}
+
\widehat{\Delta V}
]

定义：

[
\mathcal L_{\mathrm{KV}}
========================

\sum_{\ell,t}
\left|
\widehat K_{\ell,t}
-------------------

K_{\mathrm{full},\ell,t}
\right|*2^2
+
\lambda
\left|
\widehat V*{\ell,t}
-------------------

V_{\mathrm{full},\ell,t}
\right|_2^2
]

它与 delta MSE 在数学上接近，但可以分别设置标准化和权重。

对于 K，还可以加入方向损失：

[
\mathcal L_{\mathrm{K-cos}}
===========================

\sum_{\ell,t}
\left[
1-
\cos
\left(
\widehat K_{\ell,t},
K_{\mathrm{full},\ell,t}
\right)
\right]
]

因为 K 最终用于 attention dot product，方向偏差可能比单纯的绝对误差更重要。

---

# 10. 第二阶段损失：功能对齐

仅仅让 KV 数值接近，不一定保证最终 attention 行为和生成结果接近。因此需要加入纯文本 probe。

## 10.1 构造 probe continuation

训练上下文变为：

[
\text{system}+d_1+\cdots+d_n+p
]

其中 (p) 是原始长文本后续的短 continuation，例如 32–128 tokens。

注意：

* probe 位于 documents 之后；
* probe 不参与 document delta 的预测；
* 它只是用来检查预测后的 KV 是否具有正确功能。

## 10.2 Attention score loss

用 probe token 的 Q 测量 predicted K：

[
S_{\mathrm{pred}}^{(\ell)}
==========================

Q_p^{(\ell)}
\widehat K^{(\ell)\top}
]

[
S_{\mathrm{full}}^{(\ell)}
==========================

Q_p^{(\ell)}
K_{\mathrm{full}}^{(\ell)\top}
]

损失为：

[
\mathcal L_{\mathrm{score}}
===========================

\sum_\ell
\left|
S_{\mathrm{pred}}^{(\ell)}
--------------------------

S_{\mathrm{full}}^{(\ell)}
\right|_2^2
]

## 10.3 Attention output loss

进一步比较：

[
O_{\mathrm{pred}}^{(\ell)}
==========================

\operatorname{Softmax}(S_{\mathrm{pred}}^{(\ell)})
\widehat V^{(\ell)}
]

[
O_{\mathrm{full}}^{(\ell)}
==========================

\operatorname{Softmax}(S_{\mathrm{full}}^{(\ell)})
V_{\mathrm{full}}^{(\ell)}
]

[
\mathcal L_{\mathrm{attn}}
==========================

\sum_\ell
\left|
O_{\mathrm{pred}}^{(\ell)}
--------------------------

O_{\mathrm{full}}^{(\ell)}
\right|_2^2
]

## 10.4 Logit distillation

冻结原始 LLM，只对短 probe tokens 做前向传播。

Teacher 使用 full KV：

[
P_{\mathrm{full}}(y_t\mid y_{<t})
]

Student 使用预测后的 KV：

[
P_{\mathrm{pred}}(y_t\mid y_{<t})
]

优化：

[
\mathcal L_{\mathrm{logit}}
===========================

\sum_t
\operatorname{KL}
\left(
P_{\mathrm{full},t}
\parallel
P_{\mathrm{pred},t}
\right)
]

最终损失：

[
\mathcal L
==========

\lambda_\Delta\mathcal L_{\mathrm{dense}}
+
\lambda_{KV}\mathcal L_{\mathrm{KV}}
+
\lambda_c\mathcal L_{\mathrm{K-cos}}
+
\lambda_a\mathcal L_{\mathrm{attn}}
+
\lambda_l\mathcal L_{\mathrm{logit}}
]

训练早期以 (\mathcal L_{\mathrm{dense}}) 为主，模型稳定后逐步增加 functional loss。

---

# 11. 完整训练阶段

## 阶段一：Teacher 数据生成

离线生成：

* raw cache；
* preprocess cache；
* full-context KV；
* RoPE relocation 后的 aligned cache；
* 全层、全 token delta；
* 可选 probe continuation。

这一阶段不使用 selector。

## 阶段二：Dense KV 预训练

只训练：

[
\mathcal L_{\mathrm{dense}}
+
\mathcal L_{\mathrm{KV}}
+
\mathcal L_{\mathrm{K-cos}}
]

目的：验证跨文档 cached KV 是否能够预测 full-prefill delta。

## 阶段三：功能对齐

加入短 probe，优化：

[
\mathcal L_{\mathrm{attn}}
+
\mathcal L_{\mathrm{logit}}
]

只对 probe 运行冻结 LLM，不重新对 documents 做完整 prefill，因此训练成本仍可控。

## 阶段四：长度和组合 curriculum

逐步扩大：

* document 数量；
* 总序列长度；
* document 顺序变化；
* 跨来源文档比例；
* 无关文档比例。

例如：

```text
2–4 docs / 2K tokens
→ 4–8 docs / 4K tokens
→ 8–16 docs / 8K tokens
```

## 阶段五：FusionRAG 部署测试

训练完成后才使用已有 selector。

在线：

[
S=\operatorname{Selector}(query,KV_{\mathrm{cache}})
]

Predictor 计算 context summaries，然后只对 (S) 中 token 执行输出头：

[
\widehat{\Delta K}_S,\widehat{\Delta V}_S
]

写回：

[
K_S\leftarrow K_S+\widehat{\Delta K}_S
]

[
V_S\leftarrow V_S+\widehat{\Delta V}_S
]

selector 不参与训练，也不产生任何监督标签。

---

# 12. 关键评估指标

## KV gap recovery

定义：

[
R_K
===

1-
\frac{
|\widehat K-K_{\mathrm{full}}|*2
}{
|K*{\mathrm{cache}}-K_{\mathrm{full}}|_2
}
]

V 同理。

含义：

* (R=0)：没有优于原 cache；
* (R=1)：完全恢复 full KV；
* (R<0)：更新后反而更差。

应分别报告：

* 每层；
* 每个 document 位置；
* raw/preprocess；
* 不同 context length；
* 不同 document 数量。

## 功能指标

* attention-score error；
* attention-output error；
* probe-token KL；
* probe perplexity；
* 与 full prefill 的 top-1 token agreement；
* continuation generation quality。

## 系统指标

* Predictor FLOPs；
* latency；
* 额外显存；
* 相对 FusionRAG sparse reprocess 的速度提升；
* 不同 selection rate 下的质量—效率曲线。

---

# 13. 必须做的对照实验

核心对照是证明跨文档信息确实必要。

### Token-local linear

[
\Delta KV_t=X_tW'
]

### Token-local MLP

[
\Delta KV_t=\operatorname{MLP}(X_t)
]

### 加单个 document summary

[
\Delta KV_t=F(X_t,g_{<i})
]

### 加多个 summary tokens

[
\Delta KV_t=F(X_t,G_{<i})
]

### 轻量 token-level causal mixer

作为质量上限，但计算更高。

如果跨文档 Predictor 相比 token-local baseline 显著提升，就能支持你的核心假设：

[
\boxed{
\Delta KV
\text{ 是当前文档组合条件化的，而不是固定 token-local bias}
}
]

还需要比较：

* raw 与 preprocess；
* K-only、V-only、K+V；
* 每层独立与参数共享；
* 一个 summary token 与多个 summary tokens；
* 是否加入 system summary；
* 是否加入全局 position offset。

---

# 14. 建议的验收标准

第一阶段可以设置以下 go/no-go 标准。

### Dense KV 可预测性

在未见过的文档和排列上：

[
R_{KV}\ge 70%\sim80%
]

至少 preprocess cache 应达到较高恢复率。

### 跨文档信息有效

跨文档 Predictor 相对 token-local MLP，KV gap recovery 至少提升：

[
15%\sim20%
]

否则复杂 context mixer 没有必要。

### 功能恢复

相对于“不更新 cache”，预测更新至少恢复 full prefill 所带来功能改善的：

[
80%
]

例如用 perplexity gap 定义：

[
R_{\mathrm{PPL}}
================

\frac{
\mathrm{PPL}_{\mathrm{cache}}
-----------------------------

\mathrm{PPL}*{\mathrm{pred}}
}{
\mathrm{PPL}*{\mathrm{cache}}
-----------------------------

\mathrm{PPL}_{\mathrm{full}}
}
]

### FusionRAG 部署

使用完全相同的 selector 时，Delta-KV Predictor 应恢复原始 sparse reprocess 质量收益的约：

[
90%
]

同时在线计算量显著低于重新执行 Transformer blocks。

---

最终，这个项目的训练定义可以压缩为：

[
\boxed{
\begin{aligned}
&\text{输入：当前排列下所有 documents 的 cached KV}\
&\text{上下文：system 和因果前序 document KV}\
&\text{输出：所有 document token 的 dense }\Delta K,\Delta V\
&\text{监督：full-context prefill 与 cached KV 的差值}\
&\text{训练：纯文本、无 query、无 selection}\
&\text{部署：selector 只负责控制哪些预测 bias 被写回}
\end{aligned}
}
]

最核心的模型结构则是：

[
\boxed{
\text{Cached KV}
\rightarrow
\text{causal document mixer}
\rightarrow
\text{token-conditioned latent}
\rightarrow
W_K',W_V'
\rightarrow
\Delta K,\Delta V
}
]
