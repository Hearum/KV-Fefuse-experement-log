## 结论

当前结果已经足以说明：**把每个 layer/head 的 (\Delta K) 和 (\Delta V) 当成两个独立回归目标，这个建模方向不太对。**

更合理的研究对象应当改成：

[
\text{preceding context}
\rightarrow
\Delta \text{residual/normalized hidden state}
\rightarrow
\Delta K,\Delta V
]

也就是先预测“前缀给当前 token 带来的状态更新”，再通过主模型冻结的 (W_K,W_V) 产生一致的 K/V 更新，而不是分别预测两个 PCA coefficient。

系统层面还应同时保留另一条路线：**不要更新原 document KV，而是增加少量 link/adapter tokens，让后续 query 能正确读取拼接后的独立 cache。** KVLink 和近期的 KV Packet 已经表明，这条路线可能比逐 token 拟合完整 (\Delta KV) 更直接。([arXiv][1])

---

# 1. 目前实验真正说明了什么

你现在可以得出三个结论：

1. Qwen3-8B 的 (\Delta K) 存在明显的数据集级低维结构。
2. 当前特征足以预测其中一部分 K 更新。
3. 当前特征不足以确定 V 更新。

但还不能得出：

> 完整 prefix KV 对 V 没有帮助，或者 V 本身不可预测。

因为 mean-prefix 实验有 3024 条训练样本，而 cross-attention 只有 256 条，训练数据相差接近 12 倍。53.34% 对 55.68%、5.72% 对 7.16% 的比较混合了“输入形式”和“数据规模”两个变量。至少需要在相同的 256、512、1024、3024 样本上画 learning curve，才能比较两种输入。

此外，“Oracle PCA 很高”只代表：

[
\Delta K\text{ 在校准数据上的协方差近似低秩}
]

它不等价于 LoRA 意义上的：

[
\Delta K=f(A,X)
]

是一个低秩算子。需要区分三种完全不同的“低秩”：

* 样本间的 PCA covariance rank；
* 单条样本中 token × hidden 的矩阵秩；
* 从 prefix/document 到 (\Delta KV) 的映射算子秩。

当前结果主要测到了第一种。

如果 rank-64 是针对单个 128 维 head，那么 K 保留 95% 确实很强；但 V 用一半维度只能保留约 67%，只能说“有一定压缩性”，还不能称为特别强的低秩结构。还应与 cached K/V 本身、随机高斯向量和白化后的增量做对照。

---

# 2. K/V 差异有一部分来自 Qwen3 本身

Qwen3 并不是对 K、V 做完全对称的处理。其实现大致为：

[
s_{l,t}=\operatorname{RMSNorm}(h_{l,t})
]

[
\bar K_{l,t}
============

\operatorname{KNorm}\left(W_K^l s_{l,t}\right)
]

[
K_{l,t}=\operatorname{RoPE}(\bar K_{l,t}),
\qquad
V_{l,t}=W_V^l s_{l,t}
]

也就是说，Qwen3 对每个 K head 额外做 RMSNorm，而 V 没有对应的归一化。([GitHub][2])

因此：

* K 的范数和尺度受到约束；
* V 的幅度可以随着 residual state 明显变化；
* “V gap 是 K 的 1.84 倍”不能直接解释成 V 在功能上变化了 1.84 倍；
* K 的 PCA 和相对 L2 天然可能看起来更好。

建议把 K/V 指标改成：

[
E_K^{\mathrm{logit}}
====================

\left|
QK_{\mathrm{pred}}^\top-
QK_{\mathrm{full}}^\top
\right|
]

以及：

[
E_V^{\mathrm{functional}}
=========================

\left|
P_{\mathrm{full}}V_{\mathrm{pred}}
----------------------------------

P_{\mathrm{full}}V_{\mathrm{full}}
\right|
]

其中 (P_{\mathrm{full}}) 是教师 attention probability。这样测的是 K 对 attention score 的影响，以及 V 对 attention output 的影响，而不是两个不可直接比较的原始向量距离。

---

# 3. 必须先检查一个很强的 sanity check

在第一层，假设：

* cached X 和 (A+X) 中 X 的 tokenization 完全相同；
* BOS、separator 等处理相同；
* K 已正确移除旧 RoPE 并放到统一位置比较；

那么 X 的第一层输入仍然只是 token embedding：

[
h_{0,t}=E(x_t)
]

它还没有读到 A。因此理论上：

[
\Delta V^{(0)}=0
]

[
\Delta \bar K^{(0)}=0
]

只有 RoPE 位置发生变化，而你已经声称进行了 RoPE 对齐。

所以应首先画 layer-wise gap。**第 0 层必须接近数值零，然后随着层数逐渐累积。** 如果第一层仍有明显的 0.2/0.4 gap，应先排查：

* RoPE inverse/forward 的位置索引；
* cache 是否保存的是 RoPE 后 K；
* BOS 和 separator 是否不同；
* token 对齐是否错位；
* padding 或 batch position IDs；
* Qwen3 的 KNorm 是在 RoPE 前还是后处理。

这个检查比继续换 predictor 更重要。

---

# 4. 为什么只使用 KV 很可能无法预测 V

Qwen3-8B 的 hidden size 是 4096，而每层每个 token 的全部 KV 为：

[
8\times128\times2=2048
]

即使忽略 KNorm：

[
\begin{bmatrix}
K\V
\end{bmatrix}
=============

\begin{bmatrix}
W_K\W_V
\end{bmatrix}s
]

也是一个从 4096 维到至多 2048 维的投影。它至少丢失一半的 hidden-state 自由度。再加上 KNorm 会丢失每个 K head 的投影尺度，cached KV 不存在确定性的 hidden-state 逆映射。

所以问题不是简单的“MLP 不够大”，而是：

[
p(\Delta V\mid K_c,V_c,K_A,V_A)
]

可能本来就具有很高的条件方差。

应当做一个直接的可辨识性实验：

固定同一个 X，为它随机采样 16–32 个不同前缀 (A_j)，得到：

[
\Delta V_j
==========

V(A_j+X)-V(X)
]

然后测量：

* 不同 prefix 导致的 (\Delta V) 方差；
* 当前 feature 下最近邻样本之间的 target 方差；
* 加入 Q、hidden state、attention output 后条件方差下降多少。

如果加入 layer input hidden state 后 V 的 (R^2) 从 7% 跳到 40%–60%，就能明确证明问题是输入不可辨识，而不是输出 rank 或模型容量。

---

# 5. 当前 cross-attention predictor 还缺少两个关键输入

对于 (X_t)，full prefill 中的状态不仅依赖 A，还依赖已经被 A 改变的：

[
X_{<t}
]

当前结构是：

[
(KV^c_t,\ KV_A)
\rightarrow \Delta KV_t
]

它没有显式看到：

[
\Delta h_{<t}
\quad\text{或}\quad
\Delta KV_{<t}
]

更重要的是，它没有跨 layer 传播状态：

[
\Delta h^{l-1}_t
\rightarrow
\Delta h^l_t
]

而真正的 contextualization 是一个深度递归过程。完整 prefix cross-attention 只是在每层重新读取一次 A，并没有建模“上一层已经被 prefix 改变了多少”。这能解释为什么增加完整 prefix 后 V 仍然没有改善。

因此，后续模型至少需要同时具有：

* target token 之间的 causal sequence interaction；
* layer 之间的 recurrent/depth-wise propagation。

---

# 6. 最推荐的模型：Depth-Recurrent Hidden-Delta Adapter

## 6.1 预测共同 latent，而不是分别预测 K/V

定义 teacher 和 cached 的 layer input：

[
s_{l,t}^{*}
===========

\operatorname{RMSNorm}(h_{l,t}^{*}),
\qquad
s_{l,t}^{c}
===========

\operatorname{RMSNorm}(h_{l,t}^{c})
]

学习：

[
\Delta s_{l,t}
==============

s_{l,t}^{*}-s_{l,t}^{c}
\approx
B_l c_{l,t}
]

其中：

* (B_l\in\mathbb{R}^{4096\times r}) 是每层的共享 basis；
* (c_{l,t}\in\mathbb{R}^{r}) 是 predictor 输出的 coefficient；
* 所有 KV heads 共用同一个 hidden correction。

随后：

[
\widehat V_{l,t}
================

V^c_{l,t}
+
W_V^lB_lc_{l,t}
]

K 则在 KNorm 和 RoPE 之前更新：

[
u^c_{K,l,t}=W_K^ls^c_{l,t}
]

[
\widehat u_{K,l,t}
==================

u^c_{K,l,t}
+
W_K^lB_lc_{l,t}
]

[
\widehat K_{l,t}
================

\operatorname{RoPE}
\left(
\operatorname{KNorm}(\widehat u_{K,l,t})
\right)
]

这样 K 和 V 必然来自同一个上下文状态更新，不会出现分别拟合后互相不一致的问题。

由于 Qwen3 cache 中保存的是 KNorm 后的 K，需要额外缓存每个 KV head 在 KNorm 前的 RMS scale。每层每 token 只增加 8 个标量，就可以近似恢复 (u_K^c)，开销很小。

## 6.2 Predictor 结构

推荐用一个小型 depth-recurrent Transformer，而不是 layer/head 独立 MLP：

[
m_{l,t}
=======

\operatorname{CrossAttn}
\left(
q_{l,t},\ \text{PrefixAnchors}_l
\right)
]

[
c_{l,t}
=======

\operatorname{GRU/AdapterCell}
\left(
c_{l-1,t},
m_{l,t},
z^c_{l,t},
e_l,
c_{l,<t}
\right)
]

这里：

* (c_{l-1,t})：上一层已经积累的 contextual correction；
* (m_{l,t})：prefix 对当前 token 的轻量信息；
* (c_{l,<t})：前面 target tokens 的更新；
* (e_l)：layer embedding；
* (z^c)：cached token 的压缩 side state。

这个 cell 可以在 36 层共享参数，仅使用 layer embedding 区分层。输出维度从当前的：

[
36\times8\times2\times64=36864
]

个 coefficient/token，降低到例如：

[
36\times128=4608
]

而且所有 head 联合建模。

---

# 7. 是否应该额外缓存 hidden sidecar

我认为值得测试。

可以在离线生成 document KV 时，同时缓存：

[
z_{l,t}=P_lh_{l,t},\qquad z_{l,t}\in\mathbb{R}^{128\text{ 或 }256}
]

不需要保存完整 4096 维 hidden state。

原 KV 每层每 token 有 2048 个 BF16 数值。若 sidecar 为 256 维 INT8，只增加约：

[
\frac{256\text{ bytes}}{2048\times2\text{ bytes}}
=6.25%
]

的存储量。

而这个 sidecar 可以专门学习保存 KV 投影丢失、但对后续 contextual update 有用的方向。更进一步，可以令 (P_l) 专门编码：

[
\operatorname{Null}
\left(
\begin{bmatrix}W_K\W_V\end{bmatrix}
\right)
]

中最有预测价值的部分。

这比要求 predictor 从不可逆的 KV 中凭统计相关性“猜回 hidden state”合理得多。

---

# 8. 不想增加 sidecar 时，更合适的是小型文本 Delta Transformer

另一个更现实的选择是：

[
(A,X)\text{ 的 token IDs}
\rightarrow
\text{4--8 层小 Transformer}
\rightarrow
c_{l,t}
]

不要给 predictor 保存 40 MB 的完整 prefix KV，而是让一个很小的模型直接读取 384 个 token。

建议配置：

* 4–8 层；
* hidden size 512 或 768；
* causal self-attention；
* 与 Qwen3 共用 tokenizer；
* 输出每个 target token 的 context code；
* 用 layer-conditioned decoder 生成 36 层的 coefficient。

它实际执行了一次廉价的上下文建模，因此有机会预测 V，而当前 MLP/cross-attention 只是从静态投影中推断上下文作用。

EAGLE 的经验也说明，直接预测内部 feature 往往比预测最终离散输出更容易；虽然它解决的是 speculative decoding，不是 RAG cache 更新，但其“先预测模型 feature，再由主模型头处理”的范式比直接独立预测 K/V 更接近这里的问题。([arXiv][3])

还可以把 Qwen3-0.6B 或截断后的 Qwen3 风格模型作为第一版 student，再蒸馏成真正的 4 层 delta model。Qwen3 官方系列中包含 0.6B 模型，8B 模型本身是 36 层、32 个 Q heads 和 8 个 KV heads。([Qwen][4])

---

# 9. 训练目标不能只使用 PCA coefficient MSE

最终目标不是还原几何距离，而是让 query 和 decode 行为接近 full prefill。

推荐多层监督：

[
\mathcal L
==========

\lambda_h\mathcal L_{\mathrm{state}}
+
\lambda_{kv}\mathcal L_{\mathrm{KV}}
+
\lambda_a\mathcal L_{\mathrm{attn}}
+
\lambda_o\mathcal L_{\mathrm{output}}
+
\lambda_{\mathrm{KL}}\mathcal L_{\mathrm{logit}}
]

其中：

### 状态监督

[
\mathcal L_{\mathrm{state}}
===========================

\operatorname{Huber}
\left(
\widehat{\Delta h},\Delta h^*
\right)
]

### KV 监督

对每层归一化，避免某些 V 大幅度层支配训练。

### Attention 监督

[
\mathcal L_{\mathrm{attn}}
==========================

D_{\mathrm{KL}}
\left(
P_{\mathrm{full}}
\parallel
P_{\mathrm{adapter}}
\right)
]

### Attention-output 监督

[
\mathcal L_{\mathrm{output}}
============================

\left|
P_{\mathrm{adapter}}\widehat V
------------------------------

P_{\mathrm{full}}V^*
\right|_2^2
]

### 最终行为监督

让 query token 或 teacher-generated continuation 的 logits 接近 full prefill：

[
\mathcal L_{\mathrm{logit}}
===========================

D_{\mathrm{KL}}
\left(
p_{\mathrm{full}}
\parallel
p_{\mathrm{adapter}}
\right)
]

Palu、EigenAttention 和 KQ-SVD 这类低秩 KV 工作的重要启示也是：低秩目标最好围绕 attention fidelity，而不是仅围绕 K/V 的欧氏重构误差。([arXiv][5])

训练时应对全部 128 个 target token 提供密集监督，不要先做 selection。Selection 只用于 online 部署和最终成本评估，否则 8 token/sample 会浪费大量 teacher 信号。

---

# 10. 必须分开两个 teacher target

目前构造的是：

[
\Delta_{\mathrm{full}}
======================

KV_{\mathrm{full\ prefill}}-KV_{\mathrm{cached}}
]

但若目标是替换 FusionRAG 的 online partial reprocess，真正要替换的是：

[
\Delta_{\mathrm{reprocess}}
===========================

## KV_{\mathrm{after\ FusionRAG\ reprocess}}

KV_{\mathrm{cached}}
]

两者并不相同。

FusionRAG 的 online 阶段是在 reused 和 recomputed token 混合的 cache 上逐层前进，并不是 full prefill。FusionRAG 本身也明确区分 offline preprocessing 和 online selective reprocessing。([arXiv][6])

因此建议建立两个任务：

* **科学上界任务**：预测 (\Delta_{\mathrm{full}})，研究完全上下文化是否可近似；
* **系统替换任务**：预测 (\Delta_{\mathrm{reprocess}})，判断能否替换现有在线重算。

Raw KV 和 preprocess KV 也必须分开，因为它们距离 full state 的分布、方向和可预测难度不同。可以共享 predictor backbone，但至少需要 cache-type embedding 或不同 output basis。

---

# 11. 先做这些功能性 ablation，再决定是否继续 V

原始 L2 并不能判断 K/V 哪一个更影响生成。应直接构造：

1. full K + cached V；
2. cached K + full V；
3. predicted K + cached V；
4. cached K + predicted V；
5. oracle rank-64 K/V；
6. full K/V。

由此会出现三种可能：

### 情况 A：full K + cached V 已基本恢复质量

那么可以直接推进 K Adapter，V 保持 cached。你当前的 K 结果已经支持这条路线。

### 情况 B：cached K + full V 很重要

说明 V 携带了不可忽略的 contextualized content，应转向 hidden-delta/小 Transformer，而不是继续扩大 KV-only MLP。

### 情况 C：只有同时更新 K/V 才恢复

说明它们需要来自一致的 residual-state trajectory，应放弃独立 K/V predictor。

CacheBlend 的目标本来也不是最小化原始 KV L2，而是通过选择高偏差 token 降低后续 attention deviation。([ar5iv][7])


