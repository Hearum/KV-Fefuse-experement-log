
# 任务：在不训练的情况下，用 Linear Attention 替代 FusionRAG 的 selected-token MHA 重算

## 1. 研究背景

当前 FusionRAG pipeline 会预先缓存 document KV，并在 online 阶段对部分 document token 重新执行 Transformer forward，以修复离线 cache 缺少跨文档上下文交互的问题。

当前输入布局为：

```text
system prompt + doc1 + doc2 + ... + docN + query
```

由于 decoder-only 模型满足因果约束，document token 的 KV 只受到它前面的 system prompt 和前序 documents 影响，不受到末尾 query 影响。因此，本任务不研究 query-conditioned document update，也不要引入 query 作为 Linear Attention 的额外输入。

当前存在两类 cache，实验必须分开分析：

* `raw cache`：每个 document 独立 prefill 得到的 KV；
* `preprocess cache`：document 与若干相似 document 一起预处理得到的 KV。

这两种 cache 距离 full-context KV 的初始差距不同，Linear Attention 能恢复的空间也可能不同，不能混合统计。

当前 selected-token reprocess 的基本过程是：对于被选中的 token (i)，从模型底层开始，根据重新计算的 hidden state 生成该层 Q/K/V，用 (q_i) 读取当前可见的 cached prefix KV，得到新的 attention output 和下一层 hidden state，再逐层生成更新后的 KV。

已有观察表明，local/raw/preprocess cache 本身已经具有较高质量，selected token 的重算不是从零恢复 KV，而是在补充跨文档上下文导致的变化。当前观察到重算前后 K/V 的平均变化约为 20.72% 和 39.90%。因此，我们希望探索：

> 是否可以在 selected-token reprocess 中，不执行原始 Softmax MHA，而使用无需训练的 Linear Attention 读取 cached KV，从而保留部分 MHA 重算收益？

本任务只进行 zero-shot feasibility study：

* 不训练模型；
* 不微调模型；
* 不学习 feature map；
* 不使用 LoRA；
* 不改变 token selection；
* 不改变 cache 构建方式；
* 不修改 MLP、QKV projection、(W_O)、residual 或 normalization；
* 只替换 selected-token reprocess 中的 attention operator。

---

# 2. 核心研究问题

需要回答以下问题。

### RQ1：零训练 Linear Attention 能否近似 MHA reprocess？

比较：

[
KV_{i,\text{linear-reprocess}}
]

与：

[
KV_{i,\text{MHA-reprocess}}
]

重点分析这种差距如何沿层累积。

### RQ2：Linear reprocess 是否至少优于完全不重算？

即：

[
D(KV_{\text{linear}},KV_{\text{full}})
<
D(KV_{\text{cache}},KV_{\text{full}})
]

Linear Attention 不必完美复现 MHA。只要它能恢复一部分 cache-to-full gap，就值得进一步训练和优化。

### RQ3：哪种固定 feature map 最适合 MHA-trained 模型？

至少比较：

* ELU+1；
* FAVOR+；
* 可选的 local-softmax + global-linear。

### RQ4：误差主要产生在哪里？

分别分析：

* attention output；
* residual 后 hidden state；
* MLP 后 hidden state；
* 下一层 K/V；
* 最终 logits 和生成结果。

### RQ5：raw cache 和 preprocess cache 的结果是否不同？

需要验证：

* preprocess cache 是否因为初始质量更高，更容易被 Linear Attention 修复；
* 或者因为剩余 Delta 更细粒度，反而更难近似。

### RQ6：Linear Attention 什么时候可能真正带来加速？

需要区分：

* 临时从 cached KV 构造 Linear state；
* 提前保存并复用 Linear state。

第一阶段只验证质量，不应错误宣称已经实现加速。

---

# 3. 必须保留的实验基线

对于相同 example、相同 document 顺序、相同 selected-token 集合，至少运行以下四条路径。

## Baseline A：Full-context MHA prefill

完整执行：

```text
system + doc1 + ... + docN + query
```

保存 document token 的：

* attention output；
* hidden states；
* 每层 K/V；
* 最终 logits；
* 生成结果。

记为：

[
KV_{\text{full}}
]

这是最终参考，但注意它与 selected-token MHA reprocess 本身也可能有差距。

## Baseline B：Cache only，不重算

直接使用 raw/preprocess cache：

[
KV_{\text{cache}}
]

用于度量 Linear reprocess 是否真正改善了 cache。

## Baseline C：当前 MHA reprocess

运行仓库已有 selected-token reprocess：

[
KV_{\text{MHA-reprocess}}
]

这条路径是直接 teacher，也用于确认修改后没有破坏原实现。

## Candidate D：Zero-shot Linear reprocess

selected token 仍然逐层重算，但把：

[
\operatorname{softmax}
\left(
q_iK_{\le i}^{\top}/\sqrt d
\right)V_{\le i}
]

替换为固定 feature map 定义的 Linear Attention。

---

# 4. Linear Attention 背景

标准 causal Softmax Attention 对 token (i) 为：

[
o_i
===

\frac{
\sum_{j\le i}
\exp(q_i^\top k_j/\sqrt d)v_j
}{
\sum_{j\le i}
\exp(q_i^\top k_j/\sqrt d)
}
]

如果存在 feature map：

[
\phi:\mathbb R^d\rightarrow\mathbb R^r
]

使相似度近似为：

[
\exp(q^\top k/\sqrt d)
\approx
\phi(q)^\top\phi(k)
]

则可以写成：

[
o_i^{\text{linear}}
===================

\frac{
\phi(q_i)^\top
\left(
\sum_{j\le i}\phi(k_j)v_j^\top
\right)
}{
\phi(q_i)^\top
\left(
\sum_{j\le i}\phi(k_j)
\right)+\epsilon
}
]

定义：

[
S_i=\sum_{j\le i}\phi(k_j)v_j^\top
]

[
z_i=\sum_{j\le i}\phi(k_j)
]

则：

[
o_i^{\text{linear}}
===================

\frac{\phi(q_i)^\top S_i}
{\phi(q_i)^\top z_i+\epsilon}
]

这是 Linear Attention 的 recurrent state 形式。

注意本任务只重算 selected token，不转换整个模型。cached prefix token 不重新经过 Linear Attention，而只是作为当前 token 的 memory 被读取。

---

# 5. token (i) 的逐层重算流程

设第 (l) 层输入 hidden state 为：

[
h_{i,\text{linear}}^{l-1}
]

第一层从与 MHA baseline 完全相同的 embedding/input hidden state 开始。

## 5.1 使用原模型参数生成 Q/K/V

按照原模型的 pre-norm、projection、QK normalization 和 RoPE 顺序执行：

[
u_i^l=\operatorname{Norm}^l(h_{i,\text{linear}}^{l-1})
]

[
q_i^l=W_Q^lu_i^l
]

[
k_i^l=W_K^lu_i^l
]

[
v_i^l=W_V^lu_i^l
]

如果原模型使用 RoPE，则必须复用原 MHA 路径中已经应用正确全局 position 的 Q/K。

不要：

* 重复应用 RoPE；
* 对未对齐位置的 cached K 直接计算 feature map；
* 改变原始 attention scale；
* 使用旧的 cached (k_i,v_i) 替代当前重算产生的 (k_i^l,v_i^l)。

## 5.2 构造当前 token 可见的 memory

一般 decoder self-attention 允许 token (i) 看到自己，因此：

[
K_{\le i}^l=
[K_{<i,\text{cache}}^l;k_i^l]
]

[
V_{\le i}^l=
[V_{<i,\text{cache}}^l;v_i^l]
]

但最终必须检查当前仓库里的 MHA reprocess mask，以实际 baseline 为准。

原则是：

> 只替换 attention operator，不能改变 causal mask、self-token inclusion 或 cache writeback 语义。

## 5.3 执行 Linear Attention

显式形式：

[
s_{ij}^{l}
==========

\phi(q_i^l)^\top\phi(k_j^l)
]

[
a_{ij}^{l,\text{linear}}
========================

\frac{s_{ij}^l}
{\sum_{m\le i}s_{im}^l+\epsilon}
]

[
o_{i,\text{linear}}^l
=====================

\sum_{j\le i}
a_{ij}^{l,\text{linear}}v_j^l
]

状态形式：

[
S_i^l
=====

\sum_{j\le i}
\phi(k_j^l)v_j^{l\top}
]

[
z_i^l
=====

\sum_{j\le i}\phi(k_j^l)
]

[
o_{i,\text{linear}}^l
=====================

\frac{
\phi(q_i^l)^\top S_i^l
}{
\phi(q_i^l)^\top z_i^l+\epsilon
}
]

第一阶段应同时实现显式形式和状态形式，并验证二者输出一致。显式形式便于 debug 和分析 attention weights，状态形式用于确认未来加速路径在数学上正确。

## 5.4 继续原模型后续计算

Linear Attention output 之后，完全复用原模型：

[
\tilde h_i^l
============

h_{i,\text{linear}}^{l-1}
+
W_O^lo_{i,\text{linear}}^l
]

[
h_{i,\text{linear}}^l
=====================

\tilde h_i^l
+
\operatorname{MLP}^l
\left(
\operatorname{Norm}_{\text{mlp}}^l(\tilde h_i^l)
\right)
]

下一层再由 (h_i^l) 生成新的 Q/K/V。

注意：

> 第 (l) 层的 attention output 不会改变已经生成的 (k_i^l,v_i^l)，而是改变 (h_i^l)，进而影响第 (l+1) 层的 (k_i^{l+1},v_i^{l+1})。

最终保存 token (i) 在各层重算产生的 K/V，并按照当前 reprocess 流程写回 cache。

---

# 6. 必须实现的 Linear Attention 变体

## 6.1 ELU+1：最简单基线

使用：

[
\phi(x)=\operatorname{ELU}(x)+1
]

它保证 feature 非负，适合 normalized linear attention。

但要明确：

> ELU+1 不是 Softmax kernel 的严格近似，而是将 MHA 替换成另一种相似度函数。

它适合作为低成本 sanity baseline，不应作为唯一方案。

参考：

* [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
* [fast-transformers causal linear attention 实现](https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/causal_linear_attention.py)
* [FLA LinearAttention layer](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/linear_attn.py)

## 6.2 FAVOR+：主实验

FAVOR+ 使用正交随机特征近似 Softmax kernel：

[
\exp(q^\top k/\sqrt d)
\approx
\phi(q)^\top\phi(k)
]

为了对应原始 scale，使用：

[
q'=q/d^{1/4},\qquad k'=k/d^{1/4}
]

使：

[
q'^\top k'=q^\top k/\sqrt d
]

正随机特征近似形式大致为：

[
\phi_\omega(x)
==============

\frac{1}{\sqrt r}
\exp
\left(
\omega^\top x-\frac{|x|_2^2}{2}
\right)
]

但不要自己随意拼写一个不稳定版本，应参考 Performer 官方数值稳定实现，包括：

* orthogonal random projection；
* query/key 的指数稳定处理；
* FP32 accumulator；
* denominator epsilon；
* overflow/underflow 检查。

主实验至少扫描：

[
r\in{32,64,128,256,512}
]

如果成本过高，可以先跑：

[
r\in{64,128,256}
]

每个 (r) 至少测试 3 个固定 random seeds。

随机投影矩阵必须满足：

* 同一层内 Q 和 K 使用相同 projection；
* 不能每个 token 重采样；
* 不能每个 request 重采样；
* 保存到配置/checkpoint metadata 中；
* 相同实验使用完全一致的 projection。

参考：

* [Rethinking Attention with Performers](https://research.google/pubs/rethinking-attention-with-performers/)
* [Google Research FAVOR+ 实现](https://github.com/google-research/google-research/blob/master/performer/fast_attention/tensorflow/fast_attention.py)

## 6.3 Local Softmax + Global Linear：第二阶段候选

如果纯 Linear Attention 明显无法近似局部尖锐 attention，增加一个无训练 hybrid：

* 最近 (w) 个 token 使用原 Softmax；
* 更远 token 使用 FAVOR+。

建议：

[
w\in{64,128}
]

必须避免 local/global 重复计数。可以直接将 softmax kernel numerator 和 denominator 分区：

[
N_i
===

\sum_{j<i-w}
\phi(q_i)^\top\phi(k_j)v_j
+
\sum_{i-w\le j\le i}
\exp(q_i^\top k_j/\sqrt d)v_j
]

[
D_i
===

\sum_{j<i-w}
\phi(q_i)^\top\phi(k_j)
+
\sum_{i-w\le j\le i}
\exp(q_i^\top k_j/\sqrt d)
]

[
o_i=N_i/(D_i+\epsilon)
]

这与 LoLCATs/BASED 的“局部精确、全局线性”思想一致，但本任务不学习 mixing coefficient。

参考：

* [LoLCATs 论文](https://arxiv.org/abs/2410.10254)
* [LoLCATs 代码](https://github.com/HazyResearch/lolcats)
* [BASED 论文](https://arxiv.org/abs/2402.18668)
* [BASED 代码](https://github.com/HazyResearch/based)

---

# 7. Tensor shape 和 GQA 处理

当前“云端 MHA 模型”可能实际使用 GQA。本文档中的 MHA 泛指 Softmax MHA/GQA。

典型 shape：

```text
q: [batch, num_q_heads, 1, head_dim]
K: [batch, num_kv_heads, prefix_len, head_dim]
V: [batch, num_kv_heads, prefix_len, value_dim]
```

需要沿用原模型的 query-head 到 KV-head 映射：

```text
kv_head = q_head // num_q_heads_per_kv_head
```

不能把 GQA 默认为普通 MHA，也不能错误地让不同 query head 读取不对应的 KV head。

为了未来保留 GQA 的 cache 优势，建议 FAVOR+ projection 在同一个 KV group 内共享。否则，如果每个 query head 使用不同随机 feature projection，就必须为每个 query head 单独构造 (S,z)，会放大 Linear state。

第一阶段显式实现可以暂时逻辑 repeat KV，但报告中要写清楚实际状态可否按 KV head 共享。

---

# 8. 实现步骤

## 阶段 A：先理解现有 reprocess 语义

在修改代码前，先定位并记录：

1. selected token 如何产生；
2. token (i) 在每层从哪里取得 hidden state；
3. 当前 token 是否包含在自己的 attention memory 中；
4. cached K 是否已经应用 RoPE；
5. cache reuse 后如何调整 token position；
6. raw/preprocess cache 的加载路径；
7. 多个 selected token 是否在一次 batch 中并行计算；
8. 后面的 selected token 是否读取前面 selected token 的更新 KV；
9. 每层新 KV 何时写回；
10. 当前 MHA reprocess 使用的是 FlashAttention、SDPA 还是自定义 kernel。

先写一份简短的 code-path 说明，再开始替换。

## 阶段 B：增加可控 attention mode

不要直接覆盖原实现。增加配置开关，例如：

```text
--reprocess-attention-mode mha
--reprocess-attention-mode linear-elu
--reprocess-attention-mode linear-favor
--reprocess-attention-mode local-favor
```

以及：

```text
--linear-feature-dim 128
--linear-random-seed 0
--linear-window-size 64
--linear-accum-dtype fp32
```

要求：

* `mha` 模式与修改前结果一致；
* 只影响 selected-token reprocess；
* 正常 prefill、decode 和 cache construction 不受影响；
* 模型处于 `eval()` 和 `torch.no_grad()`；
* 随机 feature projection 注册为非训练 buffer；
* 不允许出现任何 optimizer、backward 或 trainable adapter。

## 阶段 C：先实现显式参考版本

先按：

[
a_{ij}^{\text{linear}}
\propto
\phi(q_i)^\top\phi(k_j)
]

显式计算 selected token 的 kernel weights。

这一步不追求速度，目的是：

* 检查 causal mask；
* 可视化权重；
* 与 MHA attention weights 对比；
* 检查 denominator；
* 检查 output norm；
* 简化 GQA 和 RoPE debug。

## 阶段 D：实现等价 state 版本

构造：

[
S=\sum_j\phi(k_j)v_j^\top,\qquad
z=\sum_j\phi(k_j)
]

验证：

[
o_{\text{explicit}}
\approx
o_{\text{state}}
]

在 FP32 下 relative error 应接近数值误差范围。若二者明显不同，不能继续做质量实验。

## 阶段 E：接入完整逐层 reprocess

将 Linear Attention 接入 selected token 的完整 layer-by-layer forward，并保存：

```text
attention_output[layer, token]
post_attention_hidden[layer, token]
post_mlp_hidden[layer, token]
K[layer, token, head]
V[layer, token, head]
```

这些 tensor 可以只对少量 debug examples 保存，避免大规模实验 I/O 过高。

---

# 9. 实验设计

## 9.1 数据

优先复用当前真实 RAG evaluation pipeline，例如 MuSiQue。

建议：

* smoke test：5 个真实 examples；
* 主实验：至少 20 个固定 examples；
* 保持 document 集合和顺序固定；
* 保存 example ID、document 顺序和 selected-token positions；
* 所有 attention mode 必须使用完全相同的 selected tokens。

不要只做随机 tensor 或 synthetic attention 测试。synthetic 只能用于单元测试，最终结论必须来自真实 RAG examples。

## 9.2 Cache 设置

必须分别运行：

```text
raw cache
preprocess cache
```

如果 preprocess 有不同 top-k，可先选择当前主配置；资源允许时再比较：

[
k\in{1,3,5,10}
]

## 9.3 Selection rate

至少测试：

[
\rho\in{0.1,0.2,0.3}
]

资源允许可以增加：

[
\rho=0.5
]

同一 example、同一 cache 类型、同一 selection rate 下，MHA 和 Linear 必须使用相同 selected-token 集合。

## 9.4 Linear 配置

第一轮推荐：

| 方法           | 配置                |
| ------------ | ----------------- |
| ELU+1        | normalized        |
| FAVOR+       | (r=64)            |
| FAVOR+       | (r=128)           |
| FAVOR+       | (r=256)           |
| Local+FAVOR+ | (w=64,r=128)，第二阶段 |

---

# 10. 评测指标

## 10.1 Attention output 误差

逐层、逐 token：

[
E_O^l
=====

\frac{
|o_{i,\text{linear}}^l-o_{i,\text{MHA}}^l|*2
}{
|o*{i,\text{MHA}}^l|_2+\epsilon
}
]

同时报告：

* cosine similarity；
* output norm ratio；
* mean/median/P90/P99。

## 10.2 Hidden-state 误差

分别记录：

[
E_{\text{post-attn}}^l
]

和：

[
E_{\text{post-MLP}}^l
]

判断 MLP/residual 是否放大或抑制 Linear Attention 误差。

## 10.3 KV 误差

分别计算：

[
E_K^l
=====

\frac{
|K_{i,\text{linear}}^l-K_{i,\text{teacher}}^l|*2
}{
|K*{i,\text{teacher}}^l|_2+\epsilon
}
]

[
E_V^l
=====

\frac{
|V_{i,\text{linear}}^l-V_{i,\text{teacher}}^l|*2
}{
|V*{i,\text{teacher}}^l|_2+\epsilon
}
]

Key 的比较必须使用现有实验里一致的 RoPE 对齐策略：

* attention 计算使用实际全局位置下的 rotated K；
* 距离分析可以 de-RoPE 或统一到相同位置；
* 不允许把位置旋转差异误判为表示差异。

Value 直接比较。

Teacher 分为两种：

* `MHA-reprocess`：度量算子替换误差；
* `full-prefill`：度量实际 cache 恢复效果。

## 10.4 Delta 方向

定义：

[
\Delta KV_{\text{MHA}}
======================

KV_{\text{MHA-reprocess}}-KV_{\text{cache}}
]

[
\Delta KV_{\text{linear}}
=========================

KV_{\text{linear-reprocess}}-KV_{\text{cache}}
]

除了 L2，还需要计算：

[
\cos
\left(
\Delta KV_{\text{linear}},
\Delta KV_{\text{MHA}}
\right)
]

这能区分：

* Linear 更新幅度偏小但方向正确；
* 更新幅度接近但方向错误；
* 完全没有恢复有效 Delta。

## 10.5 恢复比例

相对 full-prefill：

[
R_{\text{full}}
===============

\frac{
D(KV_{\text{cache}},KV_{\text{full}})
-------------------------------------

D(KV_{\text{linear}},KV_{\text{full}})
}{
D(KV_{\text{cache}},KV_{\text{full}})+\epsilon
}
]

解释：

* (R=1)：完全恢复 cache-to-full gap；
* (R=0)：与不重算相同；
* (0<R<1)：恢复一部分；
* (R<0)：Linear reprocess 反而恶化。

相对 MHA reprocess：

[
R_{\text{MHA}}
==============

1-
\frac{
D(KV_{\text{linear}},KV_{\text{MHA-reprocess}})
}{
D(KV_{\text{cache}},KV_{\text{MHA-reprocess}})+\epsilon
}
]

K 和 V 必须分别报告。

## 10.6 Attention weight 诊断

对于显式版本，保存少量 heads 的：

* MHA softmax weight；
* Linear normalized kernel weight；
* attention entropy；
* top-k overlap；
* 最大权重；
* 局部窗口质量；
* 长距离 mass。

Attention-weight matching 只是诊断指标，不是主要目标。即使权重不同，也可能有：

[
A_{\text{linear}}V
\approx
A_{\text{MHA}}V
]

所以最终仍以 output、hidden 和 KV 为主。

## 10.7 端到端指标

至少比较：

* next-token logit KL；
* perplexity 或 NLL；
* 当前 RAG benchmark 的 EM/F1/生成质量指标；
* generation 是否退化、重复或提前终止；
* no-reprocess、MHA-reprocess、Linear-reprocess 三者差异。

## 10.8 数值稳定性

记录：

* denominator 最小值和分位数；
* NaN/Inf 次数；
* feature norm；
* (S,z) norm；
* attention output norm ratio；
* 每层误差是否爆炸；
* FAVOR+ 不同 seed 的方差。

---

# 11. 正确性测试

在跑正式实验前，必须通过以下测试。

### Test 1：MHA 模式回归

新增代码后：

```text
reprocess-attention-mode=mha
```

必须与修改前结果一致。

### Test 2：显式形式与 state 形式一致

对相同 Q/K/V：

[
o_{\text{explicit}}
\approx o_{\text{state}}
]

### Test 3：因果性

修改 token (i) 之后的 K/V，不应影响 token (i) 输出。

### Test 4：当前 token inclusion

确认是否包含 token (i) 自己，并与原 MHA mask 完全一致。

### Test 5：GQA head mapping

逐 head 验证 query head 使用正确的 KV head。

### Test 6：RoPE

确认：

* 没有重复应用；
* 没有漏应用；
* cached K 已经处于正确全局位置；
* Linear 和 MHA 接收到的 Q/K 完全相同。

### Test 7：只修改 selected token

未选 token 的 cache 必须保持不变。

### Test 8：第一层输入一致

MHA 和 Linear reprocess 在第一个被替换 attention 之前，hidden/Q/K/V 必须一致。

### Test 9：可复现性

相同 seed、相同数据、相同 selected positions 的结果必须一致。

---

# 12. 关键歧义：自问自答

## Q1：本任务是在把整个模型改成 Linear Attention 吗？

不是。

只有 selected-token reprocess 的 attention read 被替换。正常 prefill、已有 cache、未选 token 和 decode 都保持原模型 MHA/GQA。

## Q2：为什么不直接使用 LoLCATs？

LoLCATs 需要 attention transfer 和 LoRA，本任务当前要验证的是：

> 在完全不训练的下界条件下，固定 Linear Attention 是否已经能恢复一部分重算收益。

LoLCATs 是后续结果成立后的自然训练方案，不属于当前阶段。

## Q3：Hedgehog 可以直接用吗？

不能把随机初始化的 Hedgehog 当成有效 zero-shot feature map。

Hedgehog 的核心是训练一个 feature-map MLP 去模仿 Softmax 的尖锐性和单调性。未训练 Hedgehog 没有合理语义。

当前可以参考其结构和诊断方法，但不能把 learnable Hedgehog 作为“无训练”主结果。

参考：[The Hedgehog & the Porcupine](https://arxiv.org/abs/2402.04347)

## Q4：ELU+1 能近似 Softmax 吗？

不能严格近似。它定义了另一种 kernel，只适合作为简单 baseline。

FAVOR+ 才是本任务中更接近原 MHA 目标的 zero-shot 方法。

## Q5：FAVOR+ 能完全复现 MHA 吗？

不能。它是有限随机特征维度下的 Monte Carlo 近似。

理论上 (r) 增大时近似改善，但实际还会受到：

* 有限 feature dimension；
* 数值稳定性；
* MHA-trained Q/K 分布；
* 多层误差累积；
* 随机 seed；
* attention 尖锐度；

影响。

因此必须扫描 (r) 并报告随机方差。

## Q6：token (i) 是否要使用新生成的 (k_i,v_i)？

是。

当前层的 (k_i^l,v_i^l) 必须由重算 hidden state：

[
h_i^{l-1}
]

通过原 (W_K,W_V) 生成。不能直接使用旧 cache 中 token (i) 的 KV，否则会破坏逐层重算链路。

## Q7：当前层的 Linear Attention 会不会更新当前层的 K/V？

不会直接更新。

当前层 K/V 在 attention 前由 (h_i^{l-1}) 产生。attention output 改变的是 (h_i^l)，从而影响下一层 K/V。

## Q8：token (i) 是否 attention 到自己？

通常会，即 (j\le i)。但必须跟随当前 MHA reprocess 的 mask，不能凭经验修改。

## Q9：多个 selected token 之间如何交互？

完全沿用现有 MHA reprocess 语义。

需要先查清楚：

* 每个 selected token 是否独立读取原 cache；
* 还是后面的 selected token 可以看到前面 selected token 的更新 KV。

本任务不能顺便改变这一行为，否则无法把差异归因于 Linear Attention。

## Q10：query 是否应该参与 Linear state？

对于 document token，不应该。

当前布局中 query 位于所有 documents 后面。因果模型中，query 不可能影响前面的 document KV。

## Q11：为什么第一阶段不直接实现 Linear state cache？

因为如果 state 尚未提前构造，针对单个 query 现场计算：

[
S=\sum_j\phi(k_j)v_j^\top
]

可能比单 query MHA 更贵。

第一阶段只验证质量。只有质量成立后，才研究：

* state checkpoint；
* selected tokens 之间的 state 增量更新；
* prefix state reuse；
* per-document state composition；
* fused kernel。

## Q12：第一阶段是否能声称 Linear Attention 加速了重算？

不能。

只有当 (S,z) 可以预先构造、缓存并在多个 token/request 间摊销时，单 token read 才能从随 prefix length 增长变成：

[
O(rd_v)
]

否则这只是算子质量实验。

## Q13：能否提前为每个 document 保存一个 Linear state？

不能直接假设可以。

如果 cached K 需要根据 document 在 full prompt 中的位置重新应用或调整 RoPE，那么：

[
\phi(R_{\Delta}k)
]

通常不能从已经聚合的：

[
\sum_j\phi(k_j)v_j^\top
]

简单恢复，因为 (\phi) 是非线性的。

第一阶段必须先在正确全局位置的 assembled cached KV 上构造 state。per-document state composition 是后续独立问题。

## Q14：为什么同时比较 full-prefill 和 MHA-reprocess？

因为二者回答不同问题：

* Linear vs MHA-reprocess：Linear 是否近似当前重算算子；
* Linear vs full-prefill：Linear 是否真正改善最终 cache 质量。

MHA partial reprocess 本身也不一定等价于完整 full prefill。

## Q15：如果 Linear Attention 的 attention weights 差很多，但 KV 较接近，算成功吗？

算。

真正需要恢复的是：

[
O=AV
]

以及后续 hidden/KV，而不是 attention matrix 本身。不同权重可能经过 (V) 投影后产生相似输出。

## Q16：如果 zero-shot 结果很差，是否意味着方向失败？

不一定。

需要先判断失败来自：

* feature map 不合适；
* (r) 太小；
* denominator 不稳定；
* 局部尖锐 attention 无法近似；
* 多层误差累积；
* raw/preprocess cache 本身不同；
* RoPE/GQA 实现错误。

如果 FAVOR+ 随 (r) 增大稳定改善，或者 local+linear 明显优于 pure linear，就仍然说明该方向值得继续训练。

---

# 13. 结果报告要求

最终提交：

## 13.1 实现说明

写清楚：

* 修改了哪些文件；
* reprocess code path；
* attention mode 开关；
* feature map 实现；
* RoPE/GQA 处理；
* 多 selected-token 语义；
* 是否包含 current token；
* explicit/state equivalence。

## 13.2 可复现命令

至少给出：

```text
MHA baseline
ELU+1
FAVOR+ r=64/128/256
local+FAVOR+（如果实现）
```

## 13.3 结果表格

raw 和 preprocess 必须分开。

至少包含：

* attention-output relative L2；
* hidden-state relative L2；
* K/V relative L2；
* Delta cosine；
* (R_{\text{MHA}})；
* (R_{\text{full}})；
* logits KL；
* 端到端生成质量；
* NaN/稳定性；
* FAVOR+ seed 方差。

## 13.4 图

建议至少生成：

1. layer-wise attention-output error；
2. layer-wise K/V recovery；
3. feature dimension (r) 与恢复率；
4. raw vs preprocess；
5. selected rate 与最终质量；
6. Linear/MHA attention entropy 或 top-k overlap；
7. output norm ratio 和 denominator 分布。

## 13.5 结论

结论必须明确回答：

1. zero-shot Linear reprocess 是否比 no-reprocess 更好；
2. 能恢复 MHA reprocess 的多少；
3. K 和 V 哪个更容易恢复；
4. 误差从哪一层开始明显累积；
5. raw/preprocess 哪个更适合；
6. FAVOR+ 是否随 (r) 稳定改善；
7. local softmax 是否必要；
8. 是否值得进入 attention transfer/LoRA 阶段；
9. 如果要优化速度，Linear state 应如何缓存和复用。

即使结果为负，也要保留实验和完整分析，不要为了得到正结论而修改 selection、数据或 teacher。

---

# 14. 参考论文与代码

按阅读优先级排列。

1. [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
   重点理解 normalized causal linear attention、(S,z) 状态形式和显式/递归等价性。

2. [fast-transformers CausalLinearAttention](https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/causal_linear_attention.py)
   适合参考最基础的 causal implementation。

3. [Rethinking Attention with Performers](https://research.google/pubs/rethinking-attention-with-performers/)
   重点理解 FAVOR+ 如何近似 Softmax kernel。

4. [Google Research FAVOR+ code](https://github.com/google-research/google-research/blob/master/performer/fast_attention/tensorflow/fast_attention.py)
   重点参考随机投影和 numerical stabilization，不要直接复制 TensorFlow 接口。

5. [LoLCATs: On Low-Rank Linearizing of Large Language Models](https://arxiv.org/abs/2410.10254)
   重点理解为什么 attention-output matching 能保留 MHA-trained 模型行为，以及 local+linear 的价值。

6. [HazyResearch/LoLCATs](https://github.com/HazyResearch/lolcats)
   参考 Llama/Mistral attention replacement、Hedgehog/T2R feature map 和 linear+window 结构。

7. [The Hedgehog & the Porcupine](https://arxiv.org/abs/2402.04347)
   重点理解 Linear Attention 为什么难以复现 Softmax 的尖锐性和单调性。本阶段不要训练 Hedgehog。

8. [BASED](https://arxiv.org/abs/2402.18668) / [代码](https://github.com/HazyResearch/based)
   重点理解 local exact attention 与 global linear state 的 recall-memory trade-off。

9. [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
   当前阶段只参考 feature map、GQA shape 和 recurrent operator；不要一开始就集成 fused kernel。

10. [Linearizing Large Language Models / SUPRA](https://arxiv.org/html/2405.06640v1)
    用于理解如果 zero-shot 失败，后续如何通过 uptraining 将 MHA-trained 模型转换为 recurrent LM。

---

# 15. 最终目标

本任务不是证明 Linear Attention 与 MHA 等价，而是量化：

[
\boxed{
\text{在不训练、不改变原模型权重的条件下，
Linear Attention 能恢复 selected-token MHA 重算收益的多少}
}
]

最重要的结果不是单个生成样例，而是以下完整链路：

[
\text{feature map}
\rightarrow
\text{attention-output error}
\rightarrow
\text{hidden-state error}
\rightarrow
\text{layer-wise KV error}
\rightarrow
\text{cache recovery}
\rightarrow
\text{generation quality}
]

完成质量验证后，再决定是否进入：

* attention transfer；
* LoRA；
* learned Hedgehog feature map；
* Linear state checkpoint；
* prefix state reuse；
* fused kernel；
* 真正的重算加速阶段。
