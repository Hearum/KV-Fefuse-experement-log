我们是要把整个 MHA 模型转换成 Linear Attention，也不准备先训练 LoLCATs（一个用于吧mha模型转化成mla都框架），而是做一个更局部、更直接的可行性实验：

> 保留原模型所有权重和已有 KV cache，只在 token (i) 的逐层重算过程中，把原来的 MHA 读取操作临时换成一个无训练的 Linear Attention，观察最终更新出来的 KV 能否接近标准 MHA 重算结果。

这件事是可行的，而且比转换整个模型简单，因为只有被选 token 的计算路径改变，前序 token 不需要重新运行模型。

## 1. 三条对照路径

对于同一个 token (i)，需要同时保留三种结果。

### Cache baseline：完全不重算

直接使用已有缓存：

[
K_{i,\text{cache}}^l,\quad V_{i,\text{cache}}^l
]

它可能来自：

* raw cache；
* preprocess cache。

这两类必须分开实验。

### MHA reprocess：标准重算

使用原模型的 MHA，对 token (i) 从第 0 层逐层重算，得到：

[
K_{i,\text{MHA}}^l,\quad V_{i,\text{MHA}}^l
]

这是当前 FusionRAG 的标准做法，也是 Linear Attention 需要近似的直接 teacher。

### Linear reprocess：你的候选方法

所有模型参数保持不变，只把 token (i) 每层的 Attention 操作替换为固定、无训练的 Linear Attention：

[
K_{i,\text{linear}}^l,\quad V_{i,\text{linear}}^l
]

最后回答两个不同的问题：

1. Linear Attention 能否复现 MHA reprocess？

[
KV_{\text{linear}}
\overset{?}{\approx}
KV_{\text{MHA-reprocess}}
]

2. Linear Attention 能否比原 cache 更接近 full-prefill KV？

[
D(KV_{\text{linear}},KV_{\text{full}})
<
D(KV_{\text{cache}},KV_{\text{full}})
]

这两个问题要分开。即使 Linear 没有完美复现 MHA，只要它比不重算明显更接近 full KV，依然有研究价值。

---

## 2. token (i) 的逐层 Linear 重算流程

设 token (i) 在第 (l) 层的重算输入为：

[
h_{i,\text{linear}}^{l-1}
]

第一层输入仍然来自原始 embedding：

[
h_{i,\text{linear}}^0=h_i^0
]

之后每一层执行以下步骤。

### 第一步：使用原模型投影生成 QKV

完全复用原模型参数：

[
q_i^l=W_Q^l\operatorname{LN}
\left(h_{i,\text{linear}}^{l-1}\right)
]

[
k_i^l=W_K^l\operatorname{LN}
\left(h_{i,\text{linear}}^{l-1}\right)
]

[
v_i^l=W_V^l\operatorname{LN}
\left(h_{i,\text{linear}}^{l-1}\right)
]

如果原模型有 RoPE，就按照原始位置 (i) 对 (q_i^l,k_i^l) 应用 RoPE。

这里没有训练新投影，也不修改 (W_Q,W_K,W_V)。

---

### 第二步：取得 token (i) 可见的缓存 KV

第 (l) 层已经保存了其他 token 的 KV：

[
\left{
K_{j,\text{cache}}^l,V_{j,\text{cache}}^l
\right}_{j<i}
]

标准 decoder attention 通常包含当前 token 自己，所以实际参与 attention 的 memory 是：

[
\mathcal K_i^l=
\left[
K_{<i,\text{cache}}^l;k_i^l
\right]
]

[
\mathcal V_i^l=
\left[
V_{<i,\text{cache}}^l;v_i^l
\right]
]

具体使用 (j<i) 还是 (j\le i)，应该完全遵循当前 MHA reprocess 的 causal mask。实验中只替换 attention operator，不改变任何可见范围和 cache writeback 语义。

---

### 第三步：用 Linear Attention 读取缓存

选择一个不需要训练的固定 feature map：

[
\phi:\mathbb R^{d}\rightarrow\mathbb R^{r}
]

将缓存 key 转换为：

[
\bar k_j^l=\phi(k_j^l)
]

query 转换为：

[
\bar q_i^l=\phi(q_i^l)
]

构造两个线性状态：

[
S_i^l
=====

\sum_{j\le i}
\bar k_j^l
v_j^{l\top}
]

[
z_i^l
=====

\sum_{j\le i}\bar k_j^l
]

Linear Attention 输出为：

[
o_{i,\text{linear}}^l
=====================

\frac{
\bar q_i^{l\top}S_i^l
}{
\bar q_i^{l\top}z_i^l+\epsilon
}
]

它等价于显式计算：

[
a_{ij}^{\text{linear}}
======================

\frac{
\phi(q_i^l)^\top\phi(k_j^l)
}{
\sum_{m\le i}
\phi(q_i^l)^\top\phi(k_m^l)+\epsilon
}
]

然后：

[
o_{i,\text{linear}}^l
=====================

\sum_{j\le i}
a_{ij}^{\text{linear}}v_j^l
]

因此第一阶段为了验证质量，完全可以显式计算 (a_{ij}^{\text{linear}})，暂时不实现 recurrent state kernel。两种写法数学等价。

---

### 第四步：继续使用原模型的后半层

得到 Linear Attention output 后，其他部分全部不改：

[
\tilde h_i^l
============

h_{i,\text{linear}}^{l-1}
+
W_O^lo_{i,\text{linear}}^l
]

再经过原始 MLP：

[
h_{i,\text{linear}}^l
=====================

\tilde h_i^l
+
\operatorname{MLP}^l
\left(
\operatorname{LN}(\tilde h_i^l)
\right)
]

这个新的 hidden state 会生成下一层的：

[
q_i^{l+1},k_i^{l+1},v_i^{l+1}
]

于是 token (i) 从底层到顶层递归更新：

[
h_i^0
\rightarrow
KV_i^1
\rightarrow
h_i^1
\rightarrow
KV_i^2
\rightarrow
\dots
\rightarrow
KV_i^L
]

最终，将每一层产生的：

[
k_{i,\text{linear}}^l,\quad
v_{i,\text{linear}}^l
]

作为 token (i) 的更新 KV 写回 cache。

---

## 3. 整个过程可以概括为

```text
原始 cache KV
       +
token i 的 embedding
       ↓
原 WQ/WK/WV 生成 q_i、k_i、v_i
       ↓
用固定 Linear Attention 读取前序 cached KV
       ↓
原 WO + residual + MLP
       ↓
得到下一层新的 hidden state
       ↓
继续生成下一层 QKV
       ↓
逐层递归到模型顶层
       ↓
写回 token i 各层的 Linear-reprocess KV
```

这里变化的只有：

[
\operatorname{softmax}
\left(
q_iK_{\le i}^\top/\sqrt d
\right)V_{\le i}
]

被替换为：

[
\frac{
\phi(q_i)^\top
\sum_{j\le i}\phi(k_j)v_j^\top
}{
\phi(q_i)^\top
\sum_{j\le i}\phi(k_j)
}
]

其他参数和计算路径都保持不变。

---

## 4. 无训练情况下应该使用哪种 Linear Attention？

这是实验成败最关键的变量。

### 不建议只做 (\phi(x)=x)

直接使用：

[
o_i=q_i^\top(K^\top V)
]

容易出现负权重、尺度爆炸和分母不稳定，与原 Softmax 差异太大。

### ELU+1 可以作为最简单基线

[
\phi(x)=\operatorname{ELU}(x)+1
]

优点是：

* 始终为正；
* 实现简单；
* 状态容易累积；
* 经典 Linear Transformer 使用过。

但它本质上换了一种 attention kernel，并不是专门近似 Softmax。

### 主实验更适合使用 FAVOR+

FAVOR+ 使用正随机特征近似：

[
\exp(q^\top k)
\approx
\phi(q)^\top\phi(k)
]

所以它和原 MHA 的目标最一致。

为了对应：

[
\exp\left(q^\top k/\sqrt d\right)
]

先缩放：

[
q'=q/d^{1/4},\qquad
k'=k/d^{1/4}
]

然后对 (q',k') 计算随机特征。

建议扫描 feature dimension：

[
r\in{32,64,128,256,512}
]

因为 (r) 越大，理论上越接近 Softmax，但状态大小和计算量也越高。

由于 FAVOR+ 有随机性，每个设置至少运行 3～5 个随机 seed。

### 可以增加一个 Local + Linear 版本

为了验证局部精确 attention 是否关键，可以再测试：

[
o_i
===

o_i^{\text{local-softmax}}
+
o_i^{\text{global-linear}}
]

例如：

* 最近 64/128 个 token 使用原 Softmax；
* 更远历史使用 FAVOR+。

这不是第一优先级，但如果纯 Linear 明显退化，可以判断问题是否主要来自局部高频、尖锐 attention 无法近似。

---

## 5. 第一阶段只测质量，不声称加速

这里存在一个重要区别。

如果只有一个 query token (i)，标准 MHA 读取缓存的成本大约是：

[
O(id)
]

如果 Linear Attention 没有提前保存 (S,z)，现场构造：

[
S_i=\sum_{j\le i}\phi(k_j)v_j^\top
]

成本是：

[
O(ird_v)
]

可能反而比单 query MHA 更贵。

所以第一阶段应该明确定位为：

> 显式计算 Linear Attention weights，只验证零训练算子替换后的重算质量。

等质量结果成立后，第二阶段才实现：

[
S_i^l,\quad z_i^l
]

的预计算与缓存。此时 token (i) 的读取复杂度才变成：

[
O(rd_v)
]

不再随前缀长度增长。

也就是说：

[
\boxed{
\text{先验证 Linear reprocess 是否有效，
再讨论 recurrent state 如何构建和复用}
}
]

---

## 6. 多个 selected tokens 的语义必须与现有重算一致

假设选中了：

[
i_1<i_2<\dots<i_m
]

有两种可能的重算语义。

### 独立重算

每个 selected token 都只读取原始 cached KV：

[
KV_{<i,\text{cache}}
]

后面的 selected token 不看前面 selected token 的更新值。

### 顺序写回

处理完 (i_1) 后，将其新 KV 写回；计算 (i_2) 时使用更新后的 (KV_{i_1})。

你现在不应该同时改变这一语义。应该完全复用当前 FusionRAG 的 MHA reprocess 流程，只把其中的 attention 计算替换掉。

如果未来使用 additive linear state，并且后续 token 要看到前面 token 的更新，可以更新：

[
S
\leftarrow
S
-

\phi(k_{\text{old}})v_{\text{old}}^\top
+
\phi(k_{\text{new}})v_{\text{new}}^\top
]

归一化状态也对应更新：

[
z
\leftarrow
z
-

\phi(k_{\text{old}})
+
\phi(k_{\text{new}})
]

但这是第二阶段的系统优化，不必在初始质量实验中实现。

---

## 7. 评估应该分成三层

### Attention 层输出

直接比较：

[
D_O^l
=====

\frac{
|o_{i,\text{linear}}^l-o_{i,\text{MHA}}^l|*2
}{
|o*{i,\text{MHA}}^l|_2
}
]

这能判断 Linear kernel 本身从哪一层开始偏离。

### Hidden state 与 KV

比较：

[
D_H^l,\qquad D_K^l,\qquad D_V^l
]

重点观察误差是否逐层累积：

[
D_O^l
\rightarrow
D_H^l
\rightarrow
D_{KV}^{l+1}
]

这比只看最终 KV 更能解释失败原因。

### 端到端效果

分别比较：

* 不重算；
* MHA 重算；
* Linear 重算；
* full prefill。

定义恢复比例：

[
R_{KV}
======

\frac{
D(KV_{\text{cache}},KV_{\text{full}})
-------------------------------------

D(KV_{\text{linear}},KV_{\text{full}})
}{
D(KV_{\text{cache}},KV_{\text{full}})
}
]

解释为：

* (R=1)：Linear 完全恢复 cache 到 full 的差距；
* (R=0)：和完全不重算一样；
* (0<R<1)：恢复了一部分；
* (R<0)：Linear 重算反而让 KV 更差。

同时计算相对于标准重算的恢复率：

[
R_{\text{MHA}}
==============

1-
\frac{
D(KV_{\text{linear}},KV_{\text{MHA-reprocess}})
}{
D(KV_{\text{cache}},KV_{\text{MHA-reprocess}})
}
]

这样可以区分：

* Linear 是否复现 MHA reprocess；
* Linear 是否实际改善 full-context KV；
* MHA reprocess 自身距离 full prefill 还有多少差距。

---

## 8. 最小实验矩阵

建议第一轮只做：

| 维度             | 设置                           |
| -------------- | ---------------------------- |
| Cache          | raw / preprocess 分开          |
| 重算             | no-reprocess / MHA / Linear  |
| Linear feature | ELU+1 / FAVOR+               |
| FAVOR+ 维度      | 64 / 128 / 256               |
| Random seed    | 3 个                          |
| Selection rate | 先固定一个，例如 20%                 |
| 指标             | (O/H/K/V) relative L2、生成质量   |
| Reference      | MHA reprocess + full prefill |

第一轮暂时不要：

* 训练 feature map；
* 训练 LoRA；
* 修改 token selection；
* 实现复杂 fused kernel；
* 同时改变 cache writeback 语义；
* 直接追求速度。

## 最终实验目标

这项实验本质上是在验证：

[
\boxed{
\text{对于已经由 MHA 训练好的模型，
固定的 Softmax-kernel Linear Approximation
是否能在不训练的情况下完成 selected-token reprocess}
}
]

完整路径是：

[
\boxed{
\text{cached KV}
\xrightarrow[\text{original projections}]
{\text{zero-shot linear read}}
\tilde h_i^1
\rightarrow
\tilde h_i^2
\rightarrow\dots\rightarrow
\widetilde{KV}_i
}
]

第一阶段的成功标准并不需要 Linear 完全复现 MHA。只要满足：

[
D(KV_{\text{linear}},KV_{\text{full}})
<
D(KV_{\text{cache}},KV_{\text{full}})
]

并且端到端生成质量随 feature dimension 或 selection rate 稳定改善，就足以说明“用 Linear Attention 替代部分重算”具有进一步训练和系统优化的价值。
