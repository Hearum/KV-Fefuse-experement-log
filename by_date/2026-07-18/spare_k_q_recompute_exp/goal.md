# 核心修订：实现 Cache-Anchored Sparse Reprocessing，而不是 Sparse KV Replacement

当前实现方向存在一个根本错误：不能把 sparse reprocessing 产生的 candidate K/V 直接覆盖原始 raw/preprocess cached K/V。

本任务的研究动机是：

对于被选中重算的 token i，cached K/V 已经包含较多有效信息。Sparse reprocessing 的作用是为 cached K/V 提供上下文修正，而不是从零生成一份新 KV 并完全覆盖 cache。

因此必须实现：

Cache KV
+
Scaled sparse correction
->
Fused working KV

## 1. 三种 KV 必须严格区分

每一层必须同时区分：

1. base_cache_k/base_cache_v

   原始 raw 或 preprocess KV。

   它们在整个实验中不可变，不允许原地覆盖。

2. candidate_k/candidate_v

   由当前 selected-token hidden state 通过原始 W_K/W_V 投影得到：

   candidate_k_i^l = W_K^l RMSNorm(h_i^(l-1))
   candidate_v_i^l = W_V^l RMSNorm(h_i^(l-1))

   candidate 的 hidden state 来自此前各层的 sparse attention，因此称为 sparse candidate，但投影本身并不稀疏。

   上式是投影级简写。Qwen3 实际写入 cache 的 Key 还要经过 head-wise `k_norm` 和对应绝对位置的 RoPE：

   candidate_k_cache_i^l = RoPE(k_norm(W_K^l RMSNorm(h_i^(l-1))), position_i)

   后续所有 base/candidate/working Key 的比较与融合均在这个实际 cached-Key 坐标系中进行；Value 不经过 RoPE。

3. working_k/working_v

   当前层真正用于 block routing 和 sparse attention 的 KV。

   对 selected token：

   working_k_i^l =
       base_cache_k_i^l
       + alpha_k^l * (
           candidate_k_i^l
           - base_cache_k_i^l
       )

   working_v_i^l =
       base_cache_v_i^l
       + alpha_v^l * (
           candidate_v_i^l
           - base_cache_v_i^l
       )

   对 unselected token：

   working_k_j^l = base_cache_k_j^l
   working_v_j^l = base_cache_v_j^l

禁止直接执行：

working_k[selected] = candidate_k
working_v[selected] = candidate_v

除非 alpha_k=alpha_v=1，并且该配置明确作为 sparse-replace baseline。

## 2. 融合必须发生在当前层 Attention 之前

第 l 层的执行顺序必须是：

A. 输入所有 selected tokens 的 h^(l-1)

B. 并行生成所有 selected tokens 的：
   q_candidate^l
   k_candidate^l
   v_candidate^l

C. 将所有 selected tokens 的 candidate KV 与第 l 层 immutable base KV 融合

D. 一次性把所有 fused KV scatter 到第 l 层 working cache

E. 使用 working K 计算 block representative 和 block routing

F. 从 working KV 中 gather selected blocks

G. 对所有 selected Query 并行执行 sparse causal attention

H. 执行 attention output projection、residual、MLP，得到所有 selected tokens 的 h^l

I. 进入第 l+1 层，重复上述流程

不能先完成 sparse attention，再在层结束后融合 KV。否则当前层产生的 hidden state不会使用 cached KV 信息。

## 3. hidden state 与 cached KV 的关系

当前方法不缓存 cached hidden state，也不尝试由 KV 反推 hidden state。

cached KV 通过以下路径影响新的 hidden state：

base K/V
    ↓
与 candidate K/V 融合
    ↓
working K/V
    ↓
sparse attention
    ↓
attention output
    ↓
residual + MLP
    ↓
new hidden state

因此只要：

1. sparse attention 使用 fused working KV；
2. current causal block 始终保留；
3. 原始 causal attention 包含 self token；

则 token i 的 cached V_i^l 会通过 self-attention 进入 h_i^l，cached K_i^l 会影响 self-attention 权重。

但不要声称 new hidden state 等于 cached hidden 与 sparse hidden 的插值，因为 cached hidden 并不存在。

## 4. 保持 FusionRAG 的 layer-parallel selected-token 更新

设 i < j 且 i、j 都是 selected tokens。

第 l 层必须先同时产生并融合所有 selected tokens 的 K/V，再执行该层所有 selected Query 的 attention。

因此 token j 如果选择了 token i 所在 block，必须读取：

working_k_i^l
working_v_i^l

其中：

working_k_i^l =
    (1-alpha_k) * base_cache_k_i^l
    + alpha_k * candidate_k_i^l

working_v_i^l =
    (1-alpha_v) * base_cache_v_i^l
    + alpha_v * candidate_v_i^l

不能让 token j 读取：

- 旧 base_cache KV；
- 未融合的 candidate KV；
- token i 下一层的 KV；
- 逐 token 串行完成后的结果。

实现仍然是 layer-parallel，而不是 token-serial。

## 5. Block representative 必须来自 working K

对 block b：

block_key_b^l =
    mean_{p∈block b}(working_k_p^l)

不能始终使用 base_cache_k 计算 block mean。

否则 selected token i 虽然更新了 KV，但它的更新不会影响后续 token j 的 block routing。

为了优化，可以使用：

block_sum_work_b =
    block_sum_base_b
    + Σ_{i∈selected∩block b}
        (working_k_i - base_cache_k_i)

block_key_b =
    block_sum_work_b / block_length_b

第一版可以直接重新计算 mean，先保证正确性。

## 6. Selected predecessor dependency

在 pure MoBA 模式下，token j 只有在 block(i) 被选择时才能看到 selected predecessor i 的 fused KV。

必须记录：

dependency_coverage =
    selected predecessor pairs whose earlier block is selected
    /
    all causal selected predecessor pairs

额外实现可选配置：

preserve_selected_dependencies = false/true

false：
    current block + normal Top-K historical blocks。

true：
    除 current block 和 Top-K 外，强制保留包含 earlier selected tokens 的 blocks。

如果 true 导致 block 数超过原预算，必须报告真实 block 数和真实 KV token 数，不能仍按原 K_B 计算加速。

## 7. Alpha 实验

20% Key change 和 40% Value change不是 alpha，禁止直接设置：

alpha_k=0.2
alpha_v=0.4

第一阶段使用固定标量：

alpha_k = alpha_v = alpha

扫描：

alpha ∈ {0, 0.25, 0.5, 0.75, 1.0}

含义：

alpha=0：
    完全保留 base KV，不写入 candidate correction。

alpha=1：
    完全使用 candidate KV，即 sparse-replace baseline。

0<alpha<1：
    cache-anchored sparse correction。

完成同 alpha 扫描后，再分别改变 alpha_k 和 alpha_v，分析 Key/Value 是否需要不同修正强度。

不要第一轮直接做完整 alpha_k × alpha_v 大网格。

## 8. 两个实验轴必须分开

本方法同时包含两个变化：

A. Attention support：
   dense causal prefix
   vs.
   MoBA Top-K blocks

B. KV write-back：
   replace
   vs.
   cache-anchored blend

因此至少实现四组对照：

1. Dense Attention + Replace
   原始 FusionRAG dense reprocessing。

2. Dense Attention + Blend
   使用完整 causal attention，但 fused KV 按 alpha 写入和参与 attention。

3. Sparse Attention + Replace
   MoBA block sparse attention，alpha=1。

4. Sparse Attention + Blend
   本任务的主要方法。

否则无法判断性能变化来自 sparse attention，还是来自 KV blending。

## 9. Delta direction 分析

定义 dense reprocessing 的目标修正：

delta_dense_k =
    k_dense_reprocess - k_cache

定义 sparse candidate 修正：

delta_sparse_k =
    k_candidate_sparse - k_cache

Value 同理。

计算：

1. delta_sparse 与 delta_dense 的 cosine similarity；
2. relative L2；
3. layer-wise 统计；
4. raw/preprocess 分开统计。

如果 delta_sparse 与 delta_dense 的方向不一致，简单 alpha blending 不可能修复问题。

额外计算 oracle alpha：

alpha_k_star =
    clip(
        <delta_dense_k, delta_sparse_k>
        /
        (||delta_sparse_k||_2^2 + epsilon),
        0,
        1
    )

Value 同理。

分别统计：

- global oracle alpha；
- per-layer oracle alpha；
- oracle alpha 的均值、方差和分布。

Oracle alpha 只用于可行性分析，不能用于正式测试数据推理。

如果 per-layer oracle alpha 比全局 alpha 显著更好，说明下一阶段应该研究 per-layer alpha。

## 10. 必须增加的正确性测试

Test A：base cache immutable

运行结束后确认 raw/preprocess base cache 没有被修改。

Test B：alpha endpoints

alpha=0 时，最终 selected-position working KV 应等于 base KV。

alpha=1 时，working KV 应等于 candidate KV。

Test C：fusion before attention

检查 sparse attention 实际读取的是 working KV，而不是 base KV 或 candidate KV。

Test D：layer-parallel update

同层所有 selected-token candidate KV 必须先完成 fusion 和 scatter，再执行该层 selected Query attention。

Test E：selected dependency

若 i<j 且 block(i) 被 j 选择，j 实际 gather 到的必须是 i 的 fused working KV。

Test F：current/self block

current block 必须保留。若原始 FusionRAG attention 包含 self，则 token i 的 working V_i 必须参与自己的 attention。

Test G：block routing

Block representative 必须根据 fused working K 计算。

Test H：no silent replacement

除 alpha=1 baseline 外，任何代码路径都不能直接用 candidate KV 覆盖 base KV。
w