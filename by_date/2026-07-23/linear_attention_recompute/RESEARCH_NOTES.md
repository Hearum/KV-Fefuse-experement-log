# Linear Attention 调研笔记：局部 document state 与 causal prefix

日期：2026-07-23

## 调研来源

- Katharopoulos et al., Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
  https://arxiv.org/abs/2006.16236
- Choromanski et al., Rethinking Attention with Performers
  https://arxiv.org/abs/2009.14794
- Zhang et al., LoLCATs: On Low-Rank Linearizing of Large Language Models
  https://arxiv.org/abs/2410.10254
- Efficient Transformers: A Survey
  https://arxiv.org/abs/2009.06732

## 结论先行

每个 document block 可以先计算局部 Linear Attention state，但局部 state 不能单独代表 token 所需的全局 causal state。

对 feature map φ，定义：

S(B) = Σ(j∈B) φ(k_j) v_j^T
z(B) = Σ(j∈B) φ(k_j)

如果可见 prefix 被分成多个 block P = B0 ∪ B1 ∪ ... ∪ Bm，则：

S(P) = Σb S(Bb)
z(P) = Σb z(Bb)

因此可以预先为每个 block 计算局部 state，再对 token i 合并它可见的前序 block state，以及当前 block 中直到 i 的 local prefix state。

但 state 数值仍然依赖 prefix。它只是固定大小的 prefix summary，不是与 prefix 无关的常量。

## 对本项目的含义

当前布局大致为：

system + doc1 + doc2 + ... + docN + query

对于 document token i，query 位于它之后，因果性意味着 query 不应进入 document token 的 state。token i 需要的是：

S(≤i) = S(system) + S(doc之前) + S(doc当前块≤i)

仅使用当前 document block 的 state 会漏掉 system 和前序 document 的贡献，因此不能直接替代完整 prefix state。

## 什么时候局部 state 可以复用

局部 state 可以复用，需要同时满足：

- 局部 block 的 K/V 与实际 assembled cache 一致；
- K 已经使用正确的全局 position 应用 RoPE；
- feature map、随机 projection、缩放和 dtype 完全一致；
- block 之间没有需要重新计算的跨 block hidden-state 依赖；
- selected token 的新 K/V 已经被纳入 state，或者能够显式更新。

如果 document 独立 prefill 使用 local position，而最终 assembled prompt 使用 global position，通常不能直接合并已经聚合好的 state。原因是 φ(R_p k) 一般不能从 φ(k) 的聚合结果中恢复，feature map 与 RoPE 不可交换。

安全做法是：在正确全局 position 下生成或变换 K，再构造 block state；或者保留 token-level K/V，在 assembled cache 上重新构造 state。

## Selected-token reprocess 的 state 更新

如果旧 state 中包含旧的 (k_i,v_i)，而 online reprocess 产生新 (k_i',v_i')，可以更新：

S' = S - φ(k_i)v_i^T + φ(k_i')v_i'^T
z' = z - φ(k_i) + φ(k_i')

但这只有在 state 的 memory 语义和当前 reprocess 顺序明确时成立。

如果 selected token 是顺序写回，后续 token 必须看到前一个 token 的更新 state；如果 selected token 是独立重算，每个 token 应从同一个原始 cache state 开始。不能因为引入 Linear state 而顺便改变现有 MHA reprocess 语义。

## Local + global hybrid

如果未来采用 local Softmax + global Linear：

- global state 可以由远距离 block states 相加得到；
- local window 仍需保留逐 token K/V，执行精确 Softmax；
- local tokens 不能同时计入 global state，否则会重复计数；
- numerator 和 denominator 必须使用相同的 local/global 分区。

形式上：

N = N_global_linear + N_local_softmax
D = D_global_linear + D_local_softmax
o = N / (D + ε)

## 对实验的建议

第一阶段不要直接假设“每个 document 一个 state 就够了”。建议：

1. 对一个真实 assembled cache，在正确全局 position 下构造 token-level Linear state；
2. 按 document/block 分组；
3. 验证 block state 相加是否等于完整 prefix state；
4. 加入 selected token 的旧值替换和新值写回；
5. 最后才测试 state reuse 和 local+global 优化。

第一阶段质量实验可以使用显式 assembled-prefix 计算；只有 state 加法、RoPE 对齐和 selected-token 更新都通过后，才讨论真正的 state-cache 加速。

## 实验报告新增口径

同时报告：

- local_block_state：仅当前 document/block；
- assembled_prefix_state：按实际全局 token 顺序合并后的可见 prefix；
- updated_prefix_state：应用 selected-token 新 KV 后的 state。

只报告 local state，不能声称已经实现 causal Linear Attention，也不能据此声称可以加速 FusionRAG online reprocess。


## Follow-up: implementation hazards found in prior experiments

### 1. The question segment may not use the full MHA path

Some earlier interfaces may reuse part of the recomputation operator for the
question segment. Then the comparison is not simply document KV
recomputation versus baseline MHA: the question-side attention path has also
changed, possibly silently.

Every experiment must record separately:

- document-token K/V construction;
- question-token Q/K/V construction;
- whether question attention uses the ordinary full-MHA operator;
- whether question tokens read the updated document KV during the same prefill;
- whether question hidden states and logits use the same path as baseline.

The primary correctness target is to keep question tokens on the ordinary MHA
path and restrict the new operator to the intended document-token region. A
question-side fast path, if tested, must be a separately named ablation.

### 2. The existing interface may not provide the required causal update operator

For causal attention, an earlier token i can affect a later token j when i < j,
but a later token j must not affect an earlier token i when j > i. A linear
state is naturally a prefix recurrence, so a token-by-token update cannot
provide reverse-direction influence without violating causality.

The current interface may also lack an efficient batched operator that updates
selected K/V while making the correct causal state visible to all later
selected positions. Before a large run, implement a small reference/fast
operator with two explicit modes:

1. independent_snapshot: every selected token reads the same pre-recompute
   cache/state;
2. causal_sequential: selected positions are processed in order, and an
   updated position can affect later positions only.

The operator should trace the input state, each rank-one state delta, the state
used at each position, and the resulting K/V. Compare it with a slow
token-by-token reference on a tiny synthetic case. If j > i appears to affect
i, treat it as an indexing, cache-write, or mask bug.

### 3. Delayed KV insertion can make recomputation a no-op for prefill

An earlier interface could compute new K/V but add them to old_kv only later in
the forward path. In that case the prefill attention still reads old_kv, so
the reported result does not measure the proposed recomputation.

The intended ordering is:

compute candidate KV
  -> blend candidate with base KV
  -> write effective KV to the cache read by this prefill
  -> run the intended MHA/linear attention
  -> produce hidden states and next-layer inputs

Each run needs an ordering assertion or trace showing that attention reads the
effective KV, rather than merely showing that a later cache write occurred.
The non-strict path needs special attention: blending after the model call
cannot affect the current query logits and can only affect later computation.

## Mandatory sanity checks for the next prototype

1. Baseline full MHA with no recomputation.
2. Same input with recomputation disabled but tracing enabled; outputs should
   match baseline within numerical tolerance.
3. independent_snapshot and causal_sequential on a tiny case, with influence
   between selected positions inspected explicitly.
4. Verify that question tokens use ordinary MHA and read the effective blended
   document KV in the same prefill.
5. Compare KV_base, KV_linear, KV_blend, attention output, post-attention
   hidden state, and next-layer K/V.

These checks are prerequisites for attributing a quality difference to linear
attention recomputation rather than to question-path reuse, causal ordering,
or delayed cache insertion.

## Clarification: layer-wise hidden-state and V propagation

The intended recurrence is correct at a high level, but it must be written
with explicit layer indices:

h_l
  -> Q_l, K_l, V_l from the current layer projections
  -> attention using the effective K_l and V_l
  -> residual/output projection/MLP
  -> h_{l+1}
  -> Q_{l+1}, K_{l+1}, V_{l+1}

Thus the current layer hidden state is not obtained by directly adding
V_old and V_compute. Attention first uses the query-dependent weights to
combine the values. If the current-layer values are blended, the intended
operation is, for example:

V_l,effective = gamma V_l,old + (1-gamma) V_l,compute

and normally K_l must be handled consistently as well:

K_l,effective = gamma K_l,old + (1-gamma) K_l,compute

The resulting attention output produces h_{l+1}; only then is the next layer
V_{l+1,compute} obtained by applying the next layer V projection to h_{l+1}.
This means an update at layer l can propagate to later layers through the
hidden state, even if only selected document tokens were directly recomputed
at layer l.

There are therefore two distinct experiments:

1. KV-level fusion: blend current-layer K/V before attention, then let the
   normal transformer block compute the next hidden state.
2. Hidden-state fusion: independently compute two attention/block outputs and
   blend hidden states after attention or after the block.

The first is the recommended implementation because it preserves the normal
attention, residual, and MLP semantics. Hidden-state fusion should be a
separate ablation. A V-only blend is also a separate ablation because changing
K changes attention weights, not just the value aggregation.

For causal propagation, an updated token at layer l can affect later tokens in
the same layer according to the causal mask, and its resulting h_{l+1} can
affect that token K/V in layer l+1. This cross-layer propagation is distinct
from allowing a later token to affect an earlier token in the same causal

## Historical link: MoBA / Working-KV sparse recomputation

The remembered old experiment was the MoBA-style sparse recomputation branch,
not Linear Attention. The closest records are:

- commit e519e24: first attempt to fuse sparse candidate K/V before attention;
- commit 44890a9: restricted fusion to explicit document reprocess calls so a
  multi-token question prefill was not mistaken for document recomputation;
- commit ea73a0e: added immutable selected-position base K/V snapshots and
  explicit causal dependency checks;
- commit 0fe51cc: moved the implementation to the Qwen3 file actually imported
  by the setup-v2 pipeline.

The historical experiment plan defines four objects for a selected token at
layer l:

1. base KV: the raw or preprocess cache;
2. dense full KV: the native dense causal recompute result;
3. sparse candidate KV: the MoBA/sparse-support recompute result;
4. working KV: the value actually written before the layer attention.

The intended working value is:

V_work = V_base + alpha_V * (V_sparse - V_base)

and not direct replacement of V_base.

### Confirmed old overwrite bug

The strongest matching discussion is in the historical
CRITICAL_BUG_FIX.md and COMPLETE_FIX_SUMMARY.md from commit e4b79e2. In Online
Lazy mode, missing documents were assigned cache positions based on the
original passages layout, even though past_key_values had already removed
missing chunks. A recomputed missing document was therefore written into a
position occupied by a loaded document. For example, a missing Doc2 at passage
positions 300:500 was written into KV positions 300:500, where loaded Doc3
actually lived. This overwrote Doc3 K/V and lost its information.

The same position mismatch also affected:

- extracting the missing document KV from the cache;
- the original positions used for RoPE correction;
- the KV saved to disk and reused by later runs.

Therefore this was a cache-position aliasing bug, not a property of sparse
attention. The repair required a passages-position to KV-position mapping in
all three places. Reusing a cache produced before the repair could make the
repair appear ineffective.

### Confirmed per-layer base-snapshot issue

The later Working-KV discussion explicitly states that base KV must remain
immutable during a selected-token recompute. At each layer, candidate K/V is
generated from the hidden state produced by the previous layer, then blended
with the original base snapshot for that layer before attention. The candidate
from one layer must not be reused as a fixed V_old for every later layer.

The failure mode is:

1. layer l writes candidate or working V into the shared cache;
2. layer l+1 reads that modified cache as if it were the original base V;
3. the next candidate is compared against or blended with the wrong baseline;
4. the original cached information is progressively lost or the layer-wise
   interpolation no longer has the intended meaning.

The correction in ea73a0e snapshots selected-position base K/V for all layers
before the reprocess call, passes those snapshots through the model, and uses
the snapshot corresponding to the current layer. This is the implementation
pattern that the new Linear experiment should preserve.

### Relation to the current Linear direction

For the Linear prototype, use distinct tensors and names:

- V_base_l: immutable cache value before layer l recompute;
- V_compute_l: candidate value produced from h_l;
- V_work_l: value used by layer l attention;
- h_{l+1}: hidden state produced by layer l;
- V_compute_{l+1}: next-layer projection from h_{l+1}.

Never let V_work_l silently become V_base_{l+1}. The latter must be obtained
from the original cache snapshot for layer l+1. Also distinguish the
cache-position mapping problem from the layer-snapshot problem: both can
