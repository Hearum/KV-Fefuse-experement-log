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

## Phase-0 execution record: 2026-07-23

Added:

scripts/test_linear_state_semantics.py

The explicit reference test passed:

- block state and normalizer additivity;
- selected-token rank-one state replacement;
- causal prefix state for every token;
- missing-document cache position mapping without overwriting a loaded document;
- KV blend alpha endpoints and intermediate values.

All checks passed with float64 maximum error around 1e-15.

The historical MoBA test
spare_k_q_recompute_exp/scripts/test_working_kv_semantics.py was also tried,

This historical test then failed before execution because the current branch no
longer exports preserve_cache_past_tokens from ktransformers.util.utils. The
failure is an interface compatibility issue, not a result for the Linear
prototype.

## Upstream implementation audit: 2026-07-23

The current qjy environment does not provide fast_transformers, fla, or
flash_linear_attention, and the FusionRAG repository does not contain a
Linear Attention implementation. The first implementation must therefore be
a local reference port, not a direct import.

The most useful reference implementations inspected are:

- Idiap fast-transformers causal linear attention:
  https://raw.githubusercontent.com/idiap/fast-transformers/master/fast_transformers/attention/causal_linear_attention.py
- FLA naive recurrent implementation:
  https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/linear_attn/naive.py
- FLA fused recurrent implementation:
  https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/linear_attn/fused_recurrent.py
- FLA normalization helper:
  https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/linear_attn/utils.py

### Implementation facts to preserve

1. The feature map is applied before the recurrent state update. The recurrent
   operator expects feature-mapped q and k; it is not automatically equivalent
   to using raw Q and K.
2. The unnormalized state is a K-by-V matrix:
   S_t = S_{t-1} + outer(phi(k_t), v_t).
3. The normalized output uses a running key state:
   z_t = z_{t-1} + phi(k_t), followed by
   o_t = (S_t^T phi(q_t)) / (phi(q_t)^T z_t + eps).
4. An initial state is part of the API. For normalized attention it is a pair
   consisting of the KV state and the cumulative-key z state.
5. A chunk implementation must include both inter-chunk prefix state and
   intra-chunk causal attention. Computing one whole block state and using it
   for every token in that block leaks future tokens inside the block.
6. FLA computes recurrent state in float32 and casts output back to input dtype
   in the naive reference. Numerical comparisons should match this policy.
7. The FLA fused recurrent implementation is a kernel for the same recurrence,
   not a drop-in replacement for Qwen MHA. The caller still chooses feature
   maps, scaling, normalization, layout, and initial-state semantics.

### Consequence for FusionRAG

The first candidate operator should expose:

linear_recurrent(q, k, v, initial_kv_state, initial_z_state,
                 normalize=True, output_final_state=True)

and return token outputs plus final state. It should have an explicit prefix
state input rather than silently reading the entire cache. This makes the
following cases testable:

- full explicit prefix versus assembled block states;
- old cache state versus updated selected-token state;
- independent snapshot versus causal sequential selected updates;
- exact FLA-style normalization versus a project-specific variant.

Only after this reference matches a token-loop implementation should we write a
fused operator. The prior MoBA experience is a reason to keep the reference
and the cache writeback path separate, not a reason to start with a Triton
kernel.
## Phase-0.5 execution record: normalized reference

Added:

- linear_attention_reference.py;
- scripts/test_linear_attention_reference.py.

The normalized recurrent implementation and the chunked implementation both
match explicit prefix computation for sequence lengths 1, 3, 7, and 16, with
chunk sizes 1, 2, 4, and 8. Tests included non-empty initial KV/z states,
final-state comparison, and bfloat16 input. Maximum observed errors were
consistent with float32 accumulation and all tests passed.

The chunk implementation explicitly separates inter-chunk state from
intra-chunk causal attention. It does not expose a whole chunk state to tokens
that occur earlier inside that chunk.

## Conceptual refinement: coefficient substitution versus KV recomputation

For a fixed layer and fixed Q/K/V, both MHA and normalized Linear Attention
produce an attention output that is a weighted sum of V:

MHA:
  o_i = sum_j softmax(q_i transpose k_j) v_j

Linear:
  o_i = [phi(q_i) transpose sum_j phi(k_j) v_j]
        / [phi(q_i) transpose sum_j phi(k_j) + eps]

The Linear expression can also be written as:

  o_i = sum_j w_linear(i,j) v_j

where the weights are induced by the feature map and the running
normalization, rather than by the exact softmax kernel. Under this fixed-QKV
view, the primary difference is the attention coefficient calculation.

This gives three distinct experimental levels:

1. coefficient-only substitution:
   keep the same Q/K/V and compare softmax attention output with normalized
   Linear output. This isolates the approximation in the attention weights.
2. attention-output blend:
   y = gamma * y_MHA + (1-gamma) * y_linear, then run the normal output
   projection, residual, and MLP. This tests a controlled coefficient
   replacement without changing the current layer K/V.
3. recursive Linear reprocess:
   use y_linear to produce the next hidden state, then project the next layer
   K/V. This changes future Q/K/V through the hidden-state recurrence and is
   no longer only a coefficient substitution.
   
The KV blend proposal is a fourth, different intervention:

  K_work = blend(K_base, K_compute)
  V_work = blend(V_base, V_compute)

It changes both the attention coefficients and the values. It must not be
described as coefficient-only Linear Attention.

### Consequence for the experiment order

The first model-level experiment should freeze current-layer Q/K/V and compare
MHA output against Linear output. It should report coefficient/output errors
before any cache writeback. Then test attention-output blending. Only after
these are understood should we enable recursive hidden-state propagation and
candidate KV generation.

This decomposition can explain a poor earlier result: a large quality drop
could come from the Linear coefficient approximation, from changed recursive

## Phase C smoke result: document-only coefficient probe

- The first end-to-end smoke exposed that setup-v2 uses a strict document
  reprocess followed by a separate query prefill. Passing ablation kwargs into
  the latter caused an old `Qwen2ForCausalLM` signature failure; the query call
  now remains dense MHA and receives no ablation kwargs.
- `linear` now forces strict mode in the setup-v2 runner, so only selected
  document reprocess tokens use the normalized Linear attention output.
- MuSiQue-v2 example 0, `online_draft`, rate 0.15, alpha=1.0 completed:
  prediction `Maria Bello`, EM=1, F1=1, inline GLM=True.
- This is only a connectivity/semantic smoke result, not a dataset-level
  quality claim. The next required result is a fixed-size comparison table
  with baseline, alpha sweep, EM/F1, and GLM Acc.

### MuSiQue-v2 first 10 results

With `online_draft`, rate=0.15, strict document-only linear coefficient
substitution, alpha=1.0, and the default inline GLM judge: n=10, EM=0.50,
F1=0.4472, GLM Acc=0.50. This is the first quality result, not merely a
connectivity check; it should be compared against the same-segment baseline
and then repeated across alpha/gamma before drawing conclusions.

### Efficiency correction

The current implementation computes standard MHA first and Linear Attention
second, then blends their outputs. This is intentionally a quality diagnostic:
it answers whether Linear output can replace MHA under the same online Q/K/V,
but it does not save MHA compute. The production candidate must branch before
SDPA: selected document queries use Linear only, while question queries remain
dense MHA. Output blending is therefore a diagnostic ablation, not the speedup
path.

## MuSiQue-v2 full 200: preprocess/raw four-group comparison

The true replacement path was run with rate=0.15 and the default inline GLM
judge. Results below are computed directly from each runs 200-row
`metrics.csv`:

| cache source | reprocess attention | EM | F1 | GLM Acc |
|---|---|---:|---:|---:|
| preprocess | MHA historical baseline | 24.00 | 37.88 | 41.00 |
| preprocess | Linear replacement | 29.50 | 37.88 | 41.00 |
| raw | MHA | 29.00 | 37.21 | 38.00 |
| raw | Linear replacement | 29.00 | 37.21 | 39.00 |

The preprocess Linear and raw Linear runs completed without errors. The
