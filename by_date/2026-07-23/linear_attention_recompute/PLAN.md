# Linear Attention Recompute 探究实验计划

日期：2026-07-23
分支：hming/linear-attention-recompute

## 1. 研究目标

本方向不是用纯 Linear Attention 替代 MHA，而是验证：

> 能否利用轻量 Linear 重算得到 selected document token 的候选 KV，
> 再与已有的 MHA/cache KV 做线性融合，以较低重算成本恢复跨文档信息。

主方法定义为：

K_work_l = K_base_l + alpha_K * (K_compute_l - K_base_l)
V_work_l = V_base_l + alpha_V * (V_compute_l - V_base_l)

第一阶段使用 alpha_K = alpha_V = alpha。alpha=0 是 cache-only，
alpha=1 是 pure candidate replacement，中间值是主要研究对象。

纯 Linear replacement 只作为下界和诊断对照，不作为主要方案。

## 2. 已知历史问题与不可违反的语义

### 2.1 question 路径

question token 必须继续使用普通完整 MHA。新的 Linear 或 sparse
recompute operator 只作用于 selected document token。必须单独记录：

- document token 的 K/V 构造路径；
- question token 的 Q/K/V 构造路径；
- question attention 是否读取当前 prefill 已生效的 working document KV；
- question hidden state 和 logits 是否与 baseline 使用同一代码路径。

任何 question-side fast path 都只能作为单独 ablation。

### 2.2 cache position

passages position 与 past_key_values position 不一定相同。存在 missing
document 时，禁止直接使用 passages 中的 token index 作为 cache_position。
所有写回、读取、RoPE 修正都必须经过明确的位置映射。

历史 e4b79e2 中，missing Doc2 被错误写入已加载 Doc3 的 cache 区间，造成
KV 覆盖和信息丢失。该问题必须用 toy mapping test 覆盖。

### 2.3 每层 immutable base snapshot

对每一层 l，维护以下不同对象：

- V_base_l：第 l 层重算前的原始 cache snapshot；
- V_compute_l：由当前 h_l 产生的 candidate；
- V_work_l：第 l 层 attention 实际读取的融合结果；
- h_{l+1}：第 l 层 block 输出；
- V_compute_{l+1}：由 h_{l+1} 经过下一层 projection 产生的 candidate。

不能让 V_work_l 静默变成 V_base_{l+1}。每层必须使用自己对应的
immutable base snapshot。

### 2.4 更新时序

正确顺序必须是：

compute candidate KV
  -> blend candidate with base KV
  -> write effective KV to the cache used by this prefill
  -> run current-layer attention
  -> produce h_{l+1}
  -> continue to next layer

如果 candidate 只在 attention 之后写回，则它不会影响当前 prefill 的
question logits。

### 2.5 causal 语义

同一层 causal attention 中，位置 i 只能影响后面的 j，不能让 j 反向影响
i。第一版显式保留两种模式：

- independent_snapshot：所有 selected token 读取同一份重算前 state；
- causal_sequential：先前 selected token 的 working KV 可以影响后续 token。

两者必须在 toy case 上分别验证，不能混为一个结果。

## 3. Linear state 定义

对于 feature map phi：

S(B) = sum(phi(k_j) outer v_j)
z(B) = sum(phi(k_j))

局部 block state 可以相加，但它仍然是 prefix-dependent summary。完整 token
所需的 state 是：

S(prefix_i) = S(previous blocks) + S(current block prefix_i)

因此实验报告区分：

- local_block_state；
- assembled_prefix_state；
- updated_prefix_state。

如果 selected token 的 K/V 改变，显式 state update 为：

S_new = S_old - phi(k_old) outer v_old + phi(k_new) outer v_new
z_new = z_old - phi(k_old) + phi(k_new)

有 RoPE 和 global position 时，不能把 local-position 聚合结果直接当成
assembled global state。第一版必须在实际 global position 下构造 state。

## 4. 实验分阶段

### Phase 0：代码与 reference correctness

实现一个慢但清晰的 reference operator，至少输出：

- cache position mapping；
- 每层 V_base、V_compute、V_work；
- attention 读取的实际 K/V；
- 每个 selected token 的 predecessor dependency；
- h_l、h_{l+1}；
- 下一层 candidate K/V。

随后实现 fast operator，并与 reference 对比：

- toy cache position mapping；
- alpha=0、alpha=1 端点；
- selected token 的 rank-one state update；
- independent_snapshot；
- causal_sequential；
- GQA、RoPE、causal mask；
- question path isolation。

在 reference 与 fast operator 数值一致前，不运行完整数据集。

### Phase 1：真实样本语义 sanity

固定 setup-v2、Qwen3-32B、MuSiQue-v2、rate=0.15，依次运行：

1. 1 条样本，保存完整 trace；
2. 10 条样本，验证输出和 KV 端点；
3. 50 条样本，比较质量和系统开销。

raw cache 与 preprocess cache 分开，不混合结论。

### Phase 2：Linear/MHA 融合矩阵

主要对照：

- cache-only；
- native dense MHA reprocess；
- pure Linear candidate；
- MHA base + Linear candidate；
- MHA base + sparse candidate 作为历史 MoBA 对照。

第一轮扫描：

alpha = {0, 0.25, 0.5, 0.75, 1}

先使用 alpha_K = alpha_V。只有观察到稳定信号后，再做 K-only、V-only 和
独立 alpha 扫描。

### Phase 3：完整 MuSiQue

只有以下条件同时满足后，才扩展到完整 200 条：

- reference/fast operator 通过 toy correctness；
- question path 与 baseline 一致；
- effective KV 已确认在 attention 前被读取；
- alpha=0 与 cache-only 端点一致；
- alpha=1 与 candidate replacement 端点一致；
- 50 条样本没有 cache overwrite 或 delayed writeback；
- 有明确的速度或 attention support 收益。

完整数据默认使用 GLM judge，并同时报告 formal EM/F1。

## 5. 必须记录的指标

质量：

- formal EM；
- formal F1；
- GLM judge；
- 与 cache-only 的逐题差异；
- 与 native dense MHA 的逐题差异。

数值：

- K/V relative L2；
- hidden-state relative L2；
- 每层 selected token delta；
- attention output delta；
- next-layer K/V delta；
- state reconstruction error；
- predecessor dependency coverage。

系统：

- document recompute 时间；
- question prefill 时间；
- 总 prefill 时间；
- attention 实际 token 数；
- 显存；
- cache 读写量；
- 当前 PyTorch prototype 与未来 fused kernel 的边界。

## 6. 当前推荐默认配置

在没有进一步讨论前，prototype 使用：

- selected document token only；
- question full MHA；
- KV 在 attention 前融合；
- alpha_K = alpha_V；
- causal_sequential 作为主语义；
- independent_snapshot 作为对照；
- raw/preprocess 分开；
- 先跑 toy、1、10、50，再跑 200；
- 不使用 dense attention score 做 Linear routing；
- 不把 pure Linear 结果当作最终方案。

## 7. 当前仍需讨论的决策

以下问题会影响实现方向，需在 Phase 0 结束前确认：

1. V_compute 是否严格定义为当前层 candidate hidden state 经过原始
   W_K/W_V projection 的结果，还是允许 Linear output 直接映射到 KV；
2. 主方法是否同时融合 K 和 V，还是优先固定 V-only 做最小 prototype；
3. 主实验采用 causal_sequential 还是 independent_snapshot；
4. Linear state 是否只用于 selected token 的 prefix，还是要维护完整
   document/global state；
5. 质量收益的最低门槛和速度收益的最低门槛；
6. 是否在所有 layer 重算，还是先选择后半层或少数 layer。

在这些决策未改变前，按本计划的默认配置推进 Phase 0 和 Phase 1。


## 8. Upstream reference implementation gate

在实际接入 Qwen3 前，先复刻并验证一个 FLA/fast-transformers 风格的
reference operator。当前 qjy 环境没有安装 fast_transformers、fla 或
flash_linear_attention，FusionRAG 也没有现成 Linear Attention 实现，因此
不能把第三方 fused kernel 直接当作正确性基线。

reference 必须明确支持：

- feature-mapped q/k，而不是默认把 raw q/k 当作 feature；
- float32 recurrent state；
- normalized numerator 和 running z state；
- initial KV state 与 initial z state；
- final state 输出；
- block 内 causal prefix，而不是把完整 block state 给所有 token；
- GQA/head layout 和 global RoPE 后的 K；
- independent snapshot 与 causal sequential 两种 selected-token 语义。

reference 与 fast operator 逐 token 对齐后，才允许修改
models/modeling_qwen3.py 或 ktransformers/util/utils.py 接入实际模型。
