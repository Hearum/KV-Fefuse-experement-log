# KV/Hidden Surrogate Update 研究计划

## 目标

目标不是继续证明 Delta-KV 存在，而是研究一个可替代完整重算的更新范式：

> 用缓存 KV、离线 token 评分和轻量 readout/update，近似恢复 full recompute 后的 document hidden/KV。

这个方向只有在计算量显著小于主模型 recompute 时才有意义。因此本计划把“近似质量”和“计算量”同时作为约束。

## 背景判断

当前 FusionRAG 的 online recompute 做法是：selector 决定 `k_need_index`，然后把这些 token 重新送入主模型 forward。即使只选 15% token，选中的 token 仍要经过完整 Transformer block，包括 attention、MLP、layernorm、K/V projection，并且高层依赖低层更新后的 hidden。

已有 hidden gap 实验显示：full prompt 下 document hidden state 和 raw/preprocess hidden state 差异很大，并集中在后层。因此如果只改 K/V 而不考虑 hidden drift，可能不足以恢复 full recompute 行为。

## 候选方案

### 方案 A：固定评分 readout，独立更新 token

对每层 `l` 和每个待更新 token `i`：

```text
r_i^l = sum_{j < i} a_{ij} * V_j^{l-1}
h_hat_i^l = g(h_old_i^{l-1}, r_i^l)
K_hat_i^l = Wk_l(h_hat_i^l)
V_hat_i^l = Wv_l(h_hat_i^l)
```

其中 `a_{ij}` 不通过主模型 QK 计算，而来自：

- offline token score 的归一化分布；
- chunk 内 top-k score 的稀疏分布；
- 距离衰减 + offline score；
- 已有 online/draft attention score 的缓存近似。

优点：所有 token 可以并行更新，不需要 token `i` 的更新影响 token `i+1`。

缺点：忽略 causal 递推，可能无法恢复真实 full recompute 的层间/ token 间传播。

### 方案 B：固定评分 readout，chunk 内串行更新

更新 token `i+1` 时，允许读取 token `i` 已更新后的 `V_hat_i^{l-1}` 或 `K_hat_i^l/V_hat_i^l`。

优点：更接近真实 causal prefill。

缺点：串行依赖会降低并行度。如果每层每 token 串行，速度可能接近甚至慢于 full recompute。除非只对少数后层、少数 token、稀疏前缀做更新，否则不应作为主路线。

### 方案 C：后层-only surrogate update

只在 hidden/KV gap 最大的后层做 surrogate，例如 Qwen3-32B 的 `56-63` 或 `59-63`。

```text
for l in selected_late_layers:
  r_i^l = sparse_readout(V_{<i}^{l-1})
  delta_h_i^l = Adapter_l([h_old_i^{l-1}, r_i^l, score_i])
  h_hat_i^l = h_old_i^{l-1} + delta_h_i^l
  K/V_hat_i^l = blend(old K/V_i^l, Wk/Wv_l(h_hat_i^l))
```

这是目前最值得优先验证的方向，因为已有 KV/hidden gap 都显示后层集中。它避免替代整条 Transformer 递归，只修补最重要的后层。

## 两个关键设计问题

### 1. token `i+1` 是否依赖 token `i` 的更新？

严格 Transformer prefill 中答案是依赖的：token `i+1` 在第 `l` 层 attention 时可以看见 token `i` 的第 `l-1` 层输出，因此如果 token `i` 的 hidden/KV 被更新，理论上会影响 token `i+1`。

但工程目标是省计算，所以需要实验比较三种近似：

| 方式 | 含义 | 计算特性 | 风险 |
|---|---|---|---|
| independent | 每个 token 只读旧 cached prefix，不读本轮更新 | 可并行，最快 | 忽略 token 间更新传播 |
| block-parallel | 按 chunk/block 迭代，block 内独立，block 间串行 | 折中 | 需要选择 block size |
| sequential | token-by-token 使用前面更新 | 最接近真实递推 | 速度风险最大 |

第一阶段建议优先做 `independent` 和 `block-parallel`，只把 `sequential` 作为 oracle-like 对照。真正可用的方法不应依赖全 token 串行。

### 2. K/V 是直接投影还是 blend？

候选：

```text
direct: K_hat = Wk(h_hat), V_hat = Wv(h_hat)
blend:  K_hat = beta * K_old + (1-beta) * Wk(h_hat)
        V_hat = beta * V_old + (1-beta) * Wv(h_hat)
delta:  K_hat = K_old + A_k(h_old, r, score)
        V_hat = V_old + A_v(h_old, r, score)
```

判断：

- `direct` 最干净，但如果 `h_hat` 误差大，会破坏原 cache。
- `blend` 更稳，能做 beta sweep，并和之前 beta 实验口径相连。
- `delta` 最像 Adapter/LoRA，但需要训练或拟合参数。

第一阶段不训练时，优先测试 `direct` 和 `blend`；如果 hidden/KV 几何指标明显改善，再考虑训练小 Adapter。

## 计算量约束

full recompute 对每个被选 token 仍跑主模型层：

```text
Cost_full ≈ selected_tokens * selected_layers * (attention + MLP + projections)
```

surrogate update 必须避免以下操作：

- 不重新计算完整 QK attention；
- 不跑 FFN/MLP；
- 不对所有层更新；
- 不对所有 token dense readout 全前缀。

可接受的第一版计算：

```text
Cost_surrogate ≈ selected_tokens * late_layers * (top_m readout over V + small adapter/projection)
```

其中 `top_m` 应该是 8/16/32/64 级别，而不是完整 `<i` 前缀长度。

## 第一阶段机制实验

### Probe 1：readout 是否能解释 full hidden delta

固定小样本，例如 MuSiQue-v2 5 examples。

对每个 selected token、每个后层：

1. 从已有 full hidden gap 脚本抓 `h_old` 和 `h_full`。
2. 构造 `r_i = sum a_{ij} V_j^{l-1}`。
3. 检查 `r_i` 与 `delta_h = h_full - h_old` 的关系：cosine、linear probe R2、ridge R2。

如果 `r_i` 对 `delta_h` 没有预测性，这条路线很难成立。

### Probe 2：old hidden + readout 能否预测后层 K/V delta

输入：`[h_old_i^{l-1}, r_i^l, score_i, position_features]`

目标：`Delta K_i^l`、`Delta V_i^l` 或 PCA coefficient。

先不训练大模型，只做 ridge / rank-limited linear probe。

### Probe 3：parallel vs block-parallel vs sequential

在小样本上构造三种更新顺序，比较 hidden/KV relative L2：

- independent：所有 token 读旧 cache。
- block-parallel：每 16/32 token 一组，组间更新。
- sequential：token-by-token 更新。

如果 independent 接近 sequential，就说明可以并行；如果差距很大，则需要 block 级折中。

### Probe 4：direct vs blend

对 `beta = 0, 0.25, 0.5, 0.75` 做几何指标比较：

- hidden relative L2 to full；
- K/V relative L2 to full；
- cosine to full；
- 后续再接端到端 accuracy。

## 推荐执行顺序

1. 不改主 pipeline，先写独立 probe 脚本，从已有 raw/preprocess/full hidden/KV 统计路径读取或生成小样本 tensor。
2. 只验证后层 `56-63`，避免全层计算。
3. 先用 offline score 的 top-m sparse readout，不算完整 attention。
4. 先做几何指标，不跑端到端。
5. 如果几何指标有效，再加一个 `FUSIONRAG_SURROGATE_UPDATE=1` 的实验接口接入 pipeline。

## 当前判断

这个方向可行性不是零，但必须满足两个条件：

1. offline/cheap score 构造的 `V` readout 对 full hidden/KV delta 有足够预测性；
2. 后层-only、top-m sparse、parallel/block-parallel 的计算量明显低于 full recompute。

如果必须使用完整 QK attention、逐 token 串行递推或完整 MLP，那就和原重算太接近，不值得继续。

