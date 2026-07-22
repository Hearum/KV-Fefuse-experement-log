# 按真实 Qwen Block 修正 Surrogate Update 设计

## 为什么要修正

之前的简化表述是：

```text
r_i^l = sum_j a_ij V_j^{l-1}
r_i^l -> hidden_i -> K/V_i
```

这个表达只说明了直觉，但不等于 Qwen3/Qwen2 的真实计算。真实模型里，Value 加权和只是 attention 分支的中间量，后面还有 `o_proj`、residual、post-attention RMSNorm、MLP、第二个 residual。K/V 也不是从 attention 后 hidden 直接来，而是从每层 attention 前的 normalized hidden 投影得到。

因此 surrogate update 必须贴合真实 block，否则做出来的近似即使数值上能动，也很难解释为“替代重算”。

## 真实 Qwen2/Qwen3 Decoder Layer 计算图

当前代码位置：

```text
ktransformers/models/modeling_qwen2.py
```

每层 `l` 的计算顺序如下。

### 1. layer input hidden

```text
x_i^l = hidden_states entering layer l
residual_1 = x_i^l
```

我们 hidden gap 脚本里保存的 `hidden_states[:-1]` 对应的就是这个 `x_i^l`，也就是每层 block 的输入 hidden。

### 2. input RMSNorm

```text
u_i^l = RMSNorm_in^l(x_i^l)
```

Q/K/V 都来自 `u_i^l`，不是直接来自 `x_i^l`。

### 3. Q/K/V projection + RoPE

```text
q_i^l = Wq^l u_i^l
k_i^l = Wk^l u_i^l
v_i^l = Wv^l u_i^l

q_i^l, k_i^l = RoPE(q_i^l, k_i^l, position_i)
```

注意：Key 受 RoPE 位置影响；Value 不受 RoPE 影响。之前 KV gap 里 raw Key 的大 gap 很多来自位置变化，所以 Key 分析要区分 RoPE-aligned 和实际写入 cache 的 RoPE key。

### 4. attention readout

```text
p_ij^l = softmax((q_i^l · k_j^l) / sqrt(d) + mask_ij)
z_i^l = sum_{j <= i} p_ij^l v_j^l
```

这里 `z_i^l` 才是 Value 的加权平均。它仍在 attention head space，不是 hidden space。

### 5. output projection + first residual

```text
a_i^l = Wo^l concat_heads(z_i^l)
y_i^l = residual_1 + a_i^l
```

所以如果我们只构造 `sum V`，还必须经过 `o_proj`，并加回原 residual，才对应 attention 子层输出。

### 6. post-attention RMSNorm + MLP + second residual

```text
m_i^l = MLP^l(RMSNorm_post^l(y_i^l))
x_i^{l+1} = y_i^l + m_i^l
```

下一层的 K/V 来自 `x_i^{l+1}` 经下一层 input RMSNorm 和 K/V projection。因此若 surrogate 不近似 MLP 分支，只更新 attention 分支，误差会进入下一层并累积。

## 对 surrogate update 的正确拆分

真实替代路径不应说“V 加权平均更新 hidden”，而应拆成三个候选近似层级。

### Level 1：只近似 attention readout，不近似 MLP

```text
z_hat_i^l = SparseReadout(V_{TopM(<i)}^l, score)
a_hat_i^l = Wo^l concat_heads(z_hat_i^l)
y_hat_i^l = x_old_i^l + a_hat_i^l
```

然后有两种选择：

```text
# 不跑 MLP，直接把 y_hat 当作下一层输入近似
x_hat_i^{l+1} = y_hat_i^l

# 或者只在少数层跑 MLP
x_hat_i^{l+1} = y_hat_i^l + MLP^l(RMSNorm_post^l(y_hat_i^l))
```

第一种更省，但偏离真实 block；第二种更准，但 MLP 是主要计算成本之一。

### Level 2：只预测 correction，不显式模拟 block

```text
delta_x_hat_i^l = A_l(x_old_i^l, z_hat_i^l, score_i, pos_i)
x_hat_i^l = x_old_i^l + delta_x_hat_i^l
```

然后：

```text
u_hat_i^l = RMSNorm_in^l(x_hat_i^l)
K_hat_i^l = RoPE(Wk^l u_hat_i^l, position_i)
V_hat_i^l = Wv^l u_hat_i^l
```

这个更像 Adapter：不强行复刻 attention+MLP，而是学习或拟合 full hidden drift。它可能更现实，因为 hidden gap 很大且后层集中。

### Level 3：直接预测 K/V correction

```text
DeltaK_hat_i^l, DeltaV_hat_i^l = A_l(x_old_i^l, z_hat_i^l, score_i, pos_i)
K_hat_i^l = K_old_i^l + DeltaK_hat_i^l
V_hat_i^l = V_old_i^l + DeltaV_hat_i^l
```

这是最便宜的路径，但解释性最弱，也最容易破坏层间一致性。可以作为下界/工程 baseline。

## K/V 更新的 direct 与 blend

按真实操作，`direct` 应该是：

```text
u_hat_i^l = RMSNorm_in^l(x_hat_i^l)
K_proj_i^l = RoPE(Wk^l u_hat_i^l, position_i)
V_proj_i^l = Wv^l u_hat_i^l

K_hat_i^l = K_proj_i^l
V_hat_i^l = V_proj_i^l
```

`blend` 应该是：

```text
K_hat_i^l = beta_k * K_old_i^l + (1 - beta_k) * K_proj_i^l
V_hat_i^l = beta_v * V_old_i^l + (1 - beta_v) * V_proj_i^l
```

建议 Key 和 Value 分开设 beta，因为已有实验显示 Key/Value gap 幅度和来源不同：Key 受 RoPE/位置影响更强，Value 更像真正 context update。

## token i+1 是否依赖 token i 的更新

真实 prefill 中依赖，原因是 token `i+1` 的 attention 可以看到 token `i` 的上一层输出和 K/V。更准确地说：

- 第 `l` 层更新 token `i+1` 时，会读 `<i+1` 的 `K^l/V^l`。
- 如果 token `i` 在第 `l` 层的 K/V 已更新，token `i+1` 的 attention readout 理论上应使用更新后的 `K_i^l/V_i^l`。
- 下一层 `l+1` 还依赖 token `i` 的 `x_i^{l+1}`。

但工程上如果完全按 token 顺序串行，就会很慢。因此需要三个版本对比：

| 版本 | 真实性 | 并行性 | 用途 |
|---|---|---|---|
| independent | 低 | 高 | 最可能加速的版本 |
| block-parallel | 中 | 中 | 折中，block size 16/32/64 |
| sequential | 高 | 低 | oracle-like 对照，不一定实用 |

如果 independent 与 sequential 差距很小，说明可以并行；如果差距很大，则考虑 block-parallel。

## 计算量红线

不能让 surrogate 退化成 full recompute。

应避免：

1. 完整 QK attention over full prefix。
2. 全层 MLP。
3. 全 token 串行更新。
4. 对所有层都做 dense top-prefix readout。

第一版建议只做：

```text
late_layers = 56-63 或 59-63
top_m = 8/16/32
update_tokens = selector/offline top 15%
readout = score-only sparse Value readout
adapter = linear/ridge 或极小 MLP
```

这样计算量才可能低于 full recompute。

## 对首个 Probe 的修正

首个 probe 不应只问 `sum V` 能不能预测 hidden，而应问三件事：

1. `z_hat = sparse weighted V` 经 `o_proj` 后，是否接近真实 attention 分支输出 `a_full` 或 hidden delta 的主方向。
2. `[x_old, o_proj(z_hat), score, position]` 是否能用轻量线性/低秩模型预测 `Delta x`。
3. 用 `x_hat` 经真实 `RMSNorm + Wk/Wv + RoPE` 得到的 `K_hat/V_hat` 是否比 old cache 更接近 full K/V。

评价仍用：relative L2、cosine、PCA coefficient R2、explained variance。

## 当前推荐路线

优先做 Level 2：

```text
score/top-m sparse Value readout -> o_proj -> tiny adapter predicts late-layer hidden correction -> real Wk/Wv projection -> blend writeback
```

理由：

- 它尊重真实模型里 `V readout -> o_proj -> residual` 的结构。
- 不需要完整 QK attention。
- 可以只做后层。
- 可以用真实 `Wk/Wv/RMSNorm/RoPE` 保持 K/V 写入格式正确。
- 比直接预测完整 K/V 更有结构，比模拟完整 block 更省计算。
