# Cache-Anchored Sparse Recompute 实验计划

## 研究问题

FusionRAG 的 native online rate=0.15 会对 selector 选中的 document token 做 dense causal recompute。对 selected token，dense recompute 产生的 full K/V 会完全覆盖原始 raw/preprocess cache。

本任务不改变 selector 和 rate。目标是验证：

> 是否可以只对 selected token 做 sparse attention recompute，用较少的前缀 attention 计算得到 sparse candidate K/V，再以 scaled correction 的形式加回原始 cache，并达到 native dense rate=0.15 的效果。

## 四种 KV 的含义

对 selected token i 的第 l 层：

1. base KV：online 已加载的 raw 或 preprocess cache，整个实验中 immutable。
2. dense full KV：native dense causal recompute 产生的目标 KV；native rate=0.15 会直接覆盖 selected token 的 cache。
3. sparse candidate KV：只读取 sparse attention support 后，由 sparse hidden state 经过原始 W_K/W_V 投影得到的候选 KV。
4. working KV：真正写入当前层 working cache 并供后续 attention 使用的 KV。

working KV 必须是：

    K_work = K_base + alpha_K * (K_sparse - K_base)
    V_work = V_base + alpha_V * (V_sparse - V_base)

alpha=0 表示不施加 sparse correction；alpha=1 表示完全采用 sparse candidate，作为 sparse-replace 对照。主要方法是 0<alpha<1，而不是直接覆盖 base KV。

## Attention context

native dense recompute 的 selected query 读取完整 causal prefix。候选方法只改变这个 attention support：

- selector 仍固定 FusionRAG rate=0.15；
- selected token 的 layer-parallel 更新顺序不变；
- 同层所有 selected token 先生成 candidate K/V、融合并 scatter，再执行 attention；
- token j 只能读取已经融合的 working KV；
- sparse support 使用 block representative + top-k blocks + current block；
- 不使用 dense full attention score 作为 online routing 输入。

因此 rate、top-k 和 alpha 的含义分别是：

- rate：重算多少个 query token；
- top-k：每个 selected query 读取多少个 KV blocks，决定 attention 计算量；
- alpha：sparse candidate correction 写回 base cache 的比例。

## 对照目标

主要 oracle 是同一批样本上的 native dense online rate=0.15，而不是 rate=1。rate=1 仅作为 full upper bound。

必须同时记录：

- rate=0 base cache；
- native dense rate=0.15；
- dense attention + alpha correction；
- sparse top-k + alpha correction；
- EM/F1/GLM；
- 逐题答案差异；
- selected-token KV/hidden gap；
- attention support token 数和 wall-clock。

## 实验阶段

### 阶段 A：10 条语义 sanity

固定 preprocess KV、MuSiQue-v2、rate=0.15。先比较 alpha={0.25,0.5,0.75}，分别测试：

- dense attention；
- sparse top-k=8；
- sparse top-k=32。

端点 alpha=0 和 alpha=1 仅作为已有对照，不把端点结果当作主要方法。

### 阶段 B：冻结配置

在阶段 A 中选择同时满足以下条件的配置：

1. GLM/EM/F1 接近 native dense rate=0.15；
2. 逐题答案差异较小；
3. sparse support 明显小于 dense causal prefix。

如果所有 sparse top-k 都明显低于 native，则结论是当前 support/alpha 形式不可行，不继续盲扫 alpha。

### 阶段 C：至少 50 条独立验证

冻结 alpha 和 top-k 后跑独立 50 条，重新与 native dense rate=0.15 对照。只有阶段 C 同时通过质量和计算量门槛，才称为可用方案。

## 禁止的混淆

- 不把 sparse candidate 直接覆盖 base KV；
- 不把 alpha=1 当作 cache-anchored 主方法；
- 不改变 selector 或 rate 来掩盖 sparse support 损失；
- 不使用真实 dense attention score 作为 online routing；
- 不把减少 FLOPs 的理论值当作端到端加速；
- 不把 full rate=1 当作 rate=0.15 的主要 baseline。

## 当前启动版本

实验代码沿用 branch exp/sparse-kv-working-blend-20260719，启动脚本为 setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py。共享 preprocess cache 为 /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2。正式启动后将把每组命令、commit、结果路径和三视角审核追加到 daily-note.md。

