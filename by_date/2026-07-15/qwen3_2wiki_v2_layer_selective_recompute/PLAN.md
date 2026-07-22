# Qwen3-32B 2WikiQA-v2 Layer-Selective Recompute 探究计划

## 目标

验证：如果 online QK `rate=0.15` 仍选择同样 token，完整重算后只写回 Delta-KV gap 较大的层，端到端性能会如何。

注意：本计划是 layer-wise **writeback ablation**，不是 strict compute-saving recompute。当前实现仍会从低层到高层完整执行重算，再按层决定是否把重算 KV 写回 cache。

## 背景依据

来自 `musique-v2` KV gap 统计：

- Key gap 主要集中在 layer 45-52。
- Value gap 主要集中在 layer 59-63。
- Value gap 比 Key gap 更大，且后 5 层占全部 Value gap energy 约 70%。

虽然这个层分布来自 MuSiQue-v2，本实验先把它迁移到 2WikiQA-v2 做探究：如果这些层是真正影响性能的层，保留这些层的重算写回应接近正常 online QK；如果性能掉很多，说明 gap 大不等于对答案关键。

## 方法设计

固定：

- dataset：`2wikimqa-v2`
- model：`Qwen3-32B`
- selection：online QK / `reprocess_method=FusionRAG`
- rate：`0.15`
- preprocess：`true`
- cache：共享 setup-v2 cache

只改变完整重算后的写回层：

| 方法名 | Key 写回层 | Value 写回层 | 目的 |
|---|---|---|---|
| `all_layers` | all | all | 正常 online QK baseline |
| `v_late_59_63` | none | 59-63 | 只验证后层 Value 是否足够 |
| `v_late_56_63` | none | 56-63 | Value 后 8 层，略宽松 |
| `k_mid_45_52` | 45-52 | none | 只验证中后层 Key |
| `kv_gap_core` | 45-52 | 59-63 | Key/Value gap 最大层组合 |
| `kv_gap_wide` | 45-52 | 56-63 | 更宽松组合 |

环境变量接口：

- `FUSIONRAG_REPROCESS_KEEP_KEY_LAYERS`
- `FUSIONRAG_REPROCESS_KEEP_VALUE_LAYERS`

`none` 表示该类 KV 全部恢复为 cache，不保留重算写回；`all` 表示保留全部层。

## 实验顺序

1. 每个组合先跑 2 examples smoke。
2. smoke 正常后跑 2WikiQA-v2 全量 200 examples。
3. 先看 EM/F1，本轮如有时间再接 GLM judge。

## 预期解读

- 如果 `v_late_59_63` 接近 `all_layers`，说明后层 Value 是主要有效更新。
- 如果 `k_mid_45_52` 很弱，说明 Key gap 虽有结构但不是主要性能来源。
- 如果 `kv_gap_core` 接近 `all_layers`，说明可用少量层写回近似 full online recompute。
- 如果所有 selective 都明显下降，说明“gap 大的层”不等价于“答案性能关键层”，需要按性能敏感度重新找层。

## 严格省算力实验应另开计划

如果目标是减少 Transformer 重算成本，不能只在写回阶段恢复旧 KV。需要另开实验，至少区分：

- `strict-skip`：真正跳过某些层的重算，验证 hidden state 断裂后还能否推理。
- `cached-hidden adapter`：用离线缓存的每层 hidden 或 KV 统计，直接预测目标层的 `Delta K/V`。
- `prefix-conditioned adapter`：用当前前序 KV/hidden 的轻量统计预测 `Delta K/V`，避免逐层递归重算。

这些实验才可以讨论 TTFT/算力节省；本实验只讨论写回层的重要性。
