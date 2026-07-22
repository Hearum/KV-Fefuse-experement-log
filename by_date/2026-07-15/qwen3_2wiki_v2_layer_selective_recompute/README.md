# Qwen3-32B 2WikiQA-v2 Layer-Selective Recompute

本实验探究：online QK `rate=0.15` 选中的 token 完整重算后，如果只写回 KV gap 较大的层，2WikiQA-v2 端到端性能如何。

**重要澄清**：本实验不是严格的计算节省实验。当前实现仍然按原始 Transformer 逻辑从低层到高层完整重算选中 token，然后在写回 cache 时恢复不保留层的旧 KV。因此它只能回答“哪些层的重算 KV 写回对最终效果重要”，不能证明“只计算这些层可以节省重算成本”。如果要验证真正的省算力方案，需要单独实现 strict layer compute skipping 或 cached-hidden/adapter 近似路径。

详细计划见 `PLAN.md`，运行记录见 `EXPERIMENT_LOG.md`。


## 2026-07-15 全量结果

实验设置：`2wikimqa-v2`，Qwen3-32B，online QK / `FusionRAG`，`rate=0.15`，`preprocess=true`。所有组合均跑满 200 examples。GLM judge 已通过统一 summarizer 拼入结果表。

| condition | Key 写回层 | Value 写回层 | EM | F1 | GLM Acc |
|---|---|---|---:|---:|---:|
| `all_layers` | all | all | 42.00 | 53.35 | 105/200 = 52.50 |
| `v_late_59_63` | none | 59-63 | 40.00 | 51.81 | 100/200 = 50.00 |
| `v_late_56_63` | none | 56-63 | 39.00 | 51.46 | 102/200 = 51.00 |
| `k_mid_45_52` | 45-52 | none | 39.00 | 51.80 | 100/200 = 50.00 |
| `kv_gap_core` | 45-52 | 59-63 | 39.00 | 52.23 | 102/200 = 51.00 |
| `kv_gap_wide` | 45-52 | 56-63 | 39.00 | 51.82 | 103/200 = 51.50 |

### 观察

1. 只保留 gap 大的层不会完全崩掉，但仍低于正常全层写回。`all_layers` 是 52.50 GLM，最好的 selective 是 `kv_gap_wide` 的 51.50，低 1 个点；EM 上 selective 普遍低 2-3 个点。
2. 只写回 Value 后层比只写回 Key 中后层没有明显优势：`v_late_59_63` 和 `k_mid_45_52` 都是 50.00 GLM。
3. Key+Value 组合略好于单独 Key/Value，尤其 `kv_gap_wide` 达到 51.50 GLM，说明 Key 中后层与 Value 后层有互补，但 gap 最大层并不能完全替代全层写回。
4. `v_late_56_63` 不比 `v_late_59_63` 稳定更好，说明单纯扩大后层 Value 范围不一定提升 EM/F1；但 GLM 从 50.00 到 51.00 有小幅改善。

### 当前结论

“KV gap 大的层”确实包含一部分有效更新信息，但不是完整性能关键集合。只写回这些层可以保留大部分性能，但无法完全达到全层写回。下一步如果继续做 layer-selective，应从两个方向探索：

- 按 2WikiQA-v2 自身的 KV gap 重新统计 layer 分布，不直接迁移 MuSiQue-v2 的 45-52/59-63。
- 做 layer ablation / leave-one-band-out，寻找性能敏感层，而不是只按 L2 gap energy 选择层。

### 对严格省算力路线的影响

这组结果只能作为“目标层选择”的先验：如果未来要把 document KV 重算改成轻量 Adapter，不能复用当前实现来宣称省算力。更严格的下一步应拆成两类：

1. **写回必要性**：继续用当前 full recompute + selective writeback 判断哪些层的 KV 更新最影响答案，这是分析性能敏感层。
2. **计算可替代性**：实现不依赖上一层重算 hidden state 的近似路径，例如用 cached hidden、前缀 KV 统计或小 Adapter 直接预测某层 `Delta K/V`，然后只生成这些层的 KV 更新。只有第二类实验才对应真实计算节省。

## 输出文件

- 统一结果：`results/full_summary_with_glm.csv`
- EM/F1-only 结果：`results/full_summary.csv`
- GLM 明细：`rejudge_glm_clean_full_20260715/rejudged_rows.csv`
- GLM 汇总：`rejudge_glm_clean_full_20260715/rejudged_summary.csv`
