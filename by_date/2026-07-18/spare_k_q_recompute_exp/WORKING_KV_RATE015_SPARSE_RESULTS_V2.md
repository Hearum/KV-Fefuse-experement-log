# Working-KV rate=0.15 稀疏重算结果 v2

## 目标

固定 FusionRAG selector 和 rate=0.15，寻找比 native online_qk selected-token 重算更低计算、同时保持 native 性能的稀疏重算方案。

## 已验证 oracle

native online_qk 与修正版 dense Working-KV alpha=1 使用相同 cache、相同 selector 和相同 selected token 集合。

50 条实验结果：

- 两者结果行：52；
- 逐题答案一致：52/52；
- native GLM：20/52；
- fixed Working-KV GLM：20/52；
- 修正 commit：1ba0b7b。

这证明 alpha=1 的修正版已经恢复 native rate=0.15 语义。

## Sparse top-k 8

配置：method=sparse_working_kv_preprocess，rate=0.15，alpha_K=1，alpha_V=1，block_size=64，topk_blocks=8，MuSiQue-v2=50 examples。

结果：

- 完整结果：50 条；
- GLM：16/50 = 32.0%；
- native 同批次：20/50 = 40.0%；
- native 与 sparse 问题集合一致；
- 逐题生成答案不一致：22/50。

结论：top-k 8 不能交付。它虽然减少了 block attention 支持集，但已经损失 native rate=0.15 的性能。

结果路径：MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_kv_search_50/

## Sparse top-k 32

原始 top-k 32 在 query chunk=1024 时 OOM。原因是 gather 中构造了未使用的 gather_idx，并且一次处理过大的 query chunk。

已完成代码修正：

1. 删除未使用的 gather_idx；
2. 使用环境变量 FUSIONRAG_SPARSE_QUERY_CHUNK=64；
3. 修正 commit：85ad6f0。

修正后 top-k 32 不再立即 OOM，但单条样本运行很慢，当前 10 条实验只完成 7 条，尚未形成正式 accuracy 结果。因此不能把它作为已验证可行方案。

## 当前可交付判断

当前唯一已经验证且稳定的方案是 native selected-query attention，它本身就是 FusionRAG rate=0.15 的 oracle，但还没有证明比 native 重算更省计算。

当前 block sparse top-k 8 已被完整 50 条实验否决为性能不可交付方案。top-k 32 需要完成后才能判断是否能恢复性能；即使恢复，也必须继续测实际计算量，否则只能算性能候选，不能算加速方案。

## 下一步

1. 完成 top-k 32 chunk64 的 10 条小样本结果；
2. 若 top-k 32 仍低于 native，测试按 query/head 分组的自适应 block 保留，而不是继续盲目增加固定 top-k；
3. 若 top-k 32 接近 native，再扩展 50 条并记录 FLOPs；
4. 只有 accuracy 接近 native 且 attention 计算明显减少，才进入完整 MuSiQue。

## 三视角审核

### 模型语义审核

PASS。报告使用 native rate=0.15 作为唯一 oracle，并区分了 alpha=1 语义修复和 sparse block 近似，不把 full rate=1 当作主目标。

### 系统实现审核

PASS。报告记录了两个 OOM 根因、query chunk 修正、代码 commit 和结果路径；没有把未完成的 top-k 32 结果写成正式 accuracy。

### 实验科学审核

PASS。top-k 8 使用完整 50 条结果；native/sparse 的问题集合已对齐；逐题 mismatch 和 GLM 同时报告；对 top-k 32 明确标记为 incomplete。

本版三方审核均 PASS，无需订正。
