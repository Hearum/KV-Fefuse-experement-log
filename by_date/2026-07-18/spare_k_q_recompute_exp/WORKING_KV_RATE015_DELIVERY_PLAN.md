# Working-KV rate=0.15 可交付验证计划

## 目标

目标不是接近 full attention rate=1，而是在固定 FusionRAG selector 和 rate=0.15 的条件下，用更少的重算计算量复现原生 FusionRAG rate=0.15 的性能。

定义唯一主基线：

- 相同 dataset、相同 document cache、相同 selector；
- 相同 selected token 集合；
- 原生 FusionRAG rate=0.15 的完整 selected-token 重算；
- 自动计算 EM、F1、GLM accuracy，并记录 prefill/recompute latency。

候选方法只有在满足以下条件后才进入完整数据集：

accuracy(candidate) ~= accuracy(native rate=0.15) 且 compute(candidate) < compute(native rate=0.15)。

## 阶段 0：固定数据与 selector

1. 固定 MuSiQue-v2 前 10 条作为调试集，再扩展到 50 条。
2. 所有方法复用同一份 Qwen3-32B/MuSiQue-v2 preprocess KV。
3. 保存每条样本的 selected absolute token positions，确保候选方法不能重新选择 token。
4. 固定 tokenizer、RoPE 设置、generation 参数和 GLM judge prompt。

## 阶段 1：建立原生 oracle

运行并保存：

1. 原生 online_qk, rate=0.15；
2. 原生 online_qk 的 strict doc/query 两阶段控制路径，但不传 Working-KV alpha；
3. 两者逐题比较生成文本，必须 100% 一致；否则先修复 pipeline。

逐层记录：

- selected token 的 K/V；
- query prefill 最后一 token 的 logits；
- first-token argmax；
- recompute time、query prefill time、总 prefill time。

## 阶段 2：验证 Working-KV 写回语义

不要把 alpha 传入 Transformer attention 主路径。先完成原生 selected-token attention，再在 cache 写回后单独执行：

K_out = K_cache + alpha_K * (K_candidate - K_cache)
V_out = V_cache + alpha_V * (V_candidate - V_cache)

先测 alpha_K=alpha_V=1：

- selected token K/V 必须与原生 oracle 逐层一致；
- query logits 必须一致到数值容差；
- 生成结果必须逐题一致。

再测 alpha=0：

- 必须等价于 rate=0 的 preprocess baseline；
- 不能使用真实 delta 作为 oracle coefficient。

如果 alpha=1 仍不一致，禁止继续测试 sparse 或 predictor。

## 阶段 3：减少 attention 计算

在 oracle 等价后，按风险从低到高测试：

1. selected-token attention 的 block sparse；
2. 只保留高贡献前缀 block；
3. layer skipping；
4. cached hidden state correction；
5. 轻量 predictor 预测 attention output 或 KV delta。

每种方法同时报告：

- GLM accuracy、EM、F1；
- selected-token K/V relative L2；
- hidden state relative L2；
- query logits cosine similarity；
- recompute FLOPs；
- 实测 latency；
- 显存和额外 cache。

## 阶段 4：50 条确认与完整 MuSiQue

50 条实验中，候选方法必须满足：

- GLM accuracy 与原生 rate=0.15 差距不超过 1 个百分点；
- EM/F1 没有明显下降；
- recompute FLOPs 至少下降 30%；
- 没有出现随 layer 深度累积的 hidden/logit 发散。

满足后才运行完整 200 条 MuSiQue-v2，并与以下方法统一比较：

- raw rate=0；
- preprocess rate=0；
- native online_qk rate=0.15；
- native online DraftModel rate=0.15；
- candidate method；
- full rate=1 作为上界参考。

## 明确禁止的比较

- 不把 full rate=1 作为 candidate 的唯一目标；
- 不把 Working-KV alpha=1 直接当作 native rate=0.15；
- 不使用真实 Delta 求 candidate coefficient；
- 不改变 selector 后再比较计算量；
- 不把 block sparse 的性能损失解释成 predictor 不可行；
- 不在 oracle 等价性未通过时扩大到完整数据集。

## 交付标准

最终报告必须给出：

1. 原生 rate=0.15 与候选方法逐题一致性；
2. 逐层 KV/hidden/logit 误差；
3. 计算量与 latency 降幅；
4. 完整 MuSiQue EM/F1/GLM 表；
5. 明确结论：
   - 可交付：性能等价且计算下降；
   - 仅可研究：有损但有速度收益；
   - 不可行：oracle 等价后仍无法在合理计算预算下保持性能。

每轮结果追加到 EXPERIMENT_LOG.md，记录启动命令、共享 cache 路径、代码 commit 和异常。
