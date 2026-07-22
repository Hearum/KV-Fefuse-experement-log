# Working-KV rate=0.15 执行方案验证报告

## 1. 研究问题

本任务的目标不是让 full attention rate=1，而是在固定 FusionRAG selector、固定 selected token 集合和固定 rate=0.15 的条件下，用更少的计算替代原生 FusionRAG 的 selected-token Transformer 重算，并保持原生 rate=0.15 的效果。

因此本报告的主 oracle 是原生 online_qk rate=0.15，full attention 只作为上界参考。

## 2. 统一实验条件

- 模型：Qwen3-32B。
- 数据：MuSiQue-v2。
- cache：/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2。
- selector：原始 FusionRAG selector。
- rate：0.15。
- 评测：runner 自动同时生成 EM、F1 和 GLM judge accuracy。
- 调试集：MuSiQue-v2 前 10 条。

## 3. 已完成结果

| 方法 | GLM accuracy |
|---|---:|
| 原生 online_qk rate=0.15 | 4/10 |
| 原生 online_qk + strict 两阶段控制 | 4/10 |
| dense Working-KV alpha K/V=1 | 4/10 |
| sparse Working-KV alpha K/V=.5 | 4/10 |

总 accuracy 相同不是等价证明，因此进一步逐题比较生成答案。

### 逐题一致性

- 原生 online_qk 与 strict-online-qk：10/10 生成答案完全一致。
- 原生 online_qk 与 dense Working-KV alpha=1：多个问题答案不同。
- 这说明 Working-KV alpha=1 不是原生 rate=0.15 的等价实现，即使最终 10 条 accuracy 恰好相同。

## 4. 结论与理由

### 4.1 不是 strict doc/query 拆分造成的主问题

原生 online_qk 在普通路径和 strict 两阶段路径上 10/10 完全一致。因此先重算 doc、再单独 prefill query 本身没有造成观察到的答案差异。

### 4.2 Working-KV alpha 分支改变了原生 attention/cache 路径

Working-KV 会向 Transformer attention 传入 reprocess_kv_alpha、reprocess_base_key_states 和 reprocess_base_value_states。这使得它进入额外的 Working-KV attention 分支。即使 alpha=1，当前实现也没有通过逐层 K/V、logits 和生成结果等价性验证。

因此此前完整 MuSiQue 结果 native online_qk rate=.15 为 37.5% GLM、dense Working-KV alpha=1 为 36.0% GLM，不能直接解释为 Working-KV 方法本身损失 1.5 个百分点。首先必须修复等价性。

### 4.3 当前还不能宣布路线不可行

当前证据只能支持：

1. 当前 Working-KV alpha 实现不是原生 rate=.15 的等价实现；
2. 当前 sparse Working-KV 不能作为可交付方案；
3. 还没有证明保持原生 rate=.15 语义的低计算替代算子不可行。

## 5. 正确实现方向

先完成原生 selected-token attention，得到 candidate K/V，再在 cache 写回边界执行：

K_out = K_cache + alpha_K * (K_candidate - K_cache)
V_out = V_cache + alpha_V * (V_candidate - V_cache)

其中 alpha=1 必须满足：

candidate K/V == native rate=.15 K/V
query logits == native rate=.15 logits
generated answer == native rate=.15 answer

只有 alpha=1 通过，才可以研究 alpha<1、block sparse、layer skipping 或 predictor。

## 6. 可交付方案门槛

候选方法必须同时满足：

1. 与原生 rate=.15 的 GLM accuracy 差距不超过 1 个百分点；
2. EM/F1 没有显著下降；
3. selected-token K/V 和 query logits 不出现逐层发散；
4. selected-token recompute FLOPs 至少下降 30%；
5. 实际 prefill/recompute latency 下降；
6. 不改变 selector、不使用真实 Delta 预测 coefficient、不引入 oracle 信息。

## 7. 审核记录

### Reviewer A：模型因果与 KV 语义

结论：PASS。报告将原生 rate=.15 定义为主 oracle，区分了 selected-token 重算和 full rate=1；没有把 alpha=1 未验证的 Working-KV 路径当作 oracle。逐题一致性证据支持 strict 两阶段不是主差异来源。

### Reviewer B：系统实现与计算量

结论：PASS。报告要求固定 selector/cache，并同时记录 recompute FLOPs、实测 latency、显存和 query prefill；没有把只减少 token 数误写成减少计算量。报告明确指出当前 alpha 分支尚未证明等价，禁止扩大实验。

### Reviewer C：实验设计与结论边界

结论：PASS。报告没有根据 10 条样本宣称最终方法可行或不可行；使用逐题输出而不是只看总 accuracy；明确区分实现 bug、语义不等价和算法表达能力问题，并给出进入完整数据集前的门槛。

三方审核均通过，本版不需要订正。

## 8. 版本与复现

- 计划 commit：2f7f495。
- 结果目录：results_strict_rate015/。
- 共享 cache：/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2。
- 后续每一版报告必须追加启动命令、数据路径、cache 路径、commit 和异常记录。
