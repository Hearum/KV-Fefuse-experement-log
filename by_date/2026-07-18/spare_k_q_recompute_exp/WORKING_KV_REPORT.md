# Cache-Anchored Sparse Reprocessing 最终研究报告

## 结论摘要

本实验回答的是：FusionRAG 对 selected document token 做 KV 更新时，能否用稀疏注意力产生 correction，并将它与 raw/preprocess cache 融合，从而替代 dense selected-token recompute。

最终结论分三层：

1. **机制实现正确。** 当前实现是 layer-parallel Working-KV，而不是错误的 Sparse KV replacement。每层先对全部 selected tokens 生成 candidate K/V，再与不可变 selected-position base snapshot 融合、scatter，随后本层 routing 和 attention 读取 working K/V。query prefill 保持 dense，且不再次 blend document KV。
2. **Dense cache-anchored blend 只有正向点估计，尚无可靠改善证据。** MuSiQue-v2 独立 test 的 Dense raw `alpha=0.75` 相比 raw `alpha=0`，EM/F1/GLM 分别变化 `+4.67/+5.23/+2.67` 个百分点，但三项 paired 置信区间都覆盖 0。即使后续证实质量收益，Dense blend 仍执行完整 causal-prefix attention，不能解决重算开销。
3. **本轮实际冻结的 Top-K=8 Sparse candidates 不可用。** 独立 test 的 Sparse raw `alpha=0.75` 相比 raw `alpha=0`，EM/F1/GLM 分别变化 `-0.67/-0.72/-3.33` 个百分点；补做的 Sparse preprocess `alpha=0.5` 在绝对值上也低于 full rate=1。5 个机制样本中，Top-K=8 只覆盖约 7.42% selected-predecessor pairs 和 29.74% dense attention mass；即使后验选择最佳 global scalar，Delta-K/V 仍保留约 83%--88% relative L2 error。

因此本阶段没有得到可部署的稀疏加速方案。得到的可复用成果是正确的 Working-KV 实验语义、可靠对照、失败边界和下一步路由设计约束。不能把“理论 attention support 降低”写成“端到端已加速”。

## 1. 方法与因果语义

对第 `l` 层 selected document token：

```text
base_KV(l)      = 从 raw/preprocess cache 抽取的 selected-position 只读快照
candidate_K(l)  = RoPE(k_norm(W_K(RMSNorm(h(l-1)))), absolute_position)
candidate_V(l)  = W_V(RMSNorm(h(l-1)))
working_KV(l)   = (1 - alpha) * base_KV(l) + alpha * candidate_KV(l)
```

本层严格执行：

```text
所有 selected h(l-1)
  -> 并行生成全部 q/k/v candidate
  -> 融合全部 selected candidate 与 base snapshot
  -> 一次性 scatter 到第 l 层 working cache
  -> 用 working K 做 block routing
  -> 从 working K/V gather
  -> selected-query attention + O projection + residual + MLP
  -> 全部 selected h(l)
```

若 `i < j` 且 token j 路由到 token i 所在 block，j 在第 l 层读取的是 i 的 `working_k_i(l), working_v_i(l)`。这不是逐 token for-loop 串行更新，也不是先算完 attention 再写 KV。

`alpha=0` 保留 base KV；`alpha=1` 等于 candidate replacement；中间值表示 cache-anchored correction。candidate 的深层 hidden 依赖此前层的 alpha，因此不能拿一次 `alpha=1` trajectory 的深层 candidate 离线线性插值来伪造其他 alpha。

## 2. 实现与正确性证据

### 2.1 生产路径

- 模型 runtime：`models/modeling_qwen3.py`
- FusionRAG pipeline：`ktransformers/util/utils.py`
- MoBA-style router：`ktransformers/operators/sparse_attention.py`
- 正式 runner：`MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py`

仓库还存在 `ktransformers/models/modeling_qwen3.py`，但 setup-v2 正式 pipeline 实际导入前者。早期曾在错误副本上测试，已撤销并统一到真实 runtime。

### 2.2 自动测试覆盖

`scripts/test_working_kv_semantics.py` 覆盖：

- alpha=0/1 endpoints；
- immutable selected-position base snapshots；
- fusion-before-attention；
- 同层全部 selected positions 在 attention 前可见；
- selected predecessor 实际读取 working V；
- current/self block 强制保留；
- routing 随 working K 改变；
- query prefill 不重新 blend；
- `past_tokens` 正常和异常路径恢复，且不替换 list 对象；
- 非 SDPA、`output_attentions=True`、selected tokens 超过 32768 时 fail-fast。

自动测试是 CPU 小模型张量测试。32B 单样本额外验证 alpha=0 对齐 rate=0 输出、alpha=1 对齐 selected replacement 输出。每层 cache index 已迁移到对应 layer cache device，但没有构造真实多设备 cache 自动测试，因此不声称“多设备语义已被直接验证”。

### 2.3 数据与评测

- 模型：Qwen3-32B；
- 数据：MuSiQue-v2 setup-standard，共 200 examples；
- validation：源数据 rows 1--50；test：rows 51--200；
- validation/test 是连续切分，不是随机或分层切分；`split_covariates.json` 显示长度、passage 数等观测量接近，但不能消除顺序偏差；
- selected-token rate：0.15；Sparse block size=64，Top-K blocks=8；
- EM/F1/GLM judge 均由 runner 自动产生；
- alpha 只在 validation 选择；full rate=1 只是 control，不参与候选选择。raw 的 Dense/Sparse 候选按 GLM、F1、较小 alpha 依次选择，test 不再调参。早期执行曾排除 preprocess 非零 alpha，属于 protocol deviation；本轮已补测原规则对应的 Dense-preprocess alpha=1.0 和 Sparse-preprocess alpha=0.5，但它们应被视作补充确认而非原始冻结 protocol 的一部分。

所有 worker 指向同一 `MODEL/DATASET` cache，不创建 worker 私有 cache。runner 在 KV 或 FAISS artifact 缺失时会持锁回填，并且目前没有覆盖所有 artifact 的严格只读 preflight；因此本轮不能证明共享 cache 完全未写，运行前后也未保存 cache 内容快照。未来复现应先单 worker warmup 并核对 cache manifest，再启动并行只读阶段。不同任务写独立 result segment；validation/test 汇总器都要求每方法达到预期唯一问题数且问题集合相同，整行完全相同的重复分片才可去重，任一字段冲突都直接失败。

alpha 冻结顺序有 `EXPERIMENT_LOG.md` 的时间顺序记录，但历史 worker 启动前没有把各自 git HEAD 写进结果目录，因此不是密码学意义的预注册。full 分片与候选使用同一数据文件、问题集合和 runner 配置，逐样本问题集合已核对一致；但同样缺少每个分片的 pre-launch HEAD 绑定。结论按这一 provenance 限制解释。

## 3. Validation：完整 alpha 矩阵（N=50）

| Attention | Load KV | alpha | EM | F1 | GLM |
|---|---|---:|---:|---:|---:|
| Dense | preprocess | 0 | 22% | 32.92% | 32% |
| Dense | preprocess | 0.25 | 24% | 33.16% | 32% |
| Dense | preprocess | 0.50 | 26% | 34.78% | 34% |
| Dense | preprocess | 0.75 | 24% | 33.55% | 34% |
| Dense | preprocess | 1.00 | 20% | 33.32% | 38% |
| Dense | raw | 0 | 14% | 23.43% | 26% |
| Dense | raw | 0.25 | 18% | 26.72% | 28% |
| Dense | raw | 0.50 | 24% | 31.11% | 30% |
| Dense | raw | 0.75 | 28% | 38.51% | 40% |
| Dense | raw | 1.00 | 26% | 37.62% | 40% |
| Sparse | preprocess | 0 | 22% | 32.92% | 32% |
| Sparse | preprocess | 0.25 | 22% | 30.81% | 30% |
| Sparse | preprocess | 0.50 | 24% | 33.71% | 34% |
| Sparse | preprocess | 0.75 | 22% | 32.10% | 32% |
| Sparse | preprocess | 1.00 | 20% | 29.96% | 30% |
| Sparse | raw | 0 | 14% | 23.43% | 26% |
| Sparse | raw | 0.25 | 18% | 25.58% | 26% |
| Sparse | raw | 0.50 | 20% | 29.56% | 30% |
| Sparse | raw | 0.75 | 26% | 33.94% | 34% |
| Sparse | raw | 1.00 | 26% | 33.49% | 32% |
| Full rate=1 control | n/a | n/a | 28% | 39.25% | 42% |

validation 上 Dense raw 0.75 相对 raw 0 的 paired GLM 为 `+14pp`，95% bootstrap CI `[+4,+26]pp`，GLM up/down=`8/1`，exact McNemar `p=0.0391`。Sparse raw 0.75 的 paired GLM 为 `+8pp`，CI `[+2,+16]pp`，up/down=`4/0`，McNemar `p=0.125`。

原始 protocol 实际冻结了 Dense raw 0.75 和 Sparse raw 0.75，preprocess 只保留 alpha=0 参考；这是早期文档与执行之间的 protocol deviation。补充实验已覆盖按原文字规则应测试的 Dense-preprocess alpha=1.0 和 Sparse-preprocess alpha=0.5，因此 preprocess 候选现在有 frozen-test 绝对结果，但不能把它们描述为原始 frozen selection 的预注册候选。

## 4. Frozen Test（N=150）

| 方法 | N | EM (95% CI) | F1 (95% CI) | GLM (95% CI) |
|---|---:|---:|---:|---:|
| preprocess alpha=0 | 150 | 20.00% [14.00, 26.67] | 34.38% [27.97, 40.66] | 33.33% [26.00, 40.67] |
| raw alpha=0 | 150 | 15.33% [10.00, 21.33] | 29.22% [23.28, 35.14] | 30.67% [23.33, 38.00] |
| Dense raw alpha=0.75 | 150 | 20.00% [14.00, 26.67] | 34.44% [28.28, 40.85] | 33.33% [26.00, 41.33] |
| Sparse raw alpha=0.75 | 150 | 14.67% [9.33, 20.67] | 28.50% [22.65, 34.66] | 27.33% [20.67, 34.67] |
| Full rate=1 | 150 | 24.67% [18.00, 32.00] | 40.29% [33.75, 47.14] | 41.33% [34.00, 49.33] |

主要 paired 对比：

| Candidate - baseline | EM delta | F1 delta | GLM delta | GLM up/down | McNemar p |
|---|---:|---:|---:|---:|---:|
| Dense raw 0.75 - raw 0 | +4.67pp [-0.67, +10.00] | +5.23pp [-0.11, +10.40] | +2.67pp [-3.33, +8.67] | 13/9 | 0.5235 |
| Sparse raw 0.75 - raw 0 | -0.67pp [-5.33, +3.33] | -0.72pp [-5.16, +3.50] | -3.33pp [-8.67, +2.00] | 6/11 | 0.3323 |
| Dense raw 0.75 - full rate=1 | -4.67pp [-10.00, 0.00] | -5.84pp [-11.06, -0.70] | -8.00pp [-14.00, -2.00] | 5/17 | 0.0169 |
| Sparse raw 0.75 - full rate=1 | -10.00pp [-16.00, -4.67] | -11.79pp [-17.44, -6.44] | -14.00pp [-20.67, -7.33] | 4/25 | 0.000104 |

Dense raw 的 validation GLM 点估计是 +14pp，而 test 点估计为 +2.67pp，validation 趋势未在 test 复现；这可能来自抽样波动或连续切分导致的顺序分布差异，不能识别为单一原因。test 的 EM/F1/GLM paired CI 均覆盖 0。实际冻结的 Sparse-raw 候选在三个指标都没有优于 raw alpha=0，补测的 Sparse-preprocess alpha=0.5 也没有达到 full rate=1 的水平；两者都不能作为当前质量可用方案。

与 full rate=1 相比，Dense raw 0.75 的 GLM 低 8pp（paired 95% CI `[-14,-2]pp`，McNemar `p=0.0169`），F1 低 5.84pp；Sparse raw 0.75 的 GLM 低 14pp、F1 低 11.79pp。Dense blend 的点估计位于 raw0 与 full 之间，但相对 raw0 的改善未被独立 test 确证；Sparse blend 则没有显示缩小缺口。

## 5. Delta 方向与 scalar 可表达性

机制 trace 使用前 5 examples；每个 example、每层固定采样前 32 个 selected tokens 的全部 KV heads/dim；Dense/Sparse 使用同 cache、selector 和 selected 顺序，trajectory 均为 alpha=1。该采样偏向 selected 序列前部，只用于机制可行性，不能视为全数据无偏统计。

| Load KV | Delta | Cosine | Sparse/Dense norm | Oracle alpha | Oracle relative L2 |
|---|---|---:|---:|---:|---:|
| raw | K | 0.498 | 1.564 | 0.318 | 0.867 |
| raw | V | 0.468 | 1.481 | 0.316 | 0.884 |
| preprocess | K | 0.545 | 1.293 | 0.421 | 0.839 |
| preprocess | V | 0.562 | 1.186 | 0.474 | 0.827 |

`oracle alpha` 是在看到 Dense target delta 后得到的后验最优全局标量，不用于推理。在这 5 个 examples、每层前 32 个 selected tokens、alpha=1 trajectory 的偏置 trace 上，即使 oracle 也残留 82.7%--88.4% relative L2；它只排除了“该采样上一个 global scalar 可重建 Dense delta”，不能解释端到端失败的原因。

层分布也不均匀。按 8 层分带聚合时，raw K/V 在 layers 0--7 的 cosine 为 0.931/0.914，oracle residual 为 0.364/0.406；layers 32--63 的 cosine 大致降到 0.44--0.48，oracle residual 大致升到 0.88--0.90。preprocess 也呈现浅层更接近、深层偏离的模式，但稍好。layer 0 的 V delta 为零，cosine 未定义；汇总时按零能量处理，未把 NaN 当 0 或 1 平均。

这提示 per-layer alpha 值得作为后续假设，但深层采样同时呈现方向偏离。尚未训练或端点评测 per-layer gate，因此不能断言 64 个 scalar gate 是否足够。

## 6. Sparse 路由支持诊断

5 个 MuSiQue-v2 examples、preprocess、alpha=0.5、block size=64、Top-K=8：

| 指标 | Mean | Median | Range |
|---|---:|---:|---:|
| selected-predecessor dependency coverage | 7.42% | 7.79% | 6.05%--8.34% |
| dense attention mass recall | 29.74% | 30.48% | 25.98%--33.05% |
| exact KV support / dense causal support | 4.80% | 4.76% | 4.50%--5.12% |
| preserve-all estimated support / dense | 80.89% | 80.79% | 79.71%--82.65% |
| effective KV tokens/query | 480.5 | 480.6 | 479.9--481.2 |
| dense causal KV tokens/query | 10034.8 | 10111.4 | 9392.1--10654.8 |

Top-K=8 确实把 exact support 降到约 4.8%，但同时漏掉约 70% dense attention mass 和约 92.6% selected-predecessor pairs。该统计是相关证据，不单独证明端到端质量下降由 router 唯一造成；但它与 delta 方向和 frozen test 同时指向“candidate 信息不足”。

强制保留全部 earlier-selected blocks 按定义可恢复 dependency coverage，但静态预算约为 dense support 的 81%，已失去大部分理论节省。该数值是 5 样本预算估算，不是 preserve-all 的正式质量/延迟实验。

## 7. 计算代价边界

当前 Sparse 实现是 correctness/profiling 用 PyTorch 原型：GQA repeat 后计算 block mean，逐 head/query gather，并使用 FP32 routing logits。它降低了 exact QK/AV token support，但引入额外 kernel、中间张量和 Python 调度；本实验没有得到可信的 TTFT 或吞吐提升，实际观察反而慢于 Dense selected-query SDPA。

因此：

- Dense blend 有质量诊断价值，但计算量仍接近原 selected-token recompute；
- Sparse prototype 只有算法 support 统计，没有端到端加速结论；
- 在实现 fused、GQA-native、无 Python loop 的 kernel 前，不能用 wall time 对方法做最终性能判断；
- 即使 kernel 理想化，当前 Top-K=8 的质量结果也已使其不具部署条件。

## 8. 对 goal.md 的逐项回答

1. **base/candidate/working 是否严格区分？** 是。production 每层保存 selected-position base K/V 快照，candidate 写入后立即按快照融合，快照本身不变。
2. **融合是否发生在本层 attention 前？** 是。自动测试直接捕获 sparse attention 的 K/V 输入并与 cache 中 working positions 对齐。
3. **selected predecessor 是否读到本层 fused KV？** 是，但仅在其 block 被路由时；构造测试直接验证，真实路由 coverage 约 7.42%。
4. **routing 是否使用 working K？** 是。测试仅修改 predecessor working K 即改变路由和读出的 V。
5. **alpha endpoints 是否正确？** 是。CPU 张量测试和 32B 单样本均验证。
6. **Dense/Sparse 与 Blend/Replace 是否分开？** 是。完整 validation 覆盖 Dense/Sparse、raw/preprocess 和 alpha 0--1；alpha=1 是 replace，alpha<1 是 blend。
7. **Sparse delta 是否和 Dense delta 同方向？** 只有中等相关，且深层更差；global oracle scalar 仍留下很高残差。
8. **能否替代原重算并加速？** 当前不能。Dense blend 未省 attention；实际测试的 Sparse-raw blend 不优于 raw rate=0，且无 fused kernel 加速证据。Sparse-preprocess alpha=0.5 已补做 frozen test，但同样没有接近 full 的质量或加速证据。

## 8.1 补充：preprocess 候选与独立 K/V alpha

第四轮审阅指出，原规则所对应的 Dense-preprocess alpha=1.0 和 Sparse-preprocess alpha=0.5 尚未进入 frozen test；同时 `goal.md` 要求在共享 alpha 之后区分 K/V 修正强度。两项已补齐。

### Frozen test（rows 51--200，N=150）

| 方法 | alpha_k | alpha_v | EM | F1 | GLM |
|---|---:|---:|---:|---:|---:|
| Dense preprocess | 1.0 | 1.0 | 21.33% | 35.71% | 36.00% |
| Sparse preprocess | 0.5 | 0.5 | 18.67% | 33.37% | 32.67% |

Dense preprocess alpha=1 相对 frozen `preprocess alpha=0` 的 paired delta 为 EM `+1.33pp`、F1 `+1.33pp`、GLM `+2.67pp`；逐行 bootstrap CI 分别为 `[-2.00,+4.67]`、`[-2.05,+4.80]`、`[-2.67,+8.00]`，因此没有确认性改善。Sparse preprocess alpha=0.5 没有对应的 frozen alpha=0 sparse control，本轮只报告绝对值，不伪造 paired 结论；它仍低于 full rate=1 的 EM 24.67%、F1 40.29%、GLM 41.33%。

### Validation K/V ablation（rows 1--50，N=50）

共享 alpha 已有结果；本轮新增 K-only `(alpha_k=a, alpha_v=0)` 与 V-only `(alpha_k=0, alpha_v=a)`：

| load KV / attention | shared a | K-only EM/F1/GLM | V-only EM/F1/GLM | K-only vs shared GLM | V-only vs shared GLM |
|---|---:|---|---|---:|---:|
| Dense raw | 0.75 | 20.00 / 29.35 / 28.00 | 20.00 / 30.31 / 34.00 | -12pp | -6pp |
| Sparse raw | 0.75 | 16.00 / 25.01 / 22.00 | 12.00 / 25.70 / 30.00 | -12pp | -4pp |
| Dense preprocess | 1.00 | 18.00 / 26.18 / 30.00 | 22.00 / 31.85 / 32.00 | -8pp | -6pp |
| Sparse preprocess | 0.50 | 18.00 / 27.10 / 24.00 | 22.00 / 36.66 / 40.00 | -10pp | +6pp |

这组结果只用于机制方向判断，N=50 且未进入 frozen test。相对 shared alpha 的 paired 95% bootstrap 结果记录在 `working_kv_alpha_kv_validation_paired.csv`：除 Sparse-preprocess V-only 外，K-only/V-only 都是负向点估计；Sparse-preprocess 上 V-only 是 `+6pp GLM`，CI `[-2,+14]pp`，尚未确认。当前证据更支持“V correction 可能比 K correction 更有用”，但不能据此宣称 K 可以完全不更新；K/V 的递归 hidden-state 耦合仍在。

### 统计敏感性

测试集存在少量词面近重复问题。按 token Jaccard `>=0.5` 做连通分量得到 134 个 cluster；cluster bootstrap 与逐行区间方向一致：Dense raw alpha=0.75 相对 raw alpha=0 的 GLM delta `+2.67pp`，cluster CI `[-3.36,+8.84]pp`；Dense preprocess alpha=1 相对 alpha=0 的 GLM delta `+2.67pp`，cluster CI `[-2.60,+8.28]pp`；Sparse raw alpha=0.75 相对 raw alpha=0 的 GLM delta `-3.33pp`，cluster CI `[-8.78,+1.99]pp`。因此结论按当前固定测试集合解释，不把逐行 bootstrap 当成无条件总体泛化保证。

## 9. 方法决策与下一步

保留：

- Working-KV 的 cache-anchored、layer-parallel 语义；
- immutable selected-position snapshot 和 pre-attention scatter；
- query dense isolation、完整性检查和 paired evaluation；
- Dense raw blend 作为“cache correction 是否有价值”的诊断上界。

不进入主线：

- Top-K=8 block-mean MoBA + global alpha；
- 仅靠 per-layer scalar 修补当前 sparse candidate；
- preserve-all selected dependencies 作为默认策略。

下一步首先应提高 sparse candidate 的信息覆盖，而不是继续扩大 alpha 网格：在新增 development split 上做 Top-K/support 的质量-成本曲线，并比较 query-aware block router、较小 block、跨 head 共享候选和保留高 mass block；本报告使用过的 rows 51--200 不再作为未来独立 test。当前 K/V 结果提示优先研究 V-aware candidate 与 V correction，但必须在新的 holdout 上验证。只有当 candidate-vs-dense delta 的深层 cosine 提高、oracle residual 降低后，才值得在 dev 上研究 per-layer K/V gate 或 fused kernel。否则应停止这条 MoBA reprocess 路线，转向低秩/预测式 correction。

## 10. 复现与证据文件

- 分支：`exp/sparse-kv-working-blend-20260719`
- 核心语义：`ea73a0e`
- Working-KV 语义 guards：`89b25a8`、`fc14285`、`9f2c6c3`
- runner 确定 GPU 绑定：`8712c8a`（后续订正保留该行为）
- validation launcher：`0437d52`
- delta trace：`92940db`、`8c5a299`
- support 统计：`06256d4`、`055e999`
- split/manifest：`23b6ec5`、`0d948de`
- sharded test 汇总：`28adeb9`

核心命令：

```bash
# 语义测试
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/test_working_kv_semantics.py

# validation 20 条件
python3 MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/launch_working_kv_fix2_50.py

# 单个 frozen-test 条件示例
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py \
  --dataset musique-v2 --method dense_working_kv_raw --rate 0.15 \
  --working-kv-alpha 0.75 --start 50 --end 200 --gpu 0 \
  --result-root MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results_working_kv_fix2_test_150

# 合并 frozen candidates 与独立 full-rate1 shards
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/summarize_working_kv_test.py \
  --root MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results_working_kv_fix2_test_150 \
  --root MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results_working_kv_fix2_test_150_full_shards \
  --summary MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/working_kv_fix2_test_summary.csv \
  --paired MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/working_kv_fix2_test_summary_paired.csv \
  --integrity MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/working_kv_fix2_test_integrity.json
```

汇总证据：

- `working_kv_fix2_validation_summary.csv`
- `working_kv_fix2_validation_summary_paired.csv`
- `working_kv_fix2_test_summary.csv`
- `working_kv_fix2_test_summary_paired.csv`
- `working_kv_fix2_test_integrity.json`
- `frozen_evidence/validation/manifest.json` 与 21 份完整逐样本 CSV
- `frozen_evidence/test/manifest.json` 与 5 份完整逐样本 CSV
- `delta_trace/raw_delta_direction_summary.csv`
- `delta_trace/preprocess_delta_direction_summary.csv`
- `router_stats_fix2/five_example_summary.csv`
- `router_stats_fix2/five_example_aggregate.json`
- `working_kv_confirmatory_preprocess_test_summary.csv`
- `working_kv_confirmatory_preprocess_test_paired_vs_alpha0.csv`
- `working_kv_alpha_kv_validation_summary.csv`
- `working_kv_alpha_kv_validation_paired.csv`
- `cluster_bootstrap_dense_raw075_vs_raw0.csv`
- `cluster_bootstrap_dense_pp_alpha1_vs_alpha0.csv`
- `cluster_bootstrap_sparse_raw075_vs_raw0.csv`
- `delta_trace/{raw,preprocess}_{dense,sparse}.pt`（primary evidence）
- `router_stats_fix2/preprocess_alpha0p5_{sample0_v3,examples1_2,examples3_4}.jsonl`（primary evidence）
- `split_covariates.json`
- `repro_manifest.json`

完整启动记录、异常、修订和 commit 映射见 `EXPERIMENT_LOG.md`；每日状态见 `daily-note.md`。
