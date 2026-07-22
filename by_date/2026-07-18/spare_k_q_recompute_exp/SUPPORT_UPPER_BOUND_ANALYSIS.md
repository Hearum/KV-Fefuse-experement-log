# Attention Support 上界分析

## 目的

在继续扩大 block top-k 之前，先判断 sparse support 是否有足够的余量。这里的上界分析不把离线真实 attention score 当作 online 算子，只用它回答一个问题：如果为了保留 selected-token 之间的因果依赖而增加 support，理论上还能省多少 KV attention。

## 数据与命令

固定 Qwen3-32B、MuSiQue-v2、preprocess cache，使用已有 router trace 的 5 个独立 example。每个 trace 是 4096 个 layer/head 记录，context block size=64，原始 router top-k=8。

运行命令：

    /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/summarize_router_examples.py --inputs router_stats_fix2/preprocess_alpha0p5_sample0_v3.jsonl router_stats_fix2/preprocess_alpha0p5_examples1_2.jsonl router_stats_fix2/preprocess_alpha0p5_examples3_4.jsonl --output-csv router_stats_fix2/support_upper_bound_examples.csv --output-json router_stats_fix2/support_upper_bound_summary.json

对应实验代码提交：4dfb016；本分析只读取已有 trace，没有重新生成 cache。

## 结果

| 指标 | mean | median | min--max |
|---|---:|---:|---:|
| selected predecessor dependency coverage（原始 top-k=8） | 7.42% | 7.79% | 6.05%--8.34% |
| attention mass recall（原始 top-k=8） | 29.74% | 30.48% | 25.98%--33.05% |
| 实际 KV support fraction（原始 top-k=8） | 4.80% | 4.76% | 4.50%--5.12% |
| 为保留 selected predecessor dependency 所需 support fraction | 80.89% | 80.79% | 79.71%--82.65% |
| 原始 sparse effective KV/query | 480.5 | 480.6 | 479.9--481.2 |
| causal KV/query | 10034.8 | 10111.4 | 9392.1--10654.8 |

## 解释

原始 top-k=8 的计算 support 很小，但只覆盖约 30% 的 dense attention mass，且 selected predecessor 的直接依赖覆盖约 7%。如果强制保留所有 selected predecessor 所在 block，support 会上升到约 81% 的 causal KV。这个结果说明：

1. top-k=8 的速度余量是真实存在的，但它删除了大量影响 hidden state 的 support，因此和 50 条 accuracy 下降相符。
2. 仅通过 preserve-selected-dependencies 纠正语义，会把 support 从约 4.8% 拉到约 80.9%，剩余计算收益很有限。
3. 因此“加大 top-k 直到准确率恢复”不太可能同时满足低计算量；top-k=32/64 的方向本质上是在逼近这个高 support 区域。
4. 这不是对所有 sparse router 的不可能性证明，因为 trace 是 5 个 example、固定 top-k=8；它是当前实现继续试验的上界约束。

## 对下一版实现的约束

- 不能把真实 dense attention score 作为 online routing 输入，否则没有省掉原始 attention 计算。
- 必须测量真实 support fraction、dependency coverage 和 attention mass recall，不能只报告 block 数。
- 若候选需要超过约 80% causal KV 才能恢复 selected-token 依赖，则不应继续把它包装为低成本 sparse reprocessing。
- 下一步更有价值的是研究预测式 support：用 cache-only 的 block representative 预测少量高价值 block，并验证是否能在不强制保留全部 dependency 的情况下提高 mass recall；同时必须用 batched kernel 测量真实收益。

## 审核

### 模型语义审核

PASS。将 dense attention mass、selected predecessor dependency 和 KV support 分开，未把三者混成同一个指标。

### 系统审核

PASS。分析只读取已有 trace，不声称使用真实 score 作为 online 算子；support fraction 按 causal KV/query 计算。

### 实验科学审核

PASS。明确样本为 5 个 example，结果只作为上界约束，不外推为完整 MuSiQue accuracy 结论。

