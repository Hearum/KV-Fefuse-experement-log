# Cache-Anchored Sparse Recompute 阶段 A 结果

## 配置

- Qwen3-32B、MuSiQue-v2、preprocess KV
- FusionRAG selector rate=0.15
- 10 条相同样本（example 0--10）
- alpha_K=alpha_V=alpha
- dense：完整 causal prefix
- sparse：block size=64、top-k=8、current block 保留
- 自动评测：EM、F1、GLM judge
- 代码和启动记录提交：23a76cc
- 计划文档提交：c9dd6db

## 结果

同批 native dense online rate=0.15 前10条参考：EM=0.2000，F1=0.3028，GLM=4/10。

| attention | alpha | EM | F1 | GLM |
|---|---:|---:|---:|---:|
| dense | 0.25 | 0.3000 | 0.3558 | 4/10 |
| dense | 0.50 | 0.3000 | 0.3558 | 4/10 |
| dense | 0.75 | 0.3000 | 0.3694 | 4/10 |
| sparse top-k=8 | 0.25 | 0.3000 | 0.3558 | 4/10 |
| sparse top-k=8 | 0.50 | 0.3000 | 0.3865 | 4/10 |
| sparse top-k=8 | 0.75 | 0.3000 | 0.3917 | 4/10 |

结果目录：

    MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results_planA_10/

每一组的 metrics.csv 都是10行，GLM 由 runner 自动执行。

## 如何解释

1. alpha 在 dense 和 sparse 中都实际生效；它不是未打开。alpha 改变了 candidate correction 写回 base cache 的幅度。
2. 在这10条 sanity 上，所有中间 alpha 的 GLM 都与 native 前10条相同，都是4/10；因此不能仅凭 GLM 说 sparse 已经恢复 native。
3. sparse top-k=8 的 F1 在 alpha=0.5/0.75 上高于 dense 同 alpha，但这是 token-overlap 指标，GLM 没有提高，不能作为端到端正确率提升结论。
4. 这个结果支持继续研究 cache-anchored correction，但尚未证明 top-k=8 能达到 native dense rate=0.15 的效果，更没有证明计算收益。top-k=8 的 attention support 仍远小于 dense prefix。
5. 下一步不是再盲扫 alpha，而是冻结 alpha=0.75 作为当前 sparse candidate，增加到独立50条，并加入 top-k=32 的同 alpha 对照，观察 F1 优势是否能转化为 GLM/逐题答案一致性。

## 复现命令

命令模板：

    /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py --dataset musique-v2 --method METHOD --rate 0.15 --working-kv-alpha-k ALPHA --working-kv-alpha-v ALPHA --start 0 --end 10 --gpu GPU --result-root MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results_planA_10

sparse 额外参数：--sparse-block-size 64 --sparse-block-topk 8。

## 三视角审核

### 模型语义审核

PASS。报告明确区分 base KV、sparse candidate KV 和 alpha correction，native dense rate=0.15 是主要参考，不把 full rate=1 混入。

### 系统实现审核

PASS。六组使用相同样本、共享 preprocess cache，所有 metrics.csv 均自动生成；结果中同时说明 support 尚未证明有端到端加速。

### 实验科学审核

PASS。F1 与 GLM 分开报告，10条结果只作 sanity，不外推完整数据集；下一步的50条验证条件已明确冻结。

三视角审核均 PASS。

