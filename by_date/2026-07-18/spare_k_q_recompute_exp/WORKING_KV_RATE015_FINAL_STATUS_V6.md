# Working-KV rate=0.15 稀疏重算阶段报告 v6

## Vectorized block representative microbenchmark

在 qjy001 GPU0、Qwen3-32B attention 形状近似的单算子测试中，设置 heads=64、context=32768、head_dim=128、block size=64，比较旧的逐 block Python loop 和新 padding + reshape + sum 的代表向量计算。

运行命令：

    CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python - <<PY
    # old/new representative benchmark; see shell history and daily-note
    PY

20 次同步平均结果：

| 实现 | 单次 block representative 时间 |
|---|---:|
| old Python loop | 25.54 ms |
| new vectorized | 1.53 ms |

单算子约 16.6 倍加速；输出 max_abs=0、mean_abs=0，说明在该测试上数值完全一致。代码提交为 a587724。

## 这项优化能说明什么

它确认了 top-k64 之前的极慢至少有一部分来自实现层面的 Python block loop，而不是必然来自 sparse attention 数学本身。但它没有改变 support：top-k8 的 support/accuracy 失败、top-k32 的逐题不等价和 top-k64 的未完成结论仍然成立。

因此这一步只提升候选实现的可测性，不等于已经找到可交付方案。下一步应在 vectorized 版本上重新跑一个 10 条 top-k32 regression，使用同一 native batch 比较答案和 wall-clock；若质量仍明显不等价，则问题是 support/router，而非 block mean 性能。

## 前序证据摘要

- native/fixed alpha=1：52/52 逐题一致，GLM 20/52；只证明语义，不省 attention。
- top-k8：50 条，GLM16/50，native20/50，答案差异22/50。
- top-k32：10 条，GLM4/10，native4/10，答案差异5/10。
- support trace：5 个 example，top-k8 只保留4.80% causal KV、mass recall29.74%；保留全部 selected predecessor dependency 需约80.89% support。

## 三视角审核

### 模型语义审核

PASS。benchmark 只替换 block mean 的实现，最后 partial block 的真实长度在正式代码中有 divisor correction；输出 exact equality 已检查。

### 系统实现审核

PASS。报告区分 microbenchmark 的 16.6 倍算子加速和端到端收益，没有把算子结果冒充 pipeline TTFT；Qwen smoke 仍有独立结果路径和自动 GLM。

### 实验科学审核

PASS。给出硬件、shape、迭代数、平均时间、误差和 commit；并明确指出 accuracy/support 尚未因该优化而被重新证明。

三视角审核均 PASS。

