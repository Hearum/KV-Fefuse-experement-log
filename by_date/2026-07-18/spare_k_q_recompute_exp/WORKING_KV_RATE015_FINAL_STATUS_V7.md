# Working-KV rate=0.15 稀疏重算阶段报告 v7

## Vectorized top-k32 回归边界

在 v6 的 block representative 向量化之后，启动了同配置的 Qwen3-32B MuSiQue-v2 top-k32 回归：preprocess KV、rate=0.15、alpha K/V=1、block size=64、query chunk=64、example 0--10、qjy001 GPU0。

命令：

    env FUSIONRAG_SPARSE_QUERY_CHUNK=64 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py --dataset musique-v2 --method sparse_working_kv_preprocess --rate 0.15 --working-kv-alpha-k 1 --working-kv-alpha-v 1 --sparse-block-size 64 --sparse-block-topk 32 --start 0 --end 10 --gpu 0 --result-root MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results_vectorized_top32_10

实际结果：运行 8 分 52 秒后只生成 3/10 条，未生成 metrics.csv，随后停止。3 条中 exact match 为 2/3。结果 CSV 和 status.log 位于 results_vectorized_top32_10/ 与 logs_vectorized_top32_10/。

这不是 accuracy 结论，而是执行边界：代表向量计算的 16.6 倍 microbenchmark 优化没有解决端到端的按 query/head KV gather 瓶颈。当前 prototype 不适合作为正式 10 条或完整数据集 pipeline。

## 当前方法结论

| 方案 | 质量证据 | 计算证据 | 判断 |
|---|---|---|---|
| native selected-token rate=0.15 | 50 条 oracle | native attention 成本 | 参考基线 |
| fixed Working-KV alpha=1 | 52/52 与 native 逐题一致 | 没有减少 attention | 语义回归基线 |
| sparse top-k=8 | 50 条，GLM16/50 vs native20/50，答案差异22/50 | support 小 | 质量失败 |
| sparse top-k=32 | 10 条，GLM4/10 vs native4/10，答案差异5/10 | Python gather 过慢 | 不等价、不可交付 |
| sparse top-k=64 | 约21分钟仅5/10，exact3/5，无 metrics | 过慢 | 不可交付 |
| vectorized top-k=32 | 3/10，exact2/3，无 metrics | block mean 加速但端到端仍超时 | 只证明优化方向 |

## 下一步实现边界

继续方向不能只是扩大 top-k 或重复 Python gather。需要实现真正的 batched sparse attention：至少把 selected query、block indices、KV gather 和 masked reduction 放入单个 GPU kernel，或者先用固定形状的 batched gather 彻底消除 head/query 的 Python 循环。实现后必须重新验证：

1. 与当前 Python prototype 的输出误差；
2. 与 native rate=0.15 的逐题答案；
3. 实际每层和端到端 wall-clock；
4. support fraction、dependency coverage 和 attention mass recall。

在 batched kernel 之前，不再把任何 sparse accuracy 小样本称为可交付方法。

## 版本和记录

- block representative 向量化：a587724
- microbenchmark 报告：304a388
- vectorized smoke 报告：3a4f9be
- timeout status 与 daily-note：03e14ec
- 共享 cache：/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2

## 三视角审核

### 模型语义审核

PASS。报告没有把 3/10 partial 结果当作 accuracy，也区分了 native oracle、sparse support 和 KV update 语义。

### 系统实现审核

PASS。报告记录了完整启动命令、GPU、结果目录、8分52秒停止边界和无 metrics.csv；没有将 block-level microbenchmark 冒充端到端加速。

### 实验科学审核

PASS。top-k8/32/64 的证据范围与 v4/v6 保持一致，vectorized top-k32 明确标为执行边界；后续 batched kernel 的验证指标已预先固定。

三视角审核均 PASS。

