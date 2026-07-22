# Working-KV rate=0.15 稀疏重算阶段报告 v5

## 本版新增内容

v4 已确认原始 Python block-gather prototype 过慢。本版只做一个保持数学语义不变的 kernel-side 优化：把 block representative 的逐 block Python loop 改为 padding、reshape、sum、length correction 的批量计算。

修改位置：ktransformers/operators/sparse_attention.py

代码提交：a587724。

## 等价性验证

对 context=128、130、32768，heads=3、head_dim=7 的随机 tensor，同时执行旧的逐 block mean 和新的批量实现，逐元素 allclose 测试 PASS。

启动命令：

    /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python - <<PY
    # compare old per-block mean and new padded batched mean
    PY

该测试只验证 block representative，没有改变 top-k 数学规则、KV blend、causal mask 或 selected-token 语义。

## Qwen3-32B smoke

配置：MuSiQue-v2、preprocess KV、rate=0.15、Working-KV alpha K/V=1、block size=64、top-k=32、query chunk=64、example 0--1、qjy001 GPU0。

启动命令：

    env FUSIONRAG_SPARSE_QUERY_CHUNK=64 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py --dataset musique-v2 --method sparse_working_kv_preprocess --rate 0.15 --working-kv-alpha-k 1 --working-kv-alpha-v 1 --sparse-block-size 64 --sparse-block-topk 32 --start 0 --end 1 --gpu 0 --result-root MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results_vectorized_smoke

结果路径：

    results_vectorized_smoke/sparse_working_kv_preprocess/musique-v2/rate_0p15/alpha_k_1p0__alpha_v_1p0/seg_0_1/

结果：EM=0、F1=0、GLM=0/1；预测为 Maggie Gyllenhaal，标准答案为 Maria Bello。这个单样本结果仅用于实现回归，不用于估计总体 accuracy。它与旧 top-k32 同一条 smoke 的预测一致，说明批量 representative 优化没有改变输出。

自动评测由 setup-v2 runner 同步调用 judge_task_csv.py 完成，metrics.csv 已生成。

## 与前序证据合并后的判断

- native 与 fixed alpha=1：52/52 逐题答案一致，GLM 均为20/52；这是语义 oracle，不省 attention。
- top-k=8 完整50条：GLM16/50，native20/50，答案差异22/50；质量不可用。
- top-k=32 10条：GLM4/10，native4/10，但答案差异5/10；不能称等价。
- top-k=64：约21分钟只生成5/10，前5条 exact=3/5，未产生完整 metrics；当前实现效率不可交付。
- support 上界 trace：原始 top-k=8 只保留约4.80% causal KV、attention mass recall约29.74%；若保留全部 selected-predecessor dependency，需要约80.89% support。

新的 vectorized representative 优化修复了一个明确的系统瓶颈，但尚未证明它能把 sparse top-k=32/64 变成端到端加速方案。下一步应先测同一条样本的 wall-clock，再考虑 10 条质量回归；如果仍不能恢复 native，应停止扩大 top-k，转向 cache-only support predictor。

## 三视角审核

### 模型语义审核

PASS。新改动只替换 block mean 的计算方式，padding 的最后 block 使用真实长度除法，没有改变 router 分数的定义。

### 系统实现审核

PASS。Qwen3-32B smoke 的自动 EM/F1/GLM 结果已落盘；报告区分了 kernel 等价性、单样本回归和完整 accuracy，不把 smoke 当总体结论。

### 实验科学审核

PASS。新结果与前序 top-k8/32/64 的证据范围一致，明确写出样本量、对标 native、代码提交和结果路径；没有用 partial top-k64 结果支持过强结论。

三视角审核均 PASS。

