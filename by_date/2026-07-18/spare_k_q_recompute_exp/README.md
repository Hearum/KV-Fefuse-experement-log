# MuSiQue-v2 Baseline 与 Sparse-K/Q Recompute 新实验

## 1. 实验目的

本目录用于在 MuSiQue-v2 setup-standard pipeline 上探究新的稀疏注意力 / sparse K-Q recompute 方法。本文档先固定已有 baseline，后续所有新方法都必须在同一数据、同一 runner、同一评测口径下比较。

当前数据集是 `musique-v2`，不是早期 reflect/main-sub 版本的 MuSiQue。MuSiQue-v2 每条样本直接使用 setup-standard 的问题和 22 个左右 document passages，最终直接回答原问题。

## 2. 已有 MuSiQue-v2 Baseline

来源文件：

```text
MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/setup_v2_summary_with_glm.csv
```

该表最后修改时间为 `2026-07-15 14:26:52 +0800`。以下是完整 200 条 MuSiQue-v2 的统一结果：

| 方法 | rate | EM | F1 | GLM Judge |
|---|---:|---:|---:|---:|
| full_rate1 | 1.00 | 25.50% | 40.03% | 未记录在该 CSV |
| raw_rate0 | 0.00 | 15.00% | 27.77% | 未记录 |
| preprocess_rate0 | 0.00 | 20.50% | 34.01% | 未记录 |
| online_qk | 0.05 | 20.50% | 33.90% | 未记录 |
| online_qk | 0.15 | 21.50% | 35.57% | 未记录 |
| online_draft | 0.05 | 20.00% | 33.80% | 未记录 |
| online_draft | 0.15 | 24.00% | 37.88% | 未记录 |
| uniform_alpha0p1_draft | 0.05 | 22.00% | 35.94% | 未记录 |
| uniform_alpha0p1_draft | 0.15 | 25.00% | 39.28% | 未记录 |
| offline3b_mean | 0.05 | 21.50% | 34.96% | 未记录 |
| offline3b_mean | 0.15 | 24.50% | 38.79% | 未记录 |
| offline3b_top2 | 0.05 | 20.50% | 34.31% | 未记录 |
| offline3b_top2 | 0.15 | 24.50% | 38.47% | 未记录 |
| offline3b_freq_boundary2 | 0.05 | 19.50% | 33.28% | 未记录 |
| offline3b_freq_boundary2 | 0.15 | 23.00% | 36.75% | 未记录 |

注意：`full_rate1` 的 GLM Judge 结果没有出现在这个统一 CSV 中。不要把旧 reflect 版本中的 `105/135` 等 main/sub 结果与这里的 200 条 setup-standard 结果混用。

## 3. 统一实验定义

| 项目 | 固定值 |
|---|---|
| model | Qwen3-32B |
| dataset | MuSiQue-v2，200 samples |
| pipeline | setup-standard v2 |
| full baseline | `rate=1.0`，所有 document token full recompute |
| no-update baselines | `raw_rate0`、`preprocess_rate0` |
| online baselines | `online_qk`、`online_draft` |
| cache | `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2` |
| source data | `/mnt/qjhs-sh-lab-01/wjh/FusionRAG/data/musique-200.jsonl`，经 v2 builder 生成 dataset manifest |
| result CSV | `results/<method>/musique-v2/rate_<rate>/seg_<start>_<end>/` |
| runner | `MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py` |
| summary | `MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/summarize_setup_v2.py` |

共享 cache 只按 `model/dataset` 维护，不按实验、worker 或 GPU 复制。新方法只允许把生成结果写到本实验目录，不能把结果 CSV 写回 cache。

## 4. Baseline 复现实验命令模板

最终实现、证据文件及对应 commit 由 `repro_manifest.json` 固化；核心 Working-KV 语义提交为
`ea73a0e`，语义回归防护提交为 `89b25a8`。不要用本节早期 smoke 模板的历史提交替代最终 manifest。

单段 smoke：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py \
  --dataset musique-v2 --method online_qk --rate 0.15 \
  --start 0 --end 5 --gpu 0 \
  --result-root MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results/smoke
```

完整任务的分片范围是 `0-25,25-50,...,175-200`。建议正式启动前先完成 shared cache warmup，
并保存 cache 内容清单；当前 runner 在 cache 缺失时仍可能回填 KV/FAISS，并没有严格只读 preflight。
本轮没有保存运行前后的完整 cache 快照，manifest 只对抽样 cache 文件做内容哈希，因此 cache 的完整不可变性
不是本轮可证明的复现保证。最终 validation/test 汇总使用本实验 Working-KV 汇总脚本，强制整行重复一致、
精确样本数和所有方法 question set 相同。

## 5. 新 sparse K/Q recompute 方法的比较要求

新方法至少需要报告：

1. 与 `preprocess_rate0`、`online_qk@0.15`、`online_draft@0.15`、`uniform_alpha0p1_draft@0.15` 和 `full_rate1` 的 EM/F1 对比。
2. selected token/block 数量和实际 recompute rate。
3. sparse attention 如何产生候选：chunk、block、token、landmark 或 router。
4. 是否先计算完整 attention score；如果先计算完整 score，则不能声称节省了 score 计算量。
5. K、Q、V 分开统计，尤其说明是否只更新 K/Q 而保留 preprocess V。
6. TTFT、prefill latency、GPU memory 和 full recompute 的相对开销。
7. 结果 CSV、启动命令、数据路径、cache 路径和 commit hash。

Working-KV 方法已经完成 validation `N=50`、冻结 test `N=150` 和机制诊断。完整定义、逐项结果、
限制与结论见 `WORKING_KV_REPORT.md`；命令和运行异常见 `EXPERIMENT_LOG.md`。当前结论是 Dense raw
`alpha=0.75` 在 test 上相对 raw alpha=0 只有未被统计确认的正向点估计，Sparse Top-K=8 没有改善质量，
且本轮没有可信的端到端加速证据。
