# offline_hybrid70 复现记录

本文档固定记录当前已经使用过的 `offline_hybrid70` 做法，避免后续新增 offline draft selector 时覆盖或混淆。

## 固定集合来源

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_hybrid_qk_draft_rate015_full/
```

关键文件：

```text
chunk_fixed_sets_npz/exampleXXX_rate0p15_chunk_local_sets.npz
hybrid_manifest.csv
offline_set_overlap_analysis/README.md
```

推理时使用的 fixed set 目录应指向：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_hybrid_qk_draft_rate015_full/chunk_fixed_sets_npz
```

## 方法名

主要复现实验使用：

```text
hybrid_draft70_qk30_score_per_chunk
```

含义：每个 chunk 内按固定 rate 选 token；候选 score 由 offline draft score 和 offline QK score 混合，权重约为 draft 70%、QK 30%。该方法是 per-chunk fixed set，不依赖 online query，因此 online selection cost 为 0。

同目录还存在：

```text
hybrid_draft70_then_qk30_per_chunk
```

这是另一个 hybrid 变体，不是前面表格默认的 `offline_hybrid70`。

## Qwen3 rate=0.15 复现实验路径

结果目录：

```text
MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/offline_hybrid70_rate015/
```

启动脚本：

```text
MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/launch_qwen3_rate015_segments.sh
```

脚本里的关键参数：

```bash
--model_type qwen3 \
--model_path /mnt/qjhs-sh-lab-01/models/Qwen3-32B \
--model_name Qwen3-32B \
--cache_dir /raid/home/hming/fusionrag-reflect-qwen3-full-cache \
--rate 0.15 \
--preprocess true \
--reprocess_method FusionRAG \
--offline_fixed_set_dir MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_hybrid_qk_draft_rate015_full/chunk_fixed_sets_npz \
--offline_fixed_set_method hybrid_draft70_qk30_score_per_chunk \
--offline_fixed_set_rate 0.15
```

已记录结果：

```text
Main Acc: 93/135 (68.89%) in qwen3_rate015_online_offline summary
Sub Acc: 196/248 (79.03%)
prefill avg: 0.8447s
selection avg: 0
```

注意：另一个集中汇总文件 `COMPLETE_OFFLINE_FIXED_SET_RESULTS.md` 中有旧表，部分数值来自 Qwen2.5/旧 judge 或旧跑法。Qwen3 当前对比优先看 `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/README.md` 和后续 Qwen3 专门实验目录。

## 其他相关 selector 路径

online draft trace：

```text
MOTIVATION_EXPERIMENTS/qwen3_online_draft_trace_rate015/
```

旧 offline draft aggregate fixed set：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_draft_score_aggregates_rate015_full/chunk_fixed_sets_npz
```

旧方法名：

```text
draft_frequency_per_chunk
draft_mean_score_per_chunk
draft_max_score_per_chunk
draft_top2_mean_score_per_chunk
draft_top4_mean_score_per_chunk
```

这些旧 offline draft 方法直接从 draft score 做 per-chunk 频率/均值/max/top-n 均值聚合，没有复用 online DraftModel 的 smart selection 后处理。
