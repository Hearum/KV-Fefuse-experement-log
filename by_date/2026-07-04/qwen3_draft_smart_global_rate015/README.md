# Qwen3 Offline Draft Smart Global 实验记录

## 背景

之前发现：

1. online DraftModel selector 在 Qwen3 rate=0.15 下质量明显高于 offline fixed set。
2. 旧 offline draft fixed set 与 online draft 的 doc-token 重合度只有约 0.32-0.33 Jaccard，覆盖 online draft 约 48%-50%。
3. 旧 offline draft 的生成方式没有复用 online DraftModel 的 smart selection 后处理，而是直接对 saved draft score 做 per-chunk frequency/mean/max 聚合。

因此本轮新增一组不覆盖旧结果的 offline draft 方法，命名为 `draft_smart_*`，用于验证：如果 offline 阶段也先模拟 online DraftModel 的 smart token selection，再聚合成 fixed set，是否能更接近 online draft 的效果。

## 固化旧 offline_hybrid70

旧 `offline_hybrid70` 复现记录已单独保存：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/repro_records/offline_hybrid70_repro.md
```

固定集合路径：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_hybrid_qk_draft_rate015_full/chunk_fixed_sets_npz
```

方法名：

```text
hybrid_draft70_qk30_score_per_chunk
```

## 新 fixed set 生成

生成脚本：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/derive_draft_smart_global_sets.py
```

输出目录：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_draft_smart_global_rate015_full/
```

输入 score cache：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_draft_rate015_full/combined/score_cache_npz
```

新方法名：

```text
draft_smart_frequency_global
draft_smart_mean_score_global
draft_smart_max_score_global
draft_smart_top2_mean_global
```

## 新旧 draft offline 的核心差异

旧 `draft_frequency_per_chunk / draft_max_score_per_chunk`：

```text
saved draft score -> 每个 chunk 内直接 top-k / frequency / mean / max -> fixed set
```

新 `draft_smart_*_global`：

```text
saved draft score -> 对每个 calibration query 先跑 online-style smart selection
                  -> 得到每个 calibration query 的 selected token set
                  -> 对 selected set / score 做 frequency、mean、max、top2-mean 聚合
                  -> 得到 global doc-level fixed set
                  -> 再拆成 runner 需要的 chunk-local npz
```

注意：新方法仍然没有用真实测试 query；calibration query 来自已有 offline draft score cache。

## Overlap 探究结果

只统计 document tokens，过滤 online trace 中的 query tokens。

| method | Jaccard vs online draft | online draft covered by fixed | fixed covered by online | avg online tokens | avg fixed tokens | avg intersection |
|---|---:|---:|---:|---:|---:|---:|
| draft_smart_frequency_global | 0.4429 | 0.6051 | 0.6290 | 213.5 | 221.9 | 126.8 |
| draft_smart_mean_score_global | 0.3872 | 0.5537 | 0.5708 | 213.5 | 219.9 | 116.3 |
| draft_smart_max_score_global | 0.3831 | 0.5578 | 0.5582 | 213.5 | 223.5 | 117.4 |
| draft_smart_top2_mean_global | 0.3870 | 0.5599 | 0.5641 | 213.5 | 222.7 | 117.7 |

对比旧 offline draft/hybrid：

```text
旧 offline_hybrid70 vs online_draft: Jaccard 0.3313, coverage 0.4890
旧 draft_max_score vs online_draft: Jaccard 0.3238, coverage 0.4807
新 draft_smart_frequency_global: Jaccard 0.4429, coverage 0.6051
```

初步结论：复用 online DraftModel 的 smart selection 后处理后，offline fixed set 明显更接近 online draft，但仍没有达到之前 full-attention stable anchor 的 0.70+ 稳定性水平。

## Accuracy 实验

启动脚本：

```text
MOTIVATION_EXPERIMENTS/qwen3_draft_smart_global_rate015_launch.sh
```

运行方式：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
MOTIVATION_EXPERIMENTS/qwen3_draft_smart_global_rate015_launch.sh
```

设置：

```text
model: /mnt/qjhs-sh-lab-01/models/Qwen3-32B
data: data/result_reflect.json
rate: 0.15
topk: 10
runtime KV: preprocess KV
reprocess_method: FusionRAG
judge: GLM-5.2 API
fixed set dir: MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_draft_smart_global_rate015_full/chunk_fixed_sets_npz
```

当前状态：2026-07-04 已启动 8 GPU 分段实验；每个 segment 顺序跑四个方法。


## Accuracy 当前结果

截至当前，`draft_smart_frequency_global` 已完成 8 个 segment，其余方法仍在跑。

| method | segments | Main Acc | Sub Acc | F1 | EM | 状态 |
|---|---:|---:|---:|---:|---:|---|
| draft_smart_frequency_global | 8/8 | 94/135 (69.63%) | 203/250 (81.20%) | 0.2138 | 0.0160 | 完成 |

参考基线：

| method | Main Acc | Sub Acc | 说明 |
|---|---:|---:|---|
| online_draft_rate015 | 99/135 (73.33%) | 209/248 (84.27%) | 当前 query-specific online DraftModel selector |
| offline_hybrid70_rate015 | 93/135 (68.89%) | 196/248 (79.03%) | 旧 offline hybrid70 fixed set |

初步结论：`draft_smart_frequency_global` 相比旧 `offline_hybrid70_rate015` 略有提升，并且 overlap 明显更接近 online draft；但仍没有达到 online draft 的准确率，说明 online draft 的 query-specific 部分仍然重要。

## Accuracy 最终结果

四个 draft_smart 方法均已完成 8 个 segment。

| method | Main Acc | Sub Acc | F1 | EM |
|---|---:|---:|---:|---:|
| draft_smart_frequency_global | 94/135 (69.63%) | 203/250 (81.20%) | 0.2138 | 0.0160 |
| draft_smart_mean_score_global | 95/135 (70.37%) | 204/250 (81.60%) | 0.1931 | 0.0080 |
| draft_smart_max_score_global | 94/135 (69.63%) | 200/250 (80.00%) | 0.2041 | 0.0160 |
| draft_smart_top2_mean_global | 94/135 (69.63%) | 201/250 (80.40%) | 0.1991 | 0.0120 |

参考基线：

| method | Main Acc | Sub Acc | 说明 |
|---|---:|---:|---|
| online_draft_rate015 | 99/135 (73.33%) | 209/248 (84.27%) | 当前 query-specific online DraftModel selector |
| offline_hybrid70_rate015 | 93/135 (68.89%) | 196/248 (79.03%) | 旧 offline hybrid70 fixed set |

最终结论：

1. 复用 online DraftModel smart selection 后处理之后，offline fixed set 明显更接近 online draft。最强 overlap 是 draft_smart_frequency_global，Jaccard 0.4429，coverage 0.6051；旧 offline_hybrid70 的 Jaccard 只有 0.3313。
2. Accuracy 也有小幅提升。最好的 accuracy 是 draft_smart_mean_score_global，Main Acc 95/135；旧 offline_hybrid70 是 93/135。
3. 但 smart offline draft 仍低于 online_draft_rate015 的 99/135，说明 query-specific online draft 仍有不可忽略的收益，单纯 offline fixed set 还不能完全复现 online draft。
4. 因此下一步如果继续沿这个方向，不应该只做 document-level fixed set；更合理的是 offline stable set + lightweight online residual selector。
