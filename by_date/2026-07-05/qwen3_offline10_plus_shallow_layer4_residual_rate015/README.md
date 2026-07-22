# Qwen3 Offline10 + Layer4 Shallow Residual5

## 实验设置

- 数据集：MuSiQue reflect，完整 200 example，其中 135 个 should_test 主问题，250 个 sub-question。
- 基座模型：Qwen3-32B。
- offline fixed set：draft_smart_frequency_global，offline rate=0.10，来源 reflect_draft_smart_global_rate010_full/chunk_fixed_sets_npz。
- online residual：DraftModel shallow layer4，residual rate=0.05。
- 总预算：offline 10% + online residual 5%。
- judge：GLM-5.2 API。

## 结果

| method | Main Acc | Sub Acc | F1 | EM | residual select time(s) | added tokens | prompt eval(s) | prompt eval tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| offline10 + layer4 shallow residual5 | 92/135 (68.15%) | 198/250 (79.20%) | 0.2021 | 0.0240 | 0.0235 | 70.8 | 0.8817 | 249.5 |

## 对照解释

- offline10 + full Draft residual5：97/135 main，208/250 sub，residual select time 0.1406s。
- pure offline15 draft_smart_frequency_global：94/135 main，203/250 sub。
- offline15 + layer4 shallow residual5：96/135 main，204/250 sub，residual select time 0.0238s。

## 初步结论

在 15% 总预算下，layer4 shallow residual 明显节省 selection 开销，但质量低于 full Draft residual，也没有超过 pure offline15。因此 layer4 单层直接替代 full Draft residual 目前不够；后续应验证 first-k layer aggregation 或训练式 shallow predictor。
