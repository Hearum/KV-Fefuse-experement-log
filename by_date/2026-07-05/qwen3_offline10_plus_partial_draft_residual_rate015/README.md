# Qwen3 Offline10 + Partial Draft Residual5

## 实验设置

- 数据集：MuSiQue reflect，完整 200 examples，其中 135 个 should_test 主问题，250 个 sub-question。
- 基座模型：Qwen3-32B。
- offline fixed set：draft_smart_frequency_global，offline rate=0.10。
- online residual：DraftModel selector residual rate=0.05。
- partial Draft selector：first12 或 middle，通过 FUSIONRAG_DRAFT_SCORE_LAYERS 指定，并在最大指定层后停止 forward。
- 总预算：offline 10% + residual 5%，用于和 online_draft_rate015 / offline10_draft005 / offline10_layer4_residual005 比较。

## 结果

| method | Main Acc | Sub Acc | F1 | EM | residual select(s) | score forward(s) | added tokens | prompt eval(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| first12 | 92/135 (68.15%) | 197/250 (78.80%) | 0.2072 | 0.0120 | 0.0524 | 0.0000 | 70.8 | 0.9139 |
| middle | 95/135 (70.37%) | 202/250 (80.80%) | 0.2044 | 0.0160 | 0.0743 | 0.0000 | 70.8 | 0.9328 |

## 对照

- offline10 + full Draft residual5：97/135 main，208/250 sub，selection 0.1406s。
- offline10 + layer4 residual5：92/135 main，198/250 sub，selection 0.0235s。
- online_draft_rate015：99/135 main，209/248 sub，selection 0.1417s。

## 初步结论

first12 和 middle 是对 low-cost Draft selector 的公平完整验证。first12 更便宜，middle selector overlap 更高；最终是否值得取决于它们相对 full Draft residual 的准确率差距和 selection time 降幅。
