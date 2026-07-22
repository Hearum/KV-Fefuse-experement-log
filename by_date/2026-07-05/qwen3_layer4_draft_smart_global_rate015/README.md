# Qwen3 Layer4 Draft Smart Global Rate015

## 实验设置

- 数据集：MuSiQue reflect，完整 200 examples，其中 135 个 should_test 主问题，250 个 sub-question。
- 基座模型：Qwen3-32B。
- fixed set 来源：reflect_draft_layer4_smart_global_rate015_full/chunk_fixed_sets_npz。
- calibration score cache：只用 DraftModel 第 4 层打分，并在第 4 层后停止前向；calibration query 来自其他 example，不使用真实测试 query。
- 推理：preprocess KV，offline fixed set rate=0.15，无 online residual。
- judge：GLM-5.2 API。

## 结果

| method | Main Acc | Sub Acc | F1 | EM | selection time(s) | prompt eval(s) | prompt eval tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| draft_smart_frequency_global | 87/135 (64.44%) | 190/250 (76.00%) | 0.1994 | 0.0040 | 0.0000 | 0.8103 | 225.4 |
| draft_smart_mean_score_global | 84/135 (62.22%) | 190/250 (76.00%) | 0.1748 | 0.0000 | 0.0000 | 0.8104 | 226.7 |

## 初步结论

这组实验检验单层 layer4 是否也能做 offline smart fixed set。它和 full Draft smart 的聚合逻辑一致，唯一差异是 calibration score 只来自 DraftModel 第 4 层。
