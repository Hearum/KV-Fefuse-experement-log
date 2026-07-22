# Qwen3 Offline Smart + Online Residual 实验记录

## 目的

前一轮 `draft_smart_*` 纯 offline fixed set 能提升 offline selector，但仍低于 online draft。Residual 诊断显示：offline smart set 覆盖 online draft 后，仍剩 39%-45% 的 online-draft tokens 没覆盖，且 residual 分散在多个 chunk。

本实验验证：在 offline fixed set 基础上，额外用轻量 online residual selector 补 5% doc tokens，是否能接近 online draft 的效果。

## 代码开关

新增环境变量，不设置时默认行为不变：

```bash
FUSIONRAG_RESIDUAL_ONLINE_RATE=0.05
FUSIONRAG_RESIDUAL_ONLINE_METHOD=DraftModel  # or FusionRAG
```

实现位置：

```text
ktransformers/util/utils.py
```

逻辑：

```text
offline fixed set -> base selected tokens
online residual selector -> 从 base 外补 residual_rate * doc_len 个 tokens
union(base, residual) -> online recompute
```

## 实验配置

启动脚本：

```text
MOTIVATION_EXPERIMENTS/qwen3_offline_smart_plus_online_residual_rate015_launch.sh
```

固定设置：

```text
model: /mnt/qjhs-sh-lab-01/models/Qwen3-32B
runtime KV: preprocess KV
base rate: 0.15
residual rate: 0.05
topk: 10
judge: GLM-5.2
```

四个配置：

| config | offline fixed set | residual selector |
|---|---|---|
| mean_draft005 | draft_smart_mean_score_global | DraftModel |
| freq_draft005 | draft_smart_frequency_global | DraftModel |
| mean_qk005 | draft_smart_mean_score_global | FusionRAG-QK |
| freq_qk005 | draft_smart_frequency_global | FusionRAG-QK |

## 当前状态

2026-07-04 已启动 8 GPU 分段实验。

## 最终结果汇总

说明：本实验的新增组别是在 offline fixed set 基础上，再额外在线挑选 residual 5% doc tokens 加入重算集合。baseline 行来自前序完整实验记录，用于对照。

| 方法 | Main Acc | Sub Acc | F1 | EM | 平均 residual tokens | residual 选择耗时(s) | 平均 prompt eval(s) | 平均 prompt tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| online_draft_rate015 | 99/135 (73.33%) | 209/248 (84.27%) | - | - | - | - | - | - |
| offline_hybrid70_rate015 | 93/135 (68.89%) | 196/248 (79.03%) | - | - | - | - | - | - |
| draft_smart_frequency_global | 94/135 (69.63%) | 203/250 (81.20%) | 0.2138 | 0.0160 | 0.0 | 0.0000 | - | - |
| draft_smart_mean_score_global | 95/135 (70.37%) | 204/250 (81.60%) | 0.1931 | 0.0080 | 0.0 | 0.0000 | - | - |
| mean_draft005 | 100/135 (74.07%) | 213/250 (85.20%) | 0.1899 | 0.0040 | 70.8 | 0.1408 | 1.1046 | 321.4 |
| freq_draft005 | 103/135 (76.30%) | 217/250 (86.80%) | 0.1884 | 0.0120 | 70.8 | 0.1410 | 1.1100 | 323.4 |
| mean_qk005 | 98/135 (72.59%) | 204/250 (81.60%) | 0.2005 | 0.0120 | 70.8 | 0.3215 | 1.2839 | 321.4 |
| freq_qk005 | 94/135 (69.63%) | 197/250 (78.80%) | 0.2047 | 0.0160 | 70.8 | 0.3221 | 1.2898 | 323.4 |

### 结论

- offline smart draft fixed set 已经比旧的 hybrid70 更好，但仍低于 online draft。
- 在 fixed set 上额外补 5% online residual 后，Draft residual 明显优于 QK residual；frequency fixed set + Draft residual 当前是这一组里最好的组合。
- Draft residual 的额外选择耗时约 0.141s，QK residual 约 0.322s；在这套实现里 Draft residual 不仅质量更高，选择开销也更低。
- 这说明固定集合不是没有价值，但纯 offline set 仍存在 query-specific 缺口；一个可行方向是 offline stable set 覆盖主体 token，online residual 只补少量 query-dependent token。
