# Offline vs Online Performance Summary

本目录把目前最常用的 offline/online selection 性能结果合并到一张表，方便写 motivation 和后续画图。

## 1. Qwen3-32B / MuSiQue

配置：

- Model: `/mnt/qjhs-sh-lab-01/models/Qwen3-32B`
- Dataset: MuSiQue reflect pipeline, 200 examples
- Retrieval: `topk=10`
- Runtime KV: preprocess KV
- Judge: GLM-5.2

| Method | Type | Rate / Budget | Main Acc | Sub Acc | Selection time | Prompt eval |
|---|---|---:|---:|---:|---:|---:|
| `full_rate1` | full | 1.00 | 105/135 (77.78%) | 215/250 (86.00%) | 0.0000s | 1.1683s |
| `online_qk_rate015` | online | 0.15 | 84/135 (62.22%) | 189/248 (76.21%) | 0.3230s | 1.1725s |
| `online_draft_rate015` | online | 0.15 | 99/135 (73.33%) | 209/248 (84.27%) | 0.1417s | 0.9924s |
| `offline_hybrid70_rate015` | offline | 0.15 | 93/135 (68.89%) | 196/248 (79.03%) | 0.0000s | 0.8447s |
| `draft_smart_mean_score_global` | offline | 0.15 | 95/135 (70.37%) | 204/250 (81.60%) | 0.0000s | - |
| `draft_smart_freq_boundary0p02_global` | offline | 0.15 | 97/135 (71.85%) | 203/250 (81.20%) | 0.0000s | - |
| `draft32b_smart_top2_mean_global` | offline | 0.15 | 98/135 (72.59%) | 202/248 (81.45%) | 0.0000s | 0.8949s |
| `offline10_draft005` | offline + online residual | 10% + 5% | 97/135 (71.85%) | 208/250 (83.20%) | 0.1406s | 1.0007s |
| `offline10_hybrid_old70_docgen30_draft005` | offline + online residual | 10% + 5% | 100/135 (74.07%) | 209/250 (83.60%) | - | - |
| `offline20_only` | offline | 0.20 | 91/135 (67.41%) | 198/250 (79.20%) | 0.0000s | 0.9801s |

结论：

- 纯 offline fixed set 可以去掉 online selection time，但 rate=0.15 下通常略低于 `online_draft_rate015`。
- `draft32b_smart_top2_mean_global` 已经接近 online draft 的 Main Acc，但 Sub Acc 仍低一些。
- `offline10 + online residual5` 比 `offline20_only` 更好，说明收益不只是来自更多 recompute token，而是来自 query-aware residual。
- 当前最好的 MuSiQue 组合之一是 `offline10_hybrid_old70_docgen30_draft005`，Main Acc 达到 100/135，Sub Acc 达到 209/250，接近 online draft。

## 2. Qwen3-32B / Cross-Dataset

配置：

- Model: Qwen3-32B
- Rate: 0.15 except `full_rate1`
- Datasets: 2WikiMQA, HotpotQA, TriviaQA

| Dataset | Method | Type | Main/Sub Acc | F1 | EM | Selection time | Prompt eval |
|---|---|---|---:|---:|---:|---:|---:|
| 2WikiMQA | `full_rate1` | full | 113/200 (56.50%) | 0.3319 | 0.0250 | 0.0000s | 3.8673s |
| 2WikiMQA | `online_qk_rate015` | online | 107/200 (53.50%) | 0.2767 | 0.0150 | 0.3244s | 2.2925s |
| 2WikiMQA | `online_draft_rate015` | online | 101/200 (50.50%) | 0.2687 | 0.0150 | 0.4012s | 2.4394s |
| 2WikiMQA | `offline3b_mean` | offline | 96/200 (48.00%) | 0.2806 | 0.0250 | 0.0000s | 2.0429s |
| 2WikiMQA | `offline3b_freq_boundary2` | offline | 99/200 (49.50%) | 0.2801 | 0.0200 | 0.0000s | 2.0302s |
| 2WikiMQA | `offline32b_top2` | offline | 103/200 (51.50%) | 0.2656 | 0.0150 | 0.0000s | 2.0097s |
| HotpotQA | `full_rate1` | full | 207/260 (79.62%) | 0.3076 | 0.0269 | 0.0000s | 1.2716s |
| HotpotQA | `online_qk_rate015` | online | 206/260 (79.23%) | 0.3007 | 0.0308 | 0.3337s | 1.2551s |
| HotpotQA | `online_draft_rate015` | online | 207/260 (79.62%) | 0.3245 | 0.0385 | 0.1516s | 1.0741s |
| HotpotQA | `offline3b_mean` | offline | 204/260 (78.46%) | 0.3193 | 0.0192 | 0.0000s | 0.9275s |
| HotpotQA | `offline3b_freq_boundary2` | offline | 201/260 (77.31%) | 0.3141 | 0.0308 | 0.0000s | 0.9278s |
| HotpotQA | `offline32b_top2` | offline | 197/260 (75.77%) | 0.3039 | 0.0346 | 0.0000s | 0.9274s |
| TriviaQA | `full_rate1` | full | 211/270 (78.15%) | 0.2036 | 0.0370 | 0.0000s | 1.2635s |
| TriviaQA | `online_qk_rate015` | online | 212/270 (78.52%) | 0.2343 | 0.0407 | 0.3268s | 1.2276s |
| TriviaQA | `online_draft_rate015` | online | 214/270 (79.26%) | 0.2137 | 0.0556 | 0.1520s | 1.0498s |
| TriviaQA | `offline3b_mean` | offline | 215/270 (79.63%) | 0.2330 | 0.0630 | 0.0000s | 0.9054s |
| TriviaQA | `offline3b_freq_boundary2` | offline | 218/270 (80.74%) | 0.2055 | 0.0556 | 0.0000s | 0.9047s |

结论：

- Offline 方法跨数据集不总是最强，但一般能去掉 0.15s-0.33s 的 online selection time。
- HotpotQA/TriviaQA 上 offline 的效果接近甚至超过 online selector；2WikiMQA 上 offline 仍有差距。
- 这说明 offline fixed set 的收益和数据集类型有关，不是单一数据集现象。

## 3. Qwen3-235B-A22B / Parameter Scaling

配置：

- Model: Qwen3-235B-A22B
- Rate: 0.15 except `full_rate1`
- Draft selector: Qwen2.5-3B-Instruct

| Dataset | Method | Type | Main Acc | Sub Acc | F1 | EM |
|---|---|---|---:|---:|---:|---:|
| MuSiQue | `full_rate1` | full | 106/135 (78.52%) | 215/248 (86.69%) | 0.5158 | 0.1734 |
| MuSiQue | `online_qk_rate015` | online | 74/135 (54.81%) | 179/248 (72.18%) | 0.4495 | 0.1331 |
| MuSiQue | `online_draft_rate015` | online | 93/135 (68.89%) | 204/248 (82.26%) | 0.4959 | 0.1532 |
| MuSiQue | `offline3b_mean_rate015` | offline | 83/135 (61.48%) | 193/248 (77.82%) | 0.4691 | 0.1331 |
| MuSiQue | `offline3b_freq_boundary2_rate015` | offline | 89/135 (65.93%) | 191/248 (77.02%) | 0.4739 | 0.1452 |
| MuSiQue | `offline32b_top2_rate015` | offline | 86/135 (63.70%) | 190/248 (76.61%) | 0.4705 | 0.1411 |
| 2WikiMQA | `full_rate1` | full | 113/200 (56.50%) | 113/200 (56.50%) | 0.4213 | 0.2700 |
| 2WikiMQA | `online_draft_rate015` | online | 106/200 (53.00%) | 106/200 (53.00%) | 0.4024 | 0.2400 |
| 2WikiMQA | `offline3b_mean_rate015` | offline | 102/200 (51.00%) | 102/200 (51.00%) | 0.3988 | 0.2000 |
| 2WikiMQA | `offline3b_freq_boundary2_rate015` | offline | 105/200 (52.50%) | 105/200 (52.50%) | 0.3963 | 0.2100 |
| HotpotQA | `full_rate1` | full | 221/260 (85.00%) | 221/260 (85.00%) | 0.6081 | 0.4692 |
| HotpotQA | `online_draft_rate015` | online | 211/260 (81.15%) | 211/260 (81.15%) | 0.5582 | 0.4077 |
| HotpotQA | `offline3b_mean_rate015` | offline | 218/260 (83.85%) | 218/260 (83.85%) | 0.5599 | 0.3846 |

结论：

- 在 235B 上，offline 方法仍然有效，并没有因为模型变大而失效。
- MuSiQue 上 online draft 仍明显强于 pure offline。
- HotpotQA 上 `offline3b_mean_rate015` 反而高于 online draft，说明 offline set 在某些数据集/大模型设置下很有竞争力。

## Data File

机器可读版本：

```text
offline_online_performance_summary.csv
```

主要来源：

- `MOTIVATION_EXPERIMENTS/qwen3_offline_improvement_probe_summary.md`
- `MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/cross_dataset_summary.csv`
- `MOTIVATION_EXPERIMENTS/qwen3_235b_param_scaling_cross_dataset/RESULTS_SUMMARY.md`
