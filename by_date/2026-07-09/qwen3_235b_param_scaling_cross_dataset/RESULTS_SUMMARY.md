# Qwen3-235B 参数量泛化实验结果汇总

更新时间：2026-07-13 17:16:50

## 主表

| Dataset | Method | Status/Progress | Rows | Main Acc | Sub Acc | F1 | EM | CSV |
|---|---|---|---:|---:|---:|---:|---:|---|
| musique | full_rate1 | done | 248 | 106/135 (78.52%) | 215/248 (86.69%) | 0.5158 | 0.1734 | `MOTIVATION_EXPERIMENTS/qwen3_235b_three_groups_unified_prompt/full_rate1/full_0_200/Qwen3-235B-A22B/musique/FusionRAG_global_topk10_bge/rate_1.0_revert_rope.csv` |
| musique | online_qk_rate015 | done | 248 | 74/135 (54.81%) | 179/248 (72.18%) | 0.4495 | 0.1331 | `MOTIVATION_EXPERIMENTS/qwen3_235b_current_rerun_20260709/online_qk_rate015/full_0_200/Qwen3-235B-A22B/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv` |
| musique | online_draft_rate015 | done | 248 | 93/135 (68.89%) | 204/248 (82.26%) | 0.4959 | 0.1532 | `MOTIVATION_EXPERIMENTS/qwen3_235b_current_rerun_20260709/online_draft_rate015/full_0_200/Qwen3-235B-A22B/musique/DraftModel_global_topk10_bge/rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv` |
| musique | offline3b_mean_rate015 | done | 248 | 83/135 (61.48%) | 193/248 (77.82%) | 0.4691 | 0.1331 | `results/musique/offline3b_mean_rate015/full_0_200/Qwen3-235B-A22B/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv` |
| musique | offline3b_freq_boundary2_rate015 | done | 248 | 89/135 (65.93%) | 191/248 (77.02%) | 0.4739 | 0.1452 | `results/musique/offline3b_freq_boundary2_rate015/full_0_200/Qwen3-235B-A22B/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv` |
| musique | offline32b_top2_rate015 | done | 248 | 86/135 (63.70%) | 190/248 (76.61%) | 0.4705 | 0.1411 | `results/musique/offline32b_top2_rate015/full_0_200/Qwen3-235B-A22B/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv` |
| 2wikimqa | full_rate1 | done | 200 | 113/200 (56.50%) | 113/200 (56.50%) | 0.4213 | 0.2700 | `results/2wikimqa/full_rate1/full_0_200/Qwen3-235B-A22B/2wikimqa/FusionRAG_global_topk10_bge/rate_1.0_revert_rope.csv` |
| 2wikimqa | online_qk_rate015 | done | 200 | 98/200 (49.00%) | 98/200 (49.00%) | 0.3728 | 0.2100 | `results/2wikimqa/online_qk_rate015/full_0_200/Qwen3-235B-A22B/2wikimqa/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv` |
| 2wikimqa | online_draft_rate015 | done | 200 | 106/200 (53.00%) | 106/200 (53.00%) | 0.4024 | 0.2400 | `results/2wikimqa/online_draft_rate015/full_0_200/Qwen3-235B-A22B/2wikimqa/DraftModel_global_topk10_bge/rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv` |
| 2wikimqa | offline3b_mean_rate015 | done | 200 | 102/200 (51.00%) | 102/200 (51.00%) | 0.3988 | 0.2000 | `results/2wikimqa/offline3b_mean_rate015/full_0_200/Qwen3-235B-A22B/2wikimqa/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv` |
| 2wikimqa | offline3b_freq_boundary2_rate015 | done | 200 | 105/200 (52.50%) | 105/200 (52.50%) | 0.3963 | 0.2100 | `results/2wikimqa/offline3b_freq_boundary2_rate015/full_0_200/Qwen3-235B-A22B/2wikimqa/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv` |
| 2wikimqa | offline32b_top2_rate015 | done | 200 | 103/200 (51.50%) | 103/200 (51.50%) | 0.3941 | 0.2250 | `results/2wikimqa/offline32b_top2_rate015/full_0_200/Qwen3-235B-A22B/2wikimqa/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv` |
| hotpotqa | full_rate1 | done | 260 | 221/260 (85.00%) | 221/260 (85.00%) | 0.6081 | 0.4692 | `results/hotpotqa/full_rate1/full_0_260/Qwen3-235B-A22B/hotpotqa/FusionRAG_global_topk10_bge/rate_1.0_revert_rope.csv` |
| hotpotqa | online_qk_rate015 | done | 260 | 209/260 (80.38%) | 209/260 (80.38%) | 0.5368 | 0.4000 | `results/hotpotqa/online_qk_rate015/full_0_260/Qwen3-235B-A22B/hotpotqa/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv` |
| hotpotqa | online_draft_rate015 | done | 260 | 211/260 (81.15%) | 211/260 (81.15%) | 0.5582 | 0.4077 | `results/hotpotqa/online_draft_rate015/full_0_260/Qwen3-235B-A22B/hotpotqa/DraftModel_global_topk10_bge/rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv` |
| hotpotqa | offline3b_mean_rate015 | done | 260 | 218/260 (83.85%) | 218/260 (83.85%) | 0.5599 | 0.3846 | `results/hotpotqa/offline3b_mean_rate015/full_0_260/Qwen3-235B-A22B/hotpotqa/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv` |
| hotpotqa | offline3b_freq_boundary2_rate015 | partial/running | 155 | 122/155 (78.71%) | 122/155 (78.71%) | 0.5531 | 0.3742 | `results/hotpotqa/offline3b_freq_boundary2_rate015/full_0_260/Qwen3-235B-A22B/hotpotqa/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv` |
| hotpotqa | offline32b_top2_rate015 | pending | - | - | - | - | - | `-` |
| triviaqa | full_rate1 | done | 270 | 235/270 (87.04%) | 235/270 (87.04%) | 0.6139 | 0.4778 | `results/triviaqa/full_rate1/full_0_270/Qwen3-235B-A22B/triviaqa/FusionRAG_global_topk10_bge/rate_1.0_revert_rope.csv` |
| triviaqa | online_qk_rate015 | done | 270 | 222/270 (82.22%) | 222/270 (82.22%) | 0.5755 | 0.4444 | `results/triviaqa/online_qk_rate015/full_0_270/Qwen3-235B-A22B/triviaqa/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv` |
| triviaqa | online_draft_rate015 | partial/running | 85 | 71/85 (83.53%) | 71/85 (83.53%) | 0.6500 | 0.5412 | `results/triviaqa/online_draft_rate015/full_0_270/Qwen3-235B-A22B/triviaqa/DraftModel_global_topk10_bge/rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv` |
| triviaqa | offline3b_mean_rate015 | pending | - | - | - | - | - | `-` |
| triviaqa | offline3b_freq_boundary2_rate015 | partial/running | 0 | - | - | - | - | `results/triviaqa/offline3b_freq_boundary2_rate015/full_0_270/Qwen3-235B-A22B/triviaqa/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv` |
| triviaqa | offline32b_top2_rate015 | pending | - | - | - | - | - | `-` |

## 方法说明

- `full_rate1`：rate=1.0，作为 full recompute/full attention 风格空白对照。
- `online_qk_rate015`：FusionRAG online QK selector，online 按真实 query 选 15% token。
- `online_draft_rate015`：online DraftModel selector，3B draft model 按真实 query 选 15% token。
- `offline3b_mean_rate015`：纯 offline fixed set，3B draft smart mean-score global，15%。
- `offline3b_freq_boundary2_rate015`：纯 offline fixed set，3B draft smart frequency + 2% boundary compensation，15%。
- `offline32b_top2_rate015`：纯 offline fixed set，32B teacher top2-mean global，15%。

## 统计口径

- CSV 先按 `(Main Question, Sub Question)` 去重，避免中断重跑产生重复行。
- `Sub Acc` 是去重后的 sub-question 级别正确率。
- `Main Acc` 是同一 `Main Question` 下所有 sub-question 都正确才计为正确。
- `F1/EM` 是去重后所有 sub-question 的平均值。
