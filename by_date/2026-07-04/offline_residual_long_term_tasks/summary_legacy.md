# Qwen3 Offline Selector Improvement Probe

<!-- QWEN3_UNIFIED_TABLE_START -->

## Unified Main Table: Qwen3-32B / MuSiQue / Reflect Pipeline

固定配置：`model=/mnt/qjhs-sh-lab-01/models/Qwen3-32B`，数据集为 MuSiQue 200 examples 的 reflect pipeline，retrieval `topk=10`，runtime 使用 preprocess KV，答案 judge 使用 GLM-5.2。除特别标注外，rate 表示文档 token 中被 online 更新/recompute 的比例。

注意：早期几轮 summary 有 `248` 个 judged sub-questions，后续 rerun/新实验 CSV 中有 `250` 个 sub-questions。主表保留每轮实验的原始分母，不强行归一化。

| group | method | rate / setting | Main Acc | Sub Acc | F1 | EM | selection time(s) | prompt eval(s) | source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| upper baseline | full_rate1 | 1.00 full recompute | 105/135 (77.78%) | 215/250 (86.00%) | 0.1938 | 0.0080 | 0.0000 | 1.1683 | `qwen3_hybrid70_online_baselines/full_rate1`, raw CSV re-aggregated |
| online selector | online_qk_rate015 | 0.15 | 84/135 (62.22%) | 189/248 (76.21%) | 0.2076 | 0.0081 | 0.3230 | 1.1725 | `qwen3_rate015_online_offline/summary.csv` |
| online selector | online_draft_rate015 | 0.15 | 99/135 (73.33%) | 209/248 (84.27%) | 0.1905 | 0.0081 | 0.1417 | 0.9924 | `qwen3_rate015_online_offline/summary.csv` |
| offline fixed set | offline_hybrid70_rate015 | 0.15 | 93/135 (68.89%) | 196/248 (79.03%) | 0.2064 | 0.0081 | 0.0000 | 0.8447 | `qwen3_rate015_online_offline/summary.csv` |
| offline fixed set | draft_frequency_per_chunk | 0.15 | 88/135 (65.19%) | 192/248 (77.42%) | - | - | - | - | early offline draft aggregation attempt |
| offline fixed set | draft_mean_score_per_chunk | 0.15 | 88/135 (65.19%) | 191/248 (77.02%) | - | - | - | - | early offline draft aggregation attempt |
| offline fixed set | draft_max_score_per_chunk | 0.15 | 92/135 (68.15%) | 195/248 (78.63%) | - | - | - | - | early offline draft aggregation attempt |
| offline fixed set | draft_top2_mean_score_per_chunk | 0.15 | 92/135 (68.15%) | 195/248 (78.63%) | - | - | - | - | early offline draft aggregation attempt |
| offline fixed set | draft_top4_mean_score_per_chunk | 0.15 | 90/135 (66.67%) | 193/248 (77.82%) | - | - | - | - | early offline draft aggregation attempt |
| offline fixed set | draft_smart_frequency_global | 0.15 | 94/135 (69.63%) | 203/250 (81.20%) | 0.2138 | 0.0160 | 0.0000 | - | `qwen3_draft_smart_global_rate015/accuracy_summary.csv` |
| offline fixed set | draft_smart_mean_score_global | 0.15 | 95/135 (70.37%) | 204/250 (81.60%) | 0.1931 | 0.0080 | 0.0000 | - | `qwen3_draft_smart_global_rate015/accuracy_summary.csv` |
| offline fixed set | draft_smart_max_score_global | 0.15 | 94/135 (69.63%) | 200/250 (80.00%) | 0.2041 | 0.0160 | 0.0000 | - | `qwen3_draft_smart_global_rate015/accuracy_summary.csv` |
| offline fixed set | draft_smart_top2_mean_global | 0.15 | 94/135 (69.63%) | 201/250 (80.40%) | 0.1991 | 0.0120 | 0.0000 | - | `qwen3_draft_smart_global_rate015/accuracy_summary.csv` |
| offline fixed set / 32B teacher | draft32b_smart_frequency_global | 0.15, 32B teacher replaces 3B offline DraftModel | 95/135 (70.37%) | 201/248 (81.05%) | 0.1858 | 0.0040 | 0.0000 | 0.9035 | `offline_draft32b_teacher_rate015/accuracy_summary_control_qwen3_32b.csv` |
| offline fixed set / 32B teacher | draft32b_smart_mean_score_global | 0.15, 32B teacher replaces 3B offline DraftModel | 97/135 (71.85%) | 202/248 (81.45%) | 0.1991 | 0.0121 | 0.0000 | 0.9003 | `offline_draft32b_teacher_rate015/accuracy_summary_control_qwen3_32b.csv` |
| offline fixed set / 32B teacher | draft32b_smart_max_score_global | 0.15, 32B teacher replaces 3B offline DraftModel | 95/135 (70.37%) | 199/248 (80.24%) | 0.1926 | 0.0242 | 0.0000 | 0.8961 | `offline_draft32b_teacher_rate015/accuracy_summary_control_qwen3_32b.csv` |
| offline fixed set / 32B teacher | draft32b_smart_top2_mean_global | 0.15, 32B teacher replaces 3B offline DraftModel | 98/135 (72.59%) | 202/248 (81.45%) | 0.1942 | 0.0161 | 0.0000 | 0.8949 | `offline_draft32b_teacher_rate015/accuracy_summary_control_qwen3_32b.csv` |
| offline boundary mix | draft_smart_freq_boundary0p02_global | 0.15 = 13% freq + 2% boundary | 97/135 (71.85%) | 203/250 (81.20%) | 0.1987 | 0.0120 | 0.0000 | - | MOTIVATION_EXPERIMENTS/offline_boundary_mix_rate015/accuracy_summary.csv |
| offline boundary mix | draft_smart_mean_boundary0p02_global | 0.15 = 13% mean + 2% boundary | 96/135 (71.11%) | 202/250 (80.80%) | 0.1864 | 0.0120 | 0.0000 | - | MOTIVATION_EXPERIMENTS/offline_boundary_mix_rate015/accuracy_summary.csv |
| offline boundary mix | draft_smart_freq_boundary0p03_global | 0.15 = 12% freq + 3% boundary | 95/135 (70.37%) | 199/250 (79.60%) | 0.1994 | 0.0120 | 0.0000 | - | MOTIVATION_EXPERIMENTS/offline_boundary_mix_rate015/accuracy_summary.csv |
| offline boundary mix | draft_smart_mean_boundary0p03_global | 0.15 = 12% mean + 3% boundary | 96/135 (71.11%) | 200/250 (80.00%) | 0.2068 | 0.0120 | 0.0000 | - | MOTIVATION_EXPERIMENTS/offline_boundary_mix_rate015/accuracy_summary.csv |
| offline boundary mix | draft_smart_freq_boundary0p05_global | 0.15 = 10% freq + 5% boundary | 94/135 (69.63%) | 199/250 (79.60%) | 0.1969 | 0.0120 | 0.0000 | - | MOTIVATION_EXPERIMENTS/offline_boundary_mix_rate015/accuracy_summary.csv |
| offline boundary mix | draft_smart_mean_boundary0p05_global | 0.15 = 10% mean + 5% boundary | 94/135 (69.63%) | 200/250 (80.00%) | 0.2054 | 0.0240 | 0.0000 | - | MOTIVATION_EXPERIMENTS/offline_boundary_mix_rate015/accuracy_summary.csv |
| offline boundary mix / 32B teacher | draft32b_smart_freq_boundary0p02_global | 0.15 = 13% 32B-freq + 2% boundary | 93/135 (68.89%) | 198/248 (79.84%) | 0.2008 | 0.0121 | 0.0000 | 0.8936 | `offline_draft32b_teacher_rate015/accuracy_summary_control_qwen3_32b_boundary_mix.csv` |
| offline boundary mix / 32B teacher | draft32b_smart_mean_boundary0p02_global | 0.15 = 13% 32B-mean + 2% boundary | 97/135 (71.85%) | 201/248 (81.05%) | 0.1869 | 0.0081 | 0.0000 | 0.8902 | `offline_draft32b_teacher_rate015/accuracy_summary_control_qwen3_32b_boundary_mix.csv` |
| offline boundary mix / 32B teacher | draft32b_smart_top2_boundary0p02_global | 0.15 = 13% 32B-top2 + 2% boundary | 94/135 (69.63%) | 199/248 (80.24%) | 0.2099 | 0.0242 | 0.0000 | 0.8931 | `offline_draft32b_teacher_rate015/accuracy_summary_control_qwen3_32b_boundary_mix.csv` |
| offline boundary mix / 32B teacher | draft32b_smart_freq_boundary0p03_global | 0.15 = 12% 32B-freq + 3% boundary | 94/135 (69.63%) | 199/248 (80.24%) | 0.2088 | 0.0121 | 0.0000 | 0.8936 | `offline_draft32b_teacher_rate015/accuracy_summary_control_qwen3_32b_boundary_mix.csv` |
| offline boundary mix / 32B teacher | draft32b_smart_mean_boundary0p03_global | 0.15 = 12% 32B-mean + 3% boundary | 96/135 (71.11%) | 202/248 (81.45%) | 0.1837 | 0.0121 | 0.0000 | 0.8930 | `offline_draft32b_teacher_rate015/accuracy_summary_control_qwen3_32b_boundary_mix.csv` |
| offline boundary mix / 32B teacher | draft32b_smart_top2_boundary0p03_global | 0.15 = 12% 32B-top2 + 3% boundary | 89/135 (65.93%) | 194/248 (78.23%) | 0.2014 | 0.0161 | 0.0000 | 0.8880 | `offline_draft32b_teacher_rate015/accuracy_summary_control_qwen3_32b_boundary_mix.csv` |
| offline boundary mix / 32B teacher | draft32b_smart_freq_boundary0p05_global | 0.15 = 10% 32B-freq + 5% boundary | 93/135 (68.89%) | 199/248 (80.24%) | 0.1798 | 0.0202 | 0.0000 | 0.8932 | `offline_draft32b_teacher_rate015/accuracy_summary_control_qwen3_32b_boundary_mix.csv` |
| offline boundary mix / 32B teacher | draft32b_smart_mean_boundary0p05_global | 0.15 = 10% 32B-mean + 5% boundary | 97/135 (71.85%) | 202/248 (81.45%) | 0.1997 | 0.0081 | 0.0000 | 0.8863 | `offline_draft32b_teacher_rate015/accuracy_summary_control_qwen3_32b_boundary_mix.csv` |
| offline boundary mix / 32B teacher | draft32b_smart_top2_boundary0p05_global | 0.15 = 10% 32B-top2 + 5% boundary | 98/135 (72.59%) | 202/248 (81.45%) | 0.1875 | 0.0121 | 0.0000 | 0.8853 | `offline_draft32b_teacher_rate015/accuracy_summary_control_qwen3_32b_boundary_mix.csv` |
| offline fixed set | layer4_draft_smart_frequency_global | 0.15, layer4-only calibration | 87/135 (64.44%) | 190/250 (76.00%) | 0.1994 | 0.0040 | 0.0000 | 0.8103 | `qwen3_layer4_draft_smart_global_rate015/accuracy_summary.csv` |
| offline fixed set | layer4_draft_smart_mean_score_global | 0.15, layer4-only calibration | 84/135 (62.22%) | 190/250 (76.00%) | 0.1748 | 0.0000 | 0.0000 | 0.8104 | `qwen3_layer4_draft_smart_global_rate015/accuracy_summary.csv` |
| diagnostic oracle | online_draft_trace_oracle_fixed | 0.15, leaky trace fixed set | 82/135 (60.74%) | 186/248 (75.00%) | - | - | - | - | true online draft traces aggregated into fixed set; diagnostic only |
| path ablation | offline_hybrid70_draftpath | 0.15, same fixed set but DraftModel recompute path | 92/135 (68.15%) | 194/248 (78.23%) | - | - | - | - | `qwen3_offline_hybrid70_draftpath_rate015` |
| offline + online residual | mean_draft005 | base 0.15 + Draft residual 0.05 | 100/135 (74.07%) | 213/250 (85.20%) | 0.1899 | 0.0040 | 0.1408 | 1.1046 | `qwen3_offline_smart_plus_online_residual_rate015/accuracy_summary.csv` |
| offline + online residual | freq_draft005 | base 0.15 + Draft residual 0.05 | 103/135 (76.30%) | 217/250 (86.80%) | 0.1884 | 0.0120 | 0.1410 | 1.1100 | `qwen3_offline_smart_plus_online_residual_rate015/accuracy_summary.csv` |
| offline + online residual | mean_qk005 | base 0.15 + QK residual 0.05 | 98/135 (72.59%) | 204/250 (81.60%) | 0.2005 | 0.0120 | 0.3215 | 1.2839 | `qwen3_offline_smart_plus_online_residual_rate015/accuracy_summary.csv` |
| offline + online residual | freq_qk005 | base 0.15 + QK residual 0.05 | 94/135 (69.63%) | 197/250 (78.80%) | 0.2047 | 0.0160 | 0.3221 | 1.2898 | `qwen3_offline_smart_plus_online_residual_rate015/accuracy_summary.csv` |
| offline + shallow residual | freq_shallow_layer4_residual005 | base 0.15 + shallow Draft layer4 residual 0.05 | 96/135 (71.11%) | 204/250 (81.60%) | 0.2034 | 0.0120 | 0.0238 | 0.9929 | `qwen3_offline_freq_plus_shallow_layer4_residual_rate015/accuracy_summary.csv` |
| fair budget | offline10_draft005 | offline 10% + Draft residual 5% | 97/135 (71.85%) | 208/250 (83.20%) | 0.1955 | 0.0200 | 0.1406 | 1.0007 | `qwen3_fair_budget_offline_vs_residual/accuracy_summary.csv` |
| fair budget | offline20_only | offline 20% only | 91/135 (67.41%) | 198/250 (79.20%) | 0.2069 | 0.0120 | 0.0000 | 0.9801 | `qwen3_fair_budget_offline_vs_residual/accuracy_summary.csv` |
| fair budget shallow | offline10_layer4_residual005 | offline 10% + shallow Draft layer4 residual 5% | 92/135 (68.15%) | 198/250 (79.20%) | 0.2021 | 0.0240 | 0.0235 | 0.8817 | `qwen3_offline10_plus_shallow_layer4_residual_rate015/accuracy_summary.csv` |
| fair budget trained attention | offline10_hs_attn005_epoch003 | offline 10% + FusionRAG-HS finetuned layer4 attention residual 5% | 93/135 (68.89%) | 201/248 (81.05%) | 0.2096 | 0.0161 | 0.1192 | 0.9886 | `qwen3_offline10_plus_hs_attn_residual_rate015/accuracy_summary.csv` |
| fair budget trained scorer | offline10_hs005_epoch003 | offline 10% + FusionRAG-HS hidden scorer residual 5% | 87/135 (64.44%) | 192/248 (77.42%) | 0.2000 | 0.0161 | 0.0540 | 0.9249 | `qwen3_offline10_plus_hs_residual_rate015/accuracy_summary.csv` |
| fair budget partial draft | offline10_first12_residual005 | offline 10% + first12 Draft residual 5% | 92/135 (68.15%) | 197/250 (78.80%) | 0.2072 | 0.0120 | 0.0524 | 0.9139 | `qwen3_offline10_plus_partial_draft_residual_rate015/accuracy_summary.csv` |
| fair budget partial draft | offline10_middle_residual005 | offline 10% + middle Draft residual 5% | 95/135 (70.37%) | 202/250 (80.80%) | 0.2044 | 0.0160 | 0.0743 | 0.9328 | `qwen3_offline10_plus_partial_draft_residual_rate015/accuracy_summary.csv` |
| fair budget docgen | offline10_docgen_draft005 | docgen offline 10% + Draft residual 5% | 99/135 (73.33%) | 207/250 (82.80%) | 0.2033 | 0.0080 | 0.1455 | - | `offline_doc_generated_query_calibration/qwen3_docgen_fair_budget/accuracy_summary_offline10_docgen_draft005.json` |
| fair budget docgen hybrid | offline10_hybrid_old90_docgen10_draft005 | old offline10 keep 90% + docgen rank fill 10% + Draft residual 5% | 95/135 (70.37%) | 205/250 (82.00%) | 0.1928 | 0.0200 | - | - | `offline_doc_generated_query_calibration/qwen3_docgen_fair_budget/accuracy_summary_offline10_hybrid_old90_docgen10_draft005.json` |
| fair budget docgen hybrid | offline10_hybrid_old70_docgen30_draft005 | old offline10 keep 70% + docgen rank fill 30% + Draft residual 5% | 100/135 (74.07%) | 209/250 (83.60%) | 0.1837 | 0.0160 | - | - | `offline_doc_generated_query_calibration/qwen3_docgen_fair_budget/accuracy_summary_offline10_hybrid_old70_docgen30_draft005.json` |
| fair budget docgen hybrid | offline10_hybrid_old50_docgen50_draft005 | old offline10 keep 50% + docgen rank fill 50% + Draft residual 5% | 95/135 (70.37%) | 203/250 (81.20%) | 0.2050 | 0.0160 | - | - | `offline_doc_generated_query_calibration/qwen3_docgen_fair_budget/accuracy_summary_offline10_hybrid_old50_docgen50_draft005.json` |
| fair budget docgen | offline15_docgen_only | docgen offline 15% only, no online residual | 87/135 (64.44%) | 192/250 (76.80%) | 0.1699 | 0.0080 | 0.0000 | - | `offline_doc_generated_query_calibration/qwen3_docgen_fair_budget/accuracy_summary_offline15_docgen_only.json` |
| fair budget docgen | offline15_docgen_mean_score_only | docgen mean-score offline 15% only, no online residual | 84/135 (62.22%) | 188/250 (75.20%) | 0.1212 | 0.0000 | 0.0000 | - | `offline_doc_generated_query_calibration/qwen3_docgen_fair_budget/accuracy_summary_offline15_docgen_mean_score_only.json` |
| rate sweep | online_qk_rate050 | 0.50 | 90/135 (66.67%) | 199/248 (80.24%) | 0.2152 | 0.0161 | 0.3183 | 2.0366 | `qwen3_hybrid70_online_baselines/summary.csv` |
| rate sweep | online_draft_rate050 | 0.50 | 108/135 (80.00%) | 219/248 (88.31%) | 0.2091 | 0.0161 | 0.1411 | 1.8645 | `qwen3_hybrid70_online_baselines/summary.csv` |
| rate sweep | offline_hybrid70_rate050 | 0.50 | 101/135 (74.81%) | 208/248 (83.87%) | 0.1950 | 0.0202 | 0.0000 | 1.7223 | `qwen3_hybrid70_online_baselines/summary.csv` |
| cross-dataset generalization | full_rate1 | 2wikimqa, rate=1.00; rate=1.0 full recompute baseline; no selector. | 113/200 (56.50%) | 113/200 (56.50%) | 0.3319 | 0.0250 | 0.0000 | 3.8673 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |
| cross-dataset generalization | online_qk_rate015 | 2wikimqa, rate=0.15; online FusionRAG-QK selector, rate=0.15. | 107/200 (53.50%) | 107/200 (53.50%) | 0.2767 | 0.0150 | 0.3244 | 2.2925 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |
| cross-dataset generalization | online_draft_rate015 | 2wikimqa, rate=0.15; online DraftModel selector with Qwen2.5-3B-Instruct, rate=0.15. | 101/200 (50.50%) | 101/200 (50.50%) | 0.2687 | 0.0150 | 0.4012 | 2.4394 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |
| cross-dataset generalization | offline3b_mean | 2wikimqa, rate=0.15; offline fixed set from 3B teacher mean-score aggregation, rate=0.15. | 96/200 (48.00%) | 96/200 (48.00%) | 0.2806 | 0.0250 | 0.0000 | 2.0429 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |
| cross-dataset generalization | offline3b_freq_boundary2 | 2wikimqa, rate=0.15; offline 3B frequency fixed set with 2% boundary replacement, total rate=0.15. | 99/200 (49.50%) | 99/200 (49.50%) | 0.2801 | 0.0200 | 0.0000 | 2.0302 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |
| cross-dataset generalization | offline32b_top2 | 2wikimqa, rate=0.15; offline fixed set from 32B teacher top2-mean aggregation, rate=0.15. | 103/200 (51.50%) | 103/200 (51.50%) | 0.2656 | 0.0150 | 0.0000 | 2.0097 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |
| cross-dataset generalization | full_rate1 | hotpotqa, rate=1.00; rate=1.0 full recompute baseline; no selector. | 207/260 (79.62%) | 207/260 (79.62%) | 0.3076 | 0.0269 | 0.0000 | 1.2716 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |
| cross-dataset generalization | online_qk_rate015 | hotpotqa, rate=0.15; online FusionRAG-QK selector, rate=0.15. | 206/260 (79.23%) | 206/260 (79.23%) | 0.3007 | 0.0308 | 0.3337 | 1.2551 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |
| cross-dataset generalization | online_draft_rate015 | hotpotqa, rate=0.15; online DraftModel selector with Qwen2.5-3B-Instruct, rate=0.15. | 207/260 (79.62%) | 207/260 (79.62%) | 0.3245 | 0.0385 | 0.1516 | 1.0741 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |
| cross-dataset generalization | offline3b_mean | hotpotqa, rate=0.15; offline fixed set from 3B teacher mean-score aggregation, rate=0.15. | 204/260 (78.46%) | 204/260 (78.46%) | 0.3193 | 0.0192 | 0.0000 | 0.9275 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |
| cross-dataset generalization | offline3b_freq_boundary2 | hotpotqa, rate=0.15; offline 3B frequency fixed set with 2% boundary replacement, total rate=0.15. | 201/260 (77.31%) | 201/260 (77.31%) | 0.3141 | 0.0308 | 0.0000 | 0.9278 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |
| cross-dataset generalization | offline32b_top2 | hotpotqa, rate=0.15; offline fixed set from 32B teacher top2-mean aggregation, rate=0.15. | 197/260 (75.77%) | 197/260 (75.77%) | 0.3039 | 0.0346 | 0.0000 | 0.9274 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |
| cross-dataset generalization | full_rate1 | triviaqa, rate=1.00; rate=1.0 full recompute baseline; no selector. | 211/270 (78.15%) | 211/270 (78.15%) | 0.2036 | 0.0370 | 0.0000 | 1.2635 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |
| cross-dataset generalization | online_qk_rate015 | triviaqa, rate=0.15; online FusionRAG-QK selector, rate=0.15. | 212/270 (78.52%) | 212/270 (78.52%) | 0.2343 | 0.0407 | 0.3268 | 1.2276 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |
| cross-dataset generalization | online_draft_rate015 | triviaqa, rate=0.15; online DraftModel selector with Qwen2.5-3B-Instruct, rate=0.15. | 214/270 (79.26%) | 214/270 (79.26%) | 0.2137 | 0.0556 | 0.1520 | 1.0498 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |
| cross-dataset generalization | offline3b_mean | triviaqa, rate=0.15; offline fixed set from 3B teacher mean-score aggregation, rate=0.15. | 215/270 (79.63%) | 215/270 (79.63%) | 0.2330 | 0.0630 | 0.0000 | 0.9054 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |
| cross-dataset generalization | offline3b_freq_boundary2 | triviaqa, rate=0.15; offline 3B frequency fixed set with 2% boundary replacement, total rate=0.15. | 218/270 (80.74%) | 218/270 (80.74%) | 0.2055 | 0.0556 | 0.0000 | 0.9047 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |
| cross-dataset generalization | offline32b_top2 | triviaqa, rate=0.15; offline fixed set from 32B teacher top2-mean aggregation, rate=0.15. Incomplete: 200/270 rows, traceback=6. | 156/200 (78.00%) | 156/200 (78.00%) | 0.2186 | 0.0500 | 0.0000 | 0.9091 | `cross_dataset_offline_generalization/cross_dataset_summary.csv` |


### Method Meaning And Data Source

- `cross-dataset generalization`: cross-dataset accuracy rerun on converted reflect-format 2WikiMQA, HotpotQA, and TriviaQA. It compares full recompute, online QK, online DraftModel, and pure-offline fixed sets at rate 0.15 using the same Qwen3-32B reflect pipeline. Source: `MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/cross_dataset_summary.csv`. Rows with `offline_qk_mean` / `offline_qk_mean_boundary2` were only pending in that experiment and are intentionally not inserted into the main table as results. TriviaQA `offline32b_top2` is partial only: 200/270 rows with traceback=6, so compare it cautiously.
- `full_rate1`: full document-token recompute / full attention upper baseline. It recomputes all document tokens online and does not use a selector. Source: `MOTIVATION_EXPERIMENTS/qwen3_hybrid70_online_baselines/full_rate1`; I re-aggregated the raw 8 segment CSVs and got 250 sub-question rows.
- `online_qk_rate015`: original FusionRAG-QK online selector. For each query, it computes importance from the current query and cached KV, then recomputes the top 15% document tokens. Source: `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/summary.csv`.
- `online_draft_rate015`: online DraftModel selector. It uses the draft model path to select query-dependent recompute tokens at rate 0.15. Source: `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/summary.csv`.
- `offline_hybrid70_rate015`: old offline fixed set. It precomputes a fixed per-chunk token set using a hybrid score, approximately draft 70% + QK 30%, then uses that fixed set at inference. Source: `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/summary.csv`.
- `draft_*_per_chunk`: early offline draft score aggregation baselines. They aggregate calibration draft scores inside each chunk using frequency, mean, max, top-k mean, etc. Source: this document's early Attempt 1 records.
- `draft_smart_*_global`: newer smart offline draft fixed set. It first applies an online-style smart draft selection on calibration queries, then aggregates selected tokens globally per document/chunk into a fixed set. Source: `MOTIVATION_EXPERIMENTS/qwen3_draft_smart_global_rate015/accuracy_summary.csv`.
- `draft32b_smart_*_global`: 32B-teacher version of `draft_smart_*_global`. It keeps the same pure-offline smart aggregation and same other-example calibration-query setting, but replaces the offline DraftModel teacher from Qwen2.5-3B-Instruct with Qwen3-32B. Online inference still uses FusionRAG + offline fixed set at rate 0.15, with no online Draft selector. Source: `MOTIVATION_EXPERIMENTS/offline_draft32b_teacher_rate015/README.md`.
- `layer4_draft_smart_*_global`: shallow-layer offline smart fixed sets. They use the same calibration-query smart aggregation as `draft_smart_*_global`, but the calibration score cache is computed only from DraftModel layer 4 and stops after layer 4. Source: `MOTIVATION_EXPERIMENTS/qwen3_layer4_draft_smart_global_rate015/accuracy_summary.csv`.
- `online_draft_trace_oracle_fixed`: diagnostic leaky oracle. It uses evaluated online draft traces to build a fixed set, so it is not a valid deployment method; it only tests whether a fixed set can preserve online draft behavior. Source: this document's Attempt 3.
- `offline_hybrid70_draftpath`: path ablation. It uses the same offline hybrid fixed set but changes recompute path to DraftModel, testing whether the gap comes from implementation path rather than selected tokens. Source: this document's Attempt 4 / `MOTIVATION_EXPERIMENTS/qwen3_offline_hybrid70_draftpath_rate015`.
- `mean_draft005` / `freq_draft005`: offline smart fixed set at base rate 0.15 plus an additional online DraftModel residual selector at rate 0.05. Final recompute set is `offline fixed tokens union online residual tokens`. Source: `MOTIVATION_EXPERIMENTS/qwen3_offline_smart_plus_online_residual_rate015/accuracy_summary.csv`.
- `mean_qk005` / `freq_qk005`: same residual idea, but the extra 5% online residual tokens are selected by FusionRAG-QK rather than DraftModel. Source: `MOTIVATION_EXPERIMENTS/qwen3_offline_smart_plus_online_residual_rate015/accuracy_summary.csv`.
- `offline10_layer4_residual005`: fair-budget shallow residual control. It uses offline `draft_smart_frequency_global` at rate 0.10 plus online shallow DraftModel layer4 residual at rate 0.05. Source: `MOTIVATION_EXPERIMENTS/qwen3_offline10_plus_shallow_layer4_residual_rate015/accuracy_summary.csv`.
- `offline10_hs_attn005_epoch003`: fair-budget trained-attention residual. It uses offline `draft_smart_frequency_global` at rate 0.10 plus online residual at rate 0.05 selected by the finetuned HS first-4-layer attention score (`FUSIONRAG_HS_SCORE_MODE=attn_prob`). It avoids the hidden scorer head and forwards each chunk separately to keep doc/query boundaries correct. Source: `MOTIVATION_EXPERIMENTS/qwen3_offline10_plus_hs_attn_residual_rate015/accuracy_summary.csv`.
- `offline10_hs005_epoch003`: fair-budget trained hidden-scorer residual. It uses offline `draft_smart_frequency_global` at rate 0.10 plus online FusionRAG-HS hidden scorer residual at rate 0.05. FusionRAG-HS is trained to mimic full DraftModel selector scores with a query-aware hidden scoring head; it is not the causal-LM draft distillation path. Source: `MOTIVATION_EXPERIMENTS/qwen3_offline10_plus_hs_residual_rate015/accuracy_summary.csv`.
- `offline10_docgen_draft005`: fair-budget doc-generated offline set. For each document chunk, GLM generates 32 diverse chunk-grounded questions offline; DraftModel scores those chunk-local generated queries; selected tokens are aggregated by frequency/mean score into a reusable offline 10% fixed set. Online then adds DraftModel residual 5%. Source: `MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/qwen3_docgen_fair_budget/accuracy_summary_offline10_docgen_draft005.json`.
- `offline10_hybrid_old{90,70,50}_docgen{10,30,50}_draft005`: fair-budget fusion of the old offline10 fixed set and the docgen ranking. It keeps the indicated fraction of the old offline10 set, fills the remaining offline10 budget from docgen rank, then adds online DraftModel residual 5%. Source: `MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/qwen3_docgen_fair_budget/accuracy_summary_docgen_ablation_main_table.csv`.
- `offline15_docgen_mean_score_only`: docgen-query variant of `draft_smart_mean_score_global`. It averages the DraftModel scores over 32 generated questions per chunk and selects top 15% tokens by mean score, with no online residual. Source: `MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/qwen3_docgen_fair_budget/accuracy_summary_offline15_docgen_mean_score_only.json`.
- `offline15_docgen_only`: pure offline docgen fixed set at 15% total budget, no online Draft residual. It tests whether docgen offline ranking can replace online residual selection. Source: `MOTIVATION_EXPERIMENTS/offline_doc_generated_query_calibration/qwen3_docgen_fair_budget/accuracy_summary_offline15_docgen_only.json`.
- `draft_smart_*_boundary*_global`: pure-offline boundary replacement at fixed total rate 0.15. It keeps the same full-Draft calibration ranking as `draft_smart_frequency_global` / `draft_smart_mean_score_global`, but reserves 2%/3%/5% of document tokens for tokens closest to retrieved passage boundaries. This tests whether the online-draft residual missed by offline sets can be cheaply compensated by boundary-biased tokens. Source: `MOTIVATION_EXPERIMENTS/offline_boundary_mix_rate015/README.md`.
- `draft32b_smart_*_boundary*_global`: 32B-teacher version of the boundary-mix ablation. It starts from `draft32b_smart_frequency_global` / `draft32b_smart_mean_score_global` / `draft32b_smart_top2_mean_global` and replaces 2%/3%/5% of the total 15% budget with boundary-near tokens. Source: `MOTIVATION_EXPERIMENTS/offline_draft32b_teacher_rate015/README.md`.
- `offline10_first12_residual005` / `offline10_middle_residual005`: fair-budget low-cost Draft residual controls. They use the same offline 10% fixed set as `offline10_draft005`, but replace full Draft residual selection with partial Draft selector (`first12` or `middle`) at residual rate 0.05. Source: `MOTIVATION_EXPERIMENTS/qwen3_offline10_plus_partial_draft_residual_rate015/accuracy_summary.csv`.
- `online_*_rate050` / `offline_hybrid70_rate050`: rate 0.50 sweep baselines for the same Qwen3/MuSiQue setup. Source: `MOTIVATION_EXPERIMENTS/qwen3_hybrid70_online_baselines/summary.csv`.


### Auxiliary Diagnostic Experiments

| diagnostic | purpose | key result | output |
|---|---|---|---|
| residual_stability_from_online_draft_trace | Test whether `online_draft_selected - offline_fixed_set` residual tokens can be predicted from other native queries of the same example. | Native-query leave-one-out residual recall is low: about 8.6%-12.4% depending on base/mode. This weakens the idea that residual can be directly replaced by a tiny number of existing native queries, and motivates 16/32-query calibration or a lightweight predictor. | `MOTIVATION_EXPERIMENTS/residual_stability_from_online_draft_trace/README.md` |
| calibration_query_selector_prediction_16q | Test whether selected-token frequency from 15 calibration queries predicts a held-out query selector output. | Strong stable component: DraftModel(raw) held-out recall 84.2%-91.0%; Target-QK(preprocess KV) recall 85.4%-92.6% across rates 0.1-0.5. This supports offline ranking / candidate-set approaches, but does not yet solve residual-only prediction. | `MOTIVATION_EXPERIMENTS/calibration_query_selector_prediction_16q/README.md` |
| residual_chunk_concentration_probe | Test whether online-draft residual tokens missed by offline fixed set are concentrated in a few chunks. | Residual is moderately spread: freq base has 8.0 nonempty chunks on average and needs 5.3 chunks for 90% residual; mean base has 9.9 nonempty chunks and needs 6.0 chunks. Chunk gating may reduce scope, but 1-2 chunk gating is unlikely to be lossless. | `MOTIVATION_EXPERIMENTS/residual_chunk_concentration_probe/README.md` |
| chunk_gate_heuristic_probe | Test deployable static chunk gates for covering residual tokens. | Oracle top-5 chunks cover 86%-91% residual, but simple deployable heuristics top-5 cover only 25%-34%. Chunk gating has potential only with query-conditioned chunk prediction. | `MOTIVATION_EXPERIMENTS/chunk_gate_heuristic_probe/README.md` |
| shallow_draft_selector_probe | Test whether early DraftModel layers can approximate full online DraftModel selector. | Layer4 is the best shallow candidate: 20-example token recall vs full DraftModel 62.1%, chunk recall 99.5%, score-forward time 0.024s vs 0.145s. Small answer sanity: layer4 14/18 main, 26/32 sub vs default 15/18, 29/32 sub. | `MOTIVATION_EXPERIMENTS/shallow_draft_selector_probe/README.md` |

### Current Cross-Experiment Takeaways

- The fairer HS-attention rerun recovers most of the gap from the flawed hidden-head path: 93/135 Main and 201/248 Sub. It is close to unfinetuned layer4 residual on Main (92/135) and slightly higher on Sub, but much slower selection (0.119s vs 0.0235s) and still below full online Draft residual. This suggests the previous HS failure was partly implementation/objective mismatch, but finetuning first-4-layer attention alone is not yet a clear win.
- FusionRAG-HS hidden-scorer distillation did not improve the real RAG pipeline in this run: offline10 + HS residual5 gets 87/135 Main and 192/248 Sub, below unfinetuned layer4 residual and far below online DraftModel. This suggests selector-overlap gains on Wiki/MuSiQue validation do not directly transfer to answer accuracy; the next distillation route should prioritize a true causal draft model (FusionRAG-CD) rather than a separate scoring head.
- Low-cost partial Draft residual improves over single layer4 only when using a deeper partial selector. `middle` reaches 95/135 Main and 202/250 Sub with 0.074s selection, between layer4 (92/135, 198/250, 0.024s) and full Draft residual (97/135, 208/250, 0.141s). `first12` does not improve over layer4 enough to justify its extra cost.
- Layer4-only offline smart is not competitive with full Draft smart: frequency gets 87/135 Main and 190/250 Sub, while full Draft smart frequency gets 94/135 and 203/250. This means the single shallow layer can be useful as a cheap residual approximation, but it does not provide enough ranking quality for a fully offline smart set.
- Fair-budget controls confirm the residual gain is not only from recomputing more tokens: `offline10 + Draft residual5` beats `offline15 only` (97/135 vs 94/135), and `offline15 + Draft residual5` beats `offline20 only` (103/135 vs 91/135). This supports query-conditioned token choice as the real source of improvement.
- Shallow Draft layer4 residual greatly reduces online residual selection time (about 0.024s vs full Draft residual about 0.141s), but current quality is much lower than full Draft residual: 96/135 Main and 204/250 Sub vs 103/135 and 217/250. This suggests a single shallow layer is not enough; next variants should test first-k-layer aggregation, slightly deeper layers, or a trained shallow predictor.
- Pure offline smart draft fixed set improves over early offline draft aggregation, but only slightly improves over old hybrid70 and remains below online draft at rate 0.15.
- Replacing the offline teacher with Qwen3-32B does not show a stable improvement over the original 3B-teacher pure offline set. The best complete 32B-teacher row is `draft32b_smart_top2_mean_global` at 98/135 Main and 202/248 Sub, which improves Main over 3B `draft_smart_mean_score_global` (95/135) but does not improve Sub Acc (204/250 -> 202/248).
- Adding boundary replacement on top of the 32B-teacher offline set does not help. The best complete 32B-boundary row reaches 98/135 Main and 202/248 Sub (`draft32b_smart_top2_boundary0p05_global`), matching `draft32b_smart_top2_mean_global` but not improving it. This suggests the earlier boundary gain is specific to the 3B frequency baseline, not a robust improvement once the offline teacher/ranking changes.
- Boundary replacement is only useful as a very small cheap bias. At fixed total rate 0.15, replacing 2% of the offline frequency set with boundary-near tokens improves Main Acc from 94/135 to 97/135 while keeping Sub Acc at 203/250. Larger boundary budgets (3%-5%) degrade Sub Acc, so boundary tokens should be treated as an auxiliary prior/candidate feature rather than a standalone replacement for draft-based ranking. This positive result is specifically on `draft_smart_frequency_global` -> `draft_smart_freq_boundary0p02_global`; the mean-score baseline does not show the same clean improvement because Sub Acc drops from 204/250 to 202/250.
- The online DraftModel selector is consistently stronger than online QK in this setup, and has lower measured selection time.
- Adding only 5% online Draft residual on top of a 15% offline smart fixed set nearly closes the gap to full recompute; `freq_draft005` even exceeds full recompute on Sub Acc in this judged run, while still slightly below full recompute on Main Acc.
- The key observation is not that a fully fixed offline set is enough. The current evidence supports a two-part structure: offline fixed set covers stable tokens, and a small online residual covers query-specific tokens.

<!-- QWEN3_UNIFIED_TABLE_END -->

## Long-Term Task Tracker

长期探究任务和实验规范维护在：

```text
MOTIVATION_EXPERIMENTS/offline_residual_long_term_tasks.md
```

后续同一模型/数据集配置的实验结果，需要先追加到本文件的 Unified Main Table，再在对应子实验目录记录细节。



本轮目标：解释为什么 `online_draft_rate015` 明显强于 `offline_hybrid70_rate015`，并尝试提升 offline fixed set 的质量。

## Baseline

| method | Main Acc | Sub Acc | note |
|---|---:|---:|---|
| online_qk_rate015 | 84/135 (62.22%) | 189/248 (76.21%) | online QK selector |
| online_draft_rate015 | 99/135 (73.33%) | 209/248 (84.27%) | online draft selector |
| offline_hybrid70_rate015 | 93/135 (68.89%) | 196/248 (79.03%) | offline hybrid draft70/qk30 |

## Attempt 1: pure offline draft score aggregation

Derived from saved draft calibration score cache, no new model forward.

| method | Main Acc | Sub Acc | conclusion |
|---|---:|---:|---|
| draft_frequency_per_chunk | 88/135 (65.19%) | 192/248 (77.42%) | worse than hybrid70 |
| draft_mean_score_per_chunk | 88/135 (65.19%) | 191/248 (77.02%) | worse than hybrid70 |
| draft_max_score_per_chunk | 92/135 (68.15%) | 195/248 (78.63%) | close but still below hybrid70 |
| draft_top2_mean_score_per_chunk | 92/135 (68.15%) | 195/248 (78.63%) | close but still below hybrid70 |
| draft_top4_mean_score_per_chunk | 90/135 (66.67%) | 193/248 (77.82%) | worse than hybrid70 |

Takeaway: simply changing offline draft aggregation from frequency/mean to max/top-k mean does not recover online draft. Existing hybrid QK+draft is still stronger.

## Attempt 2: online draft vs offline hybrid token overlap

Saved online draft selected token indices with `FUSIONRAG_SAVE_SELECTED_DIR` and compared against offline hybrid selected indices.

| subset | n subq | Jaccard | online recall by offline | offline precision vs online |
|---|---:|---:|---:|---:|
| all | 250 | 0.3004 | 0.4248 | 0.5008 |
| draft_correct_offline_wrong | 38 | 0.2902 | 0.4100 | 0.4917 |
| offline_correct_draft_wrong | 30 | 0.3046 | 0.4282 | 0.5051 |

Takeaway: online draft and offline hybrid select substantially different tokens. However, the overlap gap is not sharply larger only on `draft correct / offline wrong` cases, so token-set similarity alone is not a clean predictor of accuracy.

## Attempt 3: leaky online-draft-trace oracle fixed set

Constructed a per-document offline fixed set by aggregating true online draft selections from the evaluated queries. This uses test queries and is not a valid method; it is only a diagnostic upper-bound attempt.

| method | Main Acc | Sub Acc |
|---|---:|---:|
| online_draft_trace_oracle_fixed_rate015 | 82/135 (60.74%) | 186/248 (75.00%) |

Takeaway: aggregating true online draft selections into a fixed per-document set performs poorly. This strongly suggests online draft's advantage is query-specific rather than a stable per-document token set.

## Attempt 4: same offline hybrid token set, DraftModel recompute path

Ran the same `offline_hybrid70_rate015` token set but with `reprocess_method=DraftModel`, to test whether online draft is better because DraftModel path uses a different recompute/sparse-attention mode.

| method | Main Acc | Sub Acc |
|---|---:|---:|
| offline_hybrid70_draftpath_rate015 | 92/135 (68.15%) | 194/248 (78.23%) |

Takeaway: switching the recompute path does not close the gap. The main gap is not just FusionRAG sparse vs DraftModel path.

## Current Conclusion

The gap to online draft is mainly query adaptivity. A fixed per-document offline set, even one derived from true online draft traces, does not preserve online draft's behavior. Better offline quality likely needs query-conditioned information without full online selection, e.g. a small online supplement, query-clustered offline sets, or a lightweight predictor that chooses among several precomputed per-document token rankings.

## Oracle residual replacement experiment（集合覆盖诊断）

## 实验目的
验证 online Draft residual 这 5% token 是否能被纯 offline 候选集合替代。这里不重新跑问答 accuracy，而是复用已保存的选择结果做集合覆盖诊断：如果离线候选池本身覆盖不到 online residual，那么后续再从这个候选池中挑 5% 很难达到 online Draft residual 的效果。
## 实验定义
- 基础 offline set：`offline10_draft005` 中的 `draft_smart_frequency_global_rate0p1`，即原来的 offline 10%。
- Teacher：同一组 `offline10_draft005` 运行中保存的 online Draft 选择结果。
- 日志确认该运行使用 `residual_online_method=DraftModel residual_rate=0.05`；但实际保存的最终 selected set 平均占文档 17.26%，扣除 offline10 后 residual 平均占约 6.96%。因此本文档按保存下来的实际集合做覆盖诊断，而不是假设 residual 严格等于 5%。
- Oracle residual：`Teacher selected_abs - offline10 selected_abs`。这表示 online Draft 相对 offline10 额外补出来的 token。
- 评价指标：`residual recall = |candidate_pool ∩ oracle_residual| / |oracle_residual|`；`teacher recall = |candidate_pool ∩ teacher_selected| / |teacher_selected|`。
## 数据规模
- 样本数：250 个 question/sub-question。
- 平均文档 token 数：1457.0。
- offline10 平均 token 数：148.0。
- teacher 平均 token 数：249.5。
- teacher 与 offline10 平均重合：148.0，teacher recall=0.5731，offline10 precision-to-teacher=1.0000。
- oracle residual 平均 token 数：101.5。
## 覆盖结果
| candidate pool | pool/doc | residual recall | teacher recall | precision to residual | precision to teacher |
|---|---:|---:|---:|---:|---:|
| old10_base | 10.18% | 0.00% | 57.31% | 0.00% | 100.00% |
| docgen10 | 9.42% | 12.17% | 31.27% | 9.15% | 55.83% |
| hybrid70_old_docgen10 | 10.18% | 4.52% | 50.69% | 3.54% | 88.60% |
| old10_union_docgen10 | 15.20% | 12.17% | 62.72% | 6.04% | 71.24% |
| old10_union_hybrid70 | 11.69% | 4.52% | 59.24% | 3.06% | 90.15% |
| docgen15_frequency | 14.29% | 18.33% | 42.08% | 9.07% | 49.85% |
| docgen15_mean_score | 14.29% | 29.34% | 42.48% | 14.50% | 50.03% |
| old10_union_docgen15_frequency | 18.64% | 18.33% | 65.42% | 7.21% | 60.44% |
| old10_union_docgen15_mean | 19.39% | 29.34% | 70.29% | 11.12% | 62.23% |
| old20_base | 20.25% | 35.10% | 72.55% | 13.42% | 62.63% |
| old10_union_old20 | 20.26% | 35.10% | 72.60% | 13.42% | 62.65% |
| oracle_online_draft_selected | 17.26% | 100.00% | 100.00% | 42.69% | 100.00% |

## 初步结论
1. `old10_base` 对 oracle residual 的覆盖为 0 是定义导致的，因为 residual 已经扣除了 offline10。
2. 如果某个离线候选池的 `residual recall` 很低，说明它很难直接替代 online Draft residual；即使后续做 rerank，也缺少可选 token。
3. `old20_base` / `old10_union_old20` 是一个重要上界参照：它回答“把原 offline 预算扩大到 20% 是否自然覆盖 residual”。
4. `docgen` 系列回答“用文档生成问题得到的离线集合，是否能补到 online Draft residual”。若 union 后 residual recall 仍低，说明 docgen 与 online Draft residual 的互补性有限。

## 输出文件
- `oracle_residual_replacement_summary.csv`：候选池级汇总。
- `oracle_residual_replacement_detail.csv`：每个 example/sub-question 的详细覆盖。
- `oracle_residual_base_stats.csv`：teacher/offline10/residual 的基础统计。
- `oracle_residual_base_summary.json`：基础统计 JSON。

