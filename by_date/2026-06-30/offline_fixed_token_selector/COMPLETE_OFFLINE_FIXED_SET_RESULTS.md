# Complete Offline / Selector Accuracy Table

这张表集中记录目前已经跑过的 selector/fixed-set 结果，避免只看某个实验目录造成误判。Full attention 和 online selector 也放进来作为参考基线。

| label | rate | selector / fixed set | runtime KV | Main Acc | Sub Acc | F1 | EM | note |
|---|---:|---|---|---:|---:|---:|---:|---|
| rate1_full_attention | 1.0 | N/A | full attention | 105/135 (77.78%) | 218/248 (87.90%) | 0.5692 | 0.2298 | full compute upper baseline |
| hybrid70_rate050 | 0.50 | offline hybrid draft70/qk30 | preprocess KV | 104/135 (77.04%) | 211/248 (85.08%) | 0.5475 | 0.2218 | best offline hybrid sweep row |
| online_draft_profile_sparse | 0.15 | online draft | preprocess KV | 100/135 (74.07%) | 206/248 (83.06%) | 0.5410 | 0.2137 | clean online draft profile/sparse audit |
| online_draft_rate015 | 0.15 | online draft old table | preprocess KV | 99/135 (73.33%) | 210/248 (84.68%) | 0.5347 | 0.2177 | old complete online draft table |
| hybrid70_rate040 | 0.40 | offline hybrid draft70/qk30 | preprocess KV | 97/135 (71.85%) | 202/248 (81.45%) | 0.5399 | 0.2177 | offline hybrid sweep |
| online_qk_rate030 | 0.30 | online QK | preprocess KV | 94/135 (69.63%) | 197/248 (79.44%) | 0.5104 | 0.1935 | online QK rate sweep point |
| hybrid70_rate030 | 0.30 | offline hybrid draft70/qk30 | preprocess KV | 93/135 (68.89%) | 196/248 (79.03%) | 0.5127 | 0.2016 | offline hybrid sweep |
| hybrid70_rate015 | 0.15 | offline hybrid draft70/qk30 | preprocess KV | 88/135 (65.19%) | 186/248 (75.00%) | 0.4872 | 0.1815 | offline hybrid rate=0.15 |
| hybrid_intersection_fill_draft | 0.15 | offline hybrid intersection + draft fill | preprocess KV | 86/135 (63.70%) | 185/248 (74.60%) | 0.4935 | 0.1855 | selector candidate |
| offline_draft_frequency | 0.15 | offline draft frequency | preprocess KV | 85/135 (62.96%) | 183/248 (73.79%) | 0.4909 | 0.1935 | draft model stable set |
| online_qk_rate015 | 0.15 | online QK | preprocess KV | 84/135 (62.22%) | 187/248 (75.40%) | 0.4897 | 0.2137 | online QK baseline |
| hybrid70_rate020 | 0.20 | offline hybrid draft70/qk30 | preprocess KV | 84/135 (62.22%) | 187/248 (75.40%) | 0.4979 | 0.2016 | offline hybrid sweep |
| offline_draft_mean | 0.15 | offline draft mean-score | preprocess KV | 84/135 (62.22%) | 183/248 (73.79%) | 0.4916 | 0.1895 | draft model stable set |
| hybrid_draft50_qk50_score | 0.15 | offline hybrid draft50/qk50 | preprocess KV | 83/135 (61.48%) | 182/248 (73.39%) | 0.4794 | 0.1774 | selector candidate |
| rawqk_mean_preprocess_runtime | 0.15 | offline raw-QK mean-score | preprocess KV | 83/135 (61.48%) | 183/248 (73.79%) | 0.4812 | 0.1935 | new raw-QK selector |
| hybrid70_rate005 | 0.05 | offline hybrid draft70/qk30 | preprocess KV | 80/135 (59.26%) | 178/248 (71.77%) | 0.4663 | 0.1855 | offline hybrid sweep |
| offline_qk_frequency | 0.15 | offline preprocess-QK frequency | preprocess KV | 79/135 (58.52%) | 179/248 (72.18%) | 0.4696 | 0.1976 | QK selector built on preprocess KV |
| rawqk_freq_preprocess_runtime | 0.15 | offline raw-QK frequency | preprocess KV | 79/135 (58.52%) | 179/248 (72.18%) | 0.4867 | 0.1976 | new raw-QK selector |
| offline_position_tail_rate015 | 0.15 | position tail | preprocess KV | 79/135 (58.52%) | 176/248 (70.97%) | 0.4430 | 0.1573 | heuristic |
| raw_offline_draft_frequency | 0.15 | offline draft frequency | raw KV | 77/135 (57.04%) | 174/248 (70.16%) | 0.4709 | 0.1815 | runtime raw KV control |
| offline_fullattn_mean | 0.15 | offline full-attention mean | preprocess KV | 77/135 (57.04%) | 172/248 (69.35%) | 0.4429 | 0.1573 | calibration full attention |
| hybrid70_rate010 | 0.10 | offline hybrid draft70/qk30 | preprocess KV | 76/135 (56.30%) | 174/248 (70.16%) | 0.4719 | 0.1855 | offline hybrid sweep |
| offline_qk_mean | 0.15 | offline preprocess-QK mean-score | preprocess KV | 76/135 (56.30%) | 176/248 (70.97%) | 0.4766 | 0.2016 | QK selector built on preprocess KV |
| raw_offline_draft_mean | 0.15 | offline draft mean-score | raw KV | 76/135 (56.30%) | 175/248 (70.56%) | 0.4663 | 0.1774 | runtime raw KV control |
| offline_fullattn_frequency | 0.15 | offline full-attention frequency | preprocess KV | 76/135 (56.30%) | 171/248 (68.95%) | 0.4373 | 0.1532 | calibration full attention |
| rate0_no_doc_recompute | 0.0 | none | preprocess KV | 75/135 (55.56%) | 173/248 (69.76%) | 0.4526 | 0.1613 | no document-token recompute |
| rawqk_mean_raw_runtime | 0.15 | offline raw-QK mean-score | raw KV | 74/135 (54.81%) | 173/248 (69.76%) | 0.4735 | 0.1774 | runtime raw KV control |
| raw_hybrid_draft70_qk30 | 0.15 | offline hybrid draft70/qk30 | raw KV | 74/135 (54.81%) | 169/248 (68.15%) | 0.4568 | 0.1815 | runtime raw KV control |
| offline_random_rate015 | 0.15 | random per chunk | preprocess KV | 73/135 (54.07%) | 172/248 (69.35%) | 0.4461 | 0.1694 | heuristic |
| offline_position_boundary_rate015 | 0.15 | position boundary | preprocess KV | 71/135 (52.59%) | 169/248 (68.15%) | 0.4628 | 0.1734 | heuristic |
| raw_offline_qk_mean | 0.15 | offline preprocess-QK mean-score | raw KV | 70/135 (51.85%) | 170/248 (68.55%) | 0.4519 | 0.1734 | runtime raw KV control |
| rawqk_freq_raw_runtime | 0.15 | offline raw-QK frequency | raw KV | 68/135 (50.37%) | 166/248 (66.94%) | 0.4491 | 0.1653 | runtime raw KV control |
| raw_offline_qk_frequency | 0.15 | offline preprocess-QK frequency | raw KV | 67/135 (49.63%) | 168/248 (67.74%) | 0.4525 | 0.1734 | runtime raw KV control |
| raw_rate0_no_doc_recompute | 0.0 | none | raw KV | 64/135 (47.41%) | 159/248 (64.11%) | 0.4271 | 0.1371 | runtime raw KV control |

## Corrected Takeaway

- `rawqk_mean_preprocess_runtime` 不是全局最强 offline fixed set；它只是 raw-QK rate=0.15 这组里较强，并且接近 online QK rate=0.15。
- 目前全局 offline fixed set 中，更强的是 hybrid70 的较高 rate，尤其 `hybrid70_rate050` 和 `hybrid70_rate040`。
- 在固定 rate=0.15 时，`hybrid70_rate015` 和 `offline_draft_frequency/mean` 都强于 raw-QK rate=0.15。
- raw-QK 的价值不是“最强”，而是说明 selector 在 raw KV 上挑、runtime 用 preprocess KV 这条路线有潜力，值得做 rate sweep/hybrid。
