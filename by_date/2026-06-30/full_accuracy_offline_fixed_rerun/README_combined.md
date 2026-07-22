# Full Accuracy: Online vs Offline Fixed Token Sets

完整 `data/result_reflect.json`，同一套 reflect pipeline，同一套 GLM-5.2 judge。所有 accuracy 都按 CSV/Judge 落盘的 248 条 sub-question 计算；tail/random 曾被系统 kill，已从中断 main question 续跑，并按 `(Main Question, Sub Question)` 去重合并。

| label | rate | Main Acc | Sub Acc | F1 | EM | prompt/full-prefill mean(s) | storage mean(s) | selection mean(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rate1_full_attention | 1.00 | 105/135 (77.78%) | 218/248 (87.90%) | 0.5692 | 0.2298 | 0.1167 | 0.0000 | 0.0000 |
| online_qk_rate030 | 0.30 | 94/135 (69.63%) | 197/248 (79.44%) | 0.5104 | 0.1935 | 0.2584 | 0.0253 | 0.1043 |
| online_qk_rate015 | 0.15 | 84/135 (62.22%) | 187/248 (75.40%) | 0.4897 | 0.2137 | 0.2311 | 0.0249 | 0.1032 |
| rate0_no_doc_recompute | 0.00 | 75/135 (55.56%) | 173/248 (69.76%) | 0.4526 | 0.1613 | 0.1011 | 0.0250 | 0.0000 |
| offline_position_boundary_rate015 | 0.15 | 71/135 (52.59%) | 169/248 (68.15%) | 0.4628 | 0.1734 | 0.1293 | 0.0245 | 0.0000 |
| offline_position_tail_rate015 | 0.15 | 79/135 (58.52%) | 176/248 (70.97%) | 0.4430 | 0.1573 | 0.1316 | 0.0251 | 0.0000 |
| offline_random_rate015 | 0.15 | 73/135 (54.07%) | 172/248 (69.35%) | 0.4461 | 0.1694 | 0.1313 | 0.0248 | 0.0000 |

说明：offline fixed set 通过 `preselected_k_need_index` 直接指定重算 token，selection 时间为 0。
