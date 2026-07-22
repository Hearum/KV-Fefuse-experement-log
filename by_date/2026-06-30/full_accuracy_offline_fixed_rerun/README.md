# Full Accuracy with Offline Fixed Token Sets

完整 `data/result_reflect.json`，同一套 reflect pipeline，同一套 GLM-5.2 judge。

| label | status | rate | progress | Main Acc | Sub Acc | F1 | EM | prompt/full-prefill mean(s) | storage mean(s) | selection mean(s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rate1_full_attention | finished | 1.00 | 200/200 | 104/135 (77.04%) | 218/250 (87.20%) | 0.5666 | 0.2280 | 0.1167 | 0.0000 | 0.0000 |
| online_qk_rate030 | finished | 0.30 | 200/200 | 94/135 (69.63%) | 199/250 (79.60%) | 0.5083 | 0.1920 | 0.2584 | 0.0253 | 0.1043 |
| online_qk_rate015 | finished | 0.15 | 200/200 | 84/135 (62.22%) | 189/250 (75.60%) | 0.4878 | 0.2120 | 0.2311 | 0.0249 | 0.1032 |
| rate0_no_doc_recompute | finished | 0.00 | 200/200 | 75/135 (55.56%) | 175/250 (70.00%) | 0.4510 | 0.1600 | 0.1011 | 0.0250 | 0.0000 |
| offline_position_tail_rate015 | running | 0.15 | 167/200 | 0/0 (0.00%) | 0/0 (0.00%) | 0.0000 | 0.0000 | 0.1300 | 0.0248 | 0.0000 |
| offline_position_boundary_rate015 | finished | 0.15 | 200/200 | 71/135 (52.59%) | 171/250 (68.40%) | 0.4611 | 0.1720 | 0.1293 | 0.0245 | 0.0000 |
| offline_random_rate015 | running | 0.15 | 154/200 | 0/0 (0.00%) | 0/0 (0.00%) | 0.0000 | 0.0000 | 0.1299 | 0.0248 | 0.0000 |

说明：offline fixed set 使用 `preselected_k_need_index` 直接指定重算 token，selection 时间应为 0；当前表里的 offline QK/draft 小样本结果不在这里混入，避免和完整数据集 acc 混淆。
