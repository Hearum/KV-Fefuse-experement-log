# GLM Clean Rejudge 补充结果（rerun_20260714_1555）

本目录是 2026-07-14 使用同一份 cross-dataset 预测结果、重新调用当前 GLM-5.2 judge 服务得到的补充评测结果。为了避免覆盖原始结果，本轮写入 `rerun_20260714_1555/` 子目录，并使用独立 cache 与 prompt version。

## 输入与脚本

- 脚本：`rejudge_cross_dataset_glm_clean_rerun_20260714_1555.py`
- 原脚本来源：`rejudge_cross_dataset_glm_clean.py`
- 输入：`MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/results/*/*/seg_*`
- 加载样本数：4380；missing_segments=0
- Judge 模型：GLM-5.2，`thinking={type: disabled}`

## 输出文件

- `judge_cache_glm52_clean.jsonl`：本轮独立 judge cache
- `rejudged_rows.csv`：逐样本判定结果
- `rejudged_summary.csv/json`：本轮汇总
- `rejudge_summary_vs_original.csv`：与原 `rejudge_glm_clean_20260714/rejudged_summary.csv` 的差异对比

## 本轮 GLM Main/Sub Acc 汇总

| dataset | method | rate | rows | glm_main_correct | glm_main_total | glm_main_acc | glm_sub_correct | glm_sub_total | glm_sub_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimqa | full_rate1 | 1.0 | 200 | 118 | 200 | 0.5900 | 118 | 200 | 0.5900 |
| 2wikimqa | online_qk_rate015 | 0.15 | 200 | 111 | 200 | 0.5550 | 111 | 200 | 0.5550 |
| 2wikimqa | online_draft_rate015 | 0.15 | 200 | 107 | 200 | 0.5350 | 107 | 200 | 0.5350 |
| 2wikimqa | offline3b_mean | 0.15 | 200 | 105 | 200 | 0.5250 | 105 | 200 | 0.5250 |
| 2wikimqa | offline3b_freq_boundary2 | 0.15 | 200 | 105 | 200 | 0.5250 | 105 | 200 | 0.5250 |
| 2wikimqa | offline32b_top2 | 0.15 | 200 | 106 | 200 | 0.5300 | 106 | 200 | 0.5300 |
| hotpotqa | full_rate1 | 1.0 | 260 | 236 | 260 | 0.9077 | 236 | 260 | 0.9077 |
| hotpotqa | online_qk_rate015 | 0.15 | 260 | 226 | 260 | 0.8692 | 226 | 260 | 0.8692 |
| hotpotqa | online_draft_rate015 | 0.15 | 260 | 227 | 260 | 0.8731 | 227 | 260 | 0.8731 |
| hotpotqa | offline3b_mean | 0.15 | 260 | 229 | 260 | 0.8808 | 229 | 260 | 0.8808 |
| hotpotqa | offline3b_freq_boundary2 | 0.15 | 260 | 224 | 260 | 0.8615 | 224 | 260 | 0.8615 |
| hotpotqa | offline32b_top2 | 0.15 | 260 | 227 | 260 | 0.8731 | 227 | 260 | 0.8731 |
| triviaqa | full_rate1 | 1.0 | 270 | 247 | 270 | 0.9148 | 247 | 270 | 0.9148 |
| triviaqa | online_qk_rate015 | 0.15 | 270 | 243 | 270 | 0.9000 | 243 | 270 | 0.9000 |
| triviaqa | online_draft_rate015 | 0.15 | 270 | 240 | 270 | 0.8889 | 240 | 270 | 0.8889 |
| triviaqa | offline3b_mean | 0.15 | 270 | 242 | 270 | 0.8963 | 242 | 270 | 0.8963 |
| triviaqa | offline3b_freq_boundary2 | 0.15 | 270 | 244 | 270 | 0.9037 | 244 | 270 | 0.9037 |
| triviaqa | offline32b_top2 | 0.15 | 270 | 240 | 270 | 0.8889 | 240 | 270 | 0.8889 |

## 相比原始 rejudge 的变化

| dataset | method | rate | rows_new | rows_old | rows_delta_new_minus_old | glm_main_correct_new | glm_main_total_new | glm_main_acc_new | glm_main_correct_old | glm_main_total_old | glm_main_acc_old | glm_main_acc_delta_new_minus_old | glm_sub_correct_new | glm_sub_total_new | glm_sub_acc_new | glm_sub_correct_old | glm_sub_total_old | glm_sub_acc_old | glm_sub_acc_delta_new_minus_old |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2wikimqa | full_rate1 | 1.0 | 200 | 200 | 0 | 118 | 200 | 0.59 | 117 | 200 | 0.585 | 0.0050 | 118 | 200 | 0.59 | 117 | 200 | 0.585 | 0.0050 |
| 2wikimqa | online_qk_rate015 | 0.15 | 200 | 200 | 0 | 111 | 200 | 0.555 | 112 | 200 | 0.56 | -0.0050 | 111 | 200 | 0.555 | 112 | 200 | 0.56 | -0.0050 |
| 2wikimqa | online_draft_rate015 | 0.15 | 200 | 200 | 0 | 107 | 200 | 0.535 | 106 | 200 | 0.53 | 0.0050 | 107 | 200 | 0.535 | 106 | 200 | 0.53 | 0.0050 |
| 2wikimqa | offline3b_mean | 0.15 | 200 | 200 | 0 | 105 | 200 | 0.525 | 104 | 200 | 0.52 | 0.0050 | 105 | 200 | 0.525 | 104 | 200 | 0.52 | 0.0050 |
| 2wikimqa | offline3b_freq_boundary2 | 0.15 | 200 | 200 | 0 | 105 | 200 | 0.525 | 105 | 200 | 0.525 | 0.0000 | 105 | 200 | 0.525 | 105 | 200 | 0.525 | 0.0000 |
| 2wikimqa | offline32b_top2 | 0.15 | 200 | 200 | 0 | 106 | 200 | 0.53 | 106 | 200 | 0.53 | 0.0000 | 106 | 200 | 0.53 | 106 | 200 | 0.53 | 0.0000 |
| hotpotqa | full_rate1 | 1.0 | 260 | 260 | 0 | 236 | 260 | 0.9076923076923077 | 236 | 260 | 0.9076923076923077 | 0.0000 | 236 | 260 | 0.9076923076923077 | 236 | 260 | 0.9076923076923077 | 0.0000 |
| hotpotqa | online_qk_rate015 | 0.15 | 260 | 260 | 0 | 226 | 260 | 0.8692307692307693 | 226 | 260 | 0.8692307692307693 | 0.0000 | 226 | 260 | 0.8692307692307693 | 226 | 260 | 0.8692307692307693 | 0.0000 |
| hotpotqa | online_draft_rate015 | 0.15 | 260 | 260 | 0 | 227 | 260 | 0.8730769230769231 | 228 | 260 | 0.8769230769230769 | -0.0038 | 227 | 260 | 0.8730769230769231 | 228 | 260 | 0.8769230769230769 | -0.0038 |
| hotpotqa | offline3b_mean | 0.15 | 260 | 260 | 0 | 229 | 260 | 0.8807692307692307 | 229 | 260 | 0.8807692307692307 | 0.0000 | 229 | 260 | 0.8807692307692307 | 229 | 260 | 0.8807692307692307 | 0.0000 |
| hotpotqa | offline3b_freq_boundary2 | 0.15 | 260 | 260 | 0 | 224 | 260 | 0.8615384615384616 | 225 | 260 | 0.8653846153846154 | -0.0038 | 224 | 260 | 0.8615384615384616 | 225 | 260 | 0.8653846153846154 | -0.0038 |
| hotpotqa | offline32b_top2 | 0.15 | 260 | 260 | 0 | 227 | 260 | 0.8730769230769231 | 227 | 260 | 0.8730769230769231 | 0.0000 | 227 | 260 | 0.8730769230769231 | 227 | 260 | 0.8730769230769231 | 0.0000 |
| triviaqa | full_rate1 | 1.0 | 270 | 270 | 0 | 247 | 270 | 0.9148148148148149 | 246 | 270 | 0.9111111111111111 | 0.0037 | 247 | 270 | 0.9148148148148149 | 246 | 270 | 0.9111111111111111 | 0.0037 |
| triviaqa | online_qk_rate015 | 0.15 | 270 | 270 | 0 | 243 | 270 | 0.9 | 241 | 270 | 0.8925925925925926 | 0.0074 | 243 | 270 | 0.9 | 241 | 270 | 0.8925925925925926 | 0.0074 |
| triviaqa | online_draft_rate015 | 0.15 | 270 | 270 | 0 | 240 | 270 | 0.8888888888888888 | 239 | 270 | 0.8851851851851852 | 0.0037 | 240 | 270 | 0.8888888888888888 | 239 | 270 | 0.8851851851851852 | 0.0037 |
| triviaqa | offline3b_mean | 0.15 | 270 | 270 | 0 | 242 | 270 | 0.8962962962962963 | 241 | 270 | 0.8925925925925926 | 0.0037 | 242 | 270 | 0.8962962962962963 | 241 | 270 | 0.8925925925925926 | 0.0037 |
| triviaqa | offline3b_freq_boundary2 | 0.15 | 270 | 270 | 0 | 244 | 270 | 0.9037037037037037 | 243 | 270 | 0.9 | 0.0037 | 244 | 270 | 0.9037037037037037 | 243 | 270 | 0.9 | 0.0037 |
| triviaqa | offline32b_top2 | 0.15 | 270 | 200 | 70 | 240 | 270 | 0.8888888888888888 | 180 | 200 | 0.9 | -0.0111 | 240 | 270 | 0.8888888888888888 | 180 | 200 | 0.9 | -0.0111 |
## 补充图

- `figures/cross_dataset_glm_clean_rerun_accuracy.png`
- `figures/cross_dataset_glm_clean_rerun_accuracy.pdf`

该图使用本轮 `rejudged_summary.csv` 的 `glm_main_acc`，按数据集分成三个 panel，对比 `full_rate1`、online QK、online Draft 和三个 offline 方法；虚线表示各数据集的 full attention baseline。

## 合并 MuSiQue 的补充图

- `figures/cross_dataset_plus_musique_accuracy.png`
- `figures/cross_dataset_plus_musique_accuracy.pdf`

该图在 `cross_dataset_glm_clean_rerun_accuracy` 的基础上加入 MuSiQue panel。注意：`2wikimqa / hotpotqa / triviaqa` 使用本轮 `rerun_20260714_1555/rejudged_summary.csv` 的 GLM clean rerun 结果；MuSiQue 不在本轮 rejudge 输入中，因此使用 `MOTIVATION_EXPERIMENTS/offline_online_performance_comparison/offline_online_performance_summary.csv` 中已有的 Qwen3-32B MuSiQue main accuracy。图中仅保留四个数据集共同可比较的 6 个方法。
