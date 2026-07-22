# MuSiQue GLM Clean Rejudge 补充结果（musique_rerun_20260714_1718）

本目录是 2026-07-14 对 Qwen3-32B MuSiQue 相关方法重新调用当前 GLM-5.2 judge 服务得到的补充评测结果。结果没有覆盖旧 cross-dataset rerun，也没有覆盖原始 MuSiQue 结果。

## 配置

- Judge 模型：GLM-5.2，`thinking={type: disabled}`
- 输入方法数：10
- 加载样本行数：2500
- missing_methods=0
- 脚本：`../rejudge_musique_glm_clean_rerun_20260714_1718.py`

## 输出文件

- `judge_cache_glm52_clean.jsonl`：本轮独立 judge cache
- `rejudged_rows.csv`：逐样本判定结果
- `rejudged_summary.csv/json`：本轮汇总
- `cross_dataset_plus_musique_glm_rerun_accuracy.png/pdf`：MuSiQue 与 2WikiMQA/HotpotQA/TriviaQA 的共同方法合并图，四个数据集都使用 GLM clean rerun 结果
- `musique_glm_rerun_all_methods_accuracy.png/pdf`：MuSiQue 10 个方法完整对比图

## MuSiQue GLM Main/Sub Acc 汇总

| method | rate | rows | glm_main_correct | glm_main_total | glm_main_acc | glm_sub_correct | glm_sub_total | glm_sub_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_rate1 | 1.0 | 250 | 120 | 135 | 88.89% | 233 | 248 | 93.95% |
| online_qk_rate015 | 0.15 | 250 | 100 | 135 | 74.07% | 206 | 248 | 83.06% |
| online_draft_rate015 | 0.15 | 250 | 112 | 135 | 82.96% | 224 | 248 | 90.32% |
| offline_hybrid70_rate015 | 0.15 | 250 | 106 | 135 | 78.52% | 212 | 248 | 85.48% |
| draft_smart_mean_score_global | 0.15 | 250 | 106 | 135 | 78.52% | 215 | 248 | 86.69% |
| draft_smart_freq_boundary0p02_global | 0.15 | 250 | 103 | 135 | 76.30% | 210 | 248 | 84.68% |
| draft32b_smart_top2_mean_global | 0.15 | 250 | 102 | 135 | 75.56% | 210 | 248 | 84.68% |
| offline10_draft005 | 0.1 | 250 | 108 | 135 | 80.00% | 218 | 248 | 87.90% |
| offline10_hybrid_old70_docgen30_draft005 | 0.15 | 250 | 109 | 135 | 80.74% | 219 | 248 | 88.31% |
| offline20_only | 0.2 | 250 | 103 | 135 | 76.30% | 213 | 248 | 85.89% |
## 新旧 Judge 变化表

- `musique_glm_rerun_before_after_table.png`
- `musique_glm_rerun_before_after_table.pdf`
- `musique_glm_rerun_before_after_table.csv`

该表比较每个 MuSiQue 方法原始 `Correct` 汇总与本轮 GLM clean rerun 的 `glm_correct` 汇总，分别展示 Main accuracy 和 Sub accuracy 的前后变化。

## Judge 审核

- `glm_judge_review_samples.md`：抽样查看原始 judge 与 GLM judge 不一致的样本。
- `glm_judge_review_conclusion.md`：对 GLM 是否变宽/误判的初步结论。

## Cross-Dataset F1/EM 补充图

- `cross_dataset_plus_musique_f1.png`
- `cross_dataset_plus_musique_f1.pdf`
- `cross_dataset_plus_musique_em.png`
- `cross_dataset_plus_musique_em.pdf`
- `cross_dataset_plus_musique_f1_em_plot_data.csv`

这两张图使用 `MOTIVATION_EXPERIMENTS/offline_online_performance_comparison/offline_online_performance_summary.csv` 中 Qwen3-32B 的原始 F1/EM 字段。MuSiQue 的 offline 方法名按 accuracy 合图的对应关系映射到共同方法；TriviaQA 的 `offline32b_top2` 在主表中缺 F1/EM，因此图中标为 N/A。

## Clean Think F1/EM 补充图

前一版 `cross_dataset_plus_musique_f1.png` / `cross_dataset_plus_musique_em.png` 直接读取原 CSV 中已有 `F1/EM` 字段；这些字段的原始计算没有显式移除 `<think>...</think>` 或裸 `</think>` 占位符。

补充生成以下清理版本：先从 `Predicted` 中移除 `<think>...</think>`、裸 `<think>`/`</think>` 与开头 `Answer:` 前缀，再按标准 QA normalize 重新计算 token F1 与 exact match。

- `cross_dataset_plus_musique_f1_cleanthink.png`
- `cross_dataset_plus_musique_f1_cleanthink.pdf`
- `cross_dataset_plus_musique_em_cleanthink.png`
- `cross_dataset_plus_musique_em_cleanthink.pdf`
- `cross_dataset_plus_musique_f1_em_cleanthink_plot_data.csv`
