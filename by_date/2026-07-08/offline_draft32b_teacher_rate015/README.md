# Offline Draft-32B Teacher Rate 0.15

目的：验证 offline fixed set 构造阶段如果把 DraftModel teacher 从原来的 3B 换成 Qwen3-32B，是否能提升纯 offline selector 的 QA accuracy。

实验控制：

- 主推理模型：Qwen3-32B。
- 数据集：reflect/musique，`data/result_reflect.json`，topk=10，BGE recall，preprocess=true，global preprocess scope。
- 总重算比例：0.15。
- 只替换 offline fixed set 的 teacher；online 推理仍走 FusionRAG + offline fixed set，不额外调用 online draft selector。
- calibration query：沿用旧 offline smart 方法的 other-example native sub-question，不使用当前 example 的真实问题，避免测试问题泄漏。
- teacher 打分：Qwen3-32B 对 `[system + docs + calibration query]` 做 draft attention score，使用 `_fusionrag_compute_draft_doc_attention_scores`，默认取后半层 RRF 聚合。

待比较 baseline：

| method | Main Acc | Sub Acc | note |
|---|---:|---:|---|
| draft_smart_frequency_global | 94/135 (69.63%) | 203/250 (81.20%) | 原 3B teacher pure offline frequency |
| draft_smart_mean_score_global | 95/135 (70.37%) | 204/250 (81.60%) | 原 3B teacher pure offline mean score |

当前状态：已创建脚本，先跑 1-example smoke，确认 32B teacher score cache 格式正确后再并行跑全量。

## 32B Teacher Accuracy 结果

| method | segments | Main Acc | Sub Acc | F1 | EM | selection(s) | prefill(s) | missing csv | traceback/killed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| draft32b_smart_frequency_global | 8/8 | 95/135 (70.37%) | 201/248 (81.05%) | 0.1858 | 0.0040 | 0.0000 | 0.9035 | 0 | 0/0 |
| draft32b_smart_mean_score_global | 8/8 | 97/135 (71.85%) | 202/248 (81.45%) | 0.1991 | 0.0121 | 0.0000 | 0.9003 | 0 | 0/0 |
| draft32b_smart_max_score_global | 8/8 | 95/135 (70.37%) | 199/248 (80.24%) | 0.1926 | 0.0242 | 0.0000 | 0.8961 | 0 | 0/0 |
| draft32b_smart_top2_mean_global | 8/8 | 98/135 (72.59%) | 202/248 (81.45%) | 0.1942 | 0.0161 | 0.0000 | 0.8949 | 0 | 0/0 |

## 32B Teacher Boundary-Mix Accuracy 结果

| method | segments | Main Acc | Sub Acc | F1 | EM | selection(s) | prefill(s) | missing csv | traceback/killed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| draft32b_smart_freq_boundary0p02_global | 8/8 | 93/135 (68.89%) | 198/248 (79.84%) | 0.2008 | 0.0121 | 0.0000 | 0.8936 | 0 | 0/0 |
| draft32b_smart_mean_boundary0p02_global | 8/8 | 97/135 (71.85%) | 201/248 (81.05%) | 0.1869 | 0.0081 | 0.0000 | 0.8902 | 0 | 0/0 |
| draft32b_smart_top2_boundary0p02_global | 8/8 | 94/135 (69.63%) | 199/248 (80.24%) | 0.2099 | 0.0242 | 0.0000 | 0.8931 | 0 | 0/0 |
| draft32b_smart_freq_boundary0p03_global | 8/8 | 94/135 (69.63%) | 199/248 (80.24%) | 0.2088 | 0.0121 | 0.0000 | 0.8936 | 0 | 0/0 |
| draft32b_smart_mean_boundary0p03_global | 8/8 | 96/135 (71.11%) | 202/248 (81.45%) | 0.1837 | 0.0121 | 0.0000 | 0.8930 | 0 | 0/0 |
| draft32b_smart_top2_boundary0p03_global | 8/8 | 89/135 (65.93%) | 194/248 (78.23%) | 0.2014 | 0.0161 | 0.0000 | 0.8880 | 0 | 0/0 |
| draft32b_smart_freq_boundary0p05_global | 8/8 | 93/135 (68.89%) | 199/248 (80.24%) | 0.1798 | 0.0202 | 0.0000 | 0.8932 | 0 | 0/0 |
| draft32b_smart_mean_boundary0p05_global | 8/8 | 97/135 (71.85%) | 202/248 (81.45%) | 0.1997 | 0.0081 | 0.0000 | 0.8863 | 0 | 0/0 |
| draft32b_smart_top2_boundary0p05_global | 8/8 | 98/135 (72.59%) | 202/248 (81.45%) | 0.1875 | 0.0121 | 0.0000 | 0.8853 | 0 | 0/0 |
