# Qwen3-32B Rate=0.15 Online/Offline Selector Experiment

实验设置：reflect/musique 数据，Qwen3-32B 主模型，preprocess=true，topk=10，BGE recall，GLM-5.2 judge。样本按 8 个 segment 并行运行，汇总时按 `(Main Question, Sub Question)` 去重。

| label | rate | Main Acc | Sub Acc | F1 | EM | prefill(s) | selection(s) | rows | finished | missing csv | traceback/killed | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| online_qk_rate015 | 0.15 | 84/135 (62.22%) | 189/248 (76.21%) | 0.2076 | 0.0081 | 1.1725 | 0.3230 | 248 | 8 | 0 | 0/0 | online FusionRAG-QK selector at rate=0.15 |
| online_draft_rate015 | 0.15 | 99/135 (73.33%) | 209/248 (84.27%) | 0.1905 | 0.0081 | 0.9924 | 0.1417 | 248 | 8 | 0 | 0/0 | online draft-model selector at rate=0.15 |
| offline_hybrid70_rate015 | 0.15 | 93/135 (68.89%) | 196/248 (79.03%) | 0.2064 | 0.0081 | 0.8447 | 0.0000 | 248 | 8 | 0 | 0/0 | offline fixed set, hybrid draft70/qk30 score per chunk, rate=0.15 |

## Online Draft vs Offline Hybrid 差异诊断

基于 Qwen3 rate=0.15 已完成结果，按 main question 聚合：一个 main question 的所有 sub-question 都正确才算正确。

| category | count | meaning |
|---|---:|---|
| draft_correct_offline_wrong | 20 | online draft 答对、offline hybrid70 答错，offline 需要重点补的样本 |
| offline_correct_draft_wrong | 14 | offline hybrid70 答对、online draft 答错，offline 固定集合反而有益的样本 |
| draft_correct_qk_wrong | 22 | online draft 答对、online QK 答错，说明 draft selector 的优势样本 |
| qk_correct_draft_wrong | 7 | online QK 答对、online draft 答错，说明 QK selector 的互补样本 |

差异样本 CSV：`draft_correct_offline_wrong.csv`、`offline_correct_draft_wrong.csv`、`draft_correct_qk_wrong.csv`、`qk_correct_draft_wrong.csv`。
