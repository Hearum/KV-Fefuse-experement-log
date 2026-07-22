# Qwen3-32B Hybrid70/Online Baseline Experiment

实验设置：reflect/musique 数据，Qwen3-32B 主模型，preprocess=true，topk=10，BGE recall，GLM-5.2 judge。样本按 8 个 segment 并行运行，汇总时按 `(Main Question, Sub Question)` 去重。

| label | rate | Main Acc | Sub Acc | F1 | EM | prefill(s) | selection(s) | rows | finished | missing csv | traceback/killed | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full_rate1 | 1.00 | 105/135 (77.78%) | 215/248 (86.69%) | 0.1934 | 0.0081 | 1.1683 | 0.0000 | 248 | 8 | 0 | 0/0 | rate=1 full document-token recompute upper baseline |
| online_qk_rate050 | 0.50 | 90/135 (66.67%) | 199/248 (80.24%) | 0.2152 | 0.0161 | 2.0366 | 0.3183 | 248 | 8 | 0 | 0/0 | online FusionRAG-QK selector at rate=0.5 |
| online_draft_rate050 | 0.50 | 108/135 (80.00%) | 219/248 (88.31%) | 0.2091 | 0.0161 | 1.8645 | 0.1411 | 248 | 8 | 0 | 0/0 | online draft-model selector at rate=0.5 |
| offline_hybrid70_rate050 | 0.50 | 101/135 (74.81%) | 208/248 (83.87%) | 0.1950 | 0.0202 | 1.7223 | 0.0000 | 248 | 8 | 0 | 0/0 | offline fixed set, hybrid draft70/qk30 score per chunk, rate=0.5 |
