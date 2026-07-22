# Qwen3 Offline10 + FusionRAG-HS Residual5

实验设置：Qwen3-32B 主模型，MuSiQue/reflect 200 条样本，topk=10，preprocess=true，BGE recall，offline 固定集合使用 `draft_smart_frequency_global` 选 10% doc tokens；online residual 使用 FusionRAG-HS hidden scorer 从剩余 token 中补 5%，总预算 15%。GLM-5.2 用作 judge。

| method | Main Acc | Sub Acc | F1 | EM | added tokens | residual selection(s) | prefill(s) | csv files | missing | finished segs | traceback/killed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| offline10_draft_smart_frequency_global + FusionRAG-HS_hidden_scorer_residual005_epoch003 | 87/135 (64.44%) | 192/248 (77.42%) | 0.2000 | 0.0161 | 70.82 | 0.0540 | 0.9249 | 4 | 0 | 4 | 0/0 |
