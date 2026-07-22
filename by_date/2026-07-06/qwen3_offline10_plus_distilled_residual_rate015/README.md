# Qwen3 Offline10 + Distilled Residual5

实验设置：Qwen3-32B 主模型，MuSiQue/reflect 200 条样本，topk=10，preprocess=true，BGE recall，offline 固定集合使用 `draft_smart_frequency_global` 选 10% doc tokens；online residual 使用 WikiText-103 蒸馏得到的 4-layer selector 从剩余 token 中补 5%，总预算 15%。GLM-5.2 用作 judge。

| method | Main Acc | Sub Acc | F1 | EM | added tokens | residual selection(s) | prefill(s) | csv files | missing | traceback/killed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| offline10_draft_smart_frequency_global + distilled_4layer_residual005 | 92/135 (68.15%) | 195/248 (78.63%) | 0.2081 | 0.0202 | 70.82 | 0.0768 | 0.9382 | 8 | 0 | 0/0 |
