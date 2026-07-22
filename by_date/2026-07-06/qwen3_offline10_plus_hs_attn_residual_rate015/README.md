# Qwen3 Offline10 + FusionRAG-HS Attention Residual5

公平复现实验：固定 offline10 与 FusionRAG recompute path，只把 online residual selector 换成微调后的 HS 前 4 层 attention score。

| method | Main Acc | Sub Acc | F1 | EM | added tokens | residual selection(s) | prefill(s) | csv files | missing | finished segs | traceback/killed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| offline10_draft_smart_frequency_global + HS_attn_residual005_epoch003 | 93/135 (68.89%) | 201/248 (81.05%) | 0.2096 | 0.0161 | 70.82 | 0.1192 | 0.9886 | 4 | 0 | 4 | 0/0 |

## Notes

- `FUSIONRAG_HS_SCORE_MODE=attn_prob`：使用微调后前 4 层的 query-to-doc attention score，而不是 hidden scorer head。
- 每个 chunk 单独 forward，避免不同 chunk 长度 batch padding 导致 doc/query 边界错位。
- 对照目标：`offline10 + unfinetuned layer4 residual5`。
