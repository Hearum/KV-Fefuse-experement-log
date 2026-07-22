# Offline methods vs pure online DraftModel selection

## 目的
只比较 token selection 集合，不考虑 residual、不考虑 QA accuracy。目标是回答：在相同选择比例下，哪种 offline fixed set 最接近独立 online DraftModel selector。

## 口径
- online target: 独立 `PureDraftModel`/`DraftModel` selected indices，不使用 `offline10 + residual` 运行中的 final selected set。
- rates: 0.10, 0.15, 0.30。
- metrics: `Jaccard=|A∩B|/|A∪B|`; `online recall by offline=|A∩B|/|B|`; `offline precision vs online=|A∩B|/|A|`。
- `A` 是 offline selected set，`B` 是 pure online Draft selected set。

## 汇总结果
| rate | offline method | n | Jaccard | online recall by offline | offline precision vs online | offline size | online size |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.10 | old_unrelated_draft_smart_freq | 250 | 0.3832 | 0.5328 | 0.5743 | 148.0 | 142.2 |
| 0.10 | pipeline_old10_draft005 | 250 | 0.3832 | 0.5328 | 0.5743 | 148.0 | 142.2 |
| 0.10 | pipeline_hybrid_old70_docgen30 | 250 | 0.3485 | 0.4994 | 0.5375 | 148.0 | 142.2 |
| 0.10 | sweep_hybrid_draft70_qk30 | 250 | 0.2518 | 0.3896 | 0.4027 | 137.7 | 142.2 |
| 0.10 | sweep_draft_frequency_per_chunk | 250 | 0.2510 | 0.3886 | 0.4016 | 137.7 | 142.2 |
| 0.10 | sweep_hybrid_draft50_qk50 | 250 | 0.2439 | 0.3801 | 0.3929 | 137.7 | 142.2 |
| 0.10 | docgen_draft_smart_freq | 250 | 0.2325 | 0.3626 | 0.3745 | 137.7 | 142.2 |
| 0.10 | pipeline_docgen10 | 250 | 0.2325 | 0.3626 | 0.3745 | 137.7 | 142.2 |
| 0.10 | sweep_qk_frequency_per_chunk | 250 | 0.1954 | 0.3177 | 0.3284 | 137.7 | 142.2 |
| 0.15 | old_unrelated_draft_smart_freq | 250 | 0.3965 | 0.5248 | 0.6290 | 221.9 | 244.2 |
| 0.15 | docgen_draft_smart_freq | 250 | 0.2870 | 0.4093 | 0.4824 | 208.7 | 244.2 |
| 0.15 | pipeline_docgen15_freq | 250 | 0.2870 | 0.4093 | 0.4824 | 208.7 | 244.2 |
| 0.15 | pipeline_docgen15_mean | 250 | 0.2497 | 0.3680 | 0.4337 | 208.7 | 244.2 |
| 0.30 | sweep_draft_frequency_per_chunk | 250 | 0.4261 | 0.5926 | 0.5994 | 422.8 | 427.4 |
| 0.30 | sweep_hybrid_draft70_qk30 | 250 | 0.3936 | 0.5607 | 0.5671 | 422.8 | 427.4 |
| 0.30 | sweep_hybrid_draft50_qk50 | 250 | 0.3902 | 0.5571 | 0.5636 | 422.8 | 427.4 |
| 0.30 | docgen_draft_smart_freq_rank_top30 | 250 | 0.3683 | 0.5339 | 0.5400 | 422.8 | 427.4 |
| 0.30 | sweep_qk_frequency_per_chunk | 250 | 0.3387 | 0.5020 | 0.5078 | 422.8 | 427.4 |

## 初步结论
- rate=0.10: online recall 最高的是 `old_unrelated_draft_smart_freq`，recall=0.5328, Jaccard=0.3832。
- rate=0.15: online recall 最高的是 `old_unrelated_draft_smart_freq`，recall=0.5248, Jaccard=0.3965。
- rate=0.30: online recall 最高的是 `sweep_draft_frequency_per_chunk`，recall=0.5926, Jaccard=0.4261。

## 输出
- `offline_vs_pure_online_draft_summary.csv`
- `offline_vs_pure_online_draft_detail.csv`
