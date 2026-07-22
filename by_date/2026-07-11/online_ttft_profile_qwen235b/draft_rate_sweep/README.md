# DraftModel Dispatch-RRF18 pp=True Online Profile

本实验对齐 `/mnt/qjhs-sh-lab-01/wjh/FusionRAG/dispatch_glm_overnight.sh` 中 DraftModel 的 selection 语义：`preprocess=True`, `topk=10`, `DRAFT_LAYER_SEL=rrf`, `RRF_K=18`。selection 使用 DraftModel 后半层 query-to-doc attention，经 RRF 聚合后走 `smart_query_selection`，不是裸 top-k。

- main model: `/home/hming/models/Qwen3-235B-A22B`
- draft model: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`
- preprocess KV cache: `/home/hming/fusionrag-qwen3-235b-three-groups-unified-prompt-cache/online_draft_rate015/Qwen3-235B-A22B/musique/preprocess_kv_cache_global_topk10_bge`
- raw KV cache: `/home/hming/fusionrag-qwen3-235b-three-groups-unified-prompt-cache/online_draft_rate015/Qwen3-235B-A22B/musique/kv_cache`，本次不作为 load_path。
- rate: `1.0`
- warmup: `3`，total rows: `60`

## 结果摘要

- end_to_end_ttft_s_mean: `4.311416`
- end_to_end_ttft_s_p50: `4.259541`
- end_to_end_ttft_s_p95: `4.904382`
- kv_load_copy_s_mean: `0.000000`
- draft_selection_s_mean: `0.000000`
- draft_score_forward_s_mean: `0.000000`
- online_update_query_prefill_s_mean: `4.298899`
- extra_impl_overhead_s_mean: `0.012517`
- selected_doc_tokens_mean: `1446.140351`
- reprocess_prefill_tokens_mean: `2265.052632`
- full_tokens_mean: `2265.052632`

## 文件

- detail CSV: `/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_qwen235b/draft_rate_sweep/draft_dispatch_rrf18_pptrue_rate1.0_n60.csv`
- summary JSON: `/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_qwen235b/draft_rate_sweep/draft_dispatch_rrf18_pptrue_rate1.0_n60_summary.json`
