# DraftModel Selection + FusionRAG Sparse Update Online Profile

本实验只把 selection 换成 DraftModel：DraftModel 后半层 query-to-doc attention 经 RRF 聚合后走 `smart_query_selection`。主模型 online update/recompute 强制走 `reprocess_method=FusionRAG`，因此和 QK profile 使用同一条 sparse update 路径，用来隔离 selector 开销和重算 rate scaling。

- main model: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct`
- draft model: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`，本次不作为 load_path。
- rate: `0.15`
- warmup: `0`，total rows: `3`

## 结果摘要

- end_to_end_ttft_s_mean: `0.690788`
- end_to_end_ttft_s_p50: `0.425411`
- end_to_end_ttft_s_p95: `1.229663`
- kv_load_copy_s_mean: `0.094943`
- draft_selection_s_mean: `0.171978`
- draft_score_forward_s_mean: `0.170863`
- online_update_query_prefill_s_mean: `0.307232`
- extra_impl_overhead_s_mean: `0.116635`
- selected_doc_tokens_mean: `202.000000`
- reprocess_prefill_tokens_mean: `226.333333`
- full_tokens_mean: `2166.000000`

## 文件

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_draft_select_sparse_update_smoke/draft_select_sparse_update_rrf18_pptrue_rate0.15_n3.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_draft_select_sparse_update_smoke/draft_select_sparse_update_rrf18_pptrue_rate0.15_n3_summary.json`
