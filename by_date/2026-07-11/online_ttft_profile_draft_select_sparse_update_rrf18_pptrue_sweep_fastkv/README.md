# DraftModel Selection + FusionRAG Sparse Update Online Profile

本实验只把 selection 换成 DraftModel：DraftModel 后半层 query-to-doc attention 经 RRF 聚合后走 `smart_query_selection`。主模型 online update/recompute 强制走 `reprocess_method=FusionRAG`，因此和 QK profile 使用同一条 sparse update 路径，用来隔离 selector 开销和重算 rate scaling。

- main model: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct`
- draft model: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`，本次不作为 load_path。
- rate: `1.0`
- warmup: `8`，total rows: `80`

## 结果摘要

- end_to_end_ttft_s_mean: `0.306403`
- end_to_end_ttft_s_p50: `0.308977`
- end_to_end_ttft_s_p95: `0.390500`
- kv_load_copy_s_mean: `0.000000`
- draft_selection_s_mean: `0.000000`
- draft_score_forward_s_mean: `0.000000`
- online_update_query_prefill_s_mean: `0.305544`
- extra_impl_overhead_s_mean: `0.000859`
- selected_doc_tokens_mean: `1478.611111`
- reprocess_prefill_tokens_mean: `2297.666667`
- full_tokens_mean: `2297.666667`

## 文件

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_draft_select_sparse_update_rrf18_pptrue_sweep_fastkv/draft_select_sparse_update_rrf18_pptrue_rate1.0_n80.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_draft_select_sparse_update_rrf18_pptrue_sweep_fastkv/draft_select_sparse_update_rrf18_pptrue_rate1.0_n80_summary.json`
