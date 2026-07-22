# DraftModel Selection + FusionRAG Sparse Update Online Profile

本实验只把 selection 换成 DraftModel：DraftModel 后半层 query-to-doc attention 经 RRF 聚合后走 `smart_query_selection`。主模型 online update/recompute 强制走 `reprocess_method=FusionRAG`，因此和 QK profile 使用同一条 sparse update 路径，用来隔离 selector 开销和重算 rate scaling。

- main model: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct`
- draft model: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`，本次不作为 load_path。
- rate: `0.15`
- warmup: `4`，total rows: `16`

## 结果摘要

- end_to_end_ttft_s_mean: `0.363552`
- end_to_end_ttft_s_p50: `0.337924`
- end_to_end_ttft_s_p95: `0.440050`
- kv_load_copy_s_mean: `0.029260`
- draft_selection_s_mean: `0.147288`
- draft_score_forward_s_mean: `0.145650`
- online_update_query_prefill_s_mean: `0.106665`
- extra_impl_overhead_s_mean: `0.080339`
- selected_doc_tokens_mean: `243.750000`
- reprocess_prefill_tokens_mean: `270.333333`
- full_tokens_mean: `2447.500000`

## 文件

- detail CSV: `MOTIVATION_EXPERIMENTS/kv_load_investigation/draft_sparse_after_batched_rope_fix/draft_select_sparse_update_rrf18_pptrue_rate0.15_n16.csv`
- summary JSON: `MOTIVATION_EXPERIMENTS/kv_load_investigation/draft_sparse_after_batched_rope_fix/draft_select_sparse_update_rrf18_pptrue_rate0.15_n16_summary.json`
