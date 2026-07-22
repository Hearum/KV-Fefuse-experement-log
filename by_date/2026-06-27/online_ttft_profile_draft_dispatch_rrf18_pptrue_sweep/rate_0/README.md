# DraftModel Dispatch-RRF18 pp=True Online Profile

本实验对齐 `/mnt/qjhs-sh-lab-01/wjh/FusionRAG/dispatch_glm_overnight.sh` 中 DraftModel 的 selection 语义：`preprocess=True`, `topk=10`, `DRAFT_LAYER_SEL=rrf`, `RRF_K=18`。selection 使用 DraftModel 后半层 query-to-doc attention，经 RRF 聚合后走 `smart_query_selection`，不是裸 top-k。

- main model: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct`
- draft model: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`，本次不作为 load_path。
- rate: `0.0`
- warmup: `8`，total rows: `67`

## 结果摘要

- end_to_end_ttft_s_mean: `0.183901`
- end_to_end_ttft_s_p50: `0.182242`
- end_to_end_ttft_s_p95: `0.204978`
- kv_load_copy_s_mean: `0.025229`
- draft_selection_s_mean: `0.000000`
- draft_score_forward_s_mean: `0.000000`
- online_update_query_prefill_s_mean: `0.102887`
- extra_impl_overhead_s_mean: `0.055786`
- selected_doc_tokens_mean: `0.000000`
- reprocess_prefill_tokens_mean: `26.881356`
- full_tokens_mean: `2271.457627`

## 文件

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_draft_dispatch_rrf18_pptrue_sweep/rate_0/draft_dispatch_rrf18_pptrue_rate0.0_n120.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_draft_dispatch_rrf18_pptrue_sweep/rate_0/draft_dispatch_rrf18_pptrue_rate0.0_n120_summary.json`
