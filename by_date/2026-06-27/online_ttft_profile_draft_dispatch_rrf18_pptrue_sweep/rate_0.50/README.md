# DraftModel Dispatch-RRF18 pp=True Online Profile

本实验对齐 `/mnt/qjhs-sh-lab-01/wjh/FusionRAG/dispatch_glm_overnight.sh` 中 DraftModel 的 selection 语义：`preprocess=True`, `topk=10`, `DRAFT_LAYER_SEL=rrf`, `RRF_K=18`。selection 使用 DraftModel 后半层 query-to-doc attention，经 RRF 聚合后走 `smart_query_selection`，不是裸 top-k。

- main model: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct`
- draft model: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`，本次不作为 load_path。
- rate: `0.5`
- warmup: `8`，total rows: `67`

## 结果摘要

- end_to_end_ttft_s_mean: `0.573466`
- end_to_end_ttft_s_p50: `0.574733`
- end_to_end_ttft_s_p95: `0.772115`
- kv_load_copy_s_mean: `0.023834`
- draft_selection_s_mean: `0.139489`
- draft_score_forward_s_mean: `0.138500`
- online_update_query_prefill_s_mean: `0.349575`
- extra_impl_overhead_s_mean: `0.060569`
- selected_doc_tokens_mean: `726.016949`
- reprocess_prefill_tokens_mean: `752.898305`
- full_tokens_mean: `2271.457627`

## 文件

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_draft_dispatch_rrf18_pptrue_sweep/rate_0.50/draft_dispatch_rrf18_pptrue_rate0.5_n120.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_draft_dispatch_rrf18_pptrue_sweep/rate_0.50/draft_dispatch_rrf18_pptrue_rate0.5_n120_summary.json`
