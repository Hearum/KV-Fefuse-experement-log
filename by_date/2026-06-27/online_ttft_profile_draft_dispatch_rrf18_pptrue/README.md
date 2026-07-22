# DraftModel Dispatch-RRF18 pp=True Online Profile

本实验对齐 `/mnt/qjhs-sh-lab-01/wjh/FusionRAG/dispatch_glm_overnight.sh` 中 DraftModel 的 selection 语义：`preprocess=True`, `topk=10`, `DRAFT_LAYER_SEL=rrf`, `RRF_K=18`。selection 使用 DraftModel 后半层 query-to-doc attention，经 RRF 聚合后走 `smart_query_selection`，不是裸 top-k。

- main model: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct`
- draft model: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`，本次不作为 load_path。
- rate: `0.15`
- warmup: `8`，total rows: `67`

## 结果摘要

- end_to_end_ttft_s_mean: `0.388583`
- end_to_end_ttft_s_p50: `0.367312`
- end_to_end_ttft_s_p95: `0.519408`
- kv_load_copy_s_mean: `0.024148`
- draft_selection_s_mean: `0.139494`
- draft_score_forward_s_mean: `0.138250`
- online_update_query_prefill_s_mean: `0.169116`
- extra_impl_overhead_s_mean: `0.055825`
- selected_doc_tokens_mean: `217.457627`
- reprocess_prefill_tokens_mean: `244.338983`
- full_tokens_mean: `2271.457627`

## 文件

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_draft_dispatch_rrf18_pptrue/draft_dispatch_rrf18_pptrue_rate0.15_n120.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_draft_dispatch_rrf18_pptrue/draft_dispatch_rrf18_pptrue_rate0.15_n120_summary.json`

## 与 clean profile 对照

对照对象来自 `online_ttft_profile_clean_rate_sweep_v2`，同为 Qwen2.5-7B-Instruct、MuSiQue、topk=10、`rate=0.15`。DraftModel 本次只跑到前 50 个 main questions 中可测试的 67 个 sub-question，去掉 8 个 warmup 后统计 59 条；clean sweep 是 112 条 measured rows，因此绝对数值不是严格逐行配对，但量级已经能看出开销结构。

| 方法 | rate | end-to-end TTFT mean | selection mean | online update + query mean | KV load/copy mean | selected doc tokens |
|---|---:|---:|---:|---:|---:|---:|
| FusionRAG selector, preprocess KV | 0.15 | 0.281787s | 0.101922s | 0.101988s | 0.024036s | 216.73 |
| DraftModel-rrf18 selector, pp=True | 0.15 | 0.388583s | 0.139494s | 0.169116s | 0.024148s | 217.46 |
| Full recompute baseline | 1.00 | 0.299576s | 0.000000s | 0.299576s | 0.000000s | 1447.92 |

观察：

- DraftModel-rrf18 的 selection 本身比 FusionRAG selector 更重：`0.1395s` vs `0.1019s`。
- 两者选中的 doc token 数几乎一致，但 DraftModel 这组 online update + query 更慢：`0.1691s` vs `0.1020s`。这说明开销差异不只来自 selection，还可能来自所选 token 的分布、稀疏访问形态或当前 preselected/profile 路径的实现开销。
- 在当前 profile 条件下，DraftModel-rrf18 pp=True 的端到端 TTFT 已经慢于 full recompute baseline：`0.3886s` vs `0.2996s`。
