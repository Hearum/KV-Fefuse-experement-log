# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `0.7`
- measured sub-questions: `112`，warmup: `8`

## Summary

- Full attention TTFT compute mean: `0.000000s`, p50: `0.000000s`, p95: `0.000000s`
- Full attention TTFT wall mean: `0.000000s`, p50: `0.000000s`, p95: `0.000000s`
- Full attention exact-cache TTFT mean: `0.000000s`, p50: `0.000000s`, p95: `0.000000s`
- FusionRAG TTFT compute mean: `0.313460s`, p50: `0.303461s`, p95: `0.365403s`
- FusionRAG TTFT wall mean: `0.368646s`, p50: `0.360637s`, p95: `0.446906s`
- Mean compute speedup: `0.0000x`
- Mean compute TTFT reduction: `0.00%`
- Mean exact-cache speedup: `0.0000x`
- Mean exact-cache TTFT reduction: `0.00%`
- Mean wall speedup: `0.0000x`
- Mean wall TTFT reduction: `0.00%`

## FusionRAG Breakdown

- KV load/storage mean: `0.023990s`
- Selection mean: `0.102019s`
- Selection score forward mean: `0.101205s`
- Sparse recompute + query prefill mean: `0.187450s`
- Selected doc tokens mean: `1013.06`
- Reprocess prefill tokens mean: `1040.26`
- Full prompt tokens mean: `2267.12`

## Files

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0.70/clean_online_ttft_topk10_rate0.7_n120.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0.70/clean_online_ttft_topk10_rate0.7_n120_summary.json`
