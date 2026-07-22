# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `0.0`
- measured sub-questions: `112`，warmup: `8`

## Summary

- Full attention TTFT compute mean: `0.111543s`, p50: `0.113560s`, p95: `0.151640s`
- Full attention TTFT wall mean: `0.299471s`, p50: `0.291555s`, p95: `0.387807s`
- Full attention exact-cache TTFT mean: `0.298297s`, p50: `0.290432s`, p95: `0.386490s`
- FusionRAG TTFT compute mean: `0.097314s`, p50: `0.096703s`, p95: `0.103544s`
- FusionRAG TTFT wall mean: `0.157592s`, p50: `0.155264s`, p95: `0.189794s`
- Mean compute speedup: `1.1399x`
- Mean compute TTFT reduction: `9.77%`
- Mean exact-cache speedup: `3.0495x`
- Mean exact-cache TTFT reduction: `66.40%`
- Mean wall speedup: `1.8874x`
- Mean wall TTFT reduction: `46.40%`

## FusionRAG Breakdown

- KV load/storage mean: `0.024276s`
- Selection mean: `0.000000s`
- Selection score forward mean: `0.000000s`
- Sparse recompute + query prefill mean: `0.073037s`
- Selected doc tokens mean: `0.00`
- Reprocess prefill tokens mean: `27.20`
- Full prompt tokens mean: `2267.12`

## Files

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep/rate_0/clean_online_ttft_topk10_rate0.0_n120.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep/rate_0/clean_online_ttft_topk10_rate0.0_n120_summary.json`
