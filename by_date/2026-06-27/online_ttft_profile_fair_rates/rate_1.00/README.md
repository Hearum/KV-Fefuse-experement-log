# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `1.0`
- measured sub-questions: `112`，warmup: `8`

## Summary

- Full attention TTFT compute mean: `0.111635s`, p50: `0.113640s`, p95: `0.156378s`
- Full attention TTFT wall mean: `0.299647s`, p50: `0.291845s`, p95: `0.392470s`
- Full attention exact-cache TTFT mean: `0.298425s`, p50: `0.290613s`, p95: `0.386620s`
- FusionRAG TTFT compute mean: `0.368300s`, p50: `0.356019s`, p95: `0.453071s`
- FusionRAG TTFT wall mean: `0.424910s`, p50: `0.409994s`, p95: `0.527263s`
- Mean compute speedup: `0.3034x`
- Mean compute TTFT reduction: `-235.75%`
- Mean exact-cache speedup: `0.8119x`
- Mean exact-cache TTFT reduction: `-25.11%`
- Mean wall speedup: `0.7060x`
- Mean wall TTFT reduction: `-43.48%`

## FusionRAG Breakdown

- KV load/storage mean: `0.023884s`
- Selection mean: `0.102778s`
- Selection score forward mean: `0.101967s`
- Sparse recompute + query prefill mean: `0.241638s`
- Selected doc tokens mean: `1447.92`
- Reprocess prefill tokens mean: `1475.12`
- Full prompt tokens mean: `2267.12`

## Files

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_1.00/online_ttft_profile_topk10_rate1.0_n120.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_1.00/online_ttft_profile_topk10_rate1.0_n120_summary.json`
