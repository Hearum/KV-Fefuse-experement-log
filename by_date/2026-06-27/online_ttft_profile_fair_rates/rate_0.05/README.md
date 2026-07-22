# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `0.05`
- measured sub-questions: `112`，warmup: `8`

## Summary

- Full attention TTFT compute mean: `0.111484s`, p50: `0.113437s`, p95: `0.151749s`
- Full attention TTFT wall mean: `0.299318s`, p50: `0.291590s`, p95: `0.387763s`
- Full attention exact-cache TTFT mean: `0.298113s`, p50: `0.290184s`, p95: `0.386305s`
- FusionRAG TTFT compute mean: `0.208017s`, p50: `0.205758s`, p95: `0.227707s`
- FusionRAG TTFT wall mean: `0.265642s`, p50: `0.261851s`, p95: `0.299648s`
- Mean compute speedup: `0.5330x`
- Mean compute TTFT reduction: `-92.99%`
- Mean exact-cache speedup: `1.4260x`
- Mean exact-cache TTFT reduction: `28.12%`
- Mean wall speedup: `1.1185x`
- Mean wall TTFT reduction: `9.19%`

## FusionRAG Breakdown

- KV load/storage mean: `0.024446s`
- Selection mean: `0.102573s`
- Selection score forward mean: `0.101775s`
- Sparse recompute + query prefill mean: `0.080998s`
- Selected doc tokens mean: `71.89`
- Reprocess prefill tokens mean: `99.09`
- Full prompt tokens mean: `2267.12`

## Files

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0.05/online_ttft_profile_topk10_rate0.05_n120.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0.05/online_ttft_profile_topk10_rate0.05_n120_summary.json`
