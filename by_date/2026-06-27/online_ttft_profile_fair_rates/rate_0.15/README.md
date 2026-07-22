# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `0.15`
- measured sub-questions: `112`，warmup: `8`

## Summary

- Full attention TTFT compute mean: `0.111544s`, p50: `0.113644s`, p95: `0.151817s`
- Full attention TTFT wall mean: `0.299480s`, p50: `0.291715s`, p95: `0.388410s`
- Full attention exact-cache TTFT mean: `0.298323s`, p50: `0.290431s`, p95: `0.387482s`
- FusionRAG TTFT compute mean: `0.229595s`, p50: `0.223085s`, p95: `0.251278s`
- FusionRAG TTFT wall mean: `0.290129s`, p50: `0.280799s`, p95: `0.333042s`
- Mean compute speedup: `0.4824x`
- Mean compute TTFT reduction: `-112.25%`
- Mean exact-cache speedup: `1.2906x`
- Mean exact-cache TTFT reduction: `21.01%`
- Mean wall speedup: `1.0240x`
- Mean wall TTFT reduction: `1.11%`

## FusionRAG Breakdown

- KV load/storage mean: `0.024587s`
- Selection mean: `0.102622s`
- Selection score forward mean: `0.101818s`
- Sparse recompute + query prefill mean: `0.102387s`
- Selected doc tokens mean: `216.73`
- Reprocess prefill tokens mean: `243.93`
- Full prompt tokens mean: `2267.12`

## Files

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0.15/online_ttft_profile_topk10_rate0.15_n120.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0.15/online_ttft_profile_topk10_rate0.15_n120_summary.json`
