# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `0.0`
- measured sub-questions: `112`，warmup: `8`

## Summary

- Full attention TTFT compute mean: `0.111550s`, p50: `0.113487s`, p95: `0.151871s`
- Full attention TTFT wall mean: `0.299537s`, p50: `0.291656s`, p95: `0.387912s`
- Full attention exact-cache TTFT mean: `0.298453s`, p50: `0.290729s`, p95: `0.386683s`
- FusionRAG TTFT compute mean: `0.097640s`, p50: `0.096865s`, p95: `0.103577s`
- FusionRAG TTFT wall mean: `0.151721s`, p50: `0.150103s`, p95: `0.172686s`
- Mean compute speedup: `1.1361x`
- Mean compute TTFT reduction: `9.47%`
- Mean exact-cache speedup: `3.0410x`
- Mean exact-cache TTFT reduction: `66.31%`
- Mean wall speedup: `1.9585x`
- Mean wall TTFT reduction: `48.27%`

## FusionRAG Breakdown

- KV load/storage mean: `0.024253s`
- Selection mean: `0.000000s`
- Selection score forward mean: `0.000000s`
- Sparse recompute + query prefill mean: `0.073387s`
- Selected doc tokens mean: `0.00`
- Reprocess prefill tokens mean: `27.20`
- Full prompt tokens mean: `2267.12`

## Files

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0/online_ttft_profile_topk10_rate0.0_n120.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0/online_ttft_profile_topk10_rate0.0_n120_summary.json`
