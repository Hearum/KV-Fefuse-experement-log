# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `0.3`
- measured sub-questions: `112`，warmup: `8`

## Summary

- Full attention TTFT compute mean: `0.111761s`, p50: `0.113670s`, p95: `0.155121s`
- Full attention TTFT wall mean: `0.299666s`, p50: `0.291377s`, p95: `0.390023s`
- Full attention exact-cache TTFT mean: `0.298102s`, p50: `0.290273s`, p95: `0.386362s`
- FusionRAG TTFT compute mean: `0.253968s`, p50: `0.252336s`, p95: `0.282380s`
- FusionRAG TTFT wall mean: `0.311199s`, p50: `0.307463s`, p95: `0.360741s`
- Mean compute speedup: `0.4365x`
- Mean compute TTFT reduction: `-133.88%`
- Mean exact-cache speedup: `1.1651x`
- Mean exact-cache TTFT reduction: `12.65%`
- Mean wall speedup: `0.9549x`
- Mean wall TTFT reduction: `-5.97%`

## FusionRAG Breakdown

- KV load/storage mean: `0.024549s`
- Selection mean: `0.102587s`
- Selection score forward mean: `0.101764s`
- Sparse recompute + query prefill mean: `0.126832s`
- Selected doc tokens mean: `433.93`
- Reprocess prefill tokens mean: `461.12`
- Full prompt tokens mean: `2267.12`

## Files

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0.30/online_ttft_profile_topk10_rate0.3_n120.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0.30/online_ttft_profile_topk10_rate0.3_n120_summary.json`
