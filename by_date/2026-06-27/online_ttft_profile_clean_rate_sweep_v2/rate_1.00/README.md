# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `1.0`
- measured sub-questions: `112`，warmup: `8`

## Summary

- Full attention TTFT compute mean: `0.111595s`, p50: `0.113518s`, p95: `0.151724s`
- Full attention TTFT wall mean: `0.299576s`, p50: `0.291603s`, p95: `0.388893s`
- Full attention exact-cache TTFT mean: `0.298444s`, p50: `0.290443s`, p95: `0.386486s`
- FusionRAG TTFT compute mean: `0.299576s`, p50: `0.291603s`, p95: `0.388893s`
- FusionRAG TTFT wall mean: `0.299576s`, p50: `0.291603s`, p95: `0.388893s`
- Mean compute speedup: `0.3718x`
- Mean compute TTFT reduction: `-169.28%`
- Mean exact-cache speedup: `0.9962x`
- Mean exact-cache TTFT reduction: `-0.39%`
- Mean wall speedup: `1.0000x`
- Mean wall TTFT reduction: `0.00%`

## FusionRAG Breakdown

- KV load/storage mean: `0.000000s`
- Selection mean: `0.000000s`
- Selection score forward mean: `0.000000s`
- Sparse recompute + query prefill mean: `0.299576s`
- Selected doc tokens mean: `1447.92`
- Reprocess prefill tokens mean: `2267.12`
- Full prompt tokens mean: `2267.12`

## Files

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_1.00/clean_online_ttft_topk10_rate1.0_n120.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_1.00/clean_online_ttft_topk10_rate1.0_n120_summary.json`
