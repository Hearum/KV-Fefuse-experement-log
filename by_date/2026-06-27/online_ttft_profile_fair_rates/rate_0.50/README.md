# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `0.5`
- measured sub-questions: `112`，warmup: `8`

## Summary

- Full attention TTFT compute mean: `0.111585s`, p50: `0.113708s`, p95: `0.151830s`
- Full attention TTFT wall mean: `0.299675s`, p50: `0.291730s`, p95: `0.387907s`
- Full attention exact-cache TTFT mean: `0.298313s`, p50: `0.290566s`, p95: `0.386590s`
- FusionRAG TTFT compute mean: `0.296293s`, p50: `0.279309s`, p95: `0.335388s`
- FusionRAG TTFT wall mean: `0.352290s`, p50: `0.334555s`, p95: `0.407790s`
- Mean compute speedup: `0.3824x`
- Mean compute TTFT reduction: `-169.90%`
- Mean exact-cache speedup: `1.0230x`
- Mean exact-cache TTFT reduction: `-0.50%`
- Mean wall speedup: `0.8602x`
- Mean wall TTFT reduction: `-18.79%`

## FusionRAG Breakdown

- KV load/storage mean: `0.025375s`
- Selection mean: `0.103117s`
- Selection score forward mean: `0.102247s`
- Sparse recompute + query prefill mean: `0.167801s`
- Selected doc tokens mean: `723.70`
- Reprocess prefill tokens mean: `750.89`
- Full prompt tokens mean: `2267.12`

## Files

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0.50/online_ttft_profile_topk10_rate0.5_n120.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0.50/online_ttft_profile_topk10_rate0.5_n120_summary.json`
