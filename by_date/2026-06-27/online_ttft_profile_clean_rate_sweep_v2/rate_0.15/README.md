# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `0.15`
- measured sub-questions: `112`，warmup: `8`

## Summary

- Full attention TTFT compute mean: `0.000000s`, p50: `0.000000s`, p95: `0.000000s`
- Full attention TTFT wall mean: `0.000000s`, p50: `0.000000s`, p95: `0.000000s`
- Full attention exact-cache TTFT mean: `0.000000s`, p50: `0.000000s`, p95: `0.000000s`
- FusionRAG TTFT compute mean: `0.227947s`, p50: `0.220972s`, p95: `0.249500s`
- FusionRAG TTFT wall mean: `0.281787s`, p50: `0.274381s`, p95: `0.318646s`
- Mean compute speedup: `0.0000x`
- Mean compute TTFT reduction: `0.00%`
- Mean exact-cache speedup: `0.0000x`
- Mean exact-cache TTFT reduction: `0.00%`
- Mean wall speedup: `0.0000x`
- Mean wall TTFT reduction: `0.00%`

## FusionRAG Breakdown

- KV load/storage mean: `0.024036s`
- Selection mean: `0.101922s`
- Selection score forward mean: `0.101130s`
- Sparse recompute + query prefill mean: `0.101988s`
- Selected doc tokens mean: `216.73`
- Reprocess prefill tokens mean: `243.93`
- Full prompt tokens mean: `2267.12`

## Files

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0.15/clean_online_ttft_topk10_rate0.15_n120.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0.15/clean_online_ttft_topk10_rate0.15_n120_summary.json`
