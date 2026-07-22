# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `1.0`
- measured sub-questions: `3`，warmup: `1`

## Summary

- Full attention TTFT compute mean: `0.106304s`, p50: `0.108259s`, p95: `0.115229s`
- Full attention TTFT wall mean: `0.277275s`, p50: `0.276874s`, p95: `0.294541s`
- Full attention exact-cache TTFT mean: `0.276011s`, p50: `0.275589s`, p95: `0.293594s`
- FusionRAG TTFT compute mean: `0.277275s`, p50: `0.276874s`, p95: `0.294541s`
- FusionRAG TTFT wall mean: `0.277275s`, p50: `0.276874s`, p95: `0.294541s`
- Mean compute speedup: `0.3829x`
- Mean compute TTFT reduction: `-161.42%`
- Mean exact-cache speedup: `0.9954x`
- Mean exact-cache TTFT reduction: `-0.46%`
- Mean wall speedup: `1.0000x`
- Mean wall TTFT reduction: `0.00%`

## FusionRAG Breakdown

- KV load/storage mean: `0.000000s`
- Selection mean: `0.000000s`
- Selection score forward mean: `0.000000s`
- Sparse recompute + query prefill mean: `0.277275s`
- Selected doc tokens mean: `1296.67`
- Reprocess prefill tokens mean: `2112.67`
- Full prompt tokens mean: `2112.67`

## Files

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_smoke/rate_1.00/clean_online_ttft_topk10_rate1.0_n4.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_smoke/rate_1.00/clean_online_ttft_topk10_rate1.0_n4_summary.json`
