# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `0.15`
- measured sub-questions: `6`，warmup: `2`

## Summary

- Full attention TTFT compute mean: `0.112799s`, p50: `0.108171s`, p95: `0.153192s`
- Full attention TTFT wall mean: `0.298345s`, p50: `0.288839s`, p95: `0.392280s`
- Full attention reuse-cache TTFT mean: `0.878943s`, p50: `0.859698s`, p95: `1.109704s`
- FusionRAG TTFT compute mean: `0.229582s`, p50: `0.225826s`, p95: `0.252679s`
- FusionRAG TTFT wall mean: `0.286885s`, p50: `0.278999s`, p95: `0.328063s`
- Mean compute speedup: `0.4889x`
- Mean compute TTFT reduction: `-107.46%`
- Mean reuse-cache speedup: `3.8172x`
- Mean reuse-cache TTFT reduction: `73.66%`
- Mean wall speedup: `1.0350x`
- Mean wall TTFT reduction: `2.84%`

## FusionRAG Breakdown

- KV load/storage mean: `0.025144s`
- Selection mean: `0.104877s`
- Selection score forward mean: `0.104118s`
- Sparse recompute + query prefill mean: `0.099561s`
- Selected doc tokens mean: `216.83`
- Reprocess prefill tokens mean: `242.00`
- Full prompt tokens mean: `2265.67`

## Files

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_reuse_sanity_rate015/online_ttft_profile_topk10_rate0.15_n8.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_reuse_sanity_rate015/online_ttft_profile_topk10_rate0.15_n8_summary.json`
