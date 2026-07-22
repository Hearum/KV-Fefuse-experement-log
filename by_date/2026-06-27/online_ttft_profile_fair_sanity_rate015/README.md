# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `0.15`
- measured sub-questions: `6`，warmup: `2`

## Summary

- Full attention TTFT compute mean: `0.113540s`, p50: `0.108203s`, p95: `0.152991s`
- Full attention TTFT wall mean: `0.298593s`, p50: `0.288365s`, p95: `0.392103s`
- Full attention exact-cache TTFT mean: `0.295394s`, p50: `0.286082s`, p95: `0.389370s`
- FusionRAG TTFT compute mean: `0.229739s`, p50: `0.224050s`, p95: `0.253406s`
- FusionRAG TTFT wall mean: `0.297966s`, p50: `0.288938s`, p95: `0.343137s`
- Mean compute speedup: `0.4916x`
- Mean compute TTFT reduction: `-105.67%`
- Mean exact-cache speedup: `1.2801x`
- Mean exact-cache TTFT reduction: `21.21%`
- Mean wall speedup: `0.9974x`
- Mean wall TTFT reduction: `-0.77%`

## FusionRAG Breakdown

- KV load/storage mean: `0.025660s`
- Selection mean: `0.104395s`
- Selection score forward mean: `0.103574s`
- Sparse recompute + query prefill mean: `0.099684s`
- Selected doc tokens mean: `216.83`
- Reprocess prefill tokens mean: `242.00`
- Full prompt tokens mean: `2265.67`

## Files

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_sanity_rate015/online_ttft_profile_topk10_rate0.15_n8.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_sanity_rate015/online_ttft_profile_topk10_rate0.15_n8_summary.json`
