# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/home/hming/fusionrag-qwen3-235b-three-groups-unified-prompt-cache/online_qk_rate015/Qwen3-235B-A22B/musique/kv_cache`
- preprocess KV cache: `/home/hming/fusionrag-qwen3-235b-three-groups-unified-prompt-cache/online_qk_rate015/Qwen3-235B-A22B/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `1.0`
- measured sub-questions: `57`，warmup: `3`

## Summary

- Full attention TTFT compute mean: `4.216908s`, p50: `4.171261s`, p95: `4.743839s`
- Full attention TTFT wall mean: `4.225206s`, p50: `4.177263s`, p95: `4.751924s`
- Full attention exact-cache TTFT mean: `4.240915s`, p50: `4.159652s`, p95: `4.726302s`
- FusionRAG TTFT compute mean: `4.225206s`, p50: `4.177263s`, p95: `4.751924s`
- FusionRAG TTFT wall mean: `4.225206s`, p50: `4.177263s`, p95: `4.751924s`
- Mean compute speedup: `0.9980x`
- Mean compute TTFT reduction: `-0.20%`
- Mean exact-cache speedup: `1.0045x`
- Mean exact-cache TTFT reduction: `0.33%`
- Mean wall speedup: `1.0000x`
- Mean wall TTFT reduction: `0.00%`

## FusionRAG Breakdown

- KV load/storage mean: `0.000000s`
- Selection mean: `0.000000s`
- Selection score forward mean: `0.000000s`
- Sparse recompute + query prefill mean: `4.225206s`
- Selected doc tokens mean: `1446.14`
- Reprocess prefill tokens mean: `2265.05`
- Full prompt tokens mean: `2265.05`

## Files

- detail CSV: `/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_qwen235b/qk_rate_sweep/clean_online_ttft_topk10_rate1.0_n60.csv`
- summary JSON: `/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_qwen235b/qk_rate_sweep/clean_online_ttft_topk10_rate1.0_n60_summary.json`
