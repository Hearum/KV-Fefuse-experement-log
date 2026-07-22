# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/home/hming/fusionrag-qwen3-235b-three-groups-unified-prompt-cache/online_qk_rate015/Qwen3-235B-A22B/musique/kv_cache`
- preprocess KV cache: `/home/hming/fusionrag-qwen3-235b-three-groups-unified-prompt-cache/online_qk_rate015/Qwen3-235B-A22B/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `1.0`
- measured sub-questions: `1`，warmup: `0`

## Summary

- Full attention TTFT compute mean: `4.136543s`, p50: `4.136543s`, p95: `4.136543s`
- Full attention TTFT wall mean: `4.145804s`, p50: `4.145804s`, p95: `4.145804s`
- Full attention exact-cache TTFT mean: `5.553009s`, p50: `5.553009s`, p95: `5.553009s`
- FusionRAG TTFT compute mean: `4.145804s`, p50: `4.145804s`, p95: `4.145804s`
- FusionRAG TTFT wall mean: `4.145804s`, p50: `4.145804s`, p95: `4.145804s`
- Mean compute speedup: `0.9978x`
- Mean compute TTFT reduction: `-0.22%`
- Mean exact-cache speedup: `1.3394x`
- Mean exact-cache TTFT reduction: `25.34%`
- Mean wall speedup: `1.0000x`
- Mean wall TTFT reduction: `0.00%`

## FusionRAG Breakdown

- KV load/storage mean: `0.000000s`
- Selection mean: `0.000000s`
- Selection score forward mean: `0.000000s`
- Sparse recompute + query prefill mean: `4.145804s`
- Selected doc tokens mean: `1313.00`
- Reprocess prefill tokens mean: `2132.00`
- Full prompt tokens mean: `2132.00`

## Files

- detail CSV: `/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_qwen235b_smoke/clean_online_ttft_topk10_rate1.0_n1.csv`
- summary JSON: `/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_qwen235b_smoke/clean_online_ttft_topk10_rate1.0_n1_summary.json`
