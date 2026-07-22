# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `0.15`
- measured sub-questions: `12`，warmup: `4`

## Summary

- Full attention TTFT compute mean: `0.000000s`, p50: `0.000000s`, p95: `0.000000s`
- Full attention TTFT wall mean: `0.000000s`, p50: `0.000000s`, p95: `0.000000s`
- Full attention exact-cache TTFT mean: `0.000000s`, p50: `0.000000s`, p95: `0.000000s`
- FusionRAG TTFT compute mean: `0.314883s`, p50: `0.311702s`, p95: `0.353572s`
- FusionRAG TTFT wall mean: `0.384840s`, p50: `0.373327s`, p95: `0.442532s`
- Mean compute speedup: `0.0000x`
- Mean compute TTFT reduction: `0.00%`
- Mean exact-cache speedup: `0.0000x`
- Mean exact-cache TTFT reduction: `0.00%`
- Mean wall speedup: `0.0000x`
- Mean wall TTFT reduction: `0.00%`

## FusionRAG Breakdown

- KV load/storage mean: `0.100638s`
- Selection mean: `0.109330s`
- Selection score forward mean: `0.108464s`
- Sparse recompute + query prefill mean: `0.104914s`
- Selected doc tokens mean: `243.75`
- Reprocess prefill tokens mean: `270.33`
- Full prompt tokens mean: `2447.50`

## Files

- detail CSV: `MOTIVATION_EXPERIMENTS/kv_load_investigation/qk_current/clean_online_ttft_topk10_rate0.15_n16.csv`
- summary JSON: `MOTIVATION_EXPERIMENTS/kv_load_investigation/qk_current/clean_online_ttft_topk10_rate0.15_n16_summary.json`
