# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `0.15`
- measured sub-questions: `4`，warmup: `1`

## Summary

- Full attention TTFT mean: `0.106697s`, p50: `0.108405s`, p95: `0.114987s`
- FusionRAG TTFT mean: `0.224731s`, p50: `0.223530s`, p95: `0.230541s`
- Mean speedup: `0.4745x`
- Mean TTFT reduction: `-111.47%`

## FusionRAG Breakdown

- KV load/storage mean: `0.023976s`
- Selection mean: `0.105407s`
- Selection score forward mean: `0.104682s`
- Sparse recompute + query prefill mean: `0.095348s`
- Selected doc tokens mean: `196.25`
- Reprocess prefill tokens mean: `221.00`
- Full prompt tokens mean: `2127.50`

## Files

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_sanity/online_ttft_profile_topk10_rate0.15_n5.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_sanity/online_ttft_profile_topk10_rate0.15_n5_summary.json`
