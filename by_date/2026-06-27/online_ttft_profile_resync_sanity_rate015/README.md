# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `0.15`
- measured sub-questions: `6`，warmup: `2`

## Summary

- Full attention TTFT compute mean: `0.112505s`, p50: `0.107501s`, p95: `0.152941s`
- Full attention TTFT wall mean: `0.296666s`, p50: `0.287487s`, p95: `0.389973s`
- FusionRAG TTFT compute mean: `0.230127s`, p50: `0.225374s`, p95: `0.255255s`
- FusionRAG TTFT wall mean: `0.293214s`, p50: `0.285691s`, p95: `0.338244s`
- Mean compute speedup: `0.4865x`
- Mean compute TTFT reduction: `-108.46%`
- Mean wall speedup: `1.0074x`
- Mean wall TTFT reduction: `0.18%`

## FusionRAG Breakdown

- KV load/storage mean: `0.025635s`
- Selection mean: `0.103726s`
- Selection score forward mean: `0.102985s`
- Sparse recompute + query prefill mean: `0.100766s`
- Selected doc tokens mean: `216.83`
- Reprocess prefill tokens mean: `242.00`
- Full prompt tokens mean: `2265.67`

## Files

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_resync_sanity_rate015/online_ttft_profile_topk10_rate0.15_n8.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_resync_sanity_rate015/online_ttft_profile_topk10_rate0.15_n8_summary.json`
