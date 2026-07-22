# Online TTFT Profile

本实验只测 online 阶段。离线 raw KV cache 和 preprocess KV cache 在启动前检查存在，不在 TTFT 内计时。

- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- preprocess KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`
- topk: `10`
- FusionRAG rate: `0.15`
- measured sub-questions: `75`，warmup: `5`

## Summary

- Full attention TTFT mean: `0.113421s`, p50: `0.114281s`, p95: `0.152873s`
- FusionRAG TTFT mean: `0.230314s`, p50: `0.223396s`, p95: `0.251882s`
- Mean speedup: `0.4884x`
- Mean TTFT reduction: `-110.41%`

## FusionRAG Breakdown

- KV load/storage mean: `0.024574s`
- Selection mean: `0.102718s`
- Selection score forward mean: `0.101914s`
- Sparse recompute + query prefill mean: `0.103023s`
- Selected doc tokens mean: `221.99`
- Reprocess prefill tokens mean: `248.99`
- Full prompt tokens mean: `2301.71`

## Interpretation

这轮实验是负结果：在当前实现、当前 Musique reflect 样本长度和 `topk=10/rate=0.15` 设置下，FusionRAG 没有降低 TTFT。75 个正式计量样本中，`0/75` 个样本 FusionRAG 更快，`75/75` 个样本 full attention 更快。

FusionRAG 平均 TTFT 组成大致为：

- KV load/storage: `10.63%`
- selection: `44.75%`
- sparse recompute + query prefill: `44.63%`

因此当前瓶颈不是 offline cache 是否存在，而是 online 阶段本身：FusionRAG 需要先做一次 query-driven selection forward，随后再做 selected doc token 的 sparse recompute 和 query prefill。对于本实验里平均约 `2302` tokens 的 full prompt，dense full attention prefill 已经很快，FusionRAG 的两段 online 前向开销超过了它节省的 dense prefill 成本。

按 full prompt 长度分桶后，FusionRAG 的相对劣势会随着上下文变长而缩小，但在本轮最大 `3K-4K` token 桶内仍未反超：

| full prompt tokens | samples | full TTFT mean | FusionRAG TTFT mean | full/fusion |
| --- | ---: | ---: | ---: | ---: |
| `<=2000` | 22 | `0.0858s` | `~0.217s` | `0.3948x` |
| `2000-2500` | 31 | `0.1120s` | `~0.227s` | `0.4931x` |
| `2500-3000` | 18 | `0.1353s` | `~0.246s` | `0.5507x` |
| `3000-4000` | 4 | `0.1780s` | `~0.259s` | `0.6866x` |

当前可以支持的结论是：如果要把 FusionRAG 作为 TTFT 优化点，需要进一步降低 online selection 的固定开销，或在更长上下文、更高 full prefill 成本的设置下重新验证。否则仅靠当前 online recomputation pipeline，在这些样本上不能声称 TTFT 下降。

## Files

- detail CSV: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile/online_ttft_profile_topk10_rate0.15_n80.csv`
- summary JSON: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile/online_ttft_profile_topk10_rate0.15_n80_summary.json`
