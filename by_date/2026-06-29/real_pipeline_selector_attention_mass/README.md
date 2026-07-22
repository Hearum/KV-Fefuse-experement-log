# Real Pipeline Selector Attention Mass

## 实验目的

验证不同 selector 选出的重算 token，在真实 FusionRAG online 流程中是否承接后续 query / decode attention。

## 实验设置

- 数据：`data/result_reflect.json` 中可测试的 MuSiQue reflect 子问题。
- KV：使用 `preprocess_kv_cache_global_topk10_bge` 作为离线 preprocess KV。
- 真实流程：加载 preprocess KV -> selector 挑选 rate 比例 doc token -> 对选中 doc token 做 online KV recompute -> query prefill -> 首 token decode。
- selectors：`fusionrag_qk` 使用 FusionRAG 原 QK/importance selector；`draft` 使用 3B draft model 的后半层 query-to-doc attention，经 RRF(k=18)+smart selection；`random` 是等数量随机 token。
- rates：0.05, 0.15, 0.30, 0.50, 0.80。
- attention mass：在 recompute 后的当前 KV 上，用 full-softmax 诊断重新计算 attention 分布，统计落在 selector 选中 doc token 上的比例。该统计用于分析 token 是否真的被后续 attention 使用，不等价于 selector 自身分数。
- 统计阶段：`query_mass_*` 是 query prefill 阶段；`decode_mass_*` 是首个生成 token decode 阶段。

## 汇总表

| selector | rate | n | selected | query mass mean | query last | decode mass mean | decode last | selection s | recompute s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| draft | 0.0500 | 109 | 71.9725 | 0.0778 | 0.0713 | 0.0782 | 0.0991 | 0.1398 | 0.0362 |
| draft | 0.1500 | 109 | 216.9725 | 0.1457 | 0.1340 | 0.1473 | 0.1806 | 0.1419 | 0.0536 |
| draft | 0.3000 | 109 | 434.4037 | 0.1910 | 0.1862 | 0.1935 | 0.2475 | 0.1404 | 0.0784 |
| draft | 0.5000 | 109 | 724.4862 | 0.2297 | 0.2349 | 0.2289 | 0.2970 | 0.1412 | 0.1081 |
| draft | 0.8000 | 109 | 1159.2110 | 0.2647 | 0.2852 | 0.2636 | 0.3679 | 0.1396 | 0.1542 |
| fusionrag_qk | 0.0500 | 109 | 71.9725 | 0.0476 | 0.0563 | 0.0403 | 0.0432 | 0.0304 | 0.0342 |
| fusionrag_qk | 0.1500 | 109 | 216.9725 | 0.1054 | 0.1182 | 0.0973 | 0.1152 | 0.0303 | 0.0524 |
| fusionrag_qk | 0.3000 | 109 | 434.4037 | 0.1541 | 0.1715 | 0.1455 | 0.1706 | 0.0314 | 0.0770 |
| fusionrag_qk | 0.5000 | 109 | 724.4862 | 0.1968 | 0.2137 | 0.1860 | 0.2208 | 0.0295 | 0.1072 |
| fusionrag_qk | 0.8000 | 109 | 1159.2110 | 0.2385 | 0.2530 | 0.2239 | 0.2899 | 0.0309 | 0.1541 |
| random | 0.0500 | 109 | 71.9725 | 0.0296 | 0.0392 | 0.0269 | 0.0425 | 0.0000 | 0.0357 |
| random | 0.1500 | 109 | 216.9725 | 0.0686 | 0.0822 | 0.0615 | 0.0904 | 0.0000 | 0.0540 |
| random | 0.3000 | 109 | 434.4037 | 0.1166 | 0.1261 | 0.1049 | 0.1343 | 0.0000 | 0.0788 |
| random | 0.5000 | 109 | 724.4862 | 0.1720 | 0.1740 | 0.1619 | 0.2065 | 0.0000 | 0.1082 |
| random | 0.8000 | 109 | 1159.2110 | 0.2437 | 0.2489 | 0.2381 | 0.2995 | 0.0000 | 0.1555 |

## 相对 Random 的 attention mass 富集

| selector | rate | query mass / random | decode mass / random |
|---|---:|---:|---:|
| draft | 0.0500 | 2.6254x | 2.9051x |
| draft | 0.1500 | 2.1242x | 2.3970x |
| draft | 0.3000 | 1.6384x | 1.8445x |
| draft | 0.5000 | 1.3352x | 1.4141x |
| draft | 0.8000 | 1.0861x | 1.1072x |
| fusionrag_qk | 0.0500 | 1.6065x | 1.4972x |
| fusionrag_qk | 0.1500 | 1.5360x | 1.5837x |
| fusionrag_qk | 0.3000 | 1.3219x | 1.3868x |
| fusionrag_qk | 0.5000 | 1.1441x | 1.1489x |
| fusionrag_qk | 0.8000 | 0.9787x | 0.9402x |

## 文件

- 明细 CSV：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/real_pipeline_selector_attention_mass/detail_<selector>_rate*.csv`
- 汇总 CSV：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/real_pipeline_selector_attention_mass/summary.csv`
- JSON 汇总：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/real_pipeline_selector_attention_mass/summary.json`

## 当前备注

- 如果某个 selector 的 mass 接近 random，说明它虽然选中了 token 并触发 recompute，但这些 token 在真实 query/decode attention 中未形成明显集中。
- 如果 draft 或 FusionRAG-QK 显著高于 random，则说明 selector 的重算集合确实覆盖了后续 attention 更关注的 token。

