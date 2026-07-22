# Qwen3-235B Online TTFT Profile

本实验把主模型替换为 `Qwen3-235B-A22B`，在 MuSiQue reflect 数据上复跑 online TTFT profile。离线 raw/preprocess KV 不计入 TTFT。

- QK cache root: `/home/hming/fusionrag-qwen3-235b-three-groups-unified-prompt-cache/online_qk_rate015`
- Draft cache root: `/home/hming/fusionrag-qwen3-235b-three-groups-unified-prompt-cache/online_draft_rate015`
- 主模型加载：`model_type=qwen3_moe`, `use_multi_gpu=true`, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- Draft selector：`Qwen2.5-3B-Instruct`, RRF-18 + smart selection，和原 profile 语义对齐。

## Summary

| selector | rate | measured | e2e mean | kv load | selection | score forward | update+query | overhead | selected | full tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qk | 0.0 | 57 | 2.6613 | 0.4922 | 0.0000 | 0.0000 | 1.9390 | 0.2301 | 0.0000 | 2265.0526 |
| qk | 0.05 | 57 | 5.1657 | 0.4878 | 1.8640 | 1.8630 | 2.6470 | 0.1669 | 71.7895 | 2265.0526 |
| qk | 0.15 | 57 | 5.7991 | 0.4796 | 1.8632 | 1.8621 | 3.2904 | 0.1659 | 216.4737 | 2265.0526 |
| qk | 0.3 | 57 | 6.3977 | 0.4806 | 1.8606 | 1.8596 | 3.9060 | 0.1505 | 433.4035 | 2265.0526 |
| qk | 0.5 | 57 | 7.1963 | 0.4907 | 1.8684 | 1.8674 | 4.7212 | 0.1159 | 722.7895 | 2265.0526 |
| qk | 0.7 | 57 | 7.9366 | 0.4832 | 1.8751 | 1.8741 | 5.4427 | 0.1355 | 1011.8246 | 2265.0526 |
| qk | 0.99 | 57 | 8.9226 | 0.4824 | 1.8522 | 1.8512 | 6.4404 | 0.1476 | 1431.1754 | 2265.0526 |
| qk | 1.0 | 57 | 4.2252 | 0.0000 | 0.0000 | 0.0000 | 4.2252 | 0.0000 | 1446.1404 | 2265.0526 |
| draft | 0.0 | 57 | 2.7954 | 0.5036 | 0.0000 | 0.0000 | 1.9868 | 0.3050 | 0.0000 | 2265.0526 |
| draft | 0.05 | 57 | 3.6265 | 0.5161 | 0.1405 | 0.1398 | 2.8411 | 0.1287 | 71.7895 | 2265.0526 |
| draft | 0.15 | 57 | 4.1690 | 0.5030 | 0.1447 | 0.1434 | 3.3955 | 0.1257 | 216.4737 | 2265.0526 |
| draft | 0.3 | 57 | 5.0230 | 0.5109 | 0.1441 | 0.1427 | 4.2240 | 0.1440 | 433.4035 | 2265.0526 |
| draft | 0.5 | 57 | 5.7204 | 0.5007 | 0.1432 | 0.1408 | 4.8906 | 0.1860 | 722.7895 | 2265.0526 |
| draft | 0.7 | 57 | 6.6037 | 0.5278 | 0.1436 | 0.1408 | 5.7432 | 0.1890 | 1011.8246 | 2265.0526 |
| draft | 0.99 | 57 | 7.6017 | 0.5293 | 0.1426 | 0.1410 | 6.7818 | 0.1480 | 1431.1754 | 2265.0526 |
| draft | 1.0 | 57 | 4.3114 | 0.0000 | 0.0000 | 0.0000 | 4.2989 | 0.0125 | 1446.1404 | 2265.0526 |

## Files

- aggregate CSV: `/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_qwen235b/qwen235b_profile_summary.csv`
- detail/summary JSON: `qk_rate_sweep/`, `draft_rate_sweep/`
- logs: `logs/`
