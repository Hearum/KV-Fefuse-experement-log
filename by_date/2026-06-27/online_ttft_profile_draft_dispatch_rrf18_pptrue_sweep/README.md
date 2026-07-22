# DraftModel Selection Rate Sweep Profile

## 实验目的

这组实验固定同一个 selection method：DraftModel selection，只分析不同 rate 下 FusionRAG 在线阶段的 TTFT 和各环节开销如何变化。这里不是横向比较不同 selection method，而是回答：当 DraftModel 负责挑选重算 token 时，rate 增大后端到端收益还剩多少，以及 selection 本身是否占大头。

## 实验配置

- 仓库：`/raid/home/hming/FusionRAG-pca-analysis`
- profile 脚本：`tools_profile_draft_dispatch_rrf18_pptrue.py`
- selection 方式：DraftModel 后半层 query-to-doc attention，`DRAFT_LAYER_SEL=rrf`，`RRF_K=18`，再用 `smart_query_selection` 按 rate 选 token。
- preprocess：`preprocess=True`，`topk=10`，对齐 `dispatch_glm_overnight.sh` 的默认配置。
- rate sweep：`0, 0.05, 0.15, 0.30, 0.50, 0.70, 0.99, 1.00`。
- 样本：请求 `max_sub_questions=120`，实际前 50 个 main question 中可用子问题为 67 个；warmup 8 个，最终统计 59 个 measured sub-question。所有 rate 使用同一批样本。
- `rate=1.00`：作为 full recompute baseline，不走 DraftModel selection，不加载 cache，直接全量重算上下文。

## 汇总表

| rate | measured | E2E TTFT mean (s) | p50 (s) | p95 (s) | speedup vs full | KV load (s) | Draft selection (s) | score forward (s) | update+query (s) | extra overhead (s) | selected doc tokens | recompute/prefill tokens | full tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 59 | 0.1839 | 0.1822 | 0.2050 | 1.65x | 0.0252 | 0.0000 | 0.0000 | 0.1029 | 0.0558 | 0.0 | 26.9 | 2271.5 |
| 0.05 | 59 | 0.3368 | 0.3275 | 0.4463 | 0.90x | 0.0242 | 0.1388 | 0.1381 | 0.1155 | 0.0583 | 72.1 | 99.0 | 2271.5 |
| 0.15 | 59 | 0.3898 | 0.3680 | 0.5161 | 0.78x | 0.0246 | 0.1400 | 0.1387 | 0.1692 | 0.0560 | 217.5 | 244.3 | 2271.5 |
| 0.30 | 59 | 0.4764 | 0.4735 | 0.5880 | 0.64x | 0.0246 | 0.1393 | 0.1384 | 0.2534 | 0.0591 | 435.3 | 462.2 | 2271.5 |
| 0.50 | 59 | 0.5735 | 0.5747 | 0.7721 | 0.53x | 0.0238 | 0.1395 | 0.1385 | 0.3496 | 0.0606 | 726.0 | 752.9 | 2271.5 |
| 0.70 | 59 | 0.6844 | 0.7056 | 0.8940 | 0.44x | 0.0242 | 0.1396 | 0.1385 | 0.4576 | 0.0629 | 1016.3 | 1043.2 | 2271.5 |
| 0.99 | 59 | 0.8228 | 0.8204 | 1.1293 | 0.37x | 0.0243 | 0.1395 | 0.1383 | 0.6006 | 0.0585 | 1437.5 | 1464.4 | 2271.5 |
| 1.00 | 59 | 0.3032 | 0.2946 | 0.3914 | 1.00x | 0.0000 | 0.0000 | 0.0000 | 0.3025 | 0.0007 | 1452.6 | 2271.5 | 2271.5 |

## 结论记录

1. 在这条 DraftModel selection 路径下，非零 rate 的端到端 TTFT 全部慢于 full recompute baseline。full recompute 为 `0.3032s`，而 `rate=0.05` 已经是 `0.3368s`。因此这里的 break-even 在 `0` 和 `0.05` 之间。
2. DraftModel selection 是近似固定开销，非零 rate 下约 `0.139s`，其中大部分来自 draft score forward。这部分和选中 token 数几乎无关。
3. 在线更新/生成部分随 rate 增大而上升：`0.1155s` at 0.05，`0.1692s` at 0.15，`0.2534s` at 0.30，`0.3496s` at 0.50，`0.6006s` at 0.99。这说明稀疏重算本身会随选中 token 数增大而变慢，但系统总开销还叠加了 DraftModel selection 和 cache load。
4. `rate=0` 的 `0.1839s` 不能当成有效 FusionRAG 更新方案，只表示不选文档 token、不做 online document update 的下界。
5. 当前 profile 的客观结论是：如果 DraftModel selection 需要在线跑一遍额外 attention/forward，它在这个任务规模上不是一个划算的 selection 方案；真正可用的 selection 必须显著低于 `0.139s` 固定开销，或者被离线化/复用。

## 文件

- 汇总 CSV：`rate_sweep_summary.csv`
- 每个 rate 的原始逐样本 CSV 和 summary JSON 位于对应 `rate_*` 子目录。
