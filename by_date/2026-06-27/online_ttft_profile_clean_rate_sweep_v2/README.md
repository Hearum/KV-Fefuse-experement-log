# Clean Rate Sweep: FusionRAG Online TTFT

## 口径

- `rate < 1`：FusionRAG cache reuse 路径，即使用离线 preprocess KV，先 selection，再按 rate 更新部分 doc token，最后计算 query/first token。
- `rate = 1`：full recompute baseline，不使用离线 KV，不做 selection，直接完整上下文 prefill 到 first token。
- `端到端 TTFT`：当前路径从在线处理开始到 first token 的 wall-clock。
- `相比 rate=1 加速`：`rate=1 端到端 TTFT / 当前 rate 端到端 TTFT`。大于 1 表示快于完全重算，小于 1 表示慢于完全重算。
- 每组最多 120 个 sub-question，前 8 个 warmup 不计入均值；有效样本数为 112。

## Baseline

- `rate=1.00` full recompute 端到端 TTFT：`0.2996s`

## 主表

| rate | 语义 | 端到端 TTFT(s) | 相比 rate=1 加速 | selected doc tokens | KV load/copy(s) | selection(s) | online update + query prefill(s) | 额外实现开销(s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | cache reuse，不更新 doc KV | 0.1518 | 1.97x | 0.0 | 0.0240 | 0.0000 | 0.0725 | 0.0552 |
| 0.05 | selection 后更新 5% doc token | 0.2605 | 1.15x | 71.9 | 0.0242 | 0.1021 | 0.0807 | 0.0535 |
| 0.15 | selection 后更新 15% doc token | 0.2818 | 1.06x | 216.7 | 0.0240 | 0.1019 | 0.1020 | 0.0538 |
| 0.30 | selection 后更新 30% doc token | 0.3085 | 0.97x | 433.9 | 0.0243 | 0.1019 | 0.1265 | 0.0559 |
| 0.50 | selection 后更新 50% doc token | 0.3428 | 0.87x | 723.7 | 0.0242 | 0.1021 | 0.1564 | 0.0600 |
| 0.70 | selection 后更新 70% doc token | 0.3686 | 0.81x | 1013.1 | 0.0240 | 0.1020 | 0.1875 | 0.0552 |
| 0.99 | selection 后更新 99% doc token | 0.4134 | 0.72x | 1432.9 | 0.0239 | 0.1019 | 0.2340 | 0.0536 |
| 1.00 | full recompute，不用 cache，不 selection | 0.2996 | 1.00x | 1447.9 | 0.0000 | 0.0000 | 0.2996 | 0.0000 |

## 直接结论

- 在这批样本和当前实现下，快于 full recompute 的最大测试 rate 是 `0.15`。
- 从测试点看，`rate=0.30` 开始端到端已经慢于 full recompute。
- `rate=0.15` 仍略快于 full recompute，但 margin 很小；`rate=0.30` 已经慢于 full recompute。
- selection 基本稳定在约 0.102s，说明它是 rate>0 后的固定开销；rate 增大主要增加 `online update + query prefill`。

## 文件

- 汇总 CSV：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/clean_rate_sweep_summary.csv`
- rate=0.00 detail CSV：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0/clean_online_ttft_topk10_rate0.0_n120.csv`
- rate=0.00 summary JSON：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0/clean_online_ttft_topk10_rate0.0_n120_summary.json`
- rate=0.05 detail CSV：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0.05/clean_online_ttft_topk10_rate0.05_n120.csv`
- rate=0.05 summary JSON：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0.05/clean_online_ttft_topk10_rate0.05_n120_summary.json`
- rate=0.15 detail CSV：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0.15/clean_online_ttft_topk10_rate0.15_n120.csv`
- rate=0.15 summary JSON：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0.15/clean_online_ttft_topk10_rate0.15_n120_summary.json`
- rate=0.30 detail CSV：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0.30/clean_online_ttft_topk10_rate0.3_n120.csv`
- rate=0.30 summary JSON：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0.30/clean_online_ttft_topk10_rate0.3_n120_summary.json`
- rate=0.50 detail CSV：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0.50/clean_online_ttft_topk10_rate0.5_n120.csv`
- rate=0.50 summary JSON：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0.50/clean_online_ttft_topk10_rate0.5_n120_summary.json`
- rate=0.70 detail CSV：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0.70/clean_online_ttft_topk10_rate0.7_n120.csv`
- rate=0.70 summary JSON：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0.70/clean_online_ttft_topk10_rate0.7_n120_summary.json`
- rate=0.99 detail CSV：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0.99/clean_online_ttft_topk10_rate0.99_n120.csv`
- rate=0.99 summary JSON：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_0.99/clean_online_ttft_topk10_rate0.99_n120_summary.json`
- rate=1.00 detail CSV：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_1.00/clean_online_ttft_topk10_rate1.0_n120.csv`
- rate=1.00 summary JSON：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/rate_1.00/clean_online_ttft_topk10_rate1.0_n120_summary.json`
