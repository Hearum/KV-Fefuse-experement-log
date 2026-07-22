# FusionRAG 在线 TTFT Profile 汇总

## 实验设置

- 代码目录：`/raid/home/hming/FusionRAG-pca-analysis`。
- 脚本：`tools_profile_online_ttft.py`。
- 数据：MuSiQue reflect pipeline，最多 120 个 sub-question，其中前 8 个作为 warmup，不计入均值。
- 模型：Qwen2.5-7B-Instruct；GPU：`CUDA_VISIBLE_DEVICES=6`。
- FusionRAG cache：raw KV 与 preprocess KV 均使用已有 offline cache；在线阶段不重新构建 cache。
- topk：10；rate：0、0.05、0.15、0.30、0.50。
- `full_exact_s`：full attention forward，使用精确长度 StaticCache，计时前后做 CUDA sync，cache 分配不计入。这个是主要 full compute baseline。
- `fusion_compute_s`：FusionRAG 在线 compute 分解项之和，包含 storage/copy、selection、reprocess/query prefill，不包含磁盘 `torch.load`。
- `fusion_wall_s`：当前实现从进入 `load_kv_and_generate` 到返回的 wall-clock，包含 `torch.load`、Python 调度和额外实现开销。
- 原始 `full_ttft_compute` 是旧函数内部计时，存在同步口径问题，仅保存在 CSV 中作诊断，不作为主要结论。

## Rate 汇总

| rate | 样本数 | full tokens | selected doc tokens | full exact(s) | Fusion compute(s) | Fusion wall(s) | selection(s) | score forward(s) | reprocess/query(s) | compute speedup | wall speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 112 | 2267.1 | 0.0 | 0.2985 | 0.0976 | 0.1517 | 0.0000 | 0.0000 | 0.0734 | 3.06x | 1.97x |
| 0.05 | 112 | 2267.1 | 71.9 | 0.2981 | 0.2080 | 0.2656 | 0.1026 | 0.1018 | 0.0810 | 1.43x | 1.13x |
| 0.15 | 112 | 2267.1 | 216.7 | 0.2983 | 0.2296 | 0.2901 | 0.1026 | 0.1018 | 0.1024 | 1.30x | 1.03x |
| 0.30 | 112 | 2267.1 | 433.9 | 0.2981 | 0.2540 | 0.3112 | 0.1026 | 0.1018 | 0.1268 | 1.17x | 0.96x |
| 0.50 | 112 | 2267.1 | 723.7 | 0.2983 | 0.2963 | 0.3523 | 0.1031 | 0.1022 | 0.1678 | 1.01x | 0.85x |

## 相对 rate=0 的增量

| rate | 额外 selected doc tokens | Fusion compute 增量(s) | Fusion wall 增量(s) | reprocess/query 增量(s) | 额外 reprocess ms/token |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.0 | 0.0000 | 0.0000 | 0.0000 | nan |
| 0.05 | 71.9 | 0.1104 | 0.1139 | 0.0076 | 0.1059 |
| 0.15 | 216.7 | 0.1320 | 0.1384 | 0.0290 | 0.1338 |
| 0.30 | 433.9 | 0.1563 | 0.1595 | 0.0534 | 0.1232 |
| 0.50 | 723.7 | 0.1987 | 0.2006 | 0.0944 | 0.1305 |

## Fusion compute 内部占比

| rate | storage 占比 | selection 占比 | reprocess/query 占比 |
|---:|---:|---:|---:|
| 0.00 | 24.8% | 0.0% | 75.2% |
| 0.05 | 11.8% | 49.3% | 38.9% |
| 0.15 | 10.7% | 44.7% | 44.6% |
| 0.30 | 9.7% | 40.4% | 49.9% |
| 0.50 | 8.6% | 34.8% | 56.6% |

## 观察结论

1. cache reuse 本身是有效的：`rate=0` 时不做 selection 和 doc token 重算，Fusion compute 约为 full exact 的三分之一，wall-clock 也明显更低。
2. 一旦启用 FusionRAG 的 token selection，当前这套实现中 selection 约稳定在 0.102-0.103s，且 `score_forward_s` 基本等于 selection 总耗时，说明瓶颈主要来自用于打分/挑 token 的模型 forward，而不是排序或 Python topk。
3. `rate=0.15` 下 selection 与 reprocess/query 几乎同量级：二者都约 0.102s。因此说 selection 在该设置下“占大头”不完全准确；更精确地说，它是最大的固定新增项，并且和实际重算计算量同级。
4. 随 rate 增加，重算 token 数增加，但 reprocess/query 时间不是按 full attention 长度线性增长：从 `rate=0.05` 到 `rate=0.30`，selected doc tokens 从约 72 增到约 434，reprocess/query 从约 0.081s 增到约 0.127s。
5. 到 `rate=0.50` 时，Fusion compute 约 0.296s，已经基本贴近 full exact 的 0.298s；再加当前实现的 `torch.load`/Python 开销，Fusion wall 约 0.352s，端到端慢于 full wall。
6. 因此 `rate=0.15` 和 full attention 时间接近不是因为重算 15% token 本身等同于 full attention，而是 selection 固定开销、KV load/搬运开销、query prefill 固定开销叠加后吃掉了稀疏重算收益。

## FusionRAG 固有代价 vs 当前实现低效

### 更像 FusionRAG 方法固有的代价
- query-dependent selection 需要一次额外打分 forward；只要在线按 query 重新选 token，这部分就是算法路径上的固定开销。
- 被选 token 的 online update/recompute 随 rate 增加而增加；高 rate 下会自然逼近 full attention 的 prefill 成本。
- 稀疏/局部重算并不会按 token 数严格线性变快，因为 query prefill、kernel 调度、cache 管理和小规模计算利用率都会形成固定成本。

### 更像当前实现不够高效导致的代价
- `fusion_wall_s - fusion_compute_s` 约 0.05-0.06s，主要来自每次 query 通过 `torch.load` 读 KV、Python 调度和额外数据搬运；生产系统应常驻内存或服务化 cache，而不是每次从 `.pt` 反序列化。
- `storage_s` 约 0.024-0.025s，说明即使不算磁盘反序列化，CPU/GPU tensor copy/cache 写入也有固定开销。
- 当前 profile 里 selection 的 Python 排序/topk 不是主要问题，因为 `score_forward_s` 与 `selection_s` 几乎相等；真正要优化 selection，方向应是减少/合并打分 forward，而不是只优化 sort。

## 文件索引

- 汇总 CSV：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/summary_by_rate.csv`
- rate=0.00 CSV：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0/online_ttft_profile_topk10_rate0.0_n120.csv`
- rate=0.00 summary：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0/online_ttft_profile_topk10_rate0.0_n120_summary.json`
- rate=0.05 CSV：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0.05/online_ttft_profile_topk10_rate0.05_n120.csv`
- rate=0.05 summary：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0.05/online_ttft_profile_topk10_rate0.05_n120_summary.json`
- rate=0.15 CSV：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0.15/online_ttft_profile_topk10_rate0.15_n120.csv`
- rate=0.15 summary：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0.15/online_ttft_profile_topk10_rate0.15_n120_summary.json`
- rate=0.30 CSV：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0.30/online_ttft_profile_topk10_rate0.3_n120.csv`
- rate=0.30 summary：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0.30/online_ttft_profile_topk10_rate0.3_n120_summary.json`
- rate=0.50 CSV：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0.50/online_ttft_profile_topk10_rate0.5_n120.csv`
- rate=0.50 summary：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_fair_rates/rate_0.50/online_ttft_profile_topk10_rate0.5_n120_summary.json`
