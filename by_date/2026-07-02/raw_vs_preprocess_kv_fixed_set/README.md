# Raw KV vs Preprocess KV 对 fixed-set / online selector 的影响

## 实验问题

前面 offline fixed set 的候选 token 很多是在 raw document KV 或 selector score cache 上得到的，但生成阶段默认加载的是 preprocess 后的 KV。这里需要拆开两个因素：

1. selector / fixed set 本身是否有效；
2. runtime 使用 preprocess KV 还是 raw document KV。

核心判断：

- 如果 raw KV 明显掉点，说明 preprocess KV 本身仍然提供必要质量收益；fixed set 只是降低 online selection 开销。
- 如果 raw KV 与 preprocess KV 接近，说明 preprocess 阶段可能不是必须，可以考虑省掉 preprocess cache 构建和存储。
- 如果 raw KV 反而更好，说明 fixed set 的选择空间与 preprocess KV 存在错配，需要重新定义 fixed set 的离线构造方式。

## 固定实验设置

- 数据集：`data/result_reflect.json`，完整 200 个 examples；有效评测口径与前面一致，约 135 个 main questions / 248 个 sub questions。
- 主模型：`/mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct`。
- Draft model：`/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`。
- BGE：`/mnt/qjhs-sh-lab-01/models/bge-m3`。
- Cache root：`/raid/home/hming/fusionrag-reflect-full-cache`。
- Judge：OpenAI-compatible GLM-5.2 服务。
- topk：10。
- rate：除 rate=0 对照外，统一使用 0.15。
- preprocess=true：生成阶段加载 preprocess 后的 KV cache。
- preprocess=false：生成阶段加载 raw document KV cache，不做 preprocess KV 替换。

## 本轮新增 raw-KV 运行矩阵

| label | reprocess_method | selector / fixed set | runtime KV | rate |
|---|---|---|---|---:|
| raw_rate0_no_doc_recompute | FusionRAG | 不重算 doc token | raw doc KV | 0.00 |
| raw_online_qk_rate015 | FusionRAG | online FusionRAG-QK selector | raw doc KV | 0.15 |
| raw_offline_qk_frequency | FusionRAG | offline QK frequency fixed set | raw doc KV | 0.15 |
| raw_offline_qk_mean | FusionRAG | offline QK mean-score fixed set | raw doc KV | 0.15 |
| raw_offline_draft_frequency | FusionRAG | offline draft frequency fixed set | raw doc KV | 0.15 |
| raw_offline_draft_mean | FusionRAG | offline draft mean-score fixed set | raw doc KV | 0.15 |
| raw_hybrid_draft70_qk30 | FusionRAG | offline hybrid draft70/qk30 fixed set | raw doc KV | 0.15 |
| raw_online_draft_profile_sparse | DraftModel | online draft selector，强制 sparse attention | raw doc KV | 0.15 |

## 已有 preprocess-KV 对照

对照来自：

- `MOTIVATION_EXPERIMENTS/full_accuracy_offline_selector_reflect_summary/summary.json`
- `MOTIVATION_EXPERIMENTS/full_accuracy_offline_hybrid70_rate_sweep_reflect_summary/summary.json`
- `MOTIVATION_EXPERIMENTS/online_draft_impl_audit/full_profile_smart_sparse_rate015_summary.json`

本目录后续会补充 `raw_vs_preprocess_summary.csv/json`，把 preprocess=true 与 raw KV 新结果放到一张表里比较。

## 启动方式

见 `launch_raw_vs_preprocess.sh`。本轮按 8 张 GPU 并行启动，每个实验 full dataset 一进程。

## 实验日志

- 2026-07-02：代码已先保存 checkpoint commit `5b62705`，随后开始 raw KV vs preprocess KV 对照实验。
## 当前结果

### Raw KV 单独结果

| label | finished | Main Acc | Sub Acc | F1 | EM | prefill(s) | storage(s) | selection(s) | rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_rate0_no_doc_recompute | True | 64/135 (47.41%) | 159/248 (64.11%) | 0.4271 | 0.1371 | 0.1014 | 0.0256 | 0.0000 | 248 |
| raw_online_qk_rate015 | True | 78/135 (57.78%) | 178/248 (71.77%) | 0.4703 | 0.1935 | 0.2334 | 0.0262 | 0.1038 | 248 |
| raw_offline_qk_frequency | True | 67/135 (49.63%) | 168/248 (67.74%) | 0.4525 | 0.1734 | 0.1320 | 0.0265 | 0.0000 | 248 |
| raw_offline_qk_mean | True | 70/135 (51.85%) | 170/248 (68.55%) | 0.4519 | 0.1734 | 0.1318 | 0.0261 | 0.0000 | 248 |
| raw_offline_draft_frequency | True | 77/135 (57.04%) | 174/248 (70.16%) | 0.4709 | 0.1815 | 0.1301 | 0.0259 | 0.0000 | 248 |
| raw_offline_draft_mean | True | 76/135 (56.30%) | 175/248 (70.56%) | 0.4663 | 0.1774 | 0.1297 | 0.0255 | 0.0000 | 248 |
| raw_hybrid_draft70_qk30 | True | 74/135 (54.81%) | 169/248 (68.15%) | 0.4568 | 0.1815 | 0.1301 | 0.0257 | 0.0000 | 248 |
| raw_online_draft_profile_sparse | True | 91/135 (67.41%) | 194/248 (78.23%) | 0.5077 | 0.2137 | 0.2685 | 0.0258 | 0.1389 | 248 |

### Paired raw - preprocess 差值

| pair | rate | preprocess Main | raw Main | ΔMain | preprocess Sub | raw Sub | ΔSub | preprocess F1 | raw F1 | ΔF1 | raw rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| online_qk_rate015 | 0.15 | 62.22% | 57.78% | -4.44% | 75.40% | 71.77% | -3.63% | 0.4897 | 0.4703 | -0.0194 | 248 |
| rate0_no_doc_recompute | 0.00 | 55.56% | 47.41% | -8.15% | 69.76% | 64.11% | -5.65% | 0.4526 | 0.4271 | -0.0255 | 248 |
| offline_qk_frequency | 0.15 | 58.52% | 49.63% | -8.89% | 72.18% | 67.74% | -4.44% | 0.4696 | 0.4525 | -0.0170 | 248 |
| offline_qk_mean | 0.15 | 56.30% | 51.85% | -4.44% | 70.97% | 68.55% | -2.42% | 0.4766 | 0.4519 | -0.0247 | 248 |
| offline_draft_frequency | 0.15 | 62.96% | 57.04% | -5.93% | 73.79% | 70.16% | -3.63% | 0.4909 | 0.4709 | -0.0200 | 248 |
| offline_draft_mean | 0.15 | 62.22% | 56.30% | -5.93% | 73.79% | 70.56% | -3.23% | 0.4916 | 0.4663 | -0.0253 | 248 |
| hybrid_draft70_qk30 | 0.15 | 65.19% | 54.81% | -10.37% | 75.00% | 68.15% | -6.85% | 0.4872 | 0.4568 | -0.0304 | 248 |
| online_draft_profile_sparse | 0.15 | 74.07% | 67.41% | -6.67% | 83.06% | 78.23% | -4.84% | 0.5410 | 0.5077 | -0.0333 | 248 |


## 关于 offline 是否在 preprocess KV 上挑选

现有 `offline_qk_frequency` / `offline_qk_mean` 已经是在 preprocess KV cache 上挑选 token。对应脚本是 `tools_reflect_offline_qk_fixed_sets.py`，它加载：

`/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge`

然后用 calibration queries 计算 query-to-context attention，按每个 chunk 得到 `qk_frequency_per_chunk` 或 `qk_mean_score_per_chunk` fixed set。也就是说，这两行结果的口径是：offline 在 preprocess KV 上选 token，online 推理也使用 preprocess KV。

需要注意：`offline_draft_*` 不是 preprocess-KV selector。draft model 是小模型直接基于 token 序列算 attention/score，不能直接消费主模型的 preprocess KV cache。

## 最终结论

1. full dataset 结果已经全部完成：8 组 raw-KV 对照均为 `finished=True`，每组都有 248 个 sub-question 结果。
2. runtime 使用 raw document KV 会系统性降低效果。与 preprocess KV 相比，Main Acc 下降约 4.44% 到 10.37%，Sub Acc 下降约 2.42% 到 6.85%，F1 下降约 0.0170 到 0.0333。
3. 这说明 preprocess KV 不是可以直接省掉的工程开销。即使 fixed set 或 online selector 是在 raw/fixed score 空间里定义的，生成阶段加载 preprocess KV 仍然更稳。
4. online draft selector 仍是当前最强的 rate=0.15 selector：preprocess KV 下 Main Acc 74.07%、Sub Acc 83.06%、F1 0.5410；换成 raw KV 后降到 Main Acc 67.41%、Sub Acc 78.23%、F1 0.5077。
5. offline fixed set 方向仍有价值，但 fixed set 的构造不能绕开 preprocess KV 质量问题。更合理的下一步是：在 preprocess KV 语境下重新定义 offline set，或者研究如何离线近似 preprocess KV，而不是简单用 raw KV 替代。
