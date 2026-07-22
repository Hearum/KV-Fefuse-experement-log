# Pending Addendum: Offline QK Fixed-Set Selectors

## 目的

在当前跨数据集泛化实验之外，补充一组 offline QK fixed-set selector。它和当前已跑的 offline draft fixed set 不同：draft fixed set 使用 draft model 对文档 token 打分；offline QK fixed set 使用主模型/兼容主模型的 QK attention importance，对 calibration queries 聚合后离线确定每个 chunk 的固定更新 token。

## 待补实验组

| 实验名 | rate | fixed-set 方法 | 说明 |
|---|---:|---|---|
| `offline_qk_mean` | 0.15 | `qk_mean_score_per_chunk` | 对 calibration queries 的 QK score 做 mean aggregation，每个 chunk 选 top 15% token |
| `offline_qk_mean_boundary2` | 0.15 | `qk_mean_boundary0p02_per_chunk` | 总 rate 仍为 15%，其中 2% token budget 替换为 chunk boundary token，其余来自 QK mean score |

## Boundary 定义

保持总重算比例不变。对每个 chunk：

1. `total_budget = floor(0.15 * chunk_len)`。
2. `boundary_budget = floor(0.02 * chunk_len)`。
3. 先用 `qk_mean_score_per_chunk` 选择 `total_budget - boundary_budget` 个 token。
4. 再补充距离 chunk start/end 最近的 token，直到达到 `total_budget`。
5. 如果 boundary token 与 QK token 重合，则继续沿边界向内补齐。

这样可以测试“online/offline 差异是否集中在 chunk 边界附近”这一假设，同时保持和其他 rate=0.15 方法公平。

## 为什么当前不能直接加入正在跑的 180-task supervisor

当前 supervisor 已经启动并把任务列表写入 `logs/accuracy_tasks.tsv`；中途修改任务列表不会影响已经 fork 出来的 8 个 worker。更重要的是，offline QK 需要先生成 QK score/fixed-set cache，不能复用已经完成的 draft score cache。

因此本 addendum 应在当前 180 个任务完成后追加执行：

1. 为每个数据集生成 `fixed_sets_<dataset>_qk/chunk_fixed_sets_npz`。
2. 从 QK mean score 派生 `qk_mean_boundary0p02_per_chunk`。
3. 用同一套 accuracy pipeline 跑：
   - `offline_qk_mean`
   - `offline_qk_mean_boundary2`
4. 结果追加到 `cross_dataset_summary.csv/json` 和 README 主表。

## 实现备注

旧脚本 `tools_reflect_offline_qk_fixed_sets.py` 是 Qwen2 路线，默认使用 MuSiQue/Qwen2.5 cache。当前跨数据集主实验是 Qwen3-32B，因此需要改成 Qwen3 loader 或写一个 cross-dataset Qwen3 版本，避免把 Qwen2 fixed set 混入 Qwen3 主实验。
