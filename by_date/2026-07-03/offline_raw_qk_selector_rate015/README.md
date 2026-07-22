# Offline Raw-QK Selector at rate=0.15

## 目标

验证“offline 阶段如果在 raw document KV 上计算 QK/attention score 来挑 fixed set”，效果会不会比已经跑过的 preprocess-KV QK fixed set 更好。

这个实验专门区分两个轴：

1. **selector KV**：offline 挑 token 时使用 raw KV 还是 preprocess KV。
2. **runtime KV**：online 生成时加载 raw KV 还是 preprocess KV。

之前已经有：

- `offline_qk_frequency / offline_qk_mean`：selector 用 preprocess KV，runtime 用 preprocess KV。
- `raw_offline_qk_frequency / raw_offline_qk_mean`：selector 仍是 preprocess-KV QK fixed set，但 runtime 换成 raw KV。

本实验新增真正的 offline raw-QK selector：selector 阶段加载 raw KV cache。

## Raw-QK fixed set 构造方式

脚本：`tools_reflect_offline_qk_fixed_sets.py`

关键参数：

- `--preprocess-cache-dir /raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`

注意：脚本参数名仍叫 `preprocess-cache-dir`，但实现上只是读取 `{example}_{chunk}_key.pt/value.pt`，这里实际传入的是 raw document KV cache 目录。

固定设置：

- 数据集：`data/result_reflect.json`
- 主模型：`/mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct`
- BGE：`/mnt/qjhs-sh-lab-01/models/bge-m3`
- raw KV cache：`/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- rate：0.15
- calibration queries：每个 example 使用 8 个来自其他 examples 的 question，不使用当前真实问题，避免泄漏。
- 输出 fixed set：每个 chunk 内 local token indices。

## Fixed set 方法

### raw_qk_frequency_per_chunk

对每个 calibration query 分别取当前 chunk 内 top 15% token。统计每个 token 在 8 个 query 中被选中的次数，按：

1. 被选中频率高优先；
2. 平均 QK/attention score 高优先；
3. token index 小优先，作为稳定 tie-break。

最后每个 chunk 固定选 15% token。

### raw_qk_mean_score_per_chunk

对 8 个 calibration queries 的 QK/attention score 做平均，然后每个 chunk 内直接取平均分最高的 15% token。

## 待跑推理矩阵

| label | selector KV | fixed set method | runtime KV | rate |
|---|---|---|---|---:|
| rawqk_freq_preprocess_runtime | raw KV | raw_qk_frequency_per_chunk | preprocess KV | 0.15 |
| rawqk_mean_preprocess_runtime | raw KV | raw_qk_mean_score_per_chunk | preprocess KV | 0.15 |
| rawqk_freq_raw_runtime | raw KV | raw_qk_frequency_per_chunk | raw KV | 0.15 |
| rawqk_mean_raw_runtime | raw KV | raw_qk_mean_score_per_chunk | raw KV | 0.15 |

## 实验日志

- 2026-07-03：创建实验目录，开始生成 raw-QK fixed sets。
## 当前结果

| label | selector KV | method | runtime KV | finished | Main Acc | Sub Acc | F1 | EM | prefill(s) | storage(s) | selection(s) | rows |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| online_qk_rate015 | online_preprocess | online_qk | preprocess | True | 84/135 (62.22%) | 187/248 (75.40%) | 0.4897 | 0.2137 | 0.2311 | 0.0249 | 0.1032 | 248 |
| offline_qk_frequency | preprocess | frequency | preprocess | True | 79/135 (58.52%) | 179/248 (72.18%) | 0.4696 | 0.1976 | 0.1363 | 0.0252 | 0.0000 | 248 |
| offline_qk_mean | preprocess | mean_score | preprocess | True | 76/135 (56.30%) | 176/248 (70.97%) | 0.4766 | 0.2016 | 0.1312 | 0.0252 | 0.0000 | 248 |
| raw_offline_qk_frequency | preprocess | frequency | raw | True | 67/135 (49.63%) | 168/248 (67.74%) | 0.4525 | 0.1734 | 0.1320 | 0.0265 | 0.0000 | 248 |
| raw_offline_qk_mean | preprocess | mean_score | raw | True | 70/135 (51.85%) | 170/248 (68.55%) | 0.4519 | 0.1734 | 0.1318 | 0.0261 | 0.0000 | 248 |
| rawqk_freq_preprocess_runtime | raw | frequency | preprocess | True | 79/135 (58.52%) | 179/248 (72.18%) | 0.4867 | 0.1976 | 0.1282 | 0.0243 | 0.0000 | 248 |
| rawqk_mean_preprocess_runtime | raw | mean_score | preprocess | True | 83/135 (61.48%) | 183/248 (73.79%) | 0.4812 | 0.1935 | 0.1282 | 0.0244 | 0.0000 | 248 |
| rawqk_freq_raw_runtime | raw | frequency | raw | True | 68/135 (50.37%) | 166/248 (66.94%) | 0.4491 | 0.1653 | 0.1294 | 0.0252 | 0.0000 | 248 |
| rawqk_mean_raw_runtime | raw | mean_score | raw | True | 74/135 (54.81%) | 173/248 (69.76%) | 0.4735 | 0.1774 | 0.1303 | 0.0259 | 0.0000 | 248 |

## 结果解读

四组新增 raw-QK selector 推理均已完成。raw-QK selector 指的是 offline 阶段用 raw document KV 计算 calibration-query 的 QK/attention score，再固定选 token。

主要结论：

1. **offline raw-QK mean-score + preprocess runtime 是 raw-QK rate=0.15 这组里较强的 fixed set，不是全局最强 offline fixed set**：Main Acc 83/135 (61.48%)，Sub Acc 183/248 (73.79%)，F1 0.4812。它接近 online QK 的 Main Acc 62.22%，并明显高于之前 preprocess-QK mean fixed set 的 Main Acc 56.30%。
2. **raw-QK frequency + preprocess runtime 与 preprocess-QK frequency 在 accuracy 上几乎一致**：两者都是 Main Acc 58.52%、Sub Acc 72.18%，但 raw-QK frequency 的 F1 更高一点（0.4867 vs 0.4696）。
3. **runtime KV 仍然必须用 preprocess KV**。同样 raw-QK fixed set，如果生成阶段换成 raw KV，frequency 降到 Main Acc 50.37%，mean-score 降到 54.81%，明显低于 preprocess runtime。
4. 这说明“selector 在 raw KV 空间挑”不一定差，甚至 mean-score 可能比 preprocess-QK fixed set 更好；但“生成时直接用 raw KV”仍然不行。
5. 对后续 fixed-set 方向的启发：raw-QK 不是当前最强 offline fixed set；它的价值在于说明 offline selector 不一定要完全模拟 preprocess KV 空间，raw KV 上的 calibration score 可能更稳定/更泛化；但最终生成质量仍依赖 preprocess KV 表示。下一步可以重点 sweep raw-QK mean-score 的 rate，或者把 raw-QK mean 与 draft stable set 做 hybrid。
