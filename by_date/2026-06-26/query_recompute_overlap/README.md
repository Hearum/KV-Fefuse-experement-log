
## 运行 20260626_161124

- 时间: 2026-06-26T16:11:24
- 目的: 验证固定同一文档时，不同 query 触发的重算 token 是否稳定。
- 命令参数: `example_start=0`, `num_examples=2`, `max_passages_list=[3]`, `rate_list=[0.2]`, `native_count=3`, `control_count=3`, `block_size=16`
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - `score_cosine`: query importance score 向量余弦相似度。
  - `score_l2_rel`: score 相对 L2 差异，越小说明分数越接近。
  - `chunk_cosine`: 选中 token 落到各 chunk 的分布相似度。

## 运行 20260626_161345

- 时间: 2026-06-26T16:13:45
- 目的: 验证固定同一文档时，不同 query 触发的重算 token 是否稳定。
- 命令参数: `example_start=0`, `num_examples=2`, `max_passages_list=[3]`, `rate_list=[0.2]`, `native_count=3`, `control_count=3`, `block_size=16`
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - `score_cosine`: query importance score 向量余弦相似度。
  - `score_l2_rel`: score 相对 L2 差异，越小说明分数越接近。
  - `chunk_cosine`: 选中 token 落到各 chunk 的分布相似度。
- 完成 example 0, passages=3, rate=0.2: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 1, passages=3, rate=0.2: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan

### 汇总: passages=3, rate=0.2

- 输出: `batch_20260626_161345_2examples_3passages_rate0.2.json`
- 耗时: 7.0s
- token_jaccard mean over examples: 1.0000 (min=1.0000, max=1.0000)
- block_jaccard mean over examples: 1.0000 (min=1.0000, max=1.0000)
- score_cosine mean over examples: nan (min=nan, max=nan)
- score_l2_rel mean over examples: nan (min=nan, max=nan)

### 运行 20260626_161345 结束

- 全量输出: `batch_20260626_161345_all_cases.json`

## 运行 20260626_161517

- 时间: 2026-06-26T16:15:17
- 目的: 验证固定同一文档时，不同 query 触发的重算 token 是否稳定。
- 命令参数: `example_start=0`, `num_examples=20`, `max_passages_list=[3, 5]`, `rate_list=[0.1, 0.2]`, `native_count=5`, `control_count=8`, `block_size=16`
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - `score_cosine`: query importance score 向量余弦相似度。
  - `score_l2_rel`: score 相对 L2 差异，越小说明分数越接近。
  - `chunk_cosine`: 选中 token 落到各 chunk 的分布相似度。
- 完成 example 0, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 1, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 2, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 3, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 4, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 5, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 6, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 7, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 8, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 9, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 10, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 11, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 12, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 13, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 14, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 15, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 16, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 17, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 18, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 19, passages=3, rate=0.1: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan

### 汇总: passages=3, rate=0.1

- 输出: `batch_20260626_161517_20examples_3passages_rate0.1.json`
- 耗时: 139.8s
- token_jaccard mean over examples: 1.0000 (min=1.0000, max=1.0000)
- block_jaccard mean over examples: 1.0000 (min=1.0000, max=1.0000)
- score_cosine mean over examples: nan (min=nan, max=nan)
- score_l2_rel mean over examples: nan (min=nan, max=nan)
- 完成 example 0, passages=3, rate=0.2: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 1, passages=3, rate=0.2: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 2, passages=3, rate=0.2: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 3, passages=3, rate=0.2: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 4, passages=3, rate=0.2: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 5, passages=3, rate=0.2: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 6, passages=3, rate=0.2: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 7, passages=3, rate=0.2: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 8, passages=3, rate=0.2: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan
- 完成 example 9, passages=3, rate=0.2: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=nan

### 记录修正: 20260626_161517 结果作废

上一轮批量运行中发现 `score_cosine_mean=nan`。原因是 attention importance score 内存在 NaN/Inf，导致 top-k 选择可能被坏值污染，因此该运行中 overlap 接近 1.0 的现象不能作为实验结论。已停止该任务，并修复 `tools_batch_query_recompute_overlap.py`：在 top-k 选择、score 相似度计算和统计汇总前统一执行 `nan_to_num`，同时在 JSON 与 README 中记录 `score_bad_value_total` / `score_bad_value_count` 以便追溯。

## 运行 20260626_162029

- 时间: 2026-06-26T16:20:29
- 目的: 验证固定同一文档时，不同 query 触发的重算 token 是否稳定。
- 命令参数: `example_start=0`, `num_examples=2`, `max_passages_list=[3]`, `rate_list=[0.2]`, `native_count=3`, `control_count=3`, `block_size=16`
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - `score_cosine`: query importance score 向量余弦相似度。
  - `score_l2_rel`: score 相对 L2 差异，越小说明分数越接近。
  - `chunk_cosine`: 选中 token 落到各 chunk 的分布相似度。
- 完成 example 0, passages=3, rate=0.2: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=0.0000
- 完成 example 1, passages=3, rate=0.2: token_jaccard_mean=1.0000, block_jaccard_mean=1.0000, score_cosine_mean=0.0000

### 汇总: passages=3, rate=0.2

- 输出: `batch_20260626_162029_2examples_3passages_rate0.2.json`
- 耗时: 7.0s
- token_jaccard mean over examples: 1.0000 (min=1.0000, max=1.0000)
- block_jaccard mean over examples: 1.0000 (min=1.0000, max=1.0000)
- score_cosine mean over examples: 0.0000 (min=0.0000, max=0.0000)
- score_l2_rel mean over examples: 0.0000 (min=0.0000, max=0.0000)

### 运行 20260626_162029 结束

- 全量输出: `batch_20260626_162029_all_cases.json`

### 记录修正: 20260626_162029 结果作废

该运行虽然通过 `nan_to_num` 消除了 NaN，但 `score_bad_value_total` 显示每个 context token 的 importance score 全部来自坏值，清洗后 score 退化为全 0。因此该运行只能说明原 `importance_cache` 测量路径不可用，不能支持 query 重算区域是否稳定的结论。已将脚本改为直接使用 raw chunk KV cache 中的 K 与 query hidden state 的 Q 计算逐层 QK attention importance。

## 运行 20260626_162245

- 时间: 2026-06-26T16:22:45
- 目的: 验证固定同一文档时，不同 query 触发的重算 token 是否稳定。
- 命令参数: `example_start=0`, `num_examples=2`, `max_passages_list=[3]`, `rate_list=[0.2]`, `native_count=3`, `control_count=3`, `block_size=16`
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - score 来源: raw chunk KV cache 中的 context K，与 query hidden state 计算逐层 QK attention，softmax 后对 layer/head/query token 求和。
  - `score_cosine`: query importance score 向量余弦相似度。
  - `score_l2_rel`: score 相对 L2 差异，越小说明分数越接近。
  - `chunk_cosine`: 选中 token 落到各 chunk 的分布相似度。
- 完成 example 0, passages=3, rate=0.2: token_jaccard_mean=0.6940, block_jaccard_mean=0.8961, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 1, passages=3, rate=0.2: token_jaccard_mean=0.6719, block_jaccard_mean=0.9042, score_cosine_mean=0.9999, score_bad_value_total=0

### 汇总: passages=3, rate=0.2

- 输出: `batch_20260626_162245_2examples_3passages_rate0.2.json`
- 耗时: 8.3s
- token_jaccard mean over examples: 0.6830 (min=0.6719, max=0.6940)
- block_jaccard mean over examples: 0.9001 (min=0.8961, max=0.9042)
- score_cosine mean over examples: 0.9999 (min=0.9999, max=0.9999)
- score_l2_rel mean over examples: 0.1367 (min=0.1334, max=0.1401)

### 运行 20260626_162245 结束

- 全量输出: `batch_20260626_162245_all_cases.json`

## 运行 20260626_162330

- 时间: 2026-06-26T16:23:30
- 目的: 验证固定同一文档时，不同 query 触发的重算 token 是否稳定。
- 命令参数: `example_start=0`, `num_examples=20`, `max_passages_list=[3, 5]`, `rate_list=[0.1, 0.2]`, `native_count=8`, `control_count=8`, `block_size=16`
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - score 来源: raw chunk KV cache 中的 context K，与 query hidden state 计算逐层 QK attention，softmax 后对 layer/head/query token 求和。
  - `score_cosine`: query importance score 向量余弦相似度。
  - `score_l2_rel`: score 相对 L2 差异，越小说明分数越接近。
  - `chunk_cosine`: 选中 token 落到各 chunk 的分布相似度。
- 完成 example 0, passages=3, rate=0.1: token_jaccard_mean=0.7246, block_jaccard_mean=0.8213, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 1, passages=3, rate=0.1: token_jaccard_mean=0.6735, block_jaccard_mean=0.8146, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 2, passages=3, rate=0.1: token_jaccard_mean=0.7596, block_jaccard_mean=0.8605, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 3, passages=3, rate=0.1: token_jaccard_mean=0.7503, block_jaccard_mean=0.8639, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 4, passages=3, rate=0.1: token_jaccard_mean=0.7029, block_jaccard_mean=0.8333, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 5, passages=3, rate=0.1: token_jaccard_mean=0.6995, block_jaccard_mean=0.7905, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 6, passages=3, rate=0.1: token_jaccard_mean=0.7225, block_jaccard_mean=0.7975, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 7, passages=3, rate=0.1: token_jaccard_mean=0.7263, block_jaccard_mean=0.8274, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 8, passages=3, rate=0.1: token_jaccard_mean=0.7165, block_jaccard_mean=0.8083, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 9, passages=3, rate=0.1: token_jaccard_mean=0.6663, block_jaccard_mean=0.8149, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 10, passages=3, rate=0.1: token_jaccard_mean=0.6915, block_jaccard_mean=0.8134, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 11, passages=3, rate=0.1: token_jaccard_mean=0.7735, block_jaccard_mean=0.8573, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 12, passages=3, rate=0.1: token_jaccard_mean=0.7172, block_jaccard_mean=0.8148, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 13, passages=3, rate=0.1: token_jaccard_mean=0.7230, block_jaccard_mean=0.8809, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 14, passages=3, rate=0.1: token_jaccard_mean=0.6941, block_jaccard_mean=0.8579, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 15, passages=3, rate=0.1: token_jaccard_mean=0.7573, block_jaccard_mean=0.8588, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 16, passages=3, rate=0.1: token_jaccard_mean=0.6969, block_jaccard_mean=0.8220, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 17, passages=3, rate=0.1: token_jaccard_mean=0.7063, block_jaccard_mean=0.8549, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 18, passages=3, rate=0.1: token_jaccard_mean=0.7606, block_jaccard_mean=0.9104, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 19, passages=3, rate=0.1: token_jaccard_mean=0.7618, block_jaccard_mean=0.8531, score_cosine_mean=1.0000, score_bad_value_total=0

### 汇总: passages=3, rate=0.1

- 输出: `batch_20260626_162330_20examples_3passages_rate0.1.json`
- 耗时: 206.0s
- token_jaccard mean over examples: 0.7212 (min=0.6663, max=0.7735)
- block_jaccard mean over examples: 0.8378 (min=0.7905, max=0.9104)
- score_cosine mean over examples: 0.9999 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1283 (min=0.0793, max=0.2038)
- 完成 example 0, passages=3, rate=0.2: token_jaccard_mean=0.7241, block_jaccard_mean=0.9157, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 1, passages=3, rate=0.2: token_jaccard_mean=0.6897, block_jaccard_mean=0.9209, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 2, passages=3, rate=0.2: token_jaccard_mean=0.7730, block_jaccard_mean=0.9465, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 3, passages=3, rate=0.2: token_jaccard_mean=0.8006, block_jaccard_mean=0.9027, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 4, passages=3, rate=0.2: token_jaccard_mean=0.7542, block_jaccard_mean=0.9008, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 5, passages=3, rate=0.2: token_jaccard_mean=0.7097, block_jaccard_mean=0.9149, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 6, passages=3, rate=0.2: token_jaccard_mean=0.7709, block_jaccard_mean=0.9077, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 7, passages=3, rate=0.2: token_jaccard_mean=0.7118, block_jaccard_mean=0.8953, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 8, passages=3, rate=0.2: token_jaccard_mean=0.7244, block_jaccard_mean=0.9069, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 9, passages=3, rate=0.2: token_jaccard_mean=0.7080, block_jaccard_mean=0.9276, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 10, passages=3, rate=0.2: token_jaccard_mean=0.7279, block_jaccard_mean=0.9396, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 11, passages=3, rate=0.2: token_jaccard_mean=0.7244, block_jaccard_mean=0.8534, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 12, passages=3, rate=0.2: token_jaccard_mean=0.7435, block_jaccard_mean=0.9206, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 13, passages=3, rate=0.2: token_jaccard_mean=0.7320, block_jaccard_mean=0.9272, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 14, passages=3, rate=0.2: token_jaccard_mean=0.6837, block_jaccard_mean=0.9010, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 15, passages=3, rate=0.2: token_jaccard_mean=0.7621, block_jaccard_mean=0.9307, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 16, passages=3, rate=0.2: token_jaccard_mean=0.7280, block_jaccard_mean=0.8631, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 17, passages=3, rate=0.2: token_jaccard_mean=0.7201, block_jaccard_mean=0.8998, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 18, passages=3, rate=0.2: token_jaccard_mean=0.7923, block_jaccard_mean=0.9441, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 19, passages=3, rate=0.2: token_jaccard_mean=0.7532, block_jaccard_mean=0.9172, score_cosine_mean=1.0000, score_bad_value_total=0

### 汇总: passages=3, rate=0.2

- 输出: `batch_20260626_162330_20examples_3passages_rate0.2.json`
- 耗时: 205.0s
- token_jaccard mean over examples: 0.7367 (min=0.6837, max=0.8006)
- block_jaccard mean over examples: 0.9118 (min=0.8534, max=0.9465)
- score_cosine mean over examples: 0.9999 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1283 (min=0.0793, max=0.2038)
- 完成 example 0, passages=5, rate=0.1: token_jaccard_mean=0.7109, block_jaccard_mean=0.8034, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 1, passages=5, rate=0.1: token_jaccard_mean=0.6688, block_jaccard_mean=0.7951, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 2, passages=5, rate=0.1: token_jaccard_mean=0.7670, block_jaccard_mean=0.8451, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 3, passages=5, rate=0.1: token_jaccard_mean=0.7014, block_jaccard_mean=0.8448, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 4, passages=5, rate=0.1: token_jaccard_mean=0.6934, block_jaccard_mean=0.8291, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 5, passages=5, rate=0.1: token_jaccard_mean=0.6927, block_jaccard_mean=0.7938, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 6, passages=5, rate=0.1: token_jaccard_mean=0.7415, block_jaccard_mean=0.8328, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 7, passages=5, rate=0.1: token_jaccard_mean=0.6986, block_jaccard_mean=0.8141, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 8, passages=5, rate=0.1: token_jaccard_mean=0.7305, block_jaccard_mean=0.7822, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 9, passages=5, rate=0.1: token_jaccard_mean=0.6745, block_jaccard_mean=0.8272, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 10, passages=5, rate=0.1: token_jaccard_mean=0.6837, block_jaccard_mean=0.7979, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 11, passages=5, rate=0.1: token_jaccard_mean=0.6899, block_jaccard_mean=0.7773, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 12, passages=5, rate=0.1: token_jaccard_mean=0.7041, block_jaccard_mean=0.8080, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 13, passages=5, rate=0.1: token_jaccard_mean=0.7125, block_jaccard_mean=0.8394, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 14, passages=5, rate=0.1: token_jaccard_mean=0.7088, block_jaccard_mean=0.8740, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 15, passages=5, rate=0.1: token_jaccard_mean=0.7391, block_jaccard_mean=0.8628, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 16, passages=5, rate=0.1: token_jaccard_mean=0.6868, block_jaccard_mean=0.8055, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 17, passages=5, rate=0.1: token_jaccard_mean=0.7094, block_jaccard_mean=0.8362, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 18, passages=5, rate=0.1: token_jaccard_mean=0.7496, block_jaccard_mean=0.8577, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 19, passages=5, rate=0.1: token_jaccard_mean=0.6950, block_jaccard_mean=0.8249, score_cosine_mean=0.9999, score_bad_value_total=0

### 汇总: passages=5, rate=0.1

- 输出: `batch_20260626_162330_20examples_5passages_rate0.1.json`
- 耗时: 313.9s
- token_jaccard mean over examples: 0.7079 (min=0.6688, max=0.7670)
- block_jaccard mean over examples: 0.8226 (min=0.7773, max=0.8740)
- score_cosine mean over examples: 0.9999 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1271 (min=0.0804, max=0.2024)
- 完成 example 0, passages=5, rate=0.2: token_jaccard_mean=0.7370, block_jaccard_mean=0.8927, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 1, passages=5, rate=0.2: token_jaccard_mean=0.6919, block_jaccard_mean=0.8958, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 2, passages=5, rate=0.2: token_jaccard_mean=0.7609, block_jaccard_mean=0.9361, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 3, passages=5, rate=0.2: token_jaccard_mean=0.7370, block_jaccard_mean=0.9188, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 4, passages=5, rate=0.2: token_jaccard_mean=0.7363, block_jaccard_mean=0.9196, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 5, passages=5, rate=0.2: token_jaccard_mean=0.7044, block_jaccard_mean=0.9031, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 6, passages=5, rate=0.2: token_jaccard_mean=0.7528, block_jaccard_mean=0.9160, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 7, passages=5, rate=0.2: token_jaccard_mean=0.7194, block_jaccard_mean=0.8808, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 8, passages=5, rate=0.2: token_jaccard_mean=0.7444, block_jaccard_mean=0.8593, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 9, passages=5, rate=0.2: token_jaccard_mean=0.7076, block_jaccard_mean=0.9021, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 10, passages=5, rate=0.2: token_jaccard_mean=0.7051, block_jaccard_mean=0.8722, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 11, passages=5, rate=0.2: token_jaccard_mean=0.7126, block_jaccard_mean=0.8805, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 12, passages=5, rate=0.2: token_jaccard_mean=0.7220, block_jaccard_mean=0.9176, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 13, passages=5, rate=0.2: token_jaccard_mean=0.7129, block_jaccard_mean=0.9000, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 14, passages=5, rate=0.2: token_jaccard_mean=0.6958, block_jaccard_mean=0.8855, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 15, passages=5, rate=0.2: token_jaccard_mean=0.7662, block_jaccard_mean=0.9352, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 16, passages=5, rate=0.2: token_jaccard_mean=0.7108, block_jaccard_mean=0.8897, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 17, passages=5, rate=0.2: token_jaccard_mean=0.7285, block_jaccard_mean=0.8830, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 example 18, passages=5, rate=0.2: token_jaccard_mean=0.7377, block_jaccard_mean=0.8909, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 19, passages=5, rate=0.2: token_jaccard_mean=0.7187, block_jaccard_mean=0.8967, score_cosine_mean=0.9999, score_bad_value_total=0

### 汇总: passages=5, rate=0.2

- 输出: `batch_20260626_162330_20examples_5passages_rate0.2.json`
- 耗时: 314.0s
- token_jaccard mean over examples: 0.7251 (min=0.6919, max=0.7662)
- block_jaccard mean over examples: 0.8988 (min=0.8593, max=0.9361)
- score_cosine mean over examples: 0.9999 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1271 (min=0.0804, max=0.2024)

### 运行 20260626_162330 结束

- 全量输出: `batch_20260626_162330_all_cases.json`

### 记录修正: rate 不应触发重复前向

rate 只决定从同一条 importance score 中截取 top-k 的比例，不需要重新跑模型前向。此前 `20260626_162330` 批处理脚本按 `(passages, rate, example)` 循环，因此同一个 `(example, passages)` 在不同 rate 下重复构建 raw cache 并重复计算 QK importance，结果本身有效，但运行时间多花了一倍。已修复脚本：现在每个 `(example, passages)` 只计算一次所有 query 的 score，然后对所有 rate 离线评估 top-k overlap。

## 运行 20260626_164253

- 时间: 2026-06-26T16:42:53
- 目的: 验证固定同一文档时，不同 query 触发的重算 token 是否稳定。
- 命令参数: `example_start=0`, `num_examples=1`, `max_passages_list=[3]`, `rate_list=[0.1, 0.2]`, `native_count=3`, `control_count=2`, `block_size=16`
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - score 来源: raw chunk KV cache 中的 context K，与 query hidden state 计算逐层 QK attention，softmax 后对 layer/head/query token 求和。
  - `score_cosine`: query importance score 向量余弦相似度。
  - `score_l2_rel`: score 相对 L2 差异，越小说明分数越接近。
  - `chunk_cosine`: 选中 token 落到各 chunk 的分布相似度。
- 完成 example 0, passages=3, rate=0.1: token_jaccard_mean=0.7234, block_jaccard_mean=0.7899, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 example 0, passages=3, rate=0.2: token_jaccard_mean=0.7193, block_jaccard_mean=0.8935, score_cosine_mean=0.9999, score_bad_value_total=0

### 汇总: passages=3, rate=0.1

- 输出: `batch_20260626_164253_1examples_3passages_rate0.1.json`
- 耗时: 3.3s
- token_jaccard mean over examples: 0.7234 (min=0.7234, max=0.7234)
- block_jaccard mean over examples: 0.7899 (min=0.7899, max=0.7899)
- score_cosine mean over examples: 0.9999 (min=0.9999, max=0.9999)
- score_l2_rel mean over examples: 0.1275 (min=0.1275, max=0.1275)

### 汇总: passages=3, rate=0.2

- 输出: `batch_20260626_164253_1examples_3passages_rate0.2.json`
- 耗时: 3.3s
- token_jaccard mean over examples: 0.7193 (min=0.7193, max=0.7193)
- block_jaccard mean over examples: 0.8935 (min=0.8935, max=0.8935)
- score_cosine mean over examples: 0.9999 (min=0.9999, max=0.9999)
- score_l2_rel mean over examples: 0.1275 (min=0.1275, max=0.1275)

### 运行 20260626_164253 结束

- 全量输出: `batch_20260626_164253_all_cases.json`

### 阶段性结论: query 变化下重算区域稳定性

- 有效批量运行: `20260626_162330`，20 个 Musique example；每个 example 固定同一组 passages，使用 8 个同文档 query 变体和 8 个其他 example 的 query 作为 control。
- score 定义: 对 raw chunk KV cache 中 context K 与 query hidden state 的 Q 逐层计算 QK attention，softmax 后对 layer/head/query-token 求和，得到每个 context token 的 importance。
- 注意: 该运行脚本当时按 rate 重复前向，结果有效但耗时偏高；后续脚本已改为一次前向、多 rate 离线评估。

| passages | rate | token Jaccard | block Jaccard | score cosine | relative L2 |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.1 | 0.7212 | 0.8378 | 0.9999 | 0.1283 |
| 3 | 0.2 | 0.7367 | 0.9118 | 0.9999 | 0.1283 |
| 5 | 0.1 | 0.7079 | 0.8226 | 0.9999 | 0.1271 |
| 5 | 0.2 | 0.7251 | 0.8988 | 0.9999 | 0.1271 |

当前可支持的 observation:
- 不同 query 下，连续 importance score 的方向几乎不变，四组配置的 score cosine 都约为 0.9999。
- 精确 token-level top-k 集合并不完全稳定，Jaccard 大约在 0.71-0.74，说明 top-k 边界附近会被 query 文本扰动。
- block-level 集合明显更稳定，rate=0.1 时约 0.82-0.84，rate=0.2 时约 0.90-0.91；这更支持 block 级重算 mask / metadata 复用，而不是逐 token 精确复用。
- passages 从 3 增到 5 后，block Jaccard 略降，但没有明显崩掉，说明长一点的 context 下仍有一定稳定性。

当前不能过度声称的点:
- 不能说不同 query 会选择完全相同的 token；token-level overlap 只有中等偏高。
- 目前只是 attention-importance 层面的稳定性，还没有验证复用上一 query 的 block mask 是否能保持最终 answer accuracy。

## DraftModel selector 运行 20260626_165925

- 时间: 2026-06-26T16:59:25
- 目的: 验证固定同一文档时，不同 query 触发的 DraftModel 重算 token 是否稳定。
- 启动参考: `/mnt/qjhs-sh-lab-01/wjh/FusionRAG/run_task.py DraftModel ... --draft_model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`
- 命令参数: `example_start=0`, `num_examples=1`, `max_passages_list=[3]`, `rate_list=[0.1, 0.2]`, `native_count=2`, `control_count=2`, `block_size=16`
- DraftModel: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`, layer_selection=`rrf`, rrf_k=18
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - score 来源: draft model query→doc attention，经 `rrf` 聚合；rate 阶段复用同一 score，离线调用 `smart_query_selection`。
- 完成 DraftModel example 0, passages=3, rate=0.1: token_jaccard_mean=0.6729, block_jaccard_mean=0.7291, score_cosine_mean=0.9586, score_bad_value_total=0
- 完成 DraftModel example 0, passages=3, rate=0.2: token_jaccard_mean=0.6250, block_jaccard_mean=0.7202, score_cosine_mean=0.9586, score_bad_value_total=0

### DraftModel 汇总: passages=3, rate=0.1, mode=rrf

- 输出: `draft_batch_20260626_165925_1examples_3passages_rate0.1_rrf.json`
- 耗时: 1.3s
- token_jaccard mean over examples: 0.6729 (min=0.6729, max=0.6729)
- block_jaccard mean over examples: 0.7291 (min=0.7291, max=0.7291)
- score_cosine mean over examples: 0.9586 (min=0.9586, max=0.9586)
- score_l2_rel mean over examples: 0.2797 (min=0.2797, max=0.2797)

### DraftModel 汇总: passages=3, rate=0.2, mode=rrf

- 输出: `draft_batch_20260626_165925_1examples_3passages_rate0.2_rrf.json`
- 耗时: 1.3s
- token_jaccard mean over examples: 0.6250 (min=0.6250, max=0.6250)
- block_jaccard mean over examples: 0.7202 (min=0.7202, max=0.7202)
- score_cosine mean over examples: 0.9586 (min=0.9586, max=0.9586)
- score_l2_rel mean over examples: 0.2797 (min=0.2797, max=0.2797)

### DraftModel 运行 20260626_165925 结束

- 全量输出: `draft_batch_20260626_165925_all_cases.json`

## DraftModel selector 运行 20260626_170030

- 时间: 2026-06-26T17:00:30
- 目的: 验证固定同一文档时，不同 query 触发的 DraftModel 重算 token 是否稳定。
- 启动参考: `/mnt/qjhs-sh-lab-01/wjh/FusionRAG/run_task.py DraftModel ... --draft_model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`
- 命令参数: `example_start=0`, `num_examples=20`, `max_passages_list=[3, 5]`, `rate_list=[0.1, 0.2]`, `native_count=8`, `control_count=8`, `block_size=16`
- DraftModel: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`, layer_selection=`rrf`, rrf_k=18
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - score 来源: draft model query→doc attention，经 `rrf` 聚合；rate 阶段复用同一 score，离线调用 `smart_query_selection`。
- 完成 DraftModel example 0, passages=3, rate=0.1: token_jaccard_mean=0.7196, block_jaccard_mean=0.7539, score_cosine_mean=0.9689, score_bad_value_total=0
- 完成 DraftModel example 0, passages=3, rate=0.2: token_jaccard_mean=0.6847, block_jaccard_mean=0.7787, score_cosine_mean=0.9689, score_bad_value_total=0
- 完成 DraftModel example 1, passages=3, rate=0.1: token_jaccard_mean=0.5821, block_jaccard_mean=0.6899, score_cosine_mean=0.9358, score_bad_value_total=0
- 完成 DraftModel example 1, passages=3, rate=0.2: token_jaccard_mean=0.6305, block_jaccard_mean=0.7099, score_cosine_mean=0.9358, score_bad_value_total=0
- 完成 DraftModel example 2, passages=3, rate=0.1: token_jaccard_mean=0.7655, block_jaccard_mean=0.8372, score_cosine_mean=0.9908, score_bad_value_total=0
- 完成 DraftModel example 2, passages=3, rate=0.2: token_jaccard_mean=0.7695, block_jaccard_mean=0.8625, score_cosine_mean=0.9908, score_bad_value_total=0
- 完成 DraftModel example 3, passages=3, rate=0.1: token_jaccard_mean=0.7576, block_jaccard_mean=0.7747, score_cosine_mean=0.9708, score_bad_value_total=0
- 完成 DraftModel example 3, passages=3, rate=0.2: token_jaccard_mean=0.7511, block_jaccard_mean=0.8242, score_cosine_mean=0.9708, score_bad_value_total=0
- 完成 DraftModel example 4, passages=3, rate=0.1: token_jaccard_mean=0.6908, block_jaccard_mean=0.7797, score_cosine_mean=0.9760, score_bad_value_total=0
- 完成 DraftModel example 4, passages=3, rate=0.2: token_jaccard_mean=0.7078, block_jaccard_mean=0.8307, score_cosine_mean=0.9760, score_bad_value_total=0
- 完成 DraftModel example 5, passages=3, rate=0.1: token_jaccard_mean=0.6581, block_jaccard_mean=0.7097, score_cosine_mean=0.9801, score_bad_value_total=0
- 完成 DraftModel example 5, passages=3, rate=0.2: token_jaccard_mean=0.7107, block_jaccard_mean=0.7940, score_cosine_mean=0.9801, score_bad_value_total=0
- 完成 DraftModel example 6, passages=3, rate=0.1: token_jaccard_mean=0.6810, block_jaccard_mean=0.7430, score_cosine_mean=0.9726, score_bad_value_total=0
- 完成 DraftModel example 6, passages=3, rate=0.2: token_jaccard_mean=0.7058, block_jaccard_mean=0.7811, score_cosine_mean=0.9726, score_bad_value_total=0
- 完成 DraftModel example 7, passages=3, rate=0.1: token_jaccard_mean=0.6493, block_jaccard_mean=0.7168, score_cosine_mean=0.9705, score_bad_value_total=0
- 完成 DraftModel example 7, passages=3, rate=0.2: token_jaccard_mean=0.6562, block_jaccard_mean=0.7733, score_cosine_mean=0.9705, score_bad_value_total=0
- 完成 DraftModel example 8, passages=3, rate=0.1: token_jaccard_mean=0.7592, block_jaccard_mean=0.8161, score_cosine_mean=0.9776, score_bad_value_total=0
- 完成 DraftModel example 8, passages=3, rate=0.2: token_jaccard_mean=0.7376, block_jaccard_mean=0.8335, score_cosine_mean=0.9776, score_bad_value_total=0
- 完成 DraftModel example 9, passages=3, rate=0.1: token_jaccard_mean=0.5639, block_jaccard_mean=0.6425, score_cosine_mean=0.9476, score_bad_value_total=0
- 完成 DraftModel example 9, passages=3, rate=0.2: token_jaccard_mean=0.6626, block_jaccard_mean=0.7865, score_cosine_mean=0.9476, score_bad_value_total=0
- 完成 DraftModel example 10, passages=3, rate=0.1: token_jaccard_mean=0.6848, block_jaccard_mean=0.7733, score_cosine_mean=0.9525, score_bad_value_total=0
- 完成 DraftModel example 10, passages=3, rate=0.2: token_jaccard_mean=0.7232, block_jaccard_mean=0.8384, score_cosine_mean=0.9525, score_bad_value_total=0
- 完成 DraftModel example 11, passages=3, rate=0.1: token_jaccard_mean=0.6432, block_jaccard_mean=0.6827, score_cosine_mean=0.9621, score_bad_value_total=0
- 完成 DraftModel example 11, passages=3, rate=0.2: token_jaccard_mean=0.7596, block_jaccard_mean=0.8680, score_cosine_mean=0.9621, score_bad_value_total=0
- 完成 DraftModel example 12, passages=3, rate=0.1: token_jaccard_mean=0.7665, block_jaccard_mean=0.8185, score_cosine_mean=0.9806, score_bad_value_total=0
- 完成 DraftModel example 12, passages=3, rate=0.2: token_jaccard_mean=0.7445, block_jaccard_mean=0.8474, score_cosine_mean=0.9806, score_bad_value_total=0
- 完成 DraftModel example 13, passages=3, rate=0.1: token_jaccard_mean=0.7396, block_jaccard_mean=0.7781, score_cosine_mean=0.9761, score_bad_value_total=0
- 完成 DraftModel example 13, passages=3, rate=0.2: token_jaccard_mean=0.7061, block_jaccard_mean=0.8187, score_cosine_mean=0.9761, score_bad_value_total=0
- 完成 DraftModel example 14, passages=3, rate=0.1: token_jaccard_mean=0.6169, block_jaccard_mean=0.7120, score_cosine_mean=0.9584, score_bad_value_total=0
- 完成 DraftModel example 14, passages=3, rate=0.2: token_jaccard_mean=0.6666, block_jaccard_mean=0.8486, score_cosine_mean=0.9584, score_bad_value_total=0
- 完成 DraftModel example 15, passages=3, rate=0.1: token_jaccard_mean=0.6633, block_jaccard_mean=0.7471, score_cosine_mean=0.9674, score_bad_value_total=0
- 完成 DraftModel example 15, passages=3, rate=0.2: token_jaccard_mean=0.6915, block_jaccard_mean=0.8192, score_cosine_mean=0.9674, score_bad_value_total=0
- 完成 DraftModel example 16, passages=3, rate=0.1: token_jaccard_mean=0.6816, block_jaccard_mean=0.7648, score_cosine_mean=0.9580, score_bad_value_total=0
- 完成 DraftModel example 16, passages=3, rate=0.2: token_jaccard_mean=0.7193, block_jaccard_mean=0.8242, score_cosine_mean=0.9580, score_bad_value_total=0
- 完成 DraftModel example 17, passages=3, rate=0.1: token_jaccard_mean=0.7201, block_jaccard_mean=0.7411, score_cosine_mean=0.9684, score_bad_value_total=0
- 完成 DraftModel example 17, passages=3, rate=0.2: token_jaccard_mean=0.7508, block_jaccard_mean=0.8536, score_cosine_mean=0.9684, score_bad_value_total=0
- 完成 DraftModel example 18, passages=3, rate=0.1: token_jaccard_mean=0.5395, block_jaccard_mean=0.5945, score_cosine_mean=0.9588, score_bad_value_total=0
- 完成 DraftModel example 18, passages=3, rate=0.2: token_jaccard_mean=0.6579, block_jaccard_mean=0.7630, score_cosine_mean=0.9588, score_bad_value_total=0
- 完成 DraftModel example 19, passages=3, rate=0.1: token_jaccard_mean=0.7121, block_jaccard_mean=0.8138, score_cosine_mean=0.9605, score_bad_value_total=0
- 完成 DraftModel example 19, passages=3, rate=0.2: token_jaccard_mean=0.7160, block_jaccard_mean=0.8316, score_cosine_mean=0.9605, score_bad_value_total=0

### DraftModel 汇总: passages=3, rate=0.1, mode=rrf

- 输出: `draft_batch_20260626_170030_20examples_3passages_rate0.1_rrf.json`
- 耗时: 76.9s
- token_jaccard mean over examples: 0.6797 (min=0.5395, max=0.7665)
- block_jaccard mean over examples: 0.7445 (min=0.5945, max=0.8372)
- score_cosine mean over examples: 0.9667 (min=0.9358, max=0.9908)
- score_l2_rel mean over examples: 0.2351 (min=0.1296, max=0.3265)

### DraftModel 汇总: passages=3, rate=0.2, mode=rrf

- 输出: `draft_batch_20260626_170030_20examples_3passages_rate0.2_rrf.json`
- 耗时: 76.9s
- token_jaccard mean over examples: 0.7076 (min=0.6305, max=0.7695)
- block_jaccard mean over examples: 0.8144 (min=0.7099, max=0.8680)
- score_cosine mean over examples: 0.9667 (min=0.9358, max=0.9908)
- score_l2_rel mean over examples: 0.2351 (min=0.1296, max=0.3265)
- 完成 DraftModel example 0, passages=5, rate=0.1: token_jaccard_mean=0.6631, block_jaccard_mean=0.7359, score_cosine_mean=0.9618, score_bad_value_total=0
- 完成 DraftModel example 0, passages=5, rate=0.2: token_jaccard_mean=0.6645, block_jaccard_mean=0.7799, score_cosine_mean=0.9618, score_bad_value_total=0
- 完成 DraftModel example 1, passages=5, rate=0.1: token_jaccard_mean=0.6005, block_jaccard_mean=0.6287, score_cosine_mean=0.9466, score_bad_value_total=0
- 完成 DraftModel example 1, passages=5, rate=0.2: token_jaccard_mean=0.6495, block_jaccard_mean=0.7704, score_cosine_mean=0.9466, score_bad_value_total=0
- 完成 DraftModel example 2, passages=5, rate=0.1: token_jaccard_mean=0.7479, block_jaccard_mean=0.8466, score_cosine_mean=0.9846, score_bad_value_total=0
- 完成 DraftModel example 2, passages=5, rate=0.2: token_jaccard_mean=0.7687, block_jaccard_mean=0.8734, score_cosine_mean=0.9846, score_bad_value_total=0
- 完成 DraftModel example 3, passages=5, rate=0.1: token_jaccard_mean=0.7098, block_jaccard_mean=0.7277, score_cosine_mean=0.9381, score_bad_value_total=0
- 完成 DraftModel example 3, passages=5, rate=0.2: token_jaccard_mean=0.7296, block_jaccard_mean=0.8051, score_cosine_mean=0.9381, score_bad_value_total=0
- 完成 DraftModel example 4, passages=5, rate=0.1: token_jaccard_mean=0.7341, block_jaccard_mean=0.7889, score_cosine_mean=0.9664, score_bad_value_total=0
- 完成 DraftModel example 4, passages=5, rate=0.2: token_jaccard_mean=0.7204, block_jaccard_mean=0.8404, score_cosine_mean=0.9664, score_bad_value_total=0
- 完成 DraftModel example 5, passages=5, rate=0.1: token_jaccard_mean=0.6348, block_jaccard_mean=0.6668, score_cosine_mean=0.9748, score_bad_value_total=0
- 完成 DraftModel example 5, passages=5, rate=0.2: token_jaccard_mean=0.6817, block_jaccard_mean=0.8155, score_cosine_mean=0.9748, score_bad_value_total=0
- 完成 DraftModel example 6, passages=5, rate=0.1: token_jaccard_mean=0.7144, block_jaccard_mean=0.7488, score_cosine_mean=0.9702, score_bad_value_total=0
- 完成 DraftModel example 6, passages=5, rate=0.2: token_jaccard_mean=0.7136, block_jaccard_mean=0.8099, score_cosine_mean=0.9702, score_bad_value_total=0
- 完成 DraftModel example 7, passages=5, rate=0.1: token_jaccard_mean=0.6978, block_jaccard_mean=0.7536, score_cosine_mean=0.9702, score_bad_value_total=0
- 完成 DraftModel example 7, passages=5, rate=0.2: token_jaccard_mean=0.6778, block_jaccard_mean=0.7961, score_cosine_mean=0.9702, score_bad_value_total=0
- 完成 DraftModel example 8, passages=5, rate=0.1: token_jaccard_mean=0.7032, block_jaccard_mean=0.7989, score_cosine_mean=0.9736, score_bad_value_total=0
- 完成 DraftModel example 8, passages=5, rate=0.2: token_jaccard_mean=0.7574, block_jaccard_mean=0.8313, score_cosine_mean=0.9736, score_bad_value_total=0
- 完成 DraftModel example 9, passages=5, rate=0.1: token_jaccard_mean=0.6280, block_jaccard_mean=0.7652, score_cosine_mean=0.9326, score_bad_value_total=0
- 完成 DraftModel example 9, passages=5, rate=0.2: token_jaccard_mean=0.6949, block_jaccard_mean=0.8292, score_cosine_mean=0.9326, score_bad_value_total=0
- 完成 DraftModel example 10, passages=5, rate=0.1: token_jaccard_mean=0.6722, block_jaccard_mean=0.7720, score_cosine_mean=0.9385, score_bad_value_total=0
- 完成 DraftModel example 10, passages=5, rate=0.2: token_jaccard_mean=0.7011, block_jaccard_mean=0.8206, score_cosine_mean=0.9385, score_bad_value_total=0
- 完成 DraftModel example 11, passages=5, rate=0.1: token_jaccard_mean=0.6967, block_jaccard_mean=0.7673, score_cosine_mean=0.9587, score_bad_value_total=0
- 完成 DraftModel example 11, passages=5, rate=0.2: token_jaccard_mean=0.7809, block_jaccard_mean=0.8505, score_cosine_mean=0.9587, score_bad_value_total=0
- 完成 DraftModel example 12, passages=5, rate=0.1: token_jaccard_mean=0.6932, block_jaccard_mean=0.7321, score_cosine_mean=0.9733, score_bad_value_total=0
- 完成 DraftModel example 12, passages=5, rate=0.2: token_jaccard_mean=0.7456, block_jaccard_mean=0.8431, score_cosine_mean=0.9733, score_bad_value_total=0
- 完成 DraftModel example 13, passages=5, rate=0.1: token_jaccard_mean=0.7140, block_jaccard_mean=0.7805, score_cosine_mean=0.9746, score_bad_value_total=0
- 完成 DraftModel example 13, passages=5, rate=0.2: token_jaccard_mean=0.7116, block_jaccard_mean=0.8162, score_cosine_mean=0.9746, score_bad_value_total=0
- 完成 DraftModel example 14, passages=5, rate=0.1: token_jaccard_mean=0.6630, block_jaccard_mean=0.7475, score_cosine_mean=0.9629, score_bad_value_total=0
- 完成 DraftModel example 14, passages=5, rate=0.2: token_jaccard_mean=0.7196, block_jaccard_mean=0.8549, score_cosine_mean=0.9629, score_bad_value_total=0
- 完成 DraftModel example 15, passages=5, rate=0.1: token_jaccard_mean=0.6585, block_jaccard_mean=0.7178, score_cosine_mean=0.9557, score_bad_value_total=0
- 完成 DraftModel example 15, passages=5, rate=0.2: token_jaccard_mean=0.7360, block_jaccard_mean=0.8439, score_cosine_mean=0.9557, score_bad_value_total=0
- 完成 DraftModel example 16, passages=5, rate=0.1: token_jaccard_mean=0.6776, block_jaccard_mean=0.7537, score_cosine_mean=0.9562, score_bad_value_total=0
- 完成 DraftModel example 16, passages=5, rate=0.2: token_jaccard_mean=0.7043, block_jaccard_mean=0.7991, score_cosine_mean=0.9562, score_bad_value_total=0
- 完成 DraftModel example 17, passages=5, rate=0.1: token_jaccard_mean=0.7043, block_jaccard_mean=0.7806, score_cosine_mean=0.9728, score_bad_value_total=0
- 完成 DraftModel example 17, passages=5, rate=0.2: token_jaccard_mean=0.7214, block_jaccard_mean=0.8553, score_cosine_mean=0.9728, score_bad_value_total=0
- 完成 DraftModel example 18, passages=5, rate=0.1: token_jaccard_mean=0.6382, block_jaccard_mean=0.7094, score_cosine_mean=0.9466, score_bad_value_total=0
- 完成 DraftModel example 18, passages=5, rate=0.2: token_jaccard_mean=0.7166, block_jaccard_mean=0.8627, score_cosine_mean=0.9466, score_bad_value_total=0
- 完成 DraftModel example 19, passages=5, rate=0.1: token_jaccard_mean=0.6780, block_jaccard_mean=0.7867, score_cosine_mean=0.9621, score_bad_value_total=0
- 完成 DraftModel example 19, passages=5, rate=0.2: token_jaccard_mean=0.7098, block_jaccard_mean=0.8286, score_cosine_mean=0.9621, score_bad_value_total=0

### DraftModel 汇总: passages=5, rate=0.1, mode=rrf

- 输出: `draft_batch_20260626_170030_20examples_5passages_rate0.1_rrf.json`
- 耗时: 113.7s
- token_jaccard mean over examples: 0.6815 (min=0.6005, max=0.7479)
- block_jaccard mean over examples: 0.7504 (min=0.6287, max=0.8466)
- score_cosine mean over examples: 0.9610 (min=0.9326, max=0.9846)
- score_l2_rel mean over examples: 0.2548 (min=0.1604, max=0.3292)

### DraftModel 汇总: passages=5, rate=0.2, mode=rrf

- 输出: `draft_batch_20260626_170030_20examples_5passages_rate0.2_rrf.json`
- 耗时: 113.7s
- token_jaccard mean over examples: 0.7152 (min=0.6495, max=0.7809)
- block_jaccard mean over examples: 0.8263 (min=0.7704, max=0.8734)
- score_cosine mean over examples: 0.9610 (min=0.9326, max=0.9846)
- score_l2_rel mean over examples: 0.2548 (min=0.1604, max=0.3292)

### DraftModel 运行 20260626_170030 结束

- 全量输出: `draft_batch_20260626_170030_all_cases.json`

### 对比结论: Target QK selector vs DraftModel selector

- DraftModel 运行: `20260626_170030`，Qwen2.5-3B-Instruct，`DRAFT_LAYER_SEL=rrf`，`RRF_K=18`，20 个 Musique examples。
- 两组实验均固定同一文档，改变 query；rate 只离线决定选择比例，不重复前向。

| passages | rate | selector | token Jaccard | block Jaccard | score cosine | relative L2 |
|---:|---:|---|---:|---:|---:|---:|
| 3 | 0.1 | Target-QK | 0.7212 | 0.8378 | 0.9999 | 0.1283 |
| 3 | 0.1 | DraftModel-rrf18 | 0.6797 | 0.7445 | 0.9667 | 0.2351 |
| 3 | 0.2 | Target-QK | 0.7367 | 0.9118 | 0.9999 | 0.1283 |
| 3 | 0.2 | DraftModel-rrf18 | 0.7076 | 0.8144 | 0.9667 | 0.2351 |
| 5 | 0.1 | Target-QK | 0.7079 | 0.8226 | 0.9999 | 0.1271 |
| 5 | 0.1 | DraftModel-rrf18 | 0.6815 | 0.7504 | 0.9610 | 0.2548 |
| 5 | 0.2 | Target-QK | 0.7251 | 0.8988 | 0.9999 | 0.1271 |
| 5 | 0.2 | DraftModel-rrf18 | 0.7152 | 0.8263 | 0.9610 | 0.2548 |

当前判断:
- DraftModel selector 的跨 query 稳定性明显弱于 Target-QK selector；score cosine 约 0.96，而 Target-QK 约 0.9999。
- DraftModel 的 block-level overlap 仍高于 token-level，但整体低于 Target-QK，说明 draft attention 更受 query 表述影响。
- 如果目标是复用上一 query 的重算 mask，DraftModel 可能不是最好的 mask 生成信号；更合理的方向可能是用 Target-QK / preprocess 后主模型 KV 空间学习稳定 mask，或者只把 DraftModel 用作 query-specific 在线 selector。

## 运行 20260626_170955

- 时间: 2026-06-26T17:09:55
- 目的: 验证固定同一文档时，使用 preprocess KV 的主模型 QK selector 在不同 query 下是否稳定。
- 命令参数: `example_start=0`, `num_examples=1`, `max_passages_list=[3]`, `rate_list=[0.1, 0.2]`, `native_count=2`, `control_count=2`, `block_size=16`
- preprocess KV: `/raid/home/hming/fusionrag-pca-top1-top10-cache-20/data/musique-pca-subset-preprocess-10-revert_rope-True/Qwen2.5-7B-Instruct`
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - score 来源: preprocess top-10 KV cache 中的 context K，与 query hidden state 计算逐层 QK attention，softmax 后对 layer/head/query token 求和。
  - `score_cosine`: query importance score 向量余弦相似度。
  - `score_l2_rel`: score 相对 L2 差异，越小说明分数越接近。
  - `chunk_cosine`: 选中 token 落到各 chunk 的分布相似度。
- 完成 PreprocessKV example 0, passages=3, rate=0.1: token_jaccard_mean=0.6634, block_jaccard_mean=0.7724, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 0, passages=3, rate=0.2: token_jaccard_mean=0.6890, block_jaccard_mean=0.9041, score_cosine_mean=1.0000, score_bad_value_total=0

### PreprocessKV 汇总: passages=3, rate=0.1

- 输出: `preprocesskv_batch_20260626_170955_1examples_3passages_rate0.1.json`
- 耗时: 1.1s
- token_jaccard mean over examples: 0.6634 (min=0.6634, max=0.6634)
- block_jaccard mean over examples: 0.7724 (min=0.7724, max=0.7724)
- score_cosine mean over examples: 1.0000 (min=1.0000, max=1.0000)
- score_l2_rel mean over examples: 0.1056 (min=0.1056, max=0.1056)

### PreprocessKV 汇总: passages=3, rate=0.2

- 输出: `preprocesskv_batch_20260626_170955_1examples_3passages_rate0.2.json`
- 耗时: 1.1s
- token_jaccard mean over examples: 0.6890 (min=0.6890, max=0.6890)
- block_jaccard mean over examples: 0.9041 (min=0.9041, max=0.9041)
- score_cosine mean over examples: 1.0000 (min=1.0000, max=1.0000)
- score_l2_rel mean over examples: 0.1056 (min=0.1056, max=0.1056)

### PreprocessKV 运行 20260626_170955 结束

- 全量输出: `preprocesskv_batch_20260626_170955_all_cases.json`

## 运行 20260626_171056

- 时间: 2026-06-26T17:10:56
- 目的: 验证固定同一文档时，使用 preprocess KV 的主模型 QK selector 在不同 query 下是否稳定。
- 命令参数: `example_start=0`, `num_examples=20`, `max_passages_list=[3, 5]`, `rate_list=[0.1, 0.2]`, `native_count=8`, `control_count=8`, `block_size=16`
- preprocess KV: `/raid/home/hming/fusionrag-pca-top1-top10-cache-20/data/musique-pca-subset-preprocess-10-revert_rope-True/Qwen2.5-7B-Instruct`
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - score 来源: preprocess top-10 KV cache 中的 context K，与 query hidden state 计算逐层 QK attention，softmax 后对 layer/head/query token 求和。
  - `score_cosine`: query importance score 向量余弦相似度。
  - `score_l2_rel`: score 相对 L2 差异，越小说明分数越接近。
  - `chunk_cosine`: 选中 token 落到各 chunk 的分布相似度。
- 完成 PreprocessKV example 0, passages=3, rate=0.1: token_jaccard_mean=0.7009, block_jaccard_mean=0.8010, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 0, passages=3, rate=0.2: token_jaccard_mean=0.7240, block_jaccard_mean=0.9163, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 1, passages=3, rate=0.1: token_jaccard_mean=0.6684, block_jaccard_mean=0.8380, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 1, passages=3, rate=0.2: token_jaccard_mean=0.6916, block_jaccard_mean=0.8947, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 2, passages=3, rate=0.1: token_jaccard_mean=0.7894, block_jaccard_mean=0.8860, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 2, passages=3, rate=0.2: token_jaccard_mean=0.7999, block_jaccard_mean=0.9484, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 3, passages=3, rate=0.1: token_jaccard_mean=0.7453, block_jaccard_mean=0.8906, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 3, passages=3, rate=0.2: token_jaccard_mean=0.7997, block_jaccard_mean=0.9314, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 4, passages=3, rate=0.1: token_jaccard_mean=0.7080, block_jaccard_mean=0.8318, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 4, passages=3, rate=0.2: token_jaccard_mean=0.7569, block_jaccard_mean=0.9076, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 5, passages=3, rate=0.1: token_jaccard_mean=0.6981, block_jaccard_mean=0.8305, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 5, passages=3, rate=0.2: token_jaccard_mean=0.7125, block_jaccard_mean=0.9294, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 6, passages=3, rate=0.1: token_jaccard_mean=0.7502, block_jaccard_mean=0.8498, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 6, passages=3, rate=0.2: token_jaccard_mean=0.7662, block_jaccard_mean=0.9160, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 7, passages=3, rate=0.1: token_jaccard_mean=0.6991, block_jaccard_mean=0.8181, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 7, passages=3, rate=0.2: token_jaccard_mean=0.7308, block_jaccard_mean=0.9092, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 8, passages=3, rate=0.1: token_jaccard_mean=0.7161, block_jaccard_mean=0.8669, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 8, passages=3, rate=0.2: token_jaccard_mean=0.7428, block_jaccard_mean=0.9147, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 9, passages=3, rate=0.1: token_jaccard_mean=0.6700, block_jaccard_mean=0.8339, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 9, passages=3, rate=0.2: token_jaccard_mean=0.7239, block_jaccard_mean=0.9242, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 10, passages=3, rate=0.1: token_jaccard_mean=0.7054, block_jaccard_mean=0.8085, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 10, passages=3, rate=0.2: token_jaccard_mean=0.7516, block_jaccard_mean=0.9303, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 11, passages=3, rate=0.1: token_jaccard_mean=0.7092, block_jaccard_mean=0.7876, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 11, passages=3, rate=0.2: token_jaccard_mean=0.7052, block_jaccard_mean=0.8979, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 12, passages=3, rate=0.1: token_jaccard_mean=0.7501, block_jaccard_mean=0.8801, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 12, passages=3, rate=0.2: token_jaccard_mean=0.7512, block_jaccard_mean=0.9447, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 13, passages=3, rate=0.1: token_jaccard_mean=0.7227, block_jaccard_mean=0.8152, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 13, passages=3, rate=0.2: token_jaccard_mean=0.7556, block_jaccard_mean=0.8937, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 14, passages=3, rate=0.1: token_jaccard_mean=0.6848, block_jaccard_mean=0.7847, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 14, passages=3, rate=0.2: token_jaccard_mean=0.7144, block_jaccard_mean=0.8999, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 15, passages=3, rate=0.1: token_jaccard_mean=0.7527, block_jaccard_mean=0.8796, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 15, passages=3, rate=0.2: token_jaccard_mean=0.7946, block_jaccard_mean=0.9335, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 16, passages=3, rate=0.1: token_jaccard_mean=0.6841, block_jaccard_mean=0.8209, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 16, passages=3, rate=0.2: token_jaccard_mean=0.7371, block_jaccard_mean=0.9113, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 17, passages=3, rate=0.1: token_jaccard_mean=0.7072, block_jaccard_mean=0.8458, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 17, passages=3, rate=0.2: token_jaccard_mean=0.7367, block_jaccard_mean=0.9168, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 18, passages=3, rate=0.1: token_jaccard_mean=0.7124, block_jaccard_mean=0.8867, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 18, passages=3, rate=0.2: token_jaccard_mean=0.7343, block_jaccard_mean=0.9384, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 19, passages=3, rate=0.1: token_jaccard_mean=0.7316, block_jaccard_mean=0.8389, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 19, passages=3, rate=0.2: token_jaccard_mean=0.7588, block_jaccard_mean=0.8959, score_cosine_mean=1.0000, score_bad_value_total=0

### PreprocessKV 汇总: passages=3, rate=0.1

- 输出: `preprocesskv_batch_20260626_171056_20examples_3passages_rate0.1.json`
- 耗时: 66.3s
- token_jaccard mean over examples: 0.7153 (min=0.6684, max=0.7894)
- block_jaccard mean over examples: 0.8397 (min=0.7847, max=0.8906)
- score_cosine mean over examples: 1.0000 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1303 (min=0.0732, max=0.2100)

### PreprocessKV 汇总: passages=3, rate=0.2

- 输出: `preprocesskv_batch_20260626_171056_20examples_3passages_rate0.2.json`
- 耗时: 66.3s
- token_jaccard mean over examples: 0.7444 (min=0.6916, max=0.7999)
- block_jaccard mean over examples: 0.9177 (min=0.8937, max=0.9484)
- score_cosine mean over examples: 1.0000 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1303 (min=0.0732, max=0.2100)
- 完成 PreprocessKV example 0, passages=5, rate=0.1: token_jaccard_mean=0.6874, block_jaccard_mean=0.8057, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 0, passages=5, rate=0.2: token_jaccard_mean=0.7247, block_jaccard_mean=0.9000, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 1, passages=5, rate=0.1: token_jaccard_mean=0.6423, block_jaccard_mean=0.8202, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 1, passages=5, rate=0.2: token_jaccard_mean=0.6864, block_jaccard_mean=0.8827, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 2, passages=5, rate=0.1: token_jaccard_mean=0.7625, block_jaccard_mean=0.8970, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 2, passages=5, rate=0.2: token_jaccard_mean=0.7762, block_jaccard_mean=0.9388, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 3, passages=5, rate=0.1: token_jaccard_mean=0.7157, block_jaccard_mean=0.8745, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 3, passages=5, rate=0.2: token_jaccard_mean=0.7420, block_jaccard_mean=0.9013, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 4, passages=5, rate=0.1: token_jaccard_mean=0.6934, block_jaccard_mean=0.8119, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 4, passages=5, rate=0.2: token_jaccard_mean=0.7471, block_jaccard_mean=0.9151, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 5, passages=5, rate=0.1: token_jaccard_mean=0.6524, block_jaccard_mean=0.7979, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 5, passages=5, rate=0.2: token_jaccard_mean=0.6840, block_jaccard_mean=0.9015, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 6, passages=5, rate=0.1: token_jaccard_mean=0.7410, block_jaccard_mean=0.8193, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 6, passages=5, rate=0.2: token_jaccard_mean=0.7507, block_jaccard_mean=0.8906, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 7, passages=5, rate=0.1: token_jaccard_mean=0.6825, block_jaccard_mean=0.8136, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 7, passages=5, rate=0.2: token_jaccard_mean=0.7102, block_jaccard_mean=0.9031, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 8, passages=5, rate=0.1: token_jaccard_mean=0.7307, block_jaccard_mean=0.8709, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 8, passages=5, rate=0.2: token_jaccard_mean=0.7537, block_jaccard_mean=0.8825, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 9, passages=5, rate=0.1: token_jaccard_mean=0.7036, block_jaccard_mean=0.8380, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 9, passages=5, rate=0.2: token_jaccard_mean=0.7092, block_jaccard_mean=0.9085, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 10, passages=5, rate=0.1: token_jaccard_mean=0.7095, block_jaccard_mean=0.8325, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 10, passages=5, rate=0.2: token_jaccard_mean=0.7240, block_jaccard_mean=0.9026, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 11, passages=5, rate=0.1: token_jaccard_mean=0.6919, block_jaccard_mean=0.8094, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 11, passages=5, rate=0.2: token_jaccard_mean=0.7178, block_jaccard_mean=0.8979, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 12, passages=5, rate=0.1: token_jaccard_mean=0.7114, block_jaccard_mean=0.7970, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 12, passages=5, rate=0.2: token_jaccard_mean=0.7425, block_jaccard_mean=0.9171, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 13, passages=5, rate=0.1: token_jaccard_mean=0.7349, block_jaccard_mean=0.8607, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 13, passages=5, rate=0.2: token_jaccard_mean=0.7589, block_jaccard_mean=0.9166, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 14, passages=5, rate=0.1: token_jaccard_mean=0.6894, block_jaccard_mean=0.8070, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 14, passages=5, rate=0.2: token_jaccard_mean=0.7402, block_jaccard_mean=0.8972, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 15, passages=5, rate=0.1: token_jaccard_mean=0.7454, block_jaccard_mean=0.8618, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 15, passages=5, rate=0.2: token_jaccard_mean=0.7764, block_jaccard_mean=0.9284, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 16, passages=5, rate=0.1: token_jaccard_mean=0.6972, block_jaccard_mean=0.8275, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 16, passages=5, rate=0.2: token_jaccard_mean=0.7272, block_jaccard_mean=0.8908, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 17, passages=5, rate=0.1: token_jaccard_mean=0.6789, block_jaccard_mean=0.8266, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 17, passages=5, rate=0.2: token_jaccard_mean=0.6986, block_jaccard_mean=0.9001, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 18, passages=5, rate=0.1: token_jaccard_mean=0.6726, block_jaccard_mean=0.8096, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 18, passages=5, rate=0.2: token_jaccard_mean=0.7177, block_jaccard_mean=0.9153, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 19, passages=5, rate=0.1: token_jaccard_mean=0.7056, block_jaccard_mean=0.8264, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 19, passages=5, rate=0.2: token_jaccard_mean=0.7433, block_jaccard_mean=0.9035, score_cosine_mean=1.0000, score_bad_value_total=0

### PreprocessKV 汇总: passages=5, rate=0.1

- 输出: `preprocesskv_batch_20260626_171056_20examples_5passages_rate0.1.json`
- 耗时: 83.8s
- token_jaccard mean over examples: 0.7024 (min=0.6423, max=0.7625)
- block_jaccard mean over examples: 0.8304 (min=0.7970, max=0.8970)
- score_cosine mean over examples: 1.0000 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1288 (min=0.0727, max=0.2057)

### PreprocessKV 汇总: passages=5, rate=0.2

- 输出: `preprocesskv_batch_20260626_171056_20examples_5passages_rate0.2.json`
- 耗时: 83.8s
- token_jaccard mean over examples: 0.7315 (min=0.6840, max=0.7764)
- block_jaccard mean over examples: 0.9047 (min=0.8825, max=0.9388)
- score_cosine mean over examples: 1.0000 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1288 (min=0.0727, max=0.2057)

### PreprocessKV 运行 20260626_171056 结束

- 全量输出: `preprocesskv_batch_20260626_171056_all_cases.json`

### 指定对照: DraftModel(raw) vs Target-QK(preprocess KV)

- DraftModel(raw): 使用真实 `DraftModel` selector，Qwen2.5-3B-Instruct，raw 文档输入，`DRAFT_LAYER_SEL=rrf`, `RRF_K=18`。
- Target-QK(preprocess KV): 使用主模型 QK selector，但 context K/V 来自 top-10 preprocess KV cache：`/raid/home/hming/fusionrag-pca-top1-top10-cache-20/data/musique-pca-subset-preprocess-10-revert_rope-True/Qwen2.5-7B-Instruct`。
- 两者都使用同一批 20 个 Musique examples、相同 query 变体和 control query；rate 只离线决定选择比例，不重复前向。

| passages | rate | selector/cache | token Jaccard | block Jaccard | score cosine | relative L2 |
|---:|---:|---|---:|---:|---:|---:|
| 3 | 0.1 | DraftModel(raw) | 0.6797 | 0.7445 | 0.9667 | 0.2351 |
| 3 | 0.1 | Target-QK(preprocess KV) | 0.7153 | 0.8397 | 1.0000 | 0.1303 |
| 3 | 0.2 | DraftModel(raw) | 0.7076 | 0.8144 | 0.9667 | 0.2351 |
| 3 | 0.2 | Target-QK(preprocess KV) | 0.7444 | 0.9177 | 1.0000 | 0.1303 |
| 5 | 0.1 | DraftModel(raw) | 0.6815 | 0.7504 | 0.9610 | 0.2548 |
| 5 | 0.1 | Target-QK(preprocess KV) | 0.7024 | 0.8304 | 1.0000 | 0.1288 |
| 5 | 0.2 | DraftModel(raw) | 0.7152 | 0.8263 | 0.9610 | 0.2548 |
| 5 | 0.2 | Target-QK(preprocess KV) | 0.7315 | 0.9047 | 1.0000 | 0.1288 |

当前结论:
- 在这组设置下，Target-QK(preprocess KV) 的跨 query 稳定性明显强于 DraftModel(raw)。
- DraftModel(raw) 的 score cosine 约 0.96，说明 selector 本身更 query-sensitive；Target-QK(preprocess KV) 基本为 1.0000。
- block 粒度上，Target-QK(preprocess KV) 在 rate=0.2 时约 0.90+，而 DraftModel(raw) 约 0.81-0.83。
- 因此如果目标是寻找可跨 query 复用的重算 mask，preprocess KV 空间上的主模型 QK 信号比 DraftModel raw attention 更有希望；DraftModel 更适合保留为在线 query-specific selector。

## DraftModel selector 运行 20260626_175452

- 时间: 2026-06-26T17:54:52
- 目的: 验证固定同一文档时，不同 query 触发的 DraftModel 重算 token 是否稳定。
- 启动参考: `/mnt/qjhs-sh-lab-01/wjh/FusionRAG/run_task.py DraftModel ... --draft_model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`
- 命令参数: `example_start=0`, `num_examples=20`, `max_passages_list=[10]`, `rate_list=[0.1, 0.2]`, `native_count=8`, `control_count=8`, `block_size=16`
- DraftModel: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`, layer_selection=`rrf`, rrf_k=18
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - score 来源: draft model query→doc attention，经 `rrf` 聚合；rate 阶段复用同一 score，离线调用 `smart_query_selection`。

## 运行 20260626_175454

- 时间: 2026-06-26T17:54:54
- 目的: 验证固定同一文档时，使用 preprocess KV 的主模型 QK selector 在不同 query 下是否稳定。
- 命令参数: `example_start=0`, `num_examples=20`, `max_passages_list=[10]`, `rate_list=[0.1, 0.2]`, `native_count=8`, `control_count=8`, `block_size=16`
- preprocess KV: `/raid/home/hming/fusionrag-pca-top1-top10-cache-20/data/musique-pca-subset-preprocess-10-revert_rope-True/Qwen2.5-7B-Instruct`
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - score 来源: preprocess top-10 KV cache 中的 context K，与 query hidden state 计算逐层 QK attention，softmax 后对 layer/head/query token 求和。
  - `score_cosine`: query importance score 向量余弦相似度。
  - `score_l2_rel`: score 相对 L2 差异，越小说明分数越接近。
  - `chunk_cosine`: 选中 token 落到各 chunk 的分布相似度。
- 完成 DraftModel example 0, passages=10, rate=0.1: token_jaccard_mean=0.7084, block_jaccard_mean=0.7820, score_cosine_mean=0.9584, score_bad_value_total=0
- 完成 DraftModel example 0, passages=10, rate=0.2: token_jaccard_mean=0.7203, block_jaccard_mean=0.8260, score_cosine_mean=0.9584, score_bad_value_total=0
- 完成 PreprocessKV example 0, passages=10, rate=0.1: token_jaccard_mean=0.6854, block_jaccard_mean=0.7966, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 0, passages=10, rate=0.2: token_jaccard_mean=0.7217, block_jaccard_mean=0.8958, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 1, passages=10, rate=0.1: token_jaccard_mean=0.6029, block_jaccard_mean=0.6530, score_cosine_mean=0.9594, score_bad_value_total=0
- 完成 PreprocessKV example 1, passages=10, rate=0.1: token_jaccard_mean=0.6544, block_jaccard_mean=0.7907, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 1, passages=10, rate=0.2: token_jaccard_mean=0.6400, block_jaccard_mean=0.7819, score_cosine_mean=0.9594, score_bad_value_total=0
- 完成 PreprocessKV example 1, passages=10, rate=0.2: token_jaccard_mean=0.6933, block_jaccard_mean=0.8964, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 2, passages=10, rate=0.1: token_jaccard_mean=0.7771, block_jaccard_mean=0.8621, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 2, passages=10, rate=0.1: token_jaccard_mean=0.7273, block_jaccard_mean=0.8269, score_cosine_mean=0.9647, score_bad_value_total=0
- 完成 PreprocessKV example 2, passages=10, rate=0.2: token_jaccard_mean=0.7882, block_jaccard_mean=0.9045, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 2, passages=10, rate=0.2: token_jaccard_mean=0.7661, block_jaccard_mean=0.8437, score_cosine_mean=0.9647, score_bad_value_total=0
- 完成 DraftModel example 3, passages=10, rate=0.1: token_jaccard_mean=0.6112, block_jaccard_mean=0.6699, score_cosine_mean=0.9351, score_bad_value_total=0
- 完成 DraftModel example 3, passages=10, rate=0.2: token_jaccard_mean=0.6774, block_jaccard_mean=0.8345, score_cosine_mean=0.9351, score_bad_value_total=0
- 完成 PreprocessKV example 3, passages=10, rate=0.1: token_jaccard_mean=0.6407, block_jaccard_mean=0.8001, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 3, passages=10, rate=0.2: token_jaccard_mean=0.6861, block_jaccard_mean=0.8982, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 4, passages=10, rate=0.1: token_jaccard_mean=0.7069, block_jaccard_mean=0.8052, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 4, passages=10, rate=0.2: token_jaccard_mean=0.7486, block_jaccard_mean=0.9167, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 4, passages=10, rate=0.1: token_jaccard_mean=0.6977, block_jaccard_mean=0.7899, score_cosine_mean=0.9566, score_bad_value_total=0
- 完成 DraftModel example 4, passages=10, rate=0.2: token_jaccard_mean=0.7393, block_jaccard_mean=0.8267, score_cosine_mean=0.9566, score_bad_value_total=0
- 完成 PreprocessKV example 5, passages=10, rate=0.1: token_jaccard_mean=0.6743, block_jaccard_mean=0.7861, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 5, passages=10, rate=0.2: token_jaccard_mean=0.7009, block_jaccard_mean=0.8806, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 5, passages=10, rate=0.1: token_jaccard_mean=0.6513, block_jaccard_mean=0.7066, score_cosine_mean=0.9638, score_bad_value_total=0
- 完成 DraftModel example 5, passages=10, rate=0.2: token_jaccard_mean=0.6780, block_jaccard_mean=0.7995, score_cosine_mean=0.9638, score_bad_value_total=0
- 完成 PreprocessKV example 6, passages=10, rate=0.1: token_jaccard_mean=0.7415, block_jaccard_mean=0.8265, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 6, passages=10, rate=0.2: token_jaccard_mean=0.7776, block_jaccard_mean=0.8746, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 6, passages=10, rate=0.1: token_jaccard_mean=0.7054, block_jaccard_mean=0.7559, score_cosine_mean=0.9581, score_bad_value_total=0
- 完成 DraftModel example 6, passages=10, rate=0.2: token_jaccard_mean=0.7219, block_jaccard_mean=0.8270, score_cosine_mean=0.9581, score_bad_value_total=0
- 完成 PreprocessKV example 7, passages=10, rate=0.1: token_jaccard_mean=0.6623, block_jaccard_mean=0.8144, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 7, passages=10, rate=0.2: token_jaccard_mean=0.6973, block_jaccard_mean=0.9072, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 7, passages=10, rate=0.1: token_jaccard_mean=0.6651, block_jaccard_mean=0.7516, score_cosine_mean=0.9547, score_bad_value_total=0
- 完成 DraftModel example 7, passages=10, rate=0.2: token_jaccard_mean=0.6988, block_jaccard_mean=0.8354, score_cosine_mean=0.9547, score_bad_value_total=0
- 完成 PreprocessKV example 8, passages=10, rate=0.1: token_jaccard_mean=0.6803, block_jaccard_mean=0.8054, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 8, passages=10, rate=0.2: token_jaccard_mean=0.7063, block_jaccard_mean=0.8783, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 8, passages=10, rate=0.1: token_jaccard_mean=0.6528, block_jaccard_mean=0.7398, score_cosine_mean=0.9466, score_bad_value_total=0
- 完成 DraftModel example 8, passages=10, rate=0.2: token_jaccard_mean=0.6896, block_jaccard_mean=0.7946, score_cosine_mean=0.9466, score_bad_value_total=0
- 完成 PreprocessKV example 9, passages=10, rate=0.1: token_jaccard_mean=0.6863, block_jaccard_mean=0.8163, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 9, passages=10, rate=0.2: token_jaccard_mean=0.7012, block_jaccard_mean=0.8989, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 9, passages=10, rate=0.1: token_jaccard_mean=0.6581, block_jaccard_mean=0.7632, score_cosine_mean=0.9432, score_bad_value_total=0
- 完成 DraftModel example 9, passages=10, rate=0.2: token_jaccard_mean=0.7007, block_jaccard_mean=0.8303, score_cosine_mean=0.9432, score_bad_value_total=0
- 完成 PreprocessKV example 10, passages=10, rate=0.1: token_jaccard_mean=0.7221, block_jaccard_mean=0.8282, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 10, passages=10, rate=0.2: token_jaccard_mean=0.7396, block_jaccard_mean=0.8948, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 10, passages=10, rate=0.1: token_jaccard_mean=0.7011, block_jaccard_mean=0.7543, score_cosine_mean=0.9297, score_bad_value_total=0
- 完成 DraftModel example 10, passages=10, rate=0.2: token_jaccard_mean=0.7171, block_jaccard_mean=0.8264, score_cosine_mean=0.9297, score_bad_value_total=0
- 完成 PreprocessKV example 11, passages=10, rate=0.1: token_jaccard_mean=0.6664, block_jaccard_mean=0.7899, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 11, passages=10, rate=0.2: token_jaccard_mean=0.7035, block_jaccard_mean=0.9118, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 11, passages=10, rate=0.1: token_jaccard_mean=0.7082, block_jaccard_mean=0.7774, score_cosine_mean=0.9345, score_bad_value_total=0
- 完成 DraftModel example 11, passages=10, rate=0.2: token_jaccard_mean=0.7202, block_jaccard_mean=0.8212, score_cosine_mean=0.9345, score_bad_value_total=0
- 完成 PreprocessKV example 12, passages=10, rate=0.1: token_jaccard_mean=0.6910, block_jaccard_mean=0.8089, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 12, passages=10, rate=0.2: token_jaccard_mean=0.7226, block_jaccard_mean=0.8971, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 12, passages=10, rate=0.1: token_jaccard_mean=0.6926, block_jaccard_mean=0.7616, score_cosine_mean=0.9509, score_bad_value_total=0
- 完成 DraftModel example 12, passages=10, rate=0.2: token_jaccard_mean=0.7594, block_jaccard_mean=0.8774, score_cosine_mean=0.9509, score_bad_value_total=0
- 完成 PreprocessKV example 13, passages=10, rate=0.1: token_jaccard_mean=0.6733, block_jaccard_mean=0.7901, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 13, passages=10, rate=0.2: token_jaccard_mean=0.7159, block_jaccard_mean=0.9212, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 14, passages=10, rate=0.1: token_jaccard_mean=0.6895, block_jaccard_mean=0.8056, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 14, passages=10, rate=0.2: token_jaccard_mean=0.7022, block_jaccard_mean=0.8921, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 13, passages=10, rate=0.1: token_jaccard_mean=0.6733, block_jaccard_mean=0.7336, score_cosine_mean=0.9613, score_bad_value_total=0
- 完成 DraftModel example 13, passages=10, rate=0.2: token_jaccard_mean=0.6734, block_jaccard_mean=0.8182, score_cosine_mean=0.9613, score_bad_value_total=0
- 完成 DraftModel example 14, passages=10, rate=0.1: token_jaccard_mean=0.6386, block_jaccard_mean=0.7227, score_cosine_mean=0.9543, score_bad_value_total=0
- 完成 DraftModel example 14, passages=10, rate=0.2: token_jaccard_mean=0.6864, block_jaccard_mean=0.8143, score_cosine_mean=0.9543, score_bad_value_total=0
- 完成 PreprocessKV example 15, passages=10, rate=0.1: token_jaccard_mean=0.6886, block_jaccard_mean=0.8429, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 15, passages=10, rate=0.2: token_jaccard_mean=0.7452, block_jaccard_mean=0.9170, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 16, passages=10, rate=0.1: token_jaccard_mean=0.6642, block_jaccard_mean=0.7931, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 16, passages=10, rate=0.2: token_jaccard_mean=0.7315, block_jaccard_mean=0.8726, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 DraftModel example 15, passages=10, rate=0.1: token_jaccard_mean=0.7185, block_jaccard_mean=0.7897, score_cosine_mean=0.9495, score_bad_value_total=0
- 完成 DraftModel example 15, passages=10, rate=0.2: token_jaccard_mean=0.7645, block_jaccard_mean=0.8708, score_cosine_mean=0.9495, score_bad_value_total=0
- 完成 DraftModel example 16, passages=10, rate=0.1: token_jaccard_mean=0.6297, block_jaccard_mean=0.7253, score_cosine_mean=0.8736, score_bad_value_total=0
- 完成 DraftModel example 16, passages=10, rate=0.2: token_jaccard_mean=0.6726, block_jaccard_mean=0.8081, score_cosine_mean=0.8736, score_bad_value_total=0
- 完成 PreprocessKV example 17, passages=10, rate=0.1: token_jaccard_mean=0.7124, block_jaccard_mean=0.8156, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 17, passages=10, rate=0.2: token_jaccard_mean=0.7205, block_jaccard_mean=0.8975, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 18, passages=10, rate=0.1: token_jaccard_mean=0.6881, block_jaccard_mean=0.8157, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 18, passages=10, rate=0.2: token_jaccard_mean=0.7224, block_jaccard_mean=0.9237, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 17, passages=10, rate=0.1: token_jaccard_mean=0.6601, block_jaccard_mean=0.7392, score_cosine_mean=0.9666, score_bad_value_total=0
- 完成 DraftModel example 17, passages=10, rate=0.2: token_jaccard_mean=0.6849, block_jaccard_mean=0.8110, score_cosine_mean=0.9666, score_bad_value_total=0
- 完成 PreprocessKV example 19, passages=10, rate=0.1: token_jaccard_mean=0.7067, block_jaccard_mean=0.8467, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 19, passages=10, rate=0.2: token_jaccard_mean=0.7377, block_jaccard_mean=0.9031, score_cosine_mean=1.0000, score_bad_value_total=0

### PreprocessKV 汇总: passages=10, rate=0.1

- 输出: `preprocesskv_batch_20260626_175454_20examples_10passages_rate0.1.json`
- 耗时: 202.5s
- token_jaccard mean over examples: 0.6906 (min=0.6407, max=0.7771)
- block_jaccard mean over examples: 0.8120 (min=0.7861, max=0.8621)
- score_cosine mean over examples: 1.0000 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1262 (min=0.0741, max=0.1960)

### PreprocessKV 汇总: passages=10, rate=0.2

- 输出: `preprocesskv_batch_20260626_175454_20examples_10passages_rate0.2.json`
- 耗时: 202.5s
- token_jaccard mean over examples: 0.7231 (min=0.6861, max=0.7882)
- block_jaccard mean over examples: 0.8991 (min=0.8726, max=0.9237)
- score_cosine mean over examples: 1.0000 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1262 (min=0.0741, max=0.1960)

### PreprocessKV 运行 20260626_175454 结束

- 全量输出: `preprocesskv_batch_20260626_175454_all_cases.json`
- 完成 DraftModel example 18, passages=10, rate=0.1: token_jaccard_mean=0.7064, block_jaccard_mean=0.7940, score_cosine_mean=0.9387, score_bad_value_total=0
- 完成 DraftModel example 18, passages=10, rate=0.2: token_jaccard_mean=0.7223, block_jaccard_mean=0.8364, score_cosine_mean=0.9387, score_bad_value_total=0
- 完成 DraftModel example 19, passages=10, rate=0.1: token_jaccard_mean=0.6815, block_jaccard_mean=0.7784, score_cosine_mean=0.9453, score_bad_value_total=0
- 完成 DraftModel example 19, passages=10, rate=0.2: token_jaccard_mean=0.7190, block_jaccard_mean=0.8465, score_cosine_mean=0.9453, score_bad_value_total=0

### DraftModel 汇总: passages=10, rate=0.1, mode=rrf

- 输出: `draft_batch_20260626_175452_20examples_10passages_rate0.1_rrf.json`
- 耗时: 218.7s
- token_jaccard mean over examples: 0.6745 (min=0.6029, max=0.7273)
- block_jaccard mean over examples: 0.7508 (min=0.6530, max=0.8269)
- score_cosine mean over examples: 0.9472 (min=0.8736, max=0.9666)
- score_l2_rel mean over examples: 0.2958 (min=0.2444, max=0.4415)

### DraftModel 汇总: passages=10, rate=0.2, mode=rrf

- 输出: `draft_batch_20260626_175452_20examples_10passages_rate0.2_rrf.json`
- 耗时: 218.7s
- token_jaccard mean over examples: 0.7076 (min=0.6400, max=0.7661)
- block_jaccard mean over examples: 0.8265 (min=0.7819, max=0.8774)
- score_cosine mean over examples: 0.9472 (min=0.8736, max=0.9666)
- score_l2_rel mean over examples: 0.2958 (min=0.2444, max=0.4415)

### DraftModel 运行 20260626_175452 结束

- 全量输出: `draft_batch_20260626_175452_all_cases.json`

### 指定对照补充: 加入 passages=10

- 在原有 passages=3/5 的基础上，补跑 passages=10；其他条件保持一致。
- DraftModel(raw): Qwen2.5-3B-Instruct，`DRAFT_LAYER_SEL=rrf`, `RRF_K=18`。
- Target-QK(preprocess KV): top-10 preprocess KV cache + 主模型 QK selector。

| passages | rate | selector/cache | token Jaccard | block Jaccard | score cosine | relative L2 |
|---:|---:|---|---:|---:|---:|---:|
| 3 | 0.1 | DraftModel(raw) | 0.6797 | 0.7445 | 0.9667 | 0.2351 |
| 3 | 0.1 | Target-QK(preprocess KV) | 0.7153 | 0.8397 | 1.0000 | 0.1303 |
| 3 | 0.2 | DraftModel(raw) | 0.7076 | 0.8144 | 0.9667 | 0.2351 |
| 3 | 0.2 | Target-QK(preprocess KV) | 0.7444 | 0.9177 | 1.0000 | 0.1303 |
| 5 | 0.1 | DraftModel(raw) | 0.6815 | 0.7504 | 0.9610 | 0.2548 |
| 5 | 0.1 | Target-QK(preprocess KV) | 0.7024 | 0.8304 | 1.0000 | 0.1288 |
| 5 | 0.2 | DraftModel(raw) | 0.7152 | 0.8263 | 0.9610 | 0.2548 |
| 5 | 0.2 | Target-QK(preprocess KV) | 0.7315 | 0.9047 | 1.0000 | 0.1288 |
| 10 | 0.1 | DraftModel(raw) | 0.6745 | 0.7508 | 0.9472 | 0.2958 |
| 10 | 0.1 | Target-QK(preprocess KV) | 0.6906 | 0.8120 | 1.0000 | 0.1262 |
| 10 | 0.2 | DraftModel(raw) | 0.7076 | 0.8265 | 0.9472 | 0.2958 |
| 10 | 0.2 | Target-QK(preprocess KV) | 0.7231 | 0.8991 | 1.0000 | 0.1262 |

passages=10 新观察:
- Target-QK(preprocess KV) 在 passages=10 下仍保持较高稳定性：block Jaccard 为 0.8120 / 0.8991。
- DraftModel(raw) 在 passages=10 下 score cosine 降到 0.9472，relative L2 升到 0.2958，说明长文档下 draft attention 更容易随 query 改变。
- 加入 passages=10 后，整体结论不变：Target-QK(preprocess KV) 比 DraftModel(raw) 更适合作为跨 query 复用 mask 的信号。

## DraftModel selector 运行 20260626_182251

- 时间: 2026-06-26T18:22:51
- 目的: 验证固定同一文档时，不同 query 触发的 DraftModel 重算 token 是否稳定。
- 启动参考: `/mnt/qjhs-sh-lab-01/wjh/FusionRAG/run_task.py DraftModel ... --draft_model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`
- 命令参数: `example_start=0`, `num_examples=20`, `max_passages_list=[3, 5, 10]`, `rate_list=[0.1, 0.2]`, `native_count=8`, `control_count=8`, `block_size=16`
- DraftModel: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`, layer_selection=`rrf`, rrf_k=18
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - score 来源: draft model query→doc attention，经 `rrf` 聚合；rate 阶段复用同一 score，离线调用 `smart_query_selection`。

## 运行 20260626_182252

- 时间: 2026-06-26T18:22:52
- 目的: 验证固定同一文档时，使用 preprocess KV 的主模型 QK selector 在不同 query 下是否稳定。
- 命令参数: `example_start=0`, `num_examples=20`, `max_passages_list=[3, 5, 10]`, `rate_list=[0.1, 0.2]`, `native_count=8`, `control_count=8`, `block_size=16`
- preprocess KV: `/raid/home/hming/fusionrag-pca-top1-top10-cache-20/data/musique-pca-subset-preprocess-10-revert_rope-True/Qwen2.5-7B-Instruct`
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - score 来源: preprocess top-10 KV cache 中的 context K，与 query hidden state 计算逐层 QK attention，softmax 后对 layer/head/query token 求和。
  - `score_cosine`: query importance score 向量余弦相似度。
  - `score_l2_rel`: score 相对 L2 差异，越小说明分数越接近。
  - `chunk_cosine`: 选中 token 落到各 chunk 的分布相似度。
## 2026-06-26 流程修正：forward 与 rate 分析解耦

- 结论：不同 `rate` 下挑选哪些 token 是静态后处理，不需要重新 forward。
- 正确流程：每个 `(selector, example, passages, query)` 只做一次 forward/QK 计算，保存 token-level importance score。
- 保存内容：`details/<run_id>/*_scores.npz` 中包含 `scores[query, token]`、query 文本、label、context length、chunk 起点等元信息。
- 离线分析：使用 `tools_analyze_saved_query_scores.py` 读取 score npz，再对任意 `rate_list` 计算完整 `selected_positions`、`selected_blocks`、all-query intersection、token/block Jaccard、score cosine 等指标。
- 启动示例：

```bash
TMPDIR=/raid/home/hming/tmp /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python tools_analyze_saved_query_scores.py \
  --score-root docs/experiments/query_recompute_overlap_detail_debug/details/20260626_183241 \
  --out-dir docs/experiments/query_recompute_overlap_detail_debug/offline_from_scores \
  --rate-list 0.05,0.1,0.2,0.3 \
  --device cpu
```

- 注意：远程根分区 `/` 已满，所有后续实验都应设置 `TMPDIR=/raid/home/hming/tmp`。

## 2026-06-26 明细版：所有 query 共同选中 token 的交集占比

- 输出目录：`docs/experiments/query_recompute_overlap_detail_full`
- score cache：`details/20260626_184238/` 与 `details/20260626_184239/`
- 样本：20 个 MuSiQue examples。
- query：每个 example 固定同一组文档，使用 16 个 query，包括 1 个原始问题、7 个模板改写相关问题、8 个其他 example 的无关 control 问题。
- 指标：`all_query_common_token_ratio = |16 个 query 选中 token 集合的交集| / min_i |selected_i|`。这个指标比两两 Jaccard 更严格，表示“所有 query 都共同选中”的 token 占最小选择集合的比例。

| passages | rate | selector/cache | examples | all-query common token ratio | min | max | mean common tokens |
|---:|---:|---|---:|---:|---:|---:|---:|
| 3 | 0.1 | DraftModel(raw) | 20 | 0.5691 | 0.4211 | 0.6893 | 130.4 |
| 3 | 0.1 | Target-QK(preprocess KV) | 20 | 0.6430 | 0.5836 | 0.7571 | 146.0 |
| 3 | 0.2 | DraftModel(raw) | 20 | 0.5828 | 0.5000 | 0.6702 | 265.8 |
| 3 | 0.2 | Target-QK(preprocess KV) | 20 | 0.6661 | 0.5925 | 0.7415 | 303.3 |
| 5 | 0.1 | DraftModel(raw) | 20 | 0.5740 | 0.4471 | 0.6504 | 215.4 |
| 5 | 0.1 | Target-QK(preprocess KV) | 20 | 0.6284 | 0.5350 | 0.7076 | 234.8 |
| 5 | 0.2 | DraftModel(raw) | 20 | 0.6003 | 0.5204 | 0.7086 | 448.6 |
| 5 | 0.2 | Target-QK(preprocess KV) | 20 | 0.6468 | 0.5849 | 0.7164 | 482.8 |
| 10 | 0.1 | DraftModel(raw) | 20 | 0.5686 | 0.4846 | 0.6529 | 417.1 |
| 10 | 0.1 | Target-QK(preprocess KV) | 20 | 0.6127 | 0.5534 | 0.6913 | 450.8 |
| 10 | 0.2 | DraftModel(raw) | 20 | 0.5993 | 0.5347 | 0.6558 | 880.0 |
| 10 | 0.2 | Target-QK(preprocess KV) | 20 | 0.6429 | 0.5927 | 0.7221 | 945.0 |

初步观察：即使用 16 个 query 做严格交集，仍有大约 57%-67% 的 selected token 是所有 query 共同选中的。Target-QK(preprocess KV) 的共同交集比例稳定高于 DraftModel(raw)，说明在固定文档下，preprocess KV 上的 query-conditioned token importance 更稳定、更接近“文档自身重要 token”的静态结构。

### 追加：rate=0.5 离线分析

- 计算方式：复用 `details/*_scores.npz`，没有重新 forward。
- 输出目录：`docs/experiments/query_recompute_overlap_detail_full/offline_rate05`

| passages | rate | selector/cache | examples | all-query common token ratio | min | max | mean common tokens |
|---:|---:|---|---:|---:|---:|---:|---:|
| 3 | 0.5 | DraftModel(raw) | 20 | 0.7047 | 0.6202 | 0.7708 | 799.5 |
| 3 | 0.5 | Target-QK(preprocess KV) | 20 | 0.7690 | 0.7148 | 0.8159 | 876.9 |
| 5 | 0.5 | DraftModel(raw) | 20 | 0.7014 | 0.6366 | 0.7762 | 1312.7 |
| 5 | 0.5 | Target-QK(preprocess KV) | 20 | 0.7688 | 0.7220 | 0.8200 | 1439.5 |
| 10 | 0.5 | DraftModel(raw) | 20 | 0.7006 | 0.6481 | 0.7800 | 2577.1 |
| 10 | 0.5 | Target-QK(preprocess KV) | 20 | 0.7676 | 0.7226 | 0.8051 | 2824.9 |

rate 提高到 0.5 后，共同交集占比明显上升到约 70%-77%。这符合预期：选 token 的预算越大，不同 query 的 selected set 更容易覆盖同一批稳定高分 token。Preprocess KV 仍然比 DraftModel(raw) 高约 6-7 个百分点。

### 追加：rate=0.3 离线分析

- 计算方式：复用 `details/*_scores.npz`，没有重新 forward。
- 输出目录：`docs/experiments/query_recompute_overlap_detail_full/offline_rate03`

| passages | rate | selector/cache | examples | all-query common token ratio | min | max | mean common tokens |
|---:|---:|---|---:|---:|---:|---:|---:|
| 3 | 0.3 | DraftModel(raw) | 20 | 0.6350 | 0.5457 | 0.7076 | 431.8 |
| 3 | 0.3 | Target-QK(preprocess KV) | 20 | 0.6974 | 0.6477 | 0.7660 | 476.1 |
| 5 | 0.3 | DraftModel(raw) | 20 | 0.6388 | 0.5791 | 0.6994 | 714.4 |
| 5 | 0.3 | Target-QK(preprocess KV) | 20 | 0.6872 | 0.6327 | 0.7346 | 768.6 |
| 10 | 0.3 | DraftModel(raw) | 20 | 0.6268 | 0.5637 | 0.6970 | 1382.2 |
| 10 | 0.3 | Target-QK(preprocess KV) | 20 | 0.6842 | 0.6359 | 0.7671 | 1509.3 |

rate=0.3 的结果处在 rate=0.2 与 rate=0.5 之间，趋势连续：rate 越高，所有 query 共同选中的 token 占比越高；Target-QK(preprocess KV) 仍然稳定高于 DraftModel(raw)。
