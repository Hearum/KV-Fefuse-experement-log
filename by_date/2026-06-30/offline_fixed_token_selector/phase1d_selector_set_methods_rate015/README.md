# Offline Fixed Set Methods at rate=0.15

本实验只构造 offline fixed token sets，不做模型前向。

校准 query 只使用 `control_other_example`，不使用当前 example 的 native/native_template query。

## Methods

- `qk_frequency_per_chunk`: QK selector 分数；每个校准 query 在 chunk 内 top-rate，统计频率后选最高频 token。
- `qk_mean_score_per_chunk`: QK selector 分数；对校准 query 的分数取均值后 chunk 内 top-rate。
- `draft_frequency_per_chunk`: draft selector 分数；频率规则。
- `draft_mean_score_per_chunk`: draft selector 分数；均值规则。
- `position_tail_per_chunk`: 每个 chunk 选最后 rate token，用于测试 recency/tail bias。
- `position_boundary_per_chunk`: 每个 chunk 前后边界各取一部分。
- `random_per_chunk`: 随机固定 set。

## Held-out Overlap vs Online QK

| method | Jaccard | coverage online by offline | n |
|---|---:|---:|---:|
| qk_frequency_per_chunk | 0.6960 | 0.8152 | 2200 |
| qk_mean_score_per_chunk | 0.6910 | 0.8120 | 2200 |
| draft_frequency_per_chunk | 0.4367 | 0.6032 | 2200 |
| draft_mean_score_per_chunk | 0.4306 | 0.5973 | 2200 |
| position_boundary_per_chunk | 0.2224 | 0.3578 | 2200 |
| random_per_chunk | 0.0794 | 0.1463 | 2200 |
| position_tail_per_chunk | 0.0703 | 0.1283 | 2200 |
