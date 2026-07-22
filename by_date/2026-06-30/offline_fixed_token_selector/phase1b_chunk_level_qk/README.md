# Phase 1b: Chunk-Level Offline Fixed Token Sets

## 实验内容

修正 phase-1 的部署假设：offline 阶段无法知道 RAG 最终召回的 chunk 顺序，因此 fixed set 必须以单个 chunk 为单位保存 chunk-local token indices。

本实验复用已有 Target-QK(preprocess KV) score cache，只用 `control_other_example` queries 为每个 chunk 独立构造 fixed set，再映射回当前 held-out query 的拼接上下文位置，与 online per-document QK selection 对比。

## Offline Set 定义

`qk_calib_per_chunk_frequency`: 对每个 chunk 和每个 calibration query，在 chunk 内取 top-rate token；统计 token 频率；最终在该 chunk 内选频率最高的 top-rate local tokens。

## 防泄漏

- calibration queries: `control_other_example`
- held-out eval queries: `native + native_template`
- `uses_current_example_question_for_offline_set = false`

## Aggregate

| rate | method | Jaccard vs online per-doc QK | coverage online by offline | n |
|---:|---|---:|---:|---:|
| 0.05 | qk_calib_per_chunk_frequency | 0.6641 | 0.7964 | 220 |
| 0.05 | random_per_chunk | 0.0243 | 0.0473 | 220 |
| 0.10 | qk_calib_per_chunk_frequency | 0.6743 | 0.8040 | 220 |
| 0.10 | random_per_chunk | 0.0532 | 0.1009 | 220 |
| 0.15 | qk_calib_per_chunk_frequency | 0.6890 | 0.8145 | 220 |
| 0.15 | random_per_chunk | 0.0801 | 0.1483 | 220 |
| 0.30 | qk_calib_per_chunk_frequency | 0.7495 | 0.8559 | 220 |
| 0.30 | random_per_chunk | 0.1766 | 0.3002 | 220 |
| 0.50 | qk_calib_per_chunk_frequency | 0.8230 | 0.9025 | 220 |
| 0.50 | random_per_chunk | 0.3329 | 0.4995 | 220 |

## Files

- `chunk_fixed_set_manifest.csv`
- `offline_chunk_vs_online_detail.csv`
- `offline_chunk_vs_online_aggregate.csv`
- `chunk_fixed_sets_npz/`
