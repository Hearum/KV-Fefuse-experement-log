# Phase 1c: Chunk-Local Set Order Robustness

## 实验内容

验证 chunk-level offline fixed set 是否依赖 RAG 召回顺序。对同一个 example 的同一批 chunks，构造 original / reverse / shuffle_a / shuffle_b 四种顺序；每种顺序下只用 control_other_example queries 重新计算 QK score，并在每个 chunk 内派生 local fixed set。最后比较同一个 chunk 在扰动顺序下的 local selected set 与 original 顺序下的 Jaccard。

## 设置

- examples: `0` to `4`
- chunks per example: `10`
- control queries per example: `21`
- rates: `0.10, 0.15, 0.30, 0.50`
- no native/native_template query is used for deriving the sets

## Aggregate

| rate | order | Jaccard mean | Jaccard median | min | max | n chunks |
|---:|---|---:|---:|---:|---:|---:|
| 0.10 | reverse | 0.9811 | 0.9792 | 0.9429 | 1.0000 | 50 |
| 0.10 | shuffle_a | 0.9855 | 0.9794 | 0.9556 | 1.0000 | 50 |
| 0.10 | shuffle_b | 0.9830 | 0.9792 | 0.9429 | 1.0000 | 50 |
| 0.15 | reverse | 0.9809 | 0.9857 | 0.8909 | 1.0000 | 50 |
| 0.15 | shuffle_a | 0.9827 | 0.9856 | 0.9000 | 1.0000 | 50 |
| 0.15 | shuffle_b | 0.9837 | 0.9859 | 0.8909 | 1.0000 | 50 |
| 0.30 | reverse | 0.9853 | 0.9867 | 0.9434 | 1.0000 | 50 |
| 0.30 | shuffle_a | 0.9872 | 0.9864 | 0.9487 | 1.0000 | 50 |
| 0.30 | shuffle_b | 0.9884 | 0.9876 | 0.9487 | 1.0000 | 50 |
| 0.50 | reverse | 0.9917 | 0.9916 | 0.9692 | 1.0000 | 50 |
| 0.50 | shuffle_a | 0.9915 | 0.9920 | 0.9667 | 1.0000 | 50 |
| 0.50 | shuffle_b | 0.9919 | 0.9915 | 0.9833 | 1.0000 | 50 |
| 0.10 | all_perturbations | 0.9832 | 0.9792 | 0.9429 | 1.0000 | 150 |
| 0.15 | all_perturbations | 0.9824 | 0.9858 | 0.8909 | 1.0000 | 150 |
| 0.30 | all_perturbations | 0.9870 | 0.9867 | 0.9434 | 1.0000 | 150 |
| 0.50 | all_perturbations | 0.9917 | 0.9916 | 0.9667 | 1.0000 | 150 |

## Files

- `chunk_order_score_meta.csv`
- `chunk_order_jaccard_detail.csv`
- `chunk_order_jaccard_aggregate.csv`

Runtime seconds: `271.6`
