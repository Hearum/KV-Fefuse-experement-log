# Full-Attention Per-Document Anchor Stability

## 实验内容

复用已保存的 full-attention query-to-doc attention distribution。对比两种 top-token 选择方式：

- `global_top`: 在整个拼接文档 token 序列上直接取 attention top x%。
- `per_doc_top`: 在每个 retrieved document/chunk 内部分别取 attention top x%，再合并。这个设置控制了文档顺序/末尾文档占优的问题。

## Aggregate

| rate | method | Jaccard | related-unrelated Jaccard | all-query intersection/top-k | stable/doc | never selected | stable mean position |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.15 | global_top | 0.7452 | 0.7343 | 0.7223 | 0.1081 | 0.6494 | 0.8840 |
| 0.15 | per_doc_top | 0.6291 | 0.6090 | 0.5028 | 0.0735 | 0.5731 | 0.5332 |
| 0.30 | global_top | 0.8111 | 0.8032 | 0.7729 | 0.2317 | 0.4434 | 0.7903 |
| 0.30 | per_doc_top | 0.7547 | 0.7432 | 0.6646 | 0.1971 | 0.4014 | 0.5089 |

## Doc Attention Mass By Position

| doc position bin | mean attention mass ratio | median | n |
|---|---:|---:|---:|
| 00-10% | 0.0367 | 0.0351 | 1221 |
| 10-20% | 0.0336 | 0.0334 | 1011 |
| 20-30% | 0.0363 | 0.0359 | 1011 |
| 30-40% | 0.0377 | 0.0297 | 1081 |
| 40-50% | 0.0413 | 0.0359 | 766 |
| 50-60% | 0.0487 | 0.0393 | 1081 |
| 60-70% | 0.0580 | 0.0447 | 1116 |
| 70-80% | 0.0820 | 0.0766 | 976 |
| 80-90% | 0.0800 | 0.0789 | 1046 |
| 90-100% | 0.1808 | 0.2066 | 1256 |

## Files

- `aggregate_summary.csv`
- `example_summary.csv`
- `stable_anchor_tokens.csv`
- `stable_anchor_position_and_category.csv`
- `doc_attention_mass.csv`
- `doc_attention_mass_by_position.csv`
