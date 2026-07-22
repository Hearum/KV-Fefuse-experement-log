# Full-Attention Stable Anchor Decode

## 实验内容

复用 full-attention query-to-doc attention 分布，对每个 query 取 top-rate doc tokens，并找出所有 query 都会选中的 stable anchors。然后将这些 token index 映射回原始 doc token id，解码 token text，并做粗粒度类别统计。

## 设置

- source_dir: `MOTIVATION_EXPERIMENTS/full_attention_query_anchor_stability`
- rate: `0.15`
- stable criterion: token 被至少 `1.00` 比例的 query 选中；当前即所有 query 都选中
- examples: `20`

## Aggregate

- stable anchors / top-k mean: `0.7223`
- stable anchors / all doc tokens mean: `0.1081`
- total decoded stable anchors: `4385`

### Category distribution over stable anchors

| category | ratio | count |
|---|---:|---:|
| latin_word | 0.6091 | 2671 |
| punct_or_symbol | 0.2698 | 1183 |
| number | 0.0750 | 329 |
| space | 0.0353 | 155 |
| other_text | 0.0100 | 44 |
| newline_or_space | 0.0007 | 3 |

### Position distribution over stable anchors

| document position bin | ratio | count |
|---|---:|---:|
| 00-10% | 0.0048 | 21 |
| 10-20% | 0.0046 | 20 |
| 20-30% | 0.0059 | 26 |
| 30-40% | 0.0080 | 35 |
| 40-50% | 0.0116 | 51 |
| 50-60% | 0.0239 | 105 |
| 60-70% | 0.0410 | 180 |
| 70-80% | 0.0787 | 345 |
| 80-90% | 0.1535 | 673 |
| 90-100% | 0.6680 | 2929 |

## Files

- `stable_anchor_tokens.csv`
- `stable_anchor_category_by_example.csv`
- `stable_anchor_example_summary.csv`
- `stable_anchor_examples.md`
