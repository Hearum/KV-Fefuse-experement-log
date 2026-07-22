# Oracle Full Attention Selector Compare, rate=0.3

## 定义

`oracle_full_attention`：先跑真正 full-context forward，取 query->doc full-softmax attention 的 all-layer 平均分布，再按 rate 取 top token。
`fusionrag_qk`：FusionRAG 当前 preprocess KV 上的 QK/importance selector。

## Selector 对理想分布的覆盖

| selector | n | ideal mass | ideal last-layer mass | Jaccard vs oracle | score cosine | score JS | score Spearman | selector time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| draft_selected_set | 109 | 0.3907 | 0.3367 | 0.2595 | 0.4636 | 0.3335 | 0.0237 | 0.1382s |
| fusionrag_qk | 109 | 0.3491 | 0.2339 | 0.2348 | 0.3151 | 0.2205 | 0.2092 | 0.0271s |
| oracle_full_attention | 109 | 0.6002 | 0.9775 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.3024s |
| random | 109 | 0.2999 | 0.2938 | 0.1744 | 0.6510 | 0.0705 | -0.3914 | 0.0000s |

## 生成指标

| selector | n | F1 | EM |
|---|---:|---:|---:|
| draft_selected_set | 109 | 0.5619 | 0.2477 |
| fusionrag_qk | 109 | 0.5170 | 0.2385 |
| oracle_full_attention | 109 | 0.4747 | 0.2202 |

## 文件

- selector detail: `MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare_with_draft_answer/selector_detail_rate0p3.csv`
- answer detail: `MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare_with_draft_answer/answer_detail_rate0p3.csv`
