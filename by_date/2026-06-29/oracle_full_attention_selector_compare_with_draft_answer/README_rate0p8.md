# Oracle Full Attention Selector Compare, rate=0.8

## 定义

`oracle_full_attention`：先跑真正 full-context forward，取 query->doc full-softmax attention 的 all-layer 平均分布，再按 rate 取 top token。
`fusionrag_qk`：FusionRAG 当前 preprocess KV 上的 QK/importance selector。

## Selector 对理想分布的覆盖

| selector | n | ideal mass | ideal last-layer mass | Jaccard vs oracle | score cosine | score JS | score Spearman | selector time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| draft_selected_set | 109 | 0.8513 | 0.7654 | 0.6877 | 0.6196 | 0.1233 | -0.1812 | 0.1392s |
| fusionrag_qk | 109 | 0.8456 | 0.7359 | 0.7097 | 0.3151 | 0.2205 | 0.2092 | 0.0293s |
| oracle_full_attention | 109 | 0.9241 | 0.9990 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.3026s |
| random | 109 | 0.7991 | 0.8013 | 0.6667 | 0.6510 | 0.0705 | -0.3914 | 0.0000s |

## 生成指标

| selector | n | F1 | EM |
|---|---:|---:|---:|
| draft_selected_set | 109 | 0.5718 | 0.2569 |
| fusionrag_qk | 109 | 0.5183 | 0.2294 |
| oracle_full_attention | 109 | 0.5739 | 0.2661 |

## 文件

- selector detail: `MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare_with_draft_answer/selector_detail_rate0p8.csv`
- answer detail: `MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare_with_draft_answer/answer_detail_rate0p8.csv`
