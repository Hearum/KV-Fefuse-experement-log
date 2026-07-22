# Oracle Full Attention Selector Compare, rate=0.15

## 定义

`oracle_full_attention`：先跑真正 full-context forward，取 query->doc full-softmax attention 的 all-layer 平均分布，再按 rate 取 top token。
`fusionrag_qk`：FusionRAG 当前 preprocess KV 上的 QK/importance selector。

## Selector 对理想分布的覆盖

| selector | n | ideal mass | ideal last-layer mass | Jaccard vs oracle | score cosine | score JS | score Spearman | selector time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fusionrag_qk | 2 | 0.2057 | 0.0428 | 0.1471 | 0.2891 | 0.2079 | 0.2455 | 0.0462s |
| oracle_full_attention | 2 | 0.4315 | 0.7468 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.3726s |
| random | 2 | 0.1411 | 0.1508 | 0.0775 | 0.6563 | 0.0697 | -0.2913 | 0.0000s |

## 生成指标

| selector | n | F1 | EM |
|---|---:|---:|---:|
| fusionrag_qk | 2 | 0.6176 | 0.5000 |
| oracle_full_attention | 2 | 0.3904 | 0.0000 |

## 文件

- selector detail: `MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare_smoke/selector_detail_rate0p15.csv`
- answer detail: `MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare_smoke/answer_detail_rate0p15.csv`
