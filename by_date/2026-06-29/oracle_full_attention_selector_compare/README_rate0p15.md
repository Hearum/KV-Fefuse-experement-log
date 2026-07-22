# Oracle Full Attention Selector Compare, rate=0.15

## 定义

`oracle_full_attention`：先跑真正 full-context forward，取 query->doc full-softmax attention 的 all-layer 平均分布，再按 rate 取 top token。
`fusionrag_qk`：FusionRAG 当前 preprocess KV 上的 QK/importance selector。

## Selector 对理想分布的覆盖

| selector | n | ideal mass | ideal last-layer mass | Jaccard vs oracle | score cosine | score JS | score Spearman | selector time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| draft_selected_set | 109 | 0.2187 | 0.1327 | 0.1626 | 0.3667 | 0.4454 | -0.0884 | 0.1383s |
| fusionrag_qk | 109 | 0.1864 | 0.1186 | 0.1172 | 0.3151 | 0.2205 | 0.2092 | 0.0267s |
| oracle_full_attention | 109 | 0.4306 | 0.8150 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.3026s |
| random | 109 | 0.1501 | 0.1458 | 0.0806 | 0.6510 | 0.0705 | -0.3914 | 0.0000s |

## 生成指标

| selector | n | F1 | EM |
|---|---:|---:|---:|
| fusionrag_qk | 109 | 0.4764 | 0.2110 |
| oracle_full_attention | 109 | 0.4394 | 0.1927 |

## 文件

- selector detail: `MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare/selector_detail_rate0p15.csv`
- answer detail: `MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare/answer_detail_rate0p15.csv`
