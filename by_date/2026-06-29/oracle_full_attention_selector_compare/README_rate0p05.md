# Oracle Full Attention Selector Compare, rate=0.05

## 定义

`oracle_full_attention`：先跑真正 full-context forward，取 query->doc full-softmax attention 的 all-layer 平均分布，再按 rate 取 top token。
`fusionrag_qk`：FusionRAG 当前 preprocess KV 上的 QK/importance selector。

## Selector 对理想分布的覆盖

| selector | n | ideal mass | ideal last-layer mass | Jaccard vs oracle | score cosine | score JS | score Spearman | selector time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| draft_selected_set | 109 | 0.0852 | 0.0444 | 0.0931 | 0.2479 | 0.5630 | -0.2893 | 0.1381s |
| fusionrag_qk | 109 | 0.0644 | 0.0374 | 0.0314 | 0.3151 | 0.2205 | 0.2092 | 0.0286s |
| oracle_full_attention | 109 | 0.2377 | 0.4179 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.3032s |
| random | 109 | 0.0496 | 0.0429 | 0.0245 | 0.6510 | 0.0705 | -0.3914 | 0.0000s |

## 生成指标

| selector | n | F1 | EM |
|---|---:|---:|---:|
| fusionrag_qk | 109 | 0.4421 | 0.1927 |
| oracle_full_attention | 109 | 0.4189 | 0.1743 |

## 文件

- selector detail: `MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare/selector_detail_rate0p05.csv`
- answer detail: `MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare/answer_detail_rate0p05.csv`
