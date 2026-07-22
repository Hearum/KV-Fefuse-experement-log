# Oracle Full Attention Selector Compare, rate=0.5

## 定义

`oracle_full_attention`：先跑真正 full-context forward，取 query->doc full-softmax attention 的 all-layer 平均分布，再按 rate 取 top token。
`fusionrag_qk`：FusionRAG 当前 preprocess KV 上的 QK/importance selector。

## Selector 对理想分布的覆盖

| selector | n | ideal mass | ideal last-layer mass | Jaccard vs oracle | score cosine | score JS | score Spearman | selector time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| draft_selected_set | 109 | 0.5950 | 0.5003 | 0.3956 | 0.5473 | 0.2299 | 0.0316 | 0.1386s |
| fusionrag_qk | 109 | 0.5565 | 0.4126 | 0.3981 | 0.3151 | 0.2205 | 0.2092 | 0.0270s |
| oracle_full_attention | 109 | 0.7566 | 0.9946 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.3036s |
| random | 109 | 0.4997 | 0.5028 | 0.3341 | 0.6510 | 0.0705 | -0.3914 | 0.0000s |

## 生成指标

| selector | n | F1 | EM |
|---|---:|---:|---:|
| draft_selected_set | 109 | 0.5799 | 0.2752 |
| fusionrag_qk | 109 | 0.5386 | 0.2294 |
| oracle_full_attention | 109 | 0.4892 | 0.2110 |

## 文件

- selector detail: `MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare_with_draft_answer/selector_detail_rate0p5.csv`
- answer detail: `MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare_with_draft_answer/answer_detail_rate0p5.csv`
