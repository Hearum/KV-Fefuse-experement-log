# Selection Method vs Query Attention Mass

## 目的

验证“被选为重算 token 的位置是否承担主要 attention mass”是否依赖 selection 方法，并加入随机 selection 空白对照。

## 方法

- 评价用 attention mass: 主模型在 preprocess KV 上的 query-context QK softmax attention，排除 system tokens。
- selector:
  - `FusionRAG-QK(preprocess,32q)`: 主模型 QK selector，20 examples x 32 queries。
  - `FusionRAG-QK(preprocess,16q)`: 主模型 QK selector，20 examples x 16 queries，用于和 draft 同源对比。
  - `Draft-selector(raw,16q)`: draft model selector，评价时仍看主模型 attention mass。
  - `random_mass`: 同样 token 数随机选取，64 次取均值。
  - `stable_frequency_mass`: 对同一 example 内所有 query 的 selector top-k 投票，得到 query-agnostic stable set。
  - `oracle_attention_mass`: 直接按评价 attention score 自己 top-k，作为上界。

## 汇总

| condition | rate | queries | selector mass | stable mass | random mass | oracle mass | selector/random |
|---|---:|---:|---:|---:|---:|---:|---:|
| Draft-selector(raw,16q) | 0.05 | 320 | 0.0483 | 0.0491 | 0.0501 | 0.3657 | 0.97 |
| Draft-selector(raw,16q) | 0.10 | 320 | 0.0957 | 0.0944 | 0.1000 | 0.4638 | 0.96 |
| Draft-selector(raw,16q) | 0.15 | 320 | 0.1491 | 0.1512 | 0.1497 | 0.5341 | 1.00 |
| Draft-selector(raw,16q) | 0.20 | 320 | 0.1953 | 0.1957 | 0.1997 | 0.5910 | 0.98 |
| Draft-selector(raw,16q) | 0.30 | 320 | 0.2914 | 0.2929 | 0.3000 | 0.6825 | 0.97 |
| Draft-selector(raw,16q) | 0.50 | 320 | 0.4958 | 0.4963 | 0.5003 | 0.8165 | 0.99 |
| FusionRAG-QK(preprocess,16q) | 0.05 | 320 | 0.3657 | 0.3576 | 0.0500 | 0.3657 | 7.33 |
| FusionRAG-QK(preprocess,16q) | 0.10 | 320 | 0.4638 | 0.4535 | 0.1002 | 0.4638 | 4.64 |
| FusionRAG-QK(preprocess,16q) | 0.15 | 320 | 0.5341 | 0.5233 | 0.1500 | 0.5341 | 3.56 |
| FusionRAG-QK(preprocess,16q) | 0.20 | 320 | 0.5910 | 0.5804 | 0.1998 | 0.5910 | 2.96 |
| FusionRAG-QK(preprocess,16q) | 0.30 | 320 | 0.6825 | 0.6728 | 0.2997 | 0.6825 | 2.28 |
| FusionRAG-QK(preprocess,16q) | 0.50 | 320 | 0.8165 | 0.8096 | 0.5000 | 0.8165 | 1.63 |
| FusionRAG-QK(preprocess,32q) | 0.05 | 640 | 0.3663 | 0.3570 | 0.0498 | 0.3663 | 7.37 |
| FusionRAG-QK(preprocess,32q) | 0.10 | 640 | 0.4632 | 0.4509 | 0.0999 | 0.4632 | 4.64 |
| FusionRAG-QK(preprocess,32q) | 0.15 | 640 | 0.5331 | 0.5198 | 0.1498 | 0.5331 | 3.56 |
| FusionRAG-QK(preprocess,32q) | 0.20 | 640 | 0.5898 | 0.5767 | 0.2001 | 0.5898 | 2.95 |
| FusionRAG-QK(preprocess,32q) | 0.30 | 640 | 0.6812 | 0.6689 | 0.3000 | 0.6812 | 2.27 |
| FusionRAG-QK(preprocess,32q) | 0.50 | 640 | 0.8155 | 0.8068 | 0.5004 | 0.8155 | 1.63 |

## 初步结论

- 如果 selector mass 明显高于 random mass，说明“attention 集中到 selected tokens”不是随机选 token 的必然结果。
- 如果 draft selector 也高于 random，但低于 FusionRAG-QK，说明不同 selection 方法都能捕获一部分高 attention token，但主模型 QK selector 更贴合主模型 attention。
- stable_frequency_mass 用于观察 query-agnostic stable set 是否也能覆盖高 attention mass。

## 输出

- `selection_attention_mass_by_query.csv`
- `selection_attention_mass_summary.json`
- `selection_attention_mass.png`
- `selection_enrichment_vs_random.png`
