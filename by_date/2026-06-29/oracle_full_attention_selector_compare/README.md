# Oracle Full Attention Selector Compare

## Selector Coverage Of Full-Attention Ideal Distribution

| rate | selector | ideal mass | Jaccard vs oracle | score cosine | score JS | selector time |
|---:|---|---:|---:|---:|---:|---:|
| 0.05 | oracle_full_attention | 0.2377 | 1.0000 | 1.0000 | 0.0000 | 0.3032s |
| 0.05 | draft_selected_set | 0.0852 | 0.0931 | 0.2479 | 0.5630 | 0.1381s |
| 0.05 | fusionrag_qk | 0.0644 | 0.0314 | 0.3151 | 0.2205 | 0.0286s |
| 0.05 | random | 0.0496 | 0.0245 | 0.6510 | 0.0705 | 0.0000s |
| 0.15 | oracle_full_attention | 0.4306 | 1.0000 | 1.0000 | 0.0000 | 0.3026s |
| 0.15 | draft_selected_set | 0.2187 | 0.1626 | 0.3667 | 0.4454 | 0.1383s |
| 0.15 | fusionrag_qk | 0.1864 | 0.1172 | 0.3151 | 0.2205 | 0.0267s |
| 0.15 | random | 0.1501 | 0.0806 | 0.6510 | 0.0705 | 0.0000s |
| 0.30 | oracle_full_attention | 0.6002 | 1.0000 | 1.0000 | 0.0000 | 0.3028s |
| 0.30 | draft_selected_set | 0.3907 | 0.2595 | 0.4636 | 0.3335 | 0.1386s |
| 0.30 | fusionrag_qk | 0.3491 | 0.2348 | 0.3151 | 0.2205 | 0.0269s |
| 0.30 | random | 0.2999 | 0.1744 | 0.6510 | 0.0705 | 0.0000s |
| 0.50 | oracle_full_attention | 0.7566 | 1.0000 | 1.0000 | 0.0000 | 0.3031s |
| 0.50 | draft_selected_set | 0.5950 | 0.3956 | 0.5473 | 0.2299 | 0.1385s |
| 0.50 | fusionrag_qk | 0.5565 | 0.3981 | 0.3151 | 0.2205 | 0.0270s |
| 0.50 | random | 0.4997 | 0.3341 | 0.6510 | 0.0705 | 0.0000s |
| 0.80 | oracle_full_attention | 0.9241 | 1.0000 | 1.0000 | 0.0000 | 0.3020s |
| 0.80 | draft_selected_set | 0.8513 | 0.6877 | 0.6196 | 0.1233 | 0.1388s |
| 0.80 | fusionrag_qk | 0.8456 | 0.7097 | 0.3151 | 0.2205 | 0.0286s |
| 0.80 | random | 0.7991 | 0.6667 | 0.6510 | 0.0705 | 0.0000s |

## Generation Metrics

| rate | fusionrag_qk F1 | oracle F1 | delta F1 | fusionrag_qk EM | oracle EM | delta EM |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.4421 | 0.4189 | -0.0232 | 0.1927 | 0.1743 | -0.0183 |
| 0.15 | 0.4764 | 0.4394 | -0.0369 | 0.2110 | 0.1927 | -0.0183 |
| 0.30 | 0.5170 | 0.4747 | -0.0423 | 0.2385 | 0.2202 | -0.0183 |
| 0.50 | 0.5386 | 0.4892 | -0.0493 | 0.2294 | 0.2110 | -0.0183 |
| 0.80 | 0.5183 | 0.5739 | +0.0556 | 0.2294 | 0.2661 | +0.0367 |

## Files

- selector summary: MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare/selector_summary_all_rates.csv
- answer summary: MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare/answer_summary_all_rates.csv
