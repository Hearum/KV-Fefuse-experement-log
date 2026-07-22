# Selector Speed Comparison

这里的速度是 selector 选 token 的平均耗时，不包含后续 online recompute 和生成。`oracle_full_attention` 需要先跑一次 full-context forward 来得到 query->doc 理想 attention 分布，因此它不是可部署的在线 selector，只作为上界参考。

## Selector Time

| rate | fusionrag_qk s | draft s | oracle_full_attention s | draft / qk | oracle / qk |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.0286 | 0.1381 | 0.3032 | 4.83x | 10.61x |
| 0.15 | 0.0267 | 0.1383 | 0.3026 | 5.17x | 11.32x |
| 0.30 | 0.0269 | 0.1386 | 0.3028 | 5.15x | 11.26x |
| 0.50 | 0.0270 | 0.1385 | 0.3031 | 5.12x | 11.21x |
| 0.80 | 0.0286 | 0.1388 | 0.3020 | 4.86x | 10.57x |

## Speed And Accuracy

| rate | fusionrag_qk selector s | fusionrag_qk F1 | draft selector s | draft F1 | oracle selector s | oracle F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.0286 | 0.4421 | 0.1381 | 0.5199 | 0.3032 | 0.4189 |
| 0.15 | 0.0267 | 0.4764 | 0.1383 | 0.5383 | 0.3026 | 0.4394 |
| 0.30 | 0.0269 | 0.5170 | 0.1386 | 0.5619 | 0.3028 | 0.4747 |
| 0.50 | 0.0270 | 0.5386 | 0.1385 | 0.5799 | 0.3031 | 0.4892 |
| 0.80 | 0.0286 | 0.5183 | 0.1388 | 0.5718 | 0.3020 | 0.5739 |

## GLM Accuracy

| rate | fusionrag_qk selector s | fusionrag_qk acc | draft selector s | draft acc | oracle selector s | oracle acc |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.0286 | 0.7064 | 0.1381 | 0.7890 | 0.3032 | 0.7064 |
| 0.15 | 0.0267 | 0.7431 | 0.1383 | 0.8165 | 0.3026 | 0.6606 |
| 0.30 | 0.0269 | 0.7982 | 0.1386 | 0.8532 | 0.3028 | 0.7248 |
| 0.50 | 0.0270 | 0.8624 | 0.1385 | 0.8624 | 0.3031 | 0.7706 |
| 0.80 | 0.0286 | 0.7890 | 0.1388 | 0.8532 | 0.3020 | 0.8257 |

## Takeaway

```text
fusionrag_qk: fastest selector, about 0.027-0.029s.
draft: about 0.138s, roughly 5x slower than fusionrag_qk, but accuracy is usually higher.
oracle_full_attention: about 0.302-0.303s, roughly 11x slower than fusionrag_qk, and requires full attention, so it is an analysis upper-bound rather than an online method.
```
