# Fixed Top-k Prefix Swap

目标：固定 preprocess prefix 数量为 top-5，但替换 prefix 文档集合，观察目标 chunk 的 KV 是否仍沿相同方向变化。

## 实验设置

- model: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct`
- raw cache: `/raid/home/hming/fusionrag-pca-topk-cache-5/data/musique-pca-subset/Qwen2.5-7B-Instruct`
- baseline retrieved top5 cache: `/raid/home/hming/fusionrag-pca-topk-cache-5/data/musique-pca-subset-preprocess-5-revert_rope-True/Qwen2.5-7B-Instruct`
- examples: [1, 2]
- target chunks: [6, 8]
- generated variants: same_example_early, same_example_late, cross_next_early, cross_prev_late, mixed_same_cross
- metric: 对同一个 target chunk，比较不同 prefix 生成的 KV；重点看 `(KV_variant - raw_KV)` 之间的 cosine。

## Prefix Variants

| example | target chunk | variant | prefix docs |
|---:|---:|---|---|
| 1 | 6 | same_example_early | [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)] |
| 1 | 6 | same_example_late | [(1, 20), (1, 21), (1, 22), (1, 23), (1, 24)] |
| 1 | 6 | cross_next_early | [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5)] |
| 1 | 6 | cross_prev_late | [(5, 22), (5, 23), (5, 24), (5, 25), (5, 26)] |
| 1 | 6 | mixed_same_cross | [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3)] |
| 1 | 8 | same_example_early | [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)] |
| 1 | 8 | same_example_late | [(1, 20), (1, 21), (1, 22), (1, 23), (1, 24)] |
| 1 | 8 | cross_next_early | [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5)] |
| 1 | 8 | cross_prev_late | [(5, 22), (5, 23), (5, 24), (5, 25), (5, 26)] |
| 1 | 8 | mixed_same_cross | [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3)] |
| 2 | 6 | same_example_early | [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5)] |
| 2 | 6 | same_example_late | [(2, 15), (2, 16), (2, 17), (2, 18), (2, 19)] |
| 2 | 6 | cross_next_early | [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5)] |
| 2 | 6 | cross_prev_late | [(1, 20), (1, 21), (1, 22), (1, 23), (1, 24)] |
| 2 | 6 | mixed_same_cross | [(2, 1), (2, 2), (3, 1), (3, 2), (3, 3)] |
| 2 | 8 | same_example_early | [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5)] |
| 2 | 8 | same_example_late | [(2, 15), (2, 16), (2, 17), (2, 18), (2, 19)] |
| 2 | 8 | cross_next_early | [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5)] |
| 2 | 8 | cross_prev_late | [(1, 20), (1, 21), (1, 22), (1, 23), (1, 24)] |
| 2 | 8 | mixed_same_cross | [(2, 1), (2, 2), (3, 1), (3, 2), (3, 3)] |

## Summary

| kind | variant A | variant B | KV cosine mean | rel L2 mean | delta-from-raw cosine mean |
|---|---|---|---:|---:|---:|
| key | cross_next_early | cross_prev_late | 1.0015 | 0.0320 | 0.4499 |
| key | cross_next_early | mixed_same_cross | 1.0017 | 0.0256 | 0.7396 |
| key | cross_prev_late | mixed_same_cross | 1.0014 | 0.0348 | 0.4542 |
| key | retrieved_top5 | cross_next_early | 1.0012 | 0.0398 | 0.4274 |
| key | retrieved_top5 | cross_prev_late | 1.0012 | 0.0397 | 0.3821 |
| key | retrieved_top5 | mixed_same_cross | 1.0015 | 0.0314 | 0.6727 |
| key | retrieved_top5 | same_example_early | 1.0017 | 0.0245 | 0.8017 |
| key | retrieved_top5 | same_example_late | 1.0014 | 0.0356 | 0.5447 |
| key | same_example_early | cross_next_early | 1.0013 | 0.0387 | 0.4468 |
| key | same_example_early | cross_prev_late | 1.0013 | 0.0376 | 0.4292 |
| key | same_example_early | mixed_same_cross | 1.0016 | 0.0279 | 0.7351 |
| key | same_example_early | same_example_late | 1.0015 | 0.0314 | 0.6343 |
| key | same_example_late | cross_next_early | 1.0015 | 0.0324 | 0.4979 |
| key | same_example_late | cross_prev_late | 1.0016 | 0.0285 | 0.5240 |
| key | same_example_late | mixed_same_cross | 1.0015 | 0.0317 | 0.5820 |
| value | cross_next_early | cross_prev_late | 0.9767 | 0.2142 | 0.5178 |
| value | cross_next_early | mixed_same_cross | 0.9772 | 0.2123 | 0.6569 |
| value | cross_prev_late | mixed_same_cross | 0.9676 | 0.2522 | 0.4754 |
| value | retrieved_top5 | cross_next_early | 0.9542 | 0.2982 | 0.4080 |
| value | retrieved_top5 | cross_prev_late | 0.9555 | 0.2955 | 0.3885 |
| value | retrieved_top5 | mixed_same_cross | 0.9688 | 0.2440 | 0.6499 |
| value | retrieved_top5 | same_example_early | 0.9781 | 0.2046 | 0.7588 |
| value | retrieved_top5 | same_example_late | 0.9600 | 0.2797 | 0.4945 |
| value | same_example_early | cross_next_early | 0.9595 | 0.2805 | 0.4428 |
| value | same_example_early | cross_prev_late | 0.9612 | 0.2755 | 0.4323 |
| value | same_example_early | mixed_same_cross | 0.9772 | 0.2090 | 0.7286 |
| value | same_example_early | same_example_late | 0.9675 | 0.2518 | 0.5641 |
| value | same_example_late | cross_next_early | 0.9741 | 0.2265 | 0.5242 |
| value | same_example_late | cross_prev_late | 0.9787 | 0.2041 | 0.5697 |
| value | same_example_late | mixed_same_cross | 0.9703 | 0.2430 | 0.5534 |

## 初步解释

- 如果 `delta-from-raw cosine` 高，说明不同 prefix 文档虽然内容不同，但都把 raw KV 往相近方向推。
- 如果该指标低甚至为负，说明固定 top-k 数量不够，prefix 内容会改变 value KV 的偏移方向。
- 这组实验和 top-k 递增实验互补：前者看数量增加，后者看同数量下的内容敏感性。

## 当前结论

1. 固定 top-5 数量时，更换 prefix 文档会明显改变 value 的偏移方向。
   - 原始 retrieved top5 vs same_example_early 的 value `delta-from-raw cosine` 为 `0.7588`，方向相对接近。
   - 原始 retrieved top5 vs cross_next_early 降到 `0.4080`，vs cross_prev_late 为 `0.3885`。
   - 说明 top-k 数量相同并不保证 preprocess 作用方向相同；prefix 内容本身很重要。

2. 同 example 内部的 prefix 比跨 example prefix 更接近。
   - retrieved_top5 与 same_example_early/same_example_late 的方向相似度高于多数 cross-example 对比。
   - mixed_same_cross 介于两者之间，和 retrieved_top5 的 value `delta-from-raw cosine` 为 `0.6499`。
   - 这说明 preprocess KV 的偏移方向既包含一个较稳定的“上下文化”分量，也包含明显的文档语义/分布相关分量。

3. value 比 key 更敏感。
   - key 的 KV cosine 几乎都接近 1，relative L2 也较小。
   - value 的 KV cosine 约在 `0.95-0.98`，relative L2 可到 `0.29` 左右，且 delta 方向随 prefix 内容变化更明显。

## Figures

- `fixed_top5_prefix_swap_value_delta_cosine.png`
