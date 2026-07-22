# Preprocess Prefix KV Difference

目标：比较不同 top-k 相关文档块作为 preprocess 前缀时，生成的 preprocess KV 是否存在明显分布差异。

- cache_root: `/raid/home/hming/fusionrag-pca-topk-cache-5/data`
- top-k prefixes: [1, 3, 5, 10]
- common example/chunk pairs: 121
- metric: symmetric relative L2 and cosine between KV tensors.

| kind | pair | sym relative L2 | cosine |
|---|---|---:|---:|
| key | top1 vs top3 | 0.0356 | 1.0016 |
| key | top1 vs top5 | 0.0377 | 1.0016 |
| key | top1 vs top10 | 0.0405 | 1.0014 |
| key | top3 vs top5 | 0.0208 | 1.0021 |
| key | top3 vs top10 | 0.0268 | 1.0019 |
| key | top5 vs top10 | 0.0210 | 1.0021 |
| value | top1 vs top3 | 0.2721 | 0.9584 |
| value | top1 vs top5 | 0.2834 | 0.9554 |
| value | top1 vs top10 | 0.2955 | 0.9521 |
| value | top3 vs top5 | 0.1710 | 0.9835 |
| value | top3 vs top10 | 0.2073 | 0.9763 |
| value | top5 vs top10 | 0.1703 | 0.9839 |

## Figures

- `preprocess_topk_pairwise_kv_distance.png`
- `preprocess_topk_layer_distance_to_top10.png`

## 观察

1. 不同 top-k 前缀对 key 的整体影响较小。
   - top1 vs top10 的 key symmetric relative L2 为 `0.0405`。
   - key cosine 接近 1，说明整体方向基本稳定。
   - 表中 key cosine 略大于 1 是浮点数值误差，可视为约等于 1。

2. 不同 top-k 前缀对 value 的影响明显更大。
   - top1 vs top10 的 value symmetric relative L2 为 `0.2955`，cosine 为 `0.9521`。
   - top3 vs top10 降到 `0.2073`，top5 vs top10 降到 `0.1703`。
   - 说明随着 preprocess 使用更多相关文档块前缀，value cache 的分布持续发生明显变化，并逐步靠近 top10 的表示。

3. 差异主要集中在中后层。
   - top1 vs top10 的 key 差异最大层：layer 16, 17, 20, 22, 21, 18。
   - top1 vs top10 的 value 差异最大层：layer 18, 15, 17, 20, 16, 14。
   - 这和 online recompute KV delta 的观察一致：value 和中后层是最敏感的部分。

4. 不同 preprocess 前缀对 value 的影响趋势基本一致。
   - 将每个 top-k pair 的 value layer-wise symmetric relative L2 曲线两两计算相关系数，结果大多在 `0.98` 以上，最低也有 `0.9584`。
   - top1 vs top3、top1 vs top5、top1 vs top10 三条曲线的相关系数分别为 `0.9992`、`0.9948`、`0.9979`，说明只要以 top1 为一端，改变另一端的 top-k 主要放大/缩小差异幅度，并没有改变主要受影响的层。
   - value 差异峰值层高度重合：layer 18 始终是最大差异层，layer 15/17/20/16/19/14 也反复出现在各个 pair 的 top layers 中。
   - 位置 decile 上也呈相同趋势：越靠前的 token 差异越大，随后逐步下降，最后一个 decile 略有回升。相对于 top10，top1/top3/top5 的 value relative L2 都呈现这个形状，只是幅度依次降低。

## 含义

这个实验说明：preprocess KV 并不是只受当前 chunk 自身决定，不同的 top-k 相关文档块前缀会显著改变最终 KV，尤其是 value cache。  
因此如果要离线缓存 corrected KV，需要明确缓存对应的是哪一种 prefix context；否则同一个 chunk 在不同前缀组合下的 value 表示会存在明显分布差异。

进一步看，value 的变化不是随机噪声，而是有稳定的 layer/position 结构：不同 preprocess 设置改变的是扰动强度，而不是扰动发生的位置。因此如果后续做 correction model，可以优先针对中后层 value，以及靠前 token 的 value 表示建模。

一个可能的后续方向是把 prefix context 的信息显式纳入 correction model，例如：

- 使用 top-k prefix 的 pooled representation 作为 correction 条件；
- 按 top-k / prefix cluster 存多版本 corrected KV；
- 或者训练 `KV_before + f(prefix_summary, token_state)` 来预测更接近 online recompute / full-context 的 KV。
