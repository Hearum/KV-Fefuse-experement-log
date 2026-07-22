# FusionRAG Online Recompute KV Delta

对比对象：同一个 `past_key_values` 在 FusionRAG online recompute selected tokens 前后的 KV 差异。

- examples: 1-3
- max_passages: 10
- rate: 0.1
- preprocess cache: top-10

| scope | relative L2 | cosine | token delta mean | token delta p90 |
|---|---:|---:|---:|---:|
| key selected | 0.197211 | 0.983105 | 0.802784 | 1.056432 |
| value selected | 0.715358 | 0.753186 | 0.791670 | 1.276149 |
| key unselected | 0.000000 | 1.017989 | 0.000000 | 0.000000 |
| value unselected | 0.000000 | 1.005598 | 0.000000 | 0.000000 |

图：`online_recompute_selected_layer_delta.png`。

## 现象解释

这次比较的是 FusionRAG 真正 online 阶段的 KV 更新，不是 preprocess KV 和 full-context KV 的静态差距。

流程是：

1. 先把 preprocess top-10 KV cache 加载到 `past_key_values`。
2. 用 query 触发 FusionRAG importance，选出 rate=0.1 的 document tokens。
3. 保存 online recompute 之前的 `past_key_values`。
4. 对 `selected document tokens + query tokens` 重新 forward，并写回同一个 `past_key_values`。
5. 比较 recompute 前后的 KV。

结论：

- online recompute 只更新 selected tokens，unselected document tokens 的 delta 为 0。
- selected token 的 value 变化远大于 key：
  - key selected relative L2: `0.1972`
  - value selected relative L2: `0.7154`
- 这说明 FusionRAG online recompute 的主要作用不是轻微修正 key，而是显著改写 selected tokens 的 value 表示。
- value selected 的 cosine 只有 `0.7532`，说明更新前后的 value 方向变化也很大，不只是 norm scale 改变。
- layer 曲线显示 selected KV 的变化在大多数中间层都明显，value 在 layer 9、12、15-18 附近尤其大。

这对系统优化的含义是：如果想省掉 online recompute，不能只提前保存 selected token index，也不能只近似 key/attention score；必须想办法提前得到或预测 selected tokens recompute 后的 value cache。否则后续 generation 读出的内容表示会明显不同。

注意：unselected 的 cosine 因为 tensor 完全不变，数值上接近 1；表中略大于 1 是浮点误差，不影响“unselected delta=0”的判断。
