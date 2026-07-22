# Layer4 Selector Gap to Full DraftModel

目标：在同一套 Qwen3/MuSiQue reflect 数据上，用 full DraftModel selector 作为 teacher，比较原生 layer4 selector 与 WikiText 蒸馏 4-layer selector 的选 token 差距。

统计口径：`final` 是 offline 10% + residual 5% 的最终 15% selected set；`residual` 是从 final set 中扣掉共同 offline fixed 10% 后的在线补选 5%。`teacher recall` 表示候选方法覆盖了 full DraftModel 选中 token 的比例。

| method | items | final Jaccard | final teacher recall | residual Jaccard | residual teacher recall | selection all(s) | selection steady(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| native_layer4 | 250 | 0.6591 | 0.7912 | 0.3522 | 0.5157 | 0.0235 | 0.0200 |
| distilled_layer4 | 250 | 0.6561 | 0.7907 | 0.3411 | 0.5060 | 0.0728 | 0.0046 |
