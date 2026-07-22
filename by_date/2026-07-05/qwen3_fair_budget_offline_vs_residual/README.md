# Fair Budget: Offline vs Residual

## 目的

修正 `offline 15% + residual 5%` 和 `offline 15%` 直接比较不公平的问题。这里补两组公平预算对照：

- total 15% budget: `offline 10% + online Draft residual 5%` vs `offline 15% only`。
- total 20% budget: `offline 15% + online Draft residual 5%` vs `offline 20% only`。

所有 offline fixed set 都使用同一个 `draft_smart_frequency_global` 方法重新生成，避免混用旧 hybrid rate sweep。

## 新跑结果

| method | budget meaning | Main Acc | Sub Acc | F1 | EM | avg selected tokens | residual select time | prompt eval time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| offline10_draft005 | offline 10% + Draft residual 5% | 97/135 (71.85%) | 208/250 (83.20%) | 0.1955 | 0.0200 | 249.5 | 0.1406 | 1.0007 |
| offline20_only | offline 20% only | 91/135 (67.41%) | 198/250 (79.20%) | 0.2069 | 0.0120 | 325.0 | 0.0000 | 0.9801 |

## 对照基线（来自主表）

| method | budget meaning | Main Acc | Sub Acc | note |
|---|---|---:|---:|---|
| draft_smart_frequency_global | offline 15% only | 94/135 (69.63%) | 203/250 (81.20%) | fair total-15 offline baseline |
| freq_draft005 | offline 15% + Draft residual 5% | 103/135 (76.30%) | 217/250 (86.80%) | unfair vs offline15, fair vs offline20 |

## 结论

- 你指出的问题成立：`offline15 + residual5` 不能直接说明 residual 方法本身优于 `offline15`，因为它多用了 token budget。
- 在 total 15% 预算下，应该看 `offline10 + residual5` 是否优于 `offline15 only`。
- 在 total 20% 预算下，应该看 `offline15 + residual5` 是否优于 `offline20 only`。
- 后续所有 residual 实验都必须同时报告 matched-budget offline baseline。
