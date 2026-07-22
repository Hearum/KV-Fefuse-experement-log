# Draft Smart Global Offline Fixed Sets

目的：避免和旧 `draft_frequency_per_chunk` 混淆，新增 `draft_smart_*` 命名。

输入：`reflect_draft_rate015_full/combined/score_cache_npz` 中保存的 draft model query-to-doc scores。

核心差异：每个 calibration query 先使用 online DraftModel 相同的 smart selection 后处理，再将多个 calibration query 的选择集合聚合成 offline fixed set。

方法：

- `draft_smart_frequency_global`: 对 smart-selected token 统计频率，频率优先，mean score 打破平局。
- `draft_smart_mean_score_global`: 使用 calibration query 的 mean draft score 排序。
- `draft_smart_max_score_global`: 使用 calibration query 的 max draft score 排序。
- `draft_smart_top2_mean_global`: 使用每个 token top-2 calibration score 的均值排序。

注意：这些方法是 global doc-level selection 后再拆成 chunk-local npz；不是旧的 per-chunk top-k。
