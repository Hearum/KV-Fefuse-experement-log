# 2026-07-15 实验目录

本目录保存两个相互独立的 Qwen3-32B KV 机制实验；本日没有单独的顶层 log 文件夹，实验过程分别记录在两个实验目录自己的 `EXPERIMENT_LOG.md` 中。

## 实验列表

- `qwen3_2wiki_v2_layer_selective_recompute`：在 `2wikimqa-v2` 的 Qwen3-32B online QK `rate=0.15` pipeline 中，比较完整重算后只写回 KV gap 较大层的效果。该实验本质是 layer-selective writeback ablation，不是严格省算力重算；已完成 200 examples，结果在 `README.md` 的全量表和 `results/full_summary_with_glm.csv`，过程和启动命令在 `EXPERIMENT_LOG.md`。
- `qwen3_musique_v2_hidden_state_gap`：在 `musique-v2` 上比较 raw、preprocess 和 full 上下文的 document token hidden state gap，验证 KV 差异是否来自 hidden trajectory 的上下文漂移。已完成 5-example sanity、50-example scaling 和 200-example full scaling；统计图表与 CSV 在该目录的 `results/`、`figures/`，结论和限制在 `README.md`，运行过程在 `EXPERIMENT_LOG.md`。

每个实验目录包含 `PLAN.md`、`README.md` 和 `EXPERIMENT_LOG.md`，分别对应实验计划、中文交付结论和逐轮过程日志；复现前应先读这三个文件。
