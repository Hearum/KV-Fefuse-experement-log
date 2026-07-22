# 实验日志

## 2026-07-15 计划建立

新增 layer-selective writeback ablation 接口：

- `FUSIONRAG_REPROCESS_KEEP_KEY_LAYERS`
- `FUSIONRAG_REPROCESS_KEEP_VALUE_LAYERS`

默认不设置时行为不变。设置后，重算仍完整执行，但只保留指定层的 K/V 写回，其他层恢复为 cache 中原值。

本实验先在 `2wikimqa-v2` 上跑 online QK `rate=0.15` 的层写回消融。


## 2026-07-15 全量实验完成

运行完成 6 个条件，每个条件 200 examples，均为 `2wikimqa-v2` / Qwen3-32B / online QK / `rate=0.15`。

启动方式：

```bash
MOTIVATION_EXPERIMENTS/by_date/2026-07-15/qwen3_2wiki_v2_layer_selective_recompute/scripts/run_layer_selective_task.sh <condition> 0 200 <gpu> 0.15
```

GLM judge 已拼入 summarizer：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-15/qwen3_2wiki_v2_layer_selective_recompute/scripts/summarize_layer_selective.py \
  --with-glm \
  --out MOTIVATION_EXPERIMENTS/by_date/2026-07-15/qwen3_2wiki_v2_layer_selective_recompute/results/full_summary_with_glm.csv
```

统一结果：

| condition | EM | F1 | GLM Acc |
|---|---:|---:|---:|
| all_layers | 42.00 | 53.35 | 52.50 |
| v_late_59_63 | 40.00 | 51.81 | 50.00 |
| v_late_56_63 | 39.00 | 51.46 | 51.00 |
| k_mid_45_52 | 39.00 | 51.80 | 50.00 |
| kv_gap_core | 39.00 | 52.23 | 51.00 |
| kv_gap_wide | 39.00 | 51.82 | 51.50 |

结论：layer-selective 写回保留了大部分性能，但未达到全层写回。`kv_gap_wide` 最接近全层，GLM 只低 1 点，但 EM 仍低 3 点。gap energy 大的层和性能关键层相关但不等价。

重要澄清：这不是严格的省算力实验。当前实现仍完整重算所有层，只在写回 cache 时按层选择保留或恢复旧 KV。因此它只能作为性能敏感层分析，不能证明 layer-selective recompute 已经节省计算量。后续如果讨论 Adapter 或省算力，需要新开 strict compute-saving / adapter predictor 实验。
