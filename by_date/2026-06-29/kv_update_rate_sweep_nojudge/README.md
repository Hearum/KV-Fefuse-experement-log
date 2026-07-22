# K/V 写回模式 Rate Sweep（无 LLM judge）

## 实验设置

- 数据：data/result_reflect.json 完整 200 main questions，实际输出 135 main / 250 sub questions。
- 模型：Qwen2.5-7B-Instruct。
- KV：preprocess=True，preprocess_scope=global，topk=10，BGE recall。
- FusionRAG selection：原始 query-conditioned importance selection。
- 写回语义：strict two-stage ablation，先重算 selected document tokens，再按模式保留 K/V，然后单独 forward query。
- 模式：kv 同时更新 K/V；v_only 只保留 V 更新；k_only 只保留 K 更新。
- rate：0.0, 0.15, 0.3, 0.5, 0.8, 1.0。rate=0 和 rate=1 与写回模式无关，因此三列共享同一结果。
- 本轮设置 FUSIONRAG_SKIP_LLM_JUDGE=1，不看 LLM judge accuracy，只看本地 F1/EM。

## Avg F1

| rate | kv | v_only | k_only |
|---:|---:|---:|---:|
| 0.0 | 0.4510 | 0.4510 | 0.4510 |
| 0.15 | 0.4960 | 0.4498 | 0.4687 |
| 0.3 | 0.5122 | 0.4425 | 0.4668 |
| 0.5 | 0.5295 | 0.4465 | 0.4748 |
| 0.8 | 0.5380 | 0.4542 | 0.4879 |
| 1.0 | 0.5666 | 0.5666 | 0.5666 |

## Avg EM

| rate | kv | v_only | k_only |
|---:|---:|---:|---:|
| 0.0 | 0.1600 | 0.1600 | 0.1600 |
| 0.15 | 0.2160 | 0.1880 | 0.1880 |
| 0.3 | 0.1920 | 0.1800 | 0.1920 |
| 0.5 | 0.2040 | 0.1840 | 0.1840 |
| 0.8 | 0.2080 | 0.1840 | 0.1920 |
| 1.0 | 0.2280 | 0.2280 | 0.2280 |

## 初步观察

- kv 随 rate 增加整体提升，并在 rate=1.0 达到最高 F1/EM，说明完整 K/V online recomputation 是有效的。
- v_only 在 rate=0.15 到 0.3 下降，随后随 rate 增加恢复；到 rate=0.8 F1 接近 kv@0.8，但 EM 仍低。
- k_only 在 rate=0.3/0.5 的 F1 接近或略高于 v_only，但到 rate=0.8 明显低于 v_only 和 kv，说明只更新 K 的收益不稳定。
- 这组结果和之前 K/V delta 观察并不矛盾：Value 的数值变化更大，但只更新 Value 不能稳定替代完整 K/V；Key 的数值变化小，仍会影响 attention routing，因此 K/V 最好一起保留。

## 输出文件

- kv_update_rate_sweep_summary.csv：长表。
- kv_update_rate_sweep_pivot_f1.csv：F1 透视表。
- kv_update_rate_sweep_pivot_em.csv：EM 透视表。
- 各 run 原始 CSV 位于对应 mode_rate 子目录。
