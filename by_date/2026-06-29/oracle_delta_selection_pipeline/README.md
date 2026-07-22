# Oracle Delta Selection 实验记录

## 实验问题

原始 FusionRAG pipeline 是根据 query-to-document importance score 排序，然后按 rate 选 token 做 online recomputation。本实验验证一个理想条件：如果提前知道每个文档 token 在线重算前后的 K/V 变化大小，直接优先更新变化最大的 token，是否能提升回答质量。

## 实验设置

- 数据：`data/result_reflect.json`，完整 200 main questions；实际评测 135 main / 250 sub questions。
- 模型：`Qwen2.5-7B-Instruct`。
- KV：`preprocess=True`，`preprocess_scope=global`，`topk=10`，BGE recall。
- rate：`0.15`，与之前 strict FusionRAG 主实验对齐。
- 写回语义：开启 strict two-stage KV writeback，`FUSIONRAG_STRICT_REPROCESS_ABLATION=1`，`FUSIONRAG_CLEAN_STRICT_ABLATION=1`，`FUSIONRAG_REPROCESS_UPDATE_MODE=kv`。
- Oracle 打分方式：载入 preprocess KV 后，先对所有 document token 做一次全量 online recomputation，比较重算前后的 K/V delta；随后恢复 cache，再按 delta top-rate 选择 token，走正常 selected-token recomputation + query generation。
- 三种排序：`value_delta`、`key_delta`、`kv_delta`。其中 `kv_delta` 使用 K/V 合并相对 L2 变化，`value_delta` 和 `key_delta` 是消融。
- 评测限制：当前 DeepSeek/DashScope judge key 不可用或余额不足，因此本轮不使用 LLM judge 的 Main/Sub accuracy；只比较脚本本地计算的 F1/EM。

## 汇总结果

| run | rows | Avg F1 | Avg EM | dF1 vs importance | dEM vs importance |
|---|---:|---:|---:|---:|---:|
| `importance_strict_kv_rate0.15` | 250 | 0.4960 | 0.2160 | +0.0000 | +0.0000 |
| `oracle_value_delta_rate0.15` | 250 | 0.4581 | 0.1920 | -0.0378 | -0.0240 |
| `oracle_key_delta_rate0.15` | 250 | 0.4539 | 0.1720 | -0.0420 | -0.0440 |
| `oracle_kv_delta_rate0.15` | 250 | 0.4493 | 0.1720 | -0.0467 | -0.0440 |
| `true_rate0_no_doc_recompute` | 250 | 0.4510 | 0.1600 | -0.0450 | -0.0560 |

## 行级对比

| run | pred diff vs importance | F1 better | F1 worse | F1 same | mean dF1 |
|---|---:|---:|---:|---:|---:|
| `oracle_value_delta_rate0.15` | 105 | 40 | 52 | 158 | -0.0378 |
| `oracle_key_delta_rate0.15` | 100 | 43 | 47 | 160 | -0.0420 |
| `oracle_kv_delta_rate0.15` | 97 | 34 | 53 | 163 | -0.0467 |
| `true_rate0_no_doc_recompute` | 113 | 45 | 53 | 152 | -0.0450 |

## 结论

1. 直接用 K/V 变化大小做 token selection 没有提升质量，反而低于原始 query-conditioned importance selection。
2. 三个 oracle 中 `value_delta` 最好，但 Avg F1 仍只有 0.4581，低于 importance baseline 的 0.4960；`key_delta` 和 `kv_delta` 更接近 rate=0。
3. 这说明“变化大”不等价于“对当前 query 有用”。K/V delta 更像 context correction magnitude，而 FusionRAG 的 importance score 捕获的是 query relevance。
4. 因此，如果要优化 selection，不应该只预测 K/V delta；更合理的是预测 query-conditioned benefit，例如 `importance × expected_delta`、或者训练一个直接预测 answer-quality gain / recompute benefit 的 scorer。

## 输出文件

- `oracle_delta_summary.csv`：汇总 F1/EM。
- `oracle_delta_row_comparison_vs_importance.csv`：逐 sub-question 与 importance baseline 的文本/F1 差异。
- `oracle_value_rate015/`、`oracle_key_rate015/`、`oracle_kv_rate015/`：三组原始输出。
