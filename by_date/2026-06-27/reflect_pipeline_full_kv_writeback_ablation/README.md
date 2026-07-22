# 完整样本 FusionRAG online KV 写回消融实验

> 重要更新：复查代码后确认，最初的 `v_only` / `none` 实现不是严格的 K/V 写回消融。它们在 selected document token 和 query token 一次性 forward 之后才恢复 cache，因此 first-token logits 和 query-side KV 已经被 online recomputation 污染。下面保留原始结果用于溯源，但不能据此得出“只更新 V 可行”或“不更新 K/V 也可行”的结论。

## 实验目的
验证 FusionRAG online recomputation 之后，选中 token 的 K/V 写回是否真的影响端到端 QA 质量。这里比较三种模式：

- `kv`：正常 FusionRAG，online 重算后同时写回选中 token 的 K 和 V。
- `v_only`：online 重算后只写回 V，选中 token 的 K 恢复为重算前状态。
- `none`：online 重算后选中 token 的 K 和 V 都恢复为重算前状态，相当于不把重算结果写回文档 KV cache。

## 实验配置

- 数据：`./data/result_reflect.json` 完整 200 个 main question，其中 `llm_judge=False` 或含问题答案的样本跳过，实际评测 135 个 main question / 250 个 sub question。
- 模型：`Qwen2.5-7B-Instruct`。
- 检索与 preprocess：`preprocess=True`，`preprocess_scope=global`，`topk=10`，BGE-M3 + FAISS 全局文档相似检索。
- FusionRAG 设置：`rate=0.15`，`reprocess_method=FusionRAG`，`revert_rope=True`，`use_entropy_selection=False`。
- 评测：DeepSeek judge，`deepseek-v3.2`。API key 已在运行命令中隐藏。
- 注意：这是完整 200 条数据的 global top-k 版本，和之前 20-sample sanity run 的检索 corpus 不同，因此准确率不能直接横向对齐。

启动命令模板：

```bash
FUSIONRAG_REPROCESS_UPDATE_MODE=<kv|v_only|none> CUDA_VISIBLE_DEVICES=6 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python test_fusionrag_reflect_preprocess_exp.py --model_type qwen --model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct --model_name Qwen2.5-7B-Instruct --data_path ./data/result_reflect.json --dataset_name musique --cache_path /raid/home/hming/fusionrag-reflect-full-cache/ --result_path <mode_result_path> --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3 --rate 0.15 --topk 10 --preprocess True --recall_method bge --random_seed 42 --fixed_doc_idx 0 --reprocess_method FusionRAG --revert_rope True --use_multi_gpu False --preprocess_scope global --openai_base_url https://dashscope.aliyuncs.com/compatible-mode/v1 --openai_api_key <redacted> --openai_model deepseek-v3.2 --use_entropy_selection False --entropy_top_k 4 --draft_layer_selection entropy --vattention_topk_ratio 0.5 --epsilon 0.1 --delta 0.05 --min_rate 0.05 --max_rate 0.5 --long_decode False --long_decode_max_tokens 1000
```

## 总体结果

| mode | Main Acc | Sub Acc | Avg F1 | Avg EM | 预测文本不同于 kv | judge 正误不同于 kv | 相对 kv 变好 | 相对 kv 变差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `kv` | 85/135 (62.96%) | 189/250 (75.60%) | 0.4878 | 0.2120 | 0 | 0 | 0 | 0 |
| `v_only` | 83/135 (61.48%) | 186/250 (74.40%) | 0.4773 | 0.2080 | 48 | 9 | 3 | 6 |
| `none` | 85/135 (62.96%) | 190/250 (76.00%) | 0.4921 | 0.2120 | 59 | 15 | 8 | 7 |

## 原始结果的解释范围

- 完整样本上，`none` 的总体指标没有下降，甚至略高于 `kv`。这个现象不能解释成“不更新 K/V 更好”，因为 `none` 仍然执行了 selected document token 的 online forward，并且 first-token logits 已经在重算后的 KV 条件下产生。
- `v_only` 相比 `kv` 略低，但它同样不是严格“只更新 V”的实验，因为 first-token logits 和 query token KV 已经受重算路径影响。
- 三组结果的预测文本有明显差异，说明当前 restore 时机确实改变了后续生成轨迹；但这个差异混合了 selected document recompute、query prefill、first-token logits 和后续 decode 的影响。
- 因此这组三模式结果只能作为错误实现/污染实现的溯源记录，不能作为论文或 motivation 的核心证据。

## 正误变化样例

- row 16, `v_only`: `Who is the lead singer of the band Art Brut?`
  - kv: True | Eddie Argos
  - v_only: False | Eddie Bradley
- row 42, `v_only`: `Who are the cast members of King of the Mountain?`
  - kv: False | Harry Hamlin as Steve
  - v_only: True | Harry Hamlin as Steve Deborah Van Valkenburgh as Tina
- row 56, `v_only`: `Who founded the Janatha Vimukthi Peramuna political party?`
  - kv: True | Patabendi Don Jinadasa Nandasiri Wijeweera, known as Rohana Wijeweera, founded the Janatha Vimukthi Peramuna.
  - v_only: False | Patabendi Don Jinadasa Nandasiri Wijeweera
- row 123, `v_only`: `Who was the first nominated African American presidential candidate?`
  - kv: True | Frederick Douglass was invited to speak at the Republican National Convention in 1888 and received one vote, making him the first nominated African American presidential candidate,
  - v_only: False | Frederick Douglass was invited to speak at the Republican National Convention in 1888, though he did not formally run. The first African American to win the nomination and become p
- row 168, `v_only`: `What cities are located in Ninghai County, Zhejiang province?`
  - kv: False | Qiantong, Qiantong, Zhejiang and Qiantong, Zhejiang
  - v_only: True | Qiantong, Ninghai County, Zhejiang province
- row 185, `v_only`: `Who owned Empire Sports Network?`
  - kv: True | 21st Century Fox (through Adelphia Communications Corporation)
  - v_only: False | 21st Century Fox
- row 194, `v_only`: `In which district is the municipality of Vila Franca de Xira located?`
  - kv: False | Vila Franca de Xira is located in the Inhambane Province.   However, based on the provided documents, the correct answer should be: Answer: Vila Franca de Xira is located in the Li
  - v_only: True | Vila Franca de Xira is located in the municipality of Vila Franca de Xira, Portugal. However, it is situated in the Lisbon District.
- row 241, `v_only`: `What is Labo M?`
  - kv: True | Labo M is the third studio album by French singer-songwriter Matthieu Chedid in his persona as -M-, released in 2003.
  - v_only: False | Labo M is the third studio album by Swedish recording artist Nina Sky, released in 2003.
- row 243, `v_only`: `Who is the producer of Julius Caesar?`
  - kv: True | 无法从给定信息中确定朱利叶斯·凯撒的 producers，提供的文档没有提及该电影的制作或制作人员信息。
  - v_only: False | 无法从给定信息中确定朱利叶斯·凯撒的 producers，提供的文档主要讨论了 Julius Caesar 的戏剧和歌剧版本，以及相关的文学和艺术作品，但没有提到电影或戏剧的制作人信息。
- row 42, `none`: `Who are the cast members of King of the Mountain?`
  - kv: False | Harry Hamlin as Steve
  - none: True | Harry Hamlin as Steve Deborah Van Valkenburgh as Tina
- row 61, `none`: `Where is Bancroft located?`
  - kv: False | Bancroft is located in Ontario, Canada. Specifically, it is in the West Kootenay region and is the location of North Hastings High School, which is described as being just north of
  - none: True | Bancroft is located in Ontario, Canada. Specifically, it is in the West Kootenay region and is the location of North Hastings High School, which is described as being on the west s
- row 63, `none`: `Which counties border Hastings County in Ontario?`
  - kv: True | Renfrew County, Haliburton County
  - none: False | Renfrew County, Hastings County, and Haliburton County

## 输出文件

- `kv` 原始结果：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_full_kv/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge`
- `v_only` 原始结果：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_full_v_only/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge`
- `none` 原始结果：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_full_none/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge`
- 汇总表：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_full_kv_writeback_ablation/summary.csv`
- 行级差异：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_full_kv_writeback_ablation/row_diffs_vs_kv.csv`

## 追加实验：真正 rate=0 对照

这里补跑了一个真正 `rate=0` baseline：不选择任何文档 token 做 online recomputation，只对 query 部分进行 prefill/generate。这个设置和前面的 `none` 不同；`none` 仍然执行了 selected document token 的 online forward，只是在 forward 后恢复了文档 K/V。

| mode | Main Acc | Sub Acc | Avg F1 | Avg EM | 预测文本不同于 kv@0.15 | judge 正误不同于 kv@0.15 | 相对 kv@0.15 变好 | 相对 kv@0.15 变差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `kv_rate0.15` | 85/135 (62.96%) | 189/250 (75.60%) | 0.4878 | 0.2120 | 0 | 0 | 0 | 0 |
| `v_only_rate0.15` | 83/135 (61.48%) | 186/250 (74.40%) | 0.4773 | 0.2080 | 48 | 9 | 3 | 6 |
| `none_restore_after_recompute_rate0.15` | 85/135 (62.96%) | 190/250 (76.00%) | 0.4921 | 0.2120 | 59 | 15 | 8 | 7 |
| `true_rate0_no_doc_recompute` | 77/135 (57.04%) | 178/250 (71.20%) | 0.4510 | 0.1600 | 113 | 47 | 18 | 29 |

结论更新：真正不重算文档 token 的 `rate=0` 明显下降，Sub Acc 从 `kv@0.15` 的 75.60% 降到 71.20%，Avg F1 从 0.4878 降到 0.4510。这说明 online recomputation 本身是有用的。前面 `none` 结果不能解释成“不重算更好”，只能说明“在当前实现里，重算 forward 之后是否把文档 K/V 持久写回，对最终质量没有显示出稳定收益”。

- `rate=0` 原始结果：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_full_rate0/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge`
- 合并汇总表：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_full_kv_writeback_ablation/summary_with_rate0.csv`


## 代码复查：`v_only` / `none` 不是严格消融

复查 `ktransformers/util/utils.py::load_kv_and_generate` 后确认，前面 `v_only` 和 `none` 两组不能作为严格的 K/V 更新消融来解释。

原因如下：

- `k_need_index` 由 selected document tokens 和 query tokens 拼在一起，一次性送入 `model(...)`。
- Qwen attention 内部会先调用 `past_key_value.update(...)`，把本次 forward 的 K/V 写入 cache，然后再做 attention。
- 当前补丁是在 `logits = model(...)` 之后才 restore selected document token 的 K/V。
- 因此 first-token logits 已经是在“重算后的文档 KV”条件下得到的。
- query token 的 K/V 也在同一次 forward 中生成，并且没有被 restore；后续 decode 仍然使用了这些 query-side cache。

所以：

- `none_restore_after_recompute_rate0.15` 不是“完全不更新 K/V”，而是“执行 selected-doc online forward，并在 first-token logits 已经产生之后恢复文档 K/V”。
- `v_only_rate0.15` 同理也不是严格的“只更新 V”，因为 first-token logits 和 query KV 已经受重算路径影响。
- 这两组只能说明一个很窄的现象：当前实现里，first-token forward 之后是否保留 selected document KV 到后续 decode，没有表现出稳定收益。
- 可靠结论应收窄为：真正 `rate=0` 明显低于 `kv@0.15`，说明 online recomputation 本身有用；但 K/V 写回必要性需要重新设计严格实验。

后续如果要做严格实验，应把 selected document token recompute 和 query prefill 拆成两个阶段：先对 selected doc token 做 online recompute，再按实验条件保留/恢复 K/V，然后重新单独 forward query tokens 并从这次 query logits 开始 decode。

## 严格两阶段 K/V 写回消融

前面的 `v_only` / `none` 实现存在污染：selected document token 和 query token 被拼在同一次 forward 里，restore 发生在 first-token logits 之后。因此这里重新实现了严格两阶段版本：

1. 先只 forward selected document tokens，写入 online recomputation 后的文档 KV。
2. 按实验模式处理 selected document KV：
   - `strict_kv`：保留 K 和 V。
   - `strict_v_only`：恢复 K，保留 V。
   - `strict_none`：恢复 K 和 V。
3. 再单独 forward query tokens，从这一次 query logits 开始生成。

启动时使用：

```bash
FUSIONRAG_STRICT_REPROCESS_ABLATION=1 FUSIONRAG_REPROCESS_UPDATE_MODE=<kv|v_only|none> ...
```

其他实验条件保持一致：完整 `data/result_reflect.json`，实际评测 135 个 main question / 250 个 sub question；`Qwen2.5-7B-Instruct`；`preprocess=True`；`preprocess_scope=global`；`topk=10`；BGE-M3 检索；`rate=0.15`；`reprocess_method=FusionRAG`；`revert_rope=True`；DeepSeek `deepseek-v3.2` judge。

| mode | Main Acc | Sub Acc | Avg F1 | Avg EM | pred diff vs strict kv | judge diff vs strict kv | gain | loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `strict_kv_rate0.15` | 86/135 (63.70%) | 189/250 (75.60%) | 0.4960 | 0.2160 | 0 | 0 | 0 | 0 |
| `strict_v_only_rate0.15` | 81/135 (60.00%) | 183/250 (73.20%) | 0.4498 | 0.1880 | 98 | 36 | 15 | 21 |
| `strict_none_rate0.15` | 75/135 (55.56%) | 177/250 (70.80%) | 0.4510 | 0.1600 | 113 | 46 | 17 | 29 |
| `true_rate0_no_doc_recompute` | 77/135 (57.04%) | 178/250 (71.20%) | 0.4510 | 0.1600 | 113 | 45 | 17 | 28 |

严格实验结论：

- 严格 `strict_kv` 明显优于 `strict_v_only` 和 `strict_none`，说明 selected document token 的 online recomputation 结果需要以完整 K/V 形式保留，不能只保留 V。
- `strict_v_only` 相比 `strict_kv` 的 Sub Acc 从 75.60% 降到 73.20%，Avg F1 从 0.4960 降到 0.4498；说明 K 更新不是可忽略项。
- `strict_none` 与真正 `rate=0` 几乎同档：Sub Acc 70.80% vs 71.20%，Avg F1 同为 0.4510，EM 同为 0.1600。这说明如果文档 K/V 在 query forward 前被完全恢复，前面执行 selected-doc recompute 基本不能转化成端到端收益。
- 因此可靠结论应更新为：online recomputation 的收益主要来自被选中文档 token 的 K/V cache 状态改变，而不是仅仅来自执行了一次重算 forward；K 与 V 都对最终质量有贡献。

输出文件：

- 严格 `kv`：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_strict_kv/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge`
- 严格 `v_only`：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_strict_v_only/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge`
- 严格 `none`：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_strict_none/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge`
- 汇总表：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_full_kv_writeback_ablation/strict_summary_with_rate0.csv`
- 行级差异：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_full_kv_writeback_ablation/strict_row_diffs_vs_kv.csv`

## 追加验证：`strict_none_clean`

为了确认 `strict_none` 和 `rate=0` 的差异是不是来自 cache metadata 副作用，又补了一个 `strict_none_clean`：

- selected-doc forward 前备份 `past_key_values.past_tokens`。
- 恢复 selected document K/V 后，同时恢复 `past_tokens`。
- 然后再单独 forward query tokens。

启动方式：

```bash
FUSIONRAG_STRICT_REPROCESS_ABLATION=1 FUSIONRAG_CLEAN_STRICT_NONE=1 FUSIONRAG_REPROCESS_UPDATE_MODE=none ...
```

结果：

| mode | Main Acc | Sub Acc | Avg F1 | Avg EM |
|---|---:|---:|---:|---:|
| `strict_kv_rate0.15` | 86/135 (63.70%) | 189/250 (75.60%) | 0.4960 | 0.2160 |
| `strict_none_rate0.15` | 75/135 (55.56%) | 177/250 (70.80%) | 0.4510 | 0.1600 |
| `strict_none_clean_rate0.15` | 76/135 (56.30%) | 177/250 (70.80%) | 0.4510 | 0.1600 |
| `true_rate0_no_doc_recompute` | 77/135 (57.04%) | 178/250 (71.20%) | 0.4510 | 0.1600 |

对齐分析：

- `strict_none_clean` vs `rate=0`：预测文本差异为 0，judge 正误差异为 1。
- `strict_none_clean` vs `strict_none`：预测文本差异为 0，judge 正误差异为 2。
- `strict_none_clean` vs `strict_kv`：预测文本差异为 113，judge 正误差异为 46。

解释：

- 恢复 `past_tokens` 后，`strict_none_clean` 和 `rate=0` 的模型输出完全一致；剩下的 1 个 correct 差异来自 DeepSeek judge 的非确定性或边界样本判断波动。
- 这说明前面 `strict_none` 与 `rate=0` 不完全一致不是来自真实模型行为差异，至少不是来自输出文本差异；本质上，文档 K/V 如果在 query forward 前完全恢复，selected-doc recompute 不会贡献模型输出收益。
- 最终结论更明确：online recomputation 的端到端收益来自 selected document token 的 K/V cache 写回；如果不保留写回，它退化为 `rate=0`。

输出文件：

- `strict_none_clean` 原始结果：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_strict_none_clean/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge`
- 汇总表：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_full_kv_writeback_ablation/strict_none_clean_summary.csv`
- 对比统计：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_full_kv_writeback_ablation/strict_none_clean_comparisons.csv`
- 行级差异：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_full_kv_writeback_ablation/strict_none_clean_row_diffs.csv`



## 追加实验：干净版 K/V 单独写回消融

本轮重新跑了 clean strict ablation。旧的 `strict_v_only` 结果不能直接作为最终结论使用，因为旧实现只恢复了被选 token 的 K，但没有恢复 `past_tokens` 元数据；虽然 K/V 张量写回语义基本接近，但严格来说 query forward 的缓存状态仍可能受到 doc-token reprocess 的副作用影响。因此本轮统一开启：

- `FUSIONRAG_STRICT_REPROCESS_ABLATION=1`
- `FUSIONRAG_CLEAN_STRICT_ABLATION=1`

干净版语义如下：

- `strict_kv_rate0.15`：选中 token 的 K 和 V 都保留在线更新后的结果。
- `strict_v_only_clean_rate0.15`：恢复 K，只保留在线更新后的 V，并恢复 `past_tokens`。
- `strict_k_only_clean_rate0.15`：保留在线更新后的 K，恢复 V，并恢复 `past_tokens`。
- `strict_none_clean_rate0.15`：K/V 都恢复，且恢复 `past_tokens`，应退化到 `rate=0`。
- `true_rate0_no_doc_recompute`：完全不做 doc-token online reprocess。

指标汇总：

| run | main | sub | avg_f1 | avg_em |
| --- | --- | --- | --- | --- |
| strict_kv_rate0.15 | 86/135 (0.6370) | 189/250 (0.7560) | 0.4960 | 0.2160 |
| strict_v_only_clean_rate0.15 | 82/135 (0.6074) | 184/250 (0.7360) | 0.4498 | 0.1880 |
| strict_k_only_clean_rate0.15 | 77/135 (0.5704) | 181/250 (0.7240) | 0.4687 | 0.1880 |
| strict_none_clean_rate0.15 | 76/135 (0.5630) | 177/250 (0.7080) | 0.4510 | 0.1600 |
| true_rate0_no_doc_recompute | 77/135 (0.5704) | 178/250 (0.7120) | 0.4510 | 0.1600 |

相对 `strict_kv_rate0.15` 的逐子问题差异：

| run | prediction_diff_count | prediction_diff_rate | correct_diff_count | correct_diff_rate | mean_f1_delta_vs_kv |
| --- | --- | --- | --- | --- | --- |
| strict_v_only_clean_rate0.15 | 98 | 0.3920 | 35 | 0.1400 | -0.0462 |
| strict_k_only_clean_rate0.15 | 93 | 0.3720 | 38 | 0.1520 | -0.0273 |
| strict_none_clean_rate0.15 | 113 | 0.4520 | 46 | 0.1840 | -0.0450 |
| true_rate0_no_doc_recompute | 113 | 0.4520 | 45 | 0.1800 | -0.0450 |

当前结论：

1. `strict_none_clean_rate0.15` 与 `true_rate0_no_doc_recompute` 的预测应基本一致；如果 judge correctness 有极小差异，主要来自外部 LLM judge 的非确定性。
2. `strict_v_only_clean_rate0.15` 明显低于 `strict_kv_rate0.15`，说明只更新 V 不能替代完整 K/V 在线更新。
3. `strict_k_only_clean_rate0.15` 用来隔离 K 的贡献；需要和上表中的 `v_only_clean` 对比，看 K-only 和 V-only 哪一侧更接近完整 `kv`。

输出文件：

- `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_full_kv_writeback_ablation/strict_clean_kv_v_k_writeback_summary.csv`
- `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_full_kv_writeback_ablation/strict_clean_kv_v_k_writeback_diffs_vs_kv.csv`
