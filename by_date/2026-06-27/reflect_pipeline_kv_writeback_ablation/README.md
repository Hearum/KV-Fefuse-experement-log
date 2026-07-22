# Reflect pipeline 中 online 重算 K/V 写回消融

更新时间：2026-06-26

## 实验目的

之前用 `qwen_process_cache.py` 做过 `kv / v_only / none` 的消融，但该入口不是 `preprocess-exp` 分支里的真实 FusionRAG reflect pipeline，得到的准确率异常低，不能作为主结论使用。本实验改用 `origin/preprocess-exp:test_fusionrag_reflect.py` 对应的真实脚本，验证 online reprocess 阶段被选中文档 token 的 K/V 写回是否必要。

三个模式：

- `kv`：默认 FusionRAG 行为。online reprocess 后，被选中文档 token 的 K 和 V 都写回 cache。
- `v_only`：只保留被选中文档 token 的 V 更新，把这些 token 的 K 恢复成 reprocess 前的缓存值。
- `none`：被选中文档 token 的 K 和 V 都恢复成 reprocess 前的缓存值。query token 仍然正常 prefill/生成。

## 实验配置

- 代码位置：`/raid/home/hming/FusionRAG-pca-analysis`
- 分支：`hming/add-pca-analysis`
- 脚本：`test_fusionrag_reflect_preprocess_exp.py`，从 `origin/preprocess-exp:test_fusionrag_reflect.py` 提取
- 数据：`data/result_reflect.json`，从 `origin/preprocess-exp:data/result_reflect.json` 提取
- 模型：`/mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct`
- 数据集：`musique`
- 样本数：前 20 个 main questions；其中 2 个被原脚本跳过，所以有效 18 个 main questions、32 个 sub-questions
- 检索与 preprocess：`preprocess=True`, `preprocess_scope=global`, `topk=10`, `recall_method=bge`
- reprocess：`reprocess_method=FusionRAG`, `rate=0.15`, `revert_rope=True`
- judge：DashScope compatible endpoint + `deepseek-v3.2`
- GPU：`CUDA_VISIBLE_DEVICES=6`

## 结果汇总

| 模式 | 被选中文档 token 写回策略 | Main Acc | Sub Acc | Avg F1 | Avg EM |
|---|---|---:|---:|---:|---:|
| `kv` | 更新 K 和 V | 16/18 = 0.8889 | 29/32 = 0.9062 | 0.5296 | 0.2188 |
| `v_only` | 只更新 V，K 恢复旧 cache | 16/18 = 0.8889 | 29/32 = 0.9062 | 0.5326 | 0.2188 |
| `none` | K 和 V 都恢复旧 cache | 16/18 = 0.8889 | 29/32 = 0.9062 | 0.5077 | 0.2188 |

逐 sub-question 对比：

| 对比 | 生成文本不同 | judge 正误不同 |
|---|---:|---:|
| `v_only` vs `kv` | 4/32 | 0/32 |
| `none` vs `kv` | 4/32 | 0/32 |

输出变化的样本多数是表述长度或同义表达变化，不改变 judge 正误。例如 `National Cycle Network (NCN)` 变成 `National Cycle Network`，或者解释性回答变短/变长。

## 当前结论

在真实 `preprocess-exp` reflect pipeline、`max_samples=20`、`rate=0.15`、`topk=10` 的设置下，online reprocess 对被选中文档 token 的 K/V 写回没有影响最终 judge accuracy。`v_only` 和 `kv` 在主问题、子问题、EM 上完全一致，F1 仅有 0.003 的浮动；`none` 也保持相同准确率，但 F1 降到 0.5077，说明输出文本分布有轻微扰动。

当前更合理的解释是：这组短生成 QA 中，收益可能主要来自 query prefill 时访问被选中 token 形成的稀疏上下文，而不是必须把被选中文档 token 的更新后 K/V 写回 cache。若要把它变成强 observation，需要扩大样本数、rate、topk、数据集，并验证长生成多步 decode 是否会放大 K/V 写回差异。

## 原始结果路径

- `kv`：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_sanity/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/`
- `v_only`：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_ablation_v_only/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/`
- `none`：`/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/reflect_pipeline_ablation_none/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/`
