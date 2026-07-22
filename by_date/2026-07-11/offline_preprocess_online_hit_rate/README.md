# Offline preprocess top-k 与 online RAG 文档命中率

## 实验设置

- 仓库路径：`/home/hming/FusionRAG-pca-analysis`；qjy000 上也存在 `/raid/home/hming/FusionRAG-pca-analysis`。
- 本次主结果优先复用现有 `run.log` 中的 `Retrieved similar docs: Qx-Docy`，不删除缓存、不启动 235B。
- 统计脚本：`MOTIVATION_EXPERIMENTS/offline_preprocess_online_hit_rate/analyze_offline_preprocess_online_hit_rate.py`。
- 输出文件：`dataset_summary.json`、`aggregate_summary.csv`、`sub_question_detail.csv`、`chunk_detail.csv`。
- top-k 统计点：`1, 2, 3, 5, 10`。
- self-recall 口径：默认排除。主 runner 在非 `REPEAT_SELF` preprocess 中会跳过 self-recall，因此报告也按排除 self 统计。
- skipped example：默认排除 `should_test=False` 的 main question。

启动命令：

```bash
cd /home/hming/FusionRAG-pca-analysis
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/offline_preprocess_online_hit_rate/analyze_offline_preprocess_online_hit_rate.py \
  --datasets musique 2wikimqa triviaqa
```

如果需要完整补齐没有打印到日志的 offline top-k，可单独跑：

```bash
CUDA_VISIBLE_DEVICES="" /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/offline_preprocess_online_hit_rate/analyze_offline_preprocess_online_hit_rate.py \
  --datasets musique \
  --compute-bge \
  --bge-model-path /mnt/qjhs-sh-lab-01/models/bge-m3
```

这只跑 BGE/FAISS embedding 检索，不加载 235B；本次尝试补齐 MuSiQue 时 CPU 任务未完成，未纳入主表。

## 代码定位

- `test_fusionrag_reflect_preprocess_exp.py::prepare_reflect_data` 读取 `intermediate_context[*]["retrieve docs"]`，在每个 example 内按文档文本去重，映射为 1-indexed `chunk_id`。同一文档文本再次出现会复用同一个 chunk id。
- 同函数构建 `global_corpus` 和 `corpus_lens`。BGE/FAISS preprocess 分支生成 `context_rank`，形状是 `[total_docs, topk]`，元素是全局 doc index。用 `find_group_and_index(corpus_lens, idx)` 可还原为 `(example_id, local_doc_idx)`，日志里显示为 `Q{example+1}-Doc{chunk_id}`。
- preprocess KV 生成阶段对当前文档计算 `global_doc_idx = sum(corpus_lens[:example_id]) + doc_idx`，读取 `context_rank[global_doc_idx][:topk]`，加载这些相似文档 KV 后再追加当前 doc，保存到 `preprocess_save_path/{example_id}_{chunk_id}_key/value.pt`。
- online RAG 生成阶段，每个 sub-question 使用 `doc_chunk_ids = sub_q_info["chunk_ids"]`，构造 `kv_chunk_ids = [0] + doc_chunk_ids`，再由 `load_kv_and_generate(..., chunk_ids=kv_chunk_ids)` 载入 system chunk 0 和 online 实际使用的文档 chunks。

## 统计口径

- `Jaccard = |offline_topk ∩ online_docs| / |offline_topk ∪ online_docs|`。
- `offline_covered_by_online = |intersection| / |offline_topk|`：offline 预取 top-k 中真正被 online 使用到的比例。
- `online_covered_by_offline = |intersection| / |online_docs|`：online 实际 docs 中落入 offline top-k 的比例。
- `chunk` 层：对 online sub-question 中每个 source chunk，比较该 chunk 的 offline top-k 和同一 sub-question 的 online docs。
- `sub_question` 层：把该 sub-question 所有 online chunks 的 offline top-k 取并集，再与 online docs 比较。

## 数据覆盖

| dataset | examples | docs | sub_questions_used | offline_rows | offline_coverage | source |
|---|---:|---:|---:|---:|---:|---|
| musique | 200 | 3408 | 250 | 1528 | 0.4484 | qjy001 现有 run.log |
| 2wikimqa | 200 | 1991 | 200 | 1088 | 0.5465 | qjy001 现有 run.log，当前 235B 仍在继续跑 |
| triviaqa | 270 | 2700 | 270 | 0 | 0.0000 | 当前没有对应 offline top-k 日志 |
| hotpotqa | 260 | 2600 | - | 0 | 0.0000 | qjy000 当前 offline32B run.log 尚未产出 |

注意：由于 runner 命中已有 preprocess KV 时会跳过对应 chunk，不会再次打印 `Retrieved similar docs`，所以 MuSiQue/2Wiki 的日志结果是“现有可观测 preprocess rows”上的估计，不是全 corpus 完整估计。

## 主要结果

top-10 口径：

| dataset | level | n | Jaccard mean/p50/p90 | offline覆盖 mean/p50/p90 | online覆盖 mean/p50/p90 |
|---|---|---:|---:|---:|---:|
| musique | chunk | 1791 | 0.1032 / 0.0556 / 0.2667 | 0.1779 / 0.1111 / 0.4444 | 0.1592 / 0.1000 / 0.4000 |
| musique | sub_question | 250 | 0.0893 / 0.0633 / 0.1957 | 0.0937 / 0.0677 / 0.2000 | 0.4292 / 0.4000 / 0.9000 |
| 2wikimqa | chunk | 1088 | 0.1495 / 0.0556 / 0.4615 | 0.2336 / 0.1111 / 0.6667 | 0.2105 / 0.1000 / 0.6000 |
| 2wikimqa | sub_question | 200 | 0.0714 / 0.0326 / 0.1670 | 0.0738 / 0.0362 / 0.1732 | 0.3620 / 0.3000 / 0.9000 |

完整 top-k 分布见 `aggregate_summary.csv`。明细行见 `sub_question_detail.csv` 和 `chunk_detail.csv`。

## 初步结论

- 在现有可观测日志 rows 上，offline preprocess top-10 与 online 实际 docs 的重合偏低。MuSiQue chunk 层 offline 预取文档真正被 online 使用的均值约 17.8%，sub-question 并集层约 9.4%；2WikiMQA 分别约 23.4% 和 7.4%。
- 从 `online_covered_by_offline` 看，MuSiQue sub-question 层 top-10 均值约 42.9%、p50 约 40.0%；2WikiMQA 均值约 36.2%、p50 约 30.0%。这说明一个 query 的多个 source chunks 的 offline top-k 并集能覆盖一部分 online docs，但仍不是稳定高覆盖。
- offline top-k 作为 preprocess KV 预取假设有明显浪费：top-10 下大量预取文档不会在对应 online RAG query 中出现。当前结果支持继续做 query-aware 或 residual online selector，而不是只依赖文档间静态相似度。
- 由于 MuSiQue/2Wiki 只覆盖日志可见 rows，完整数值需要跑 `--compute-bge` 补齐；脚本已经支持该路径，但建议在不影响 235B 任务时单独排队运行。
