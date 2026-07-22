# Offline preprocess top-k 与 online RAG 文档命中率

## 实验设置

- 仓库路径：`/home/hming/FusionRAG-pca-analysis`；qjy001 常用路径是 `/home/hming/FusionRAG-pca-analysis`，qjy000 也可能有 `/raid/home/hming/FusionRAG-pca-analysis`。
- offline top-k 来源：`existing run.log first; optional BGE/FAISS recompute with --compute-bge`。
- top-k 统计点：`1, 2, 3, 5, 10`。
- self-recall 口径：`exclude`；主 runner 在非 `REPEAT_SELF` preprocess 中会跳过 self-recall，因此默认报告使用 exclude。
- skipped example：`excluded`。

启动命令：

```bash
cd /home/hming/FusionRAG-pca-analysis  # 或 /raid/home/hming/FusionRAG-pca-analysis
python3 MOTIVATION_EXPERIMENTS/offline_preprocess_online_hit_rate/analyze_offline_preprocess_online_hit_rate.py \
  --datasets musique 2wikimqa hotpotqa triviaqa
```

如现有 `run.log` 不完整，可显式追加 `--compute-bge --bge-model-path /mnt/qjhs-sh-lab-01/models/bge-m3`；这只跑 BGE/FAISS embedding 检索，不启动 235B。

## 代码定位

- `test_fusionrag_reflect_preprocess_exp.py::prepare_reflect_data`：读取 `intermediate_context[*]['retrieve docs']`，在每个 example 内按文档文本去重，映射为 1-indexed `chunk_id`；同时构建全局 `global_corpus` 和 `corpus_lens`。
- 同函数中 BGE/FAISS 分支生成 `context_rank`，形状为 `[total_docs, topk]`，元素是全局 doc index；`find_group_and_index(corpus_lens, idx)` 可还原为 `(example_id, local_doc_idx)`，在线打印为 `Q{example+1}-Doc{chunk_id}`。
- preprocess 生成 KV 时，对当前 doc 的 `global_doc_idx = sum(corpus_lens[:example_id]) + doc_idx` 读取 `context_rank[global_doc_idx][:topk]`，加载这些相似文档 KV 后再把当前 doc 追加进去，保存到 `preprocess_save_path/{example_id}_{chunk_id}_key/value.pt`。
- online RAG 生成时，每个 sub-question 使用 `doc_chunk_ids = sub_q_info['chunk_ids']`，`kv_chunk_ids = [0] + doc_chunk_ids`，随后 `load_kv_and_generate(..., chunk_ids=kv_chunk_ids)` 载入 system chunk 0 和这些实际检索/使用的文档 chunk。

## 统计口径

- `Jaccard = |offline_topk ∩ online_docs| / |offline_topk ∪ online_docs|`。
- `offline_covered_by_online = |intersection| / |offline_topk|`：offline 预取的 top-k 中有多少被 online 实际用到。
- `online_covered_by_offline = |intersection| / |online_docs|`：online 实际文档中有多少落在 offline top-k 内。
- `chunk` 层：以 online sub-question 中每个 source chunk 为单位，比较该 chunk 的 offline top-k 和同一 sub-question 的 online docs。
- `sub_question` 层：把该 sub-question 所有 online chunks 的 offline top-k 取并集，再与 online docs 比较。

## 数据覆盖

| dataset | examples | docs | sub_questions_used | offline_rows | offline_coverage | source |
|---|---:|---:|---:|---:|---:|---|
| musique | 200 | 3408 | 250 | 1528 | 0.4484 | log:MOTIVATION_EXPERIMENTS/qwen3_235b_param_scaling_cross_dataset/results/musique/offline32b_top2_rate015/full_0_200/run.log |
| 2wikimqa | 200 | 1991 | 200 | 987 | 0.4957 | log:MOTIVATION_EXPERIMENTS/qwen3_235b_param_scaling_cross_dataset/results/2wikimqa/offline32b_top2_rate015/full_0_200/run.log |

## 主要结果

| dataset | level | topk | n | Jaccard mean/p50/p90 | offline覆盖 mean/p50/p90 | online覆盖 mean/p50/p90 |
|---|---|---:|---:|---:|---:|---:|
| 2wikimqa | chunk | 1 | 987 | 0.0496/0.0000/0.1000 | 0.4944/0.0000/1.0000 | 0.0496/0.0000/0.1000 |
| 2wikimqa | chunk | 2 | 987 | 0.0833/0.0909/0.2000 | 0.4245/0.5000/1.0000 | 0.0851/0.1000/0.2000 |
| 2wikimqa | chunk | 5 | 987 | 0.1420/0.0714/0.5000 | 0.3228/0.2000/1.0000 | 0.1616/0.1000/0.5000 |
| 2wikimqa | chunk | 10 | 987 | 0.1553/0.0588/0.4615 | 0.2405/0.1111/0.6667 | 0.2166/0.1111/0.6000 |
| 2wikimqa | sub_question | 1 | 200 | 0.1320/0.0000/0.4545 | 0.2240/0.0000/0.7528 | 0.1714/0.0000/0.5000 |
| 2wikimqa | sub_question | 2 | 200 | 0.1328/0.0000/0.4444 | 0.1726/0.0000/0.5726 | 0.2225/0.0000/0.7000 |
| 2wikimqa | sub_question | 5 | 200 | 0.0970/0.0000/0.2834 | 0.1067/0.0000/0.3032 | 0.2885/0.0000/0.8000 |
| 2wikimqa | sub_question | 10 | 200 | 0.0655/0.0000/0.1639 | 0.0679/0.0000/0.1705 | 0.3262/0.0000/0.9000 |
| musique | chunk | 1 | 1791 | 0.0164/0.0000/0.1000 | 0.1647/0.0000/1.0000 | 0.0164/0.0000/0.1000 |
| musique | chunk | 2 | 1791 | 0.0323/0.0000/0.0909 | 0.1720/0.0000/0.5000 | 0.0343/0.0000/0.1000 |
| musique | chunk | 5 | 1791 | 0.0750/0.0714/0.1538 | 0.1885/0.2000/0.4000 | 0.0938/0.1000/0.2000 |
| musique | chunk | 10 | 1791 | 0.1032/0.0556/0.2667 | 0.1779/0.1111/0.4444 | 0.1592/0.1000/0.4000 |
| musique | sub_question | 1 | 250 | 0.0617/0.0000/0.2000 | 0.1086/0.0000/0.3750 | 0.0936/0.0000/0.3000 |
| musique | sub_question | 2 | 250 | 0.0798/0.0345/0.2513 | 0.1074/0.0500/0.3353 | 0.1625/0.0955/0.5000 |
| musique | sub_question | 5 | 250 | 0.0994/0.0632/0.2667 | 0.1101/0.0741/0.2857 | 0.3244/0.3000/0.8000 |
| musique | sub_question | 10 | 250 | 0.0893/0.0633/0.1957 | 0.0937/0.0677/0.2000 | 0.4292/0.4000/0.9000 |

## 初步结论

- 以当前口径看，offline preprocess 的 BGE top-k 相似文档与 online query 实际使用文档的重合通常偏低；`offline_covered_by_online` 直接表示预取 KV 中真正被在线 RAG 用到的比例。
- `sub_question` 层的 `online_covered_by_offline` 高于 `chunk` 层时，说明一个 query 的多个 source chunks 的 offline top-k 并集能覆盖更多在线 docs，但单个 chunk 的预取仍较分散。
- 如果某个 dataset 的 `offline_coverage` 明显小于 1，说明只能基于现有日志中打印过的 preprocess rows 统计；需要完整结论时请用 `--compute-bge` 补齐。

详细明细见：`sub_question_detail.csv`、`chunk_detail.csv`、`aggregate_summary.csv`、`dataset_summary.json`。
