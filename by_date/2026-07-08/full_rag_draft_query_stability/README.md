# 完整 RAG 召回文档下 DraftModel selector 的 Query 稳定性

## 实验目的

验证：对同一个 example 的完整 RAG 召回文档序列（top-10 passages）固定不变时，只替换 query，DraftModel selector 选出的重算 token 集合是否稳定；并判断它是否和之前较短文档/单块分析中观察到的 stable set 现象一致。

## 实验设置

- 数据：MuSiQue 前 20 个 example。
- 文档：每个 example 使用完整 RAG 召回的 top-10 passages 拼接后的文档 token 序列；同一个 example 内文档完全固定。
- Query：每个 example 16 条 query，包括原始问题/改写问题以及来自其他样本的 control query。保存的 query 示例见 `example000_queries.csv`。
- Selector：DraftModel `/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`，后半层 query-to-doc attention，经 RRF(k=18) 聚合，然后调用原始 `smart_query_selection`。
- 本轮没有重新跑 DraftModel forward；0.10/0.20/0.30/0.50 读取既有 JSON，0.15 从保存的 score NPZ 复用原始 `smart_query_selection` 离线补算。

复现 0.15 的命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python tools_analyze_saved_query_scores.py \
  --score-root MOTIVATION_EXPERIMENTS/query_recompute_overlap_detail_full/details/20260626_184238 \
  --out-dir MOTIVATION_EXPERIMENTS/full_rag_draft_query_stability/recomputed_from_saved_scores \
  --rate-list 0.15 --block-size 16 --device cpu
```

## 指标解释

- `token Jaccard`：两条 query 选中 token 集合的交并比，越高说明集合越像。
- `token overlap`：交集除以较小集合大小；这里各 query 选择数量基本一致，可理解为互相覆盖比例。
- `block Jaccard`：把 token 映射到 16-token block 后计算 Jaccard，用来观察稳定区域。
- `all-query common / selected`：16 条 query 全部共同选中的 token 数量，占单次选择 token 数的比例，是更严格的 stable core 指标。
- `freq=0 doc frac`：在 16 条 query 中一次都没有被选中的 doc token 占比。
- `freq=16 / selected`：被 16 条 query 全部选中的 token 数，占单次选择预算的比例。
- `score cosine`：不同 query 的 DraftModel importance score 向量余弦相似度，衡量排序分数是否稳定。

## 完整 top-10 RAG 文档结果

| rate | token Jaccard | token overlap | block Jaccard | all-query common/selected | freq=0 doc frac | freq=16/selected | score cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.6745 | 0.7972 | 0.7508 | 0.5686 | 0.8061 | 0.5686 | 0.9472 |
| 0.15 | 0.6936 | 0.8120 | 0.7783 | 0.5830 | 0.7224 | 0.5830 | 0.9472 |
| 0.20 | 0.7076 | 0.8224 | 0.8265 | 0.5993 | 0.6451 | 0.5993 | 0.9472 |
| 0.30 | 0.7386 | 0.8447 | 0.8827 | 0.6268 | 0.5107 | 0.6268 | 0.9472 |
| 0.50 | 0.8009 | 0.8867 | 0.9531 | 0.7006 | 0.2975 | 0.7006 | 0.9472 |

## 和较短文档长度的对比

| rate | passages | token Jaccard | block Jaccard | score cosine |
|---:|---:|---:|---:|---:|
| 0.10 | 3 | 0.6797 | 0.7445 | 0.9667 |
| 0.10 | 5 | 0.6815 | 0.7504 | 0.9610 |
| 0.10 | 10 | 0.6745 | 0.7508 | 0.9472 |
| 0.15 | 3 | 0.6890 | 0.7865 | 0.9667 |
| 0.15 | 5 | 0.6856 | 0.7911 | 0.9610 |
| 0.15 | 10 | 0.6936 | 0.7783 | 0.9472 |
| 0.20 | 3 | 0.7076 | 0.8144 | 0.9667 |
| 0.20 | 5 | 0.7152 | 0.8263 | 0.9610 |
| 0.20 | 10 | 0.7076 | 0.8265 | 0.9472 |

## 结论

1. 完整 top-10 RAG 文档下，DraftModel selector 仍然存在明显的 query 稳定性：rate=0.15 时 token Jaccard 为 0.6936，互相覆盖比例为 0.8120。
2. rate=0.30 时 token Jaccard 进一步提升到 0.7386，block Jaccard 达到 0.8827。
3. 严格共同交集也不小：rate=0.15 时，16 条 query 全部共同选中的 token 约占单次选择预算的 0.5830。这支持完整 RAG 文档中也存在稳定 anchor/update core 的说法。
4. 但它不是完全 query-invariant：native-vs-control 的重合度低于全体平均，score cosine 约 0.947，说明无关 query 会带来一部分真正的选择差异。
5. 和 3/5 passage 设置相比，top-10 的 token Jaccard 没有崩掉，数值处在同一量级；因此之前在较短文档上观察到的 stable set 现象，在完整 RAG 召回文档上依然成立。

## 文件

- `full_rag_top10_draft_query_stability_summary.csv`：top-10 完整 RAG 文档的 rate 级汇总。
- `full_rag_top10_draft_query_stability_detail.csv`：每个 example 的明细。
- `passage_count_comparison.csv`：3/5/10 passage 的对比。
- `example000_queries.csv`：示例 query 内容。
