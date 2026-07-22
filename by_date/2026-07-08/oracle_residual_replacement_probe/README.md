# Oracle residual replacement experiment
## 实验目的
验证 online Draft residual 这 5% token 是否能被纯 offline 候选集合替代。这里不重新跑问答 accuracy，而是复用已保存的选择结果做集合覆盖诊断：如果离线候选池本身覆盖不到 online residual，那么后续再从这个候选池中挑 5% 很难达到 online Draft residual 的效果。
## 实验定义
- 基础 offline set：`offline10_draft005` 中的 `draft_smart_frequency_global_rate0p1`，即原来的 offline 10%。
- Teacher：同一组 `offline10_draft005` 运行中保存的 online Draft 选择结果。
- 日志确认该运行使用 `residual_online_method=DraftModel residual_rate=0.05`；但实际保存的最终 selected set 平均占文档 17.26%，扣除 offline10 后 residual 平均占约 6.96%。因此本文档按保存下来的实际集合做覆盖诊断，而不是假设 residual 严格等于 5%。
- Oracle residual：`Teacher selected_abs - offline10 selected_abs`。这表示 online Draft 相对 offline10 额外补出来的 token。
- 评价指标：`residual recall = |candidate_pool ∩ oracle_residual| / |oracle_residual|`；`teacher recall = |candidate_pool ∩ teacher_selected| / |teacher_selected|`。
## 数据规模
- 样本数：250 个 question/sub-question。
- 平均文档 token 数：1457.0。
- offline10 平均 token 数：148.0。
- teacher 平均 token 数：249.5。
- teacher 与 offline10 平均重合：148.0，teacher recall=0.5731，offline10 precision-to-teacher=1.0000。
- oracle residual 平均 token 数：101.5。
## 覆盖结果
| candidate pool | pool/doc | residual recall | teacher recall | precision to residual | precision to teacher |
|---|---:|---:|---:|---:|---:|
| old10_base | 10.18% | 0.00% | 57.31% | 0.00% | 100.00% |
| docgen10 | 9.42% | 12.17% | 31.27% | 9.15% | 55.83% |
| hybrid70_old_docgen10 | 10.18% | 4.52% | 50.69% | 3.54% | 88.60% |
| old10_union_docgen10 | 15.20% | 12.17% | 62.72% | 6.04% | 71.24% |
| old10_union_hybrid70 | 11.69% | 4.52% | 59.24% | 3.06% | 90.15% |
| docgen15_frequency | 14.29% | 18.33% | 42.08% | 9.07% | 49.85% |
| docgen15_mean_score | 14.29% | 29.34% | 42.48% | 14.50% | 50.03% |
| old10_union_docgen15_frequency | 18.64% | 18.33% | 65.42% | 7.21% | 60.44% |
| old10_union_docgen15_mean | 19.39% | 29.34% | 70.29% | 11.12% | 62.23% |
| old20_base | 20.25% | 35.10% | 72.55% | 13.42% | 62.63% |
| old10_union_old20 | 20.26% | 35.10% | 72.60% | 13.42% | 62.65% |
| oracle_online_draft_selected | 17.26% | 100.00% | 100.00% | 42.69% | 100.00% |

## 初步结论
1. `old10_base` 对 oracle residual 的覆盖为 0 是定义导致的，因为 residual 已经扣除了 offline10。
2. 如果某个离线候选池的 `residual recall` 很低，说明它很难直接替代 online Draft residual；即使后续做 rerank，也缺少可选 token。
3. `old20_base` / `old10_union_old20` 是一个重要上界参照：它回答“把原 offline 预算扩大到 20% 是否自然覆盖 residual”。
4. `docgen` 系列回答“用文档生成问题得到的离线集合，是否能补到 online Draft residual”。若 union 后 residual recall 仍低，说明 docgen 与 online Draft residual 的互补性有限。

## 输出文件
- `oracle_residual_replacement_summary.csv`：候选池级汇总。
- `oracle_residual_replacement_detail.csv`：每个 example/sub-question 的详细覆盖。
- `oracle_residual_base_stats.csv`：teacher/offline10/residual 的基础统计。
- `oracle_residual_base_summary.json`：基础统计 JSON。
