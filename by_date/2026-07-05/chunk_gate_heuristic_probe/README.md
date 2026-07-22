# Chunk Gate Heuristic Probe

## 目的

测试可部署的 chunk gate 启发式能覆盖多少 `online_draft_selected - offline_fixed_set` residual tokens。这里不跑模型，只复用已有 online draft trace 和 offline base set。

## Gate 方法

- `retrieval_first`: 按 chunk/retrieval 顺序取前 k 个。
- `longest`: 取最长的 k 个 chunk。
- `highest_base_count`: 取 offline fixed set 选中 token 数最多的 k 个 chunk。
- `highest_base_density`: 取 offline fixed set token 密度最高的 k 个 chunk。
- `lowest_base_density`: 取 offline fixed set token 密度最低的 k 个 chunk。
- `oracle_residual`: 按真实 residual count 取 top-k，是不可部署 oracle 上界。

## 汇总结果：avg residual coverage

### Base: freq

| chunks | retrieval_first | longest | highest_base_count | highest_base_density | lowest_base_density | oracle_residual |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.36% | 3.26% | 5.40% | 5.16% | 28.70% | 35.27% |
| 2 | 22.23% | 8.86% | 10.57% | 11.74% | 32.65% | 59.25% |
| 3 | 26.35% | 13.44% | 17.97% | 18.34% | 38.46% | 75.39% |
| 4 | 28.99% | 19.13% | 24.18% | 24.71% | 43.63% | 84.63% |
| 5 | 31.54% | 25.19% | 29.59% | 32.04% | 49.29% | 90.64% |
| 6 | 33.76% | 32.97% | 34.86% | 39.76% | 55.83% | 94.58% |

### Base: mean

| chunks | retrieval_first | longest | highest_base_count | highest_base_density | lowest_base_density | oracle_residual |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16.44% | 3.63% | 6.71% | 6.68% | 25.85% | 32.14% |
| 2 | 23.72% | 9.29% | 12.44% | 14.06% | 29.69% | 54.78% |
| 3 | 28.14% | 14.12% | 21.17% | 21.79% | 35.16% | 70.61% |
| 4 | 31.16% | 20.13% | 27.14% | 28.76% | 41.15% | 80.28% |
| 5 | 34.07% | 26.50% | 32.64% | 35.77% | 47.03% | 86.64% |
| 6 | 36.60% | 34.38% | 38.70% | 42.81% | 53.27% | 91.11% |

## 结论

- 简单启发式能覆盖一部分 residual，但和 oracle chunk gate 仍有明显差距。
- `longest` 和 `highest_base_count/density` 如果接近 oracle，说明可以用 offline chunk summary 做 coarse gate；如果明显低于 oracle，则需要 query-conditioned chunk predictor。
- 这组结果用于决定是否值得实现 chunk-gated DraftModel：如果用 3-5 个 chunk 已经覆盖大部分 residual，下一步可以跑真实 pipeline 验证只在这些 chunk 上做 residual selection。

## 产物

- `chunk_gate_detail.csv`
- `chunk_gate_summary.csv`
