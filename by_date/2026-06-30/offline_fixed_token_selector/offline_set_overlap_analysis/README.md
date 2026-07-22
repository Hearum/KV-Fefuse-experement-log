# Offline Fixed Set Overlap Analysis

每个值按 example/chunk 计算后取平均。`coverage_b_by_a` 表示 a 覆盖 b 的比例。

## Coverage To Strong References

| candidate | reference | chunks | Jaccard | coverage reference by candidate |
|---|---|---:|---:|---:|
| draft_score_lexical_tiebreak | draft_freq | 2098 | 0.9764 | 0.9874 |
| hybrid_draft70_qk30 | draft_freq | 2098 | 0.9244 | 0.9590 |
| hybrid_intersection_fill | draft_freq | 2098 | 0.8976 | 0.9437 |
| draft90_lexical10 | draft_freq | 2098 | 0.8235 | 0.9032 |
| draft85_lexical15 | draft_freq | 2098 | 0.7508 | 0.8567 |
| draft80_lexical20 | draft_freq | 2098 | 0.6934 | 0.8177 |
| hybrid_rrf | draft_freq | 2098 | 0.6952 | 0.8147 |
| hybrid_draft50_qk50 | draft_freq | 2098 | 0.6689 | 0.7945 |
| qk_freq | draft_freq | 2098 | 0.4385 | 0.5976 |
| fullattn_freq | draft_freq | 2098 | 0.1740 | 0.2895 |
| lexical_entity | draft_freq | 2098 | 0.1681 | 0.2823 |
| lexical_idf_boundary | draft_freq | 2098 | 0.1369 | 0.2364 |
| lexical_idf_entity | draft_freq | 2098 | 0.1183 | 0.2074 |
| hybrid_draft50_qk50 | qk_freq | 2098 | 0.6763 | 0.8003 |
| hybrid_rrf | qk_freq | 2098 | 0.5983 | 0.7396 |
| hybrid_intersection_fill | qk_freq | 2098 | 0.4956 | 0.6512 |
| hybrid_draft70_qk30 | qk_freq | 2098 | 0.4778 | 0.6348 |
| draft_freq | qk_freq | 2098 | 0.4385 | 0.5976 |
| draft_score_lexical_tiebreak | qk_freq | 2098 | 0.4377 | 0.5969 |
| draft90_lexical10 | qk_freq | 2098 | 0.4207 | 0.5820 |
| draft85_lexical15 | qk_freq | 2098 | 0.4104 | 0.5716 |
| draft80_lexical20 | qk_freq | 2098 | 0.4011 | 0.5628 |
| fullattn_freq | qk_freq | 2098 | 0.1683 | 0.2811 |
| lexical_entity | qk_freq | 2098 | 0.1619 | 0.2743 |
| lexical_idf_boundary | qk_freq | 2098 | 0.1058 | 0.1875 |
| lexical_idf_entity | qk_freq | 2098 | 0.0940 | 0.1680 |
