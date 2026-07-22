# Reflect Offline Hybrid Fixed Sets

- rate: 0.15
- source: existing offline QK + draft calibration caches

| method | chunks | Jaccard vs draft freq | Jaccard vs QK freq |
|---|---:|---:|---:|
| hybrid_draft50_qk50_score_per_chunk | 2098 | 0.6689 | 0.6763 |
| hybrid_draft70_qk30_score_per_chunk | 2098 | 0.9244 | 0.4778 |
| hybrid_draft70_then_qk30_per_chunk | 2098 | 0.6136 | 0.6897 |
| hybrid_qk_draft_intersection_fill_draft_per_chunk | 2098 | 0.8976 | 0.4956 |
| hybrid_qk_draft_rrf_per_chunk | 2098 | 0.6952 | 0.5983 |
