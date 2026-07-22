# Derived Draft Score Aggregate Fixed Sets

- source score cache: `reflect_draft_rate015_full/combined/score_cache_npz`
- rate: 0.15
- calibration queries: existing 8 unrelated control queries per example
- methods: `draft_frequency_per_chunk`, `draft_mean_score_per_chunk`, `draft_max_score_per_chunk`, `draft_top2_mean_score_per_chunk`, `draft_top4_mean_score_per_chunk`
- no model forward was rerun; this is derived from saved score tensors.
