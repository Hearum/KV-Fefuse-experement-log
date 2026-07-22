# Docgen Draft Mean Score Fixed Set

Method: `docgen_draft_smart_mean_score_global`

This derives a docgen offline fixed set from existing chunk-local score cache. For each document chunk, scores from 32 generated questions are averaged token-wise, then top-rate tokens by mean score are selected. This matches the spirit of `draft_smart_mean_score_global`, but replaces the original calibration queries with document-generated queries.

Rates: [0.15]
Output: `rate_0p15/chunk_fixed_sets_npz/`
Manifest: `manifest.csv`
