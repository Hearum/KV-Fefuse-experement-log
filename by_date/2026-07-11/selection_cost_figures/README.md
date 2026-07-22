# Selection Cost Figures

Updated with the fast-KV DraftModel-selector + FusionRAG sparse-update sweep.

- QK source: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_clean_rate_sweep_v2/clean_rate_sweep_summary.csv`
- Draft source: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/online_ttft_profile_draft_select_sparse_update_rrf18_pptrue_sweep_fastkv`
- `qk_e2e_ttft_vs_rate.png/pdf`: end-to-end TTFT vs update rate; rate=1 is shown only as the full recompute dashed baseline.
- `qk_ttft_breakdown_by_rate.png/pdf`: FusionRAG-QK TTFT breakdown; shares the same y-axis range as the DraftModel breakdown.
- `draft_ttft_breakdown_by_rate.png/pdf`: DraftModel selector TTFT breakdown; rate=1 is shown only as the full recompute dashed baseline.
