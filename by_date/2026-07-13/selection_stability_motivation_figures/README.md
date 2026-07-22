# Selection Stability Motivation Figures

This folder collects the figures used to support the motivation that document-token selection contains a large query-stable component.

Generated figures:

- `token_selection_frequency_hist_rate010.png/pdf`: token frequency histogram over 16 queries, rate=0.1.
- `heldout_query_selection_overlap.png/pdf`: leave-one-query-out prediction using the other 15 queries.
- `stable_set_convergence_16q.png/pdf`: strict intersection ratio as more queries are used, for DraftModel and FusionRAG-QK.
- `stable_set_convergence_32q_qk.png/pdf`: 32-query convergence for FusionRAG-QK/preprocess KV.
- `full_rag_top10_draft_stability.png/pdf`: DraftModel stability under full top-10 RAG context.
- `stable_core_vs_residual_contrast.png/pdf`: stable full selected set versus query-specific residual.
- `selection_stability_motivation_2x2.png/pdf`: compact four-panel figure for the motivation section.

Key source files:

- `MOTIVATION_EXPERIMENTS/query_selection_frequency_stability/query_selection_frequency_stability.json`
- `MOTIVATION_EXPERIMENTS/query_selection_frequency_stability_32q_preprocess/query_selection_frequency_stability.json`
- `MOTIVATION_EXPERIMENTS/calibration_query_selector_prediction_16q/loo_summary.csv`
- `MOTIVATION_EXPERIMENTS/full_rag_draft_query_stability/full_rag_top10_draft_query_stability_summary.csv`
- `MOTIVATION_EXPERIMENTS/residual_stability_from_online_draft_trace/summary.csv`
- `MOTIVATION_EXPERIMENTS/residual_stability_from_online_draft_trace/leave_one_query_out_summary.csv`
