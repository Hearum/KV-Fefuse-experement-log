# Offline fixed token selector generation check

- examples: 0..19
- topk passages: 10
- rate: 0.15
- kv cache: `/raid/home/hming/fusionrag-pca-top1-top10-cache-20/data/musique-pca-subset-preprocess-10-revert_rope-True/Qwen2.5-7B-Instruct`
- fixed set source: `MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/phase1d_selector_set_methods_rate015/chunk_fixed_sets_npz`
- selectors: `online_fusionrag_qk`, `offline_qk_freq`, `offline_qk_mean`, `offline_draft_freq`, `offline_draft_mean`, `position_boundary`, `position_tail`, `random_chunk`

## Summary

```json
{
  "online_fusionrag_qk": {
    "count": 20,
    "rouge1_mean": 0.2434482742631629,
    "em_mean": 0.2,
    "wall_time_mean": 1.569997013336979,
    "ttft_mean": 0.41132737121079116,
    "storage_time_mean": 0.06038677915930748,
    "selection_time_mean": 0.11357042731251568,
    "reprocess_prefill_time_mean": 0.23737016473896802,
    "selected_doc_tokens_mean": 1103.2
  },
  "offline_qk_freq": {
    "count": 20,
    "rouge1_mean": 0.24921568455181467,
    "em_mean": 0.2,
    "wall_time_mean": 1.9659250350901858,
    "ttft_mean": 0.27204076743219047,
    "storage_time_mean": 0.05678724623285234,
    "selection_time_mean": 0.0,
    "reprocess_prefill_time_mean": 0.21525352119933813,
    "selected_doc_tokens_mean": 1098.6
  },
  "offline_qk_mean": {
    "count": 20,
    "rouge1_mean": 0.2493306270767519,
    "em_mean": 0.2,
    "wall_time_mean": 1.7376217833720147,
    "ttft_mean": 0.2717025579884648,
    "storage_time_mean": 0.056950916559435426,
    "selection_time_mean": 0.0,
    "reprocess_prefill_time_mean": 0.21475164142902942,
    "selected_doc_tokens_mean": 1098.6
  },
  "offline_draft_freq": {
    "count": 20,
    "rouge1_mean": 0.24921568455181467,
    "em_mean": 0.2,
    "wall_time_mean": 1.975701812771149,
    "ttft_mean": 0.2715765873901546,
    "storage_time_mean": 0.05671703999396414,
    "selection_time_mean": 0.0,
    "reprocess_prefill_time_mean": 0.21485954739619045,
    "selected_doc_tokens_mean": 1098.6
  },
  "offline_draft_mean": {
    "count": 20,
    "rouge1_mean": 0.24588235135737024,
    "em_mean": 0.2,
    "wall_time_mean": 1.8221420405898243,
    "ttft_mean": 0.2722387895220891,
    "storage_time_mean": 0.057286346517503264,
    "selection_time_mean": 0.0,
    "reprocess_prefill_time_mean": 0.2149524430045858,
    "selected_doc_tokens_mean": 1098.6
  },
  "position_boundary": {
    "count": 20,
    "rouge1_mean": 0.18702293512888213,
    "em_mean": 0.15,
    "wall_time_mean": 1.8580803586868568,
    "ttft_mean": 0.2715189496986568,
    "storage_time_mean": 0.05639569575432688,
    "selection_time_mean": 0.0,
    "reprocess_prefill_time_mean": 0.2151232539443299,
    "selected_doc_tokens_mean": 1098.6
  },
  "position_tail": {
    "count": 20,
    "rouge1_mean": 0.2419079924206115,
    "em_mean": 0.2,
    "wall_time_mean": 2.002711918996647,
    "ttft_mean": 0.27268446490634235,
    "storage_time_mean": 0.05638621635735035,
    "selection_time_mean": 0.0,
    "reprocess_prefill_time_mean": 0.216298248548992,
    "selected_doc_tokens_mean": 1098.6
  },
  "random_chunk": {
    "count": 20,
    "rouge1_mean": 0.22588235159737025,
    "em_mean": 0.2,
    "wall_time_mean": 1.77357672767248,
    "ttft_mean": 0.27229006262496114,
    "storage_time_mean": 0.05671059188898653,
    "selection_time_mean": 0.0,
    "reprocess_prefill_time_mean": 0.2155794707359746,
    "selected_doc_tokens_mean": 1098.6
  }
}
```
