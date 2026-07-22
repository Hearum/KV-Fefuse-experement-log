# Offline fixed token selector generation check

- examples: 0..19
- topk passages: 10
- rate: 0.15
- kv cache: `/raid/home/hming/fusionrag-pca-top1-top10-cache-20/data/musique-pca-subset-preprocess-10-revert_rope-True/Qwen2.5-7B-Instruct`
- fixed set source: `/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/phase1b_chunk_level_qk/chunk_fixed_sets_npz`
- selectors: `online_fusionrag_qk`, `offline_chunk_qk`, `random_chunk`

## Summary

```json
{
  "online_fusionrag_qk": {
    "count": 20,
    "rouge1_mean": 0.2434482742631629,
    "em_mean": 0.2,
    "wall_time_mean": 1.556822544708848,
    "ttft_mean": 0.4116435744101182,
    "storage_time_mean": 0.059947035671211776,
    "selection_time_mean": 0.11393944937735796,
    "reprocess_prefill_time_mean": 0.23775708936154843,
    "selected_doc_tokens_mean": 1103.2
  },
  "offline_chunk_qk": {
    "count": 20,
    "rouge1_mean": 0.24921568455181467,
    "em_mean": 0.2,
    "wall_time_mean": 1.9450330427149312,
    "ttft_mean": 0.2705979062709957,
    "storage_time_mean": 0.05563571958336979,
    "selection_time_mean": 0.0,
    "reprocess_prefill_time_mean": 0.21496218668762596,
    "selected_doc_tokens_mean": 1098.6
  },
  "random_chunk": {
    "count": 20,
    "rouge1_mean": 0.22588235159737025,
    "em_mean": 0.2,
    "wall_time_mean": 1.7558591478038579,
    "ttft_mean": 0.2709764037514105,
    "storage_time_mean": 0.055881472979672255,
    "selection_time_mean": 0.0,
    "reprocess_prefill_time_mean": 0.2150949307717383,
    "selected_doc_tokens_mean": 1098.6
  }
}
```
