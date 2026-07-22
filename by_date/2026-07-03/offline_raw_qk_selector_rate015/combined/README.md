# Combined Offline Raw-QK Fixed Sets

- dataset: data/result_reflect.json
- rate: 0.15
- calibration queries per example: 8, from other examples only
- selector KV: raw document KV cache
- raw KV cache: `/raid/home/hming/fusionrag-reflect-full-cache/Qwen2.5-7B-Instruct/musique/kv_cache`
- methods stored in npz: `qk_frequency_per_chunk`, `qk_mean_score_per_chunk`
- generated from 8 shards under `../shard_*`
- fixed-set files: 135 npz, matching should_test examples
