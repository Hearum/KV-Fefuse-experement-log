# Qwen3 MuSiQue Online QK/Draft Rate Sweep

## 2026-07-14 launch plan

Goal:
- Rerun the two online methods after the judge-cleaning audit: Online QK and Online DraftModel.
- Sweep recompute rates `0.0, 0.1, 0.3, 0.5, 0.8, 0.9` on MuSiQue with Qwen3-32B.

Methods:
- `online_qk`: `--reprocess_method FusionRAG`
- `online_draft`: `--reprocess_method DraftModel --draft_model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`

Shared settings:
- model: `/mnt/qjhs-sh-lab-01/models/Qwen3-32B`
- dataset: `data/result_reflect.json`
- dataset name: `musique`
- samples: `0-200`, split into 8 segments of 25 examples
- topk: `10`
- preprocess: `true`
- recall: `bge`
- preprocess scope: `global`
- BGE model: `/mnt/qjhs-sh-lab-01/models/bge-m3`
- judge: `GLM-5.2`

Runtime commit:
- `3ec6deda9df1afc38f070fae199fdcff910c1a9c`

Output root:
- `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate_sweep_20260714`

Cache root:
- `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-rate-sweep-cache-20260714`

Cache isolation:
- Each worker uses `${CACHE_ROOT}/${host}/worker_gpu${gpu}/musique`.
- `FUSIONRAG_PREPROCESS_CACHE_READONLY` is unset so missing cache can be generated, but workers do not write the same cache subtree.

Launcher:
- `scripts/run_musique_qk_draft_rate_sweep.py`

Launch command:

```bash
ssh qjy000 'cd /raid/home/hming/FusionRAG-pca-analysis && FUSIONRAG_QWEN3_RATE_SWEEP_GPUS=0,1,2,3,4,5,6,7 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_rate_sweep_online_qk_draft/scripts/run_musique_qk_draft_rate_sweep.py > /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate_sweep_20260714/logs/launcher_qjy000.log 2>&1 < /dev/null &'
```

Expected task count:
- 6 rates x 2 methods x 8 segments = 96 segment runs.

Status:
- Not launched yet at document creation time.

## 2026-07-14 actual launch

Initial qjy000-only launch:
- Started briefly, then stopped to avoid running all 96 tasks on one host while qjy001/qjy003 were idle.
- The parent launcher and orphaned child segment processes were terminated before the sharded launch.
- No partial output directories were deleted; incomplete segments will be rerun because the launcher only skips logs containing `FINAL RESULTS` plus the expected CSV.

Sharded launch:

| Host | Shard | PID | GPUs |
|---|---:|---:|---|
| qjy000 / qjhs-sh-lab-01 | 0/3 | 1993151 | 0,1,2,3,4,5,6,7 |
| qjy001 / qjhs-sh-lab-02 | 1/3 | 4166695 | 0,1,2,3,4,5,6,7 |
| qjy003 / qjhs-sh-lab-04 | 2/3 | 3614797 | 0,1,2,3,4,5,6,7 |

Commands:

```bash
ssh qjy000 'cd /raid/home/hming/FusionRAG-pca-analysis && FUSIONRAG_QWEN3_RATE_SWEEP_GPUS=0,1,2,3,4,5,6,7 FUSIONRAG_QWEN3_RATE_SWEEP_SHARD_COUNT=3 FUSIONRAG_QWEN3_RATE_SWEEP_SHARD_INDEX=0 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_rate_sweep_online_qk_draft/scripts/run_musique_qk_draft_rate_sweep.py > /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate_sweep_20260714/logs/launcher_qjy000_shard0.log 2>&1 < /dev/null &'

ssh qjy001 'cd /home/hming/FusionRAG-pca-analysis && FUSIONRAG_QWEN3_RATE_SWEEP_GPUS=0,1,2,3,4,5,6,7 FUSIONRAG_QWEN3_RATE_SWEEP_SHARD_COUNT=3 FUSIONRAG_QWEN3_RATE_SWEEP_SHARD_INDEX=1 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_rate_sweep_online_qk_draft/scripts/run_musique_qk_draft_rate_sweep.py > /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate_sweep_20260714/logs/launcher_qjy001_shard1.log 2>&1 < /dev/null &'

ssh qjy003 'cd /home/hming/FusionRAG-pca-analysis && FUSIONRAG_QWEN3_RATE_SWEEP_GPUS=0,1,2,3,4,5,6,7 FUSIONRAG_QWEN3_RATE_SWEEP_SHARD_COUNT=3 FUSIONRAG_QWEN3_RATE_SWEEP_SHARD_INDEX=2 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_rate_sweep_online_qk_draft/scripts/run_musique_qk_draft_rate_sweep.py > /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate_sweep_20260714/logs/launcher_qjy003_shard2.log 2>&1 < /dev/null &'
```

Initial status:
- All three hosts started worker logs under `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate_sweep_20260714/logs/`.
- Each worker owns 4 segment runs after sharding.
- First tasks include both `online_qk` and `online_draft` at `rate=0.0`, plus early `rate=0.1` segments.

## 2026-07-14 relaunch with reused preprocess cache

Reason:
+- The first sharded launch used an empty cache root: `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-rate-sweep-cache-20260714`.
+- It immediately spent most time in on-demand BGE preprocess KV generation, which is not the target of this rate sweep.
+- That launch was stopped. Its incomplete output root is kept for provenance and should not be used for metrics: `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate_sweep_20260714`.
+
+Relaunch settings:
+- Output root: `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate_sweep_reusecache_20260714`
+- Cache root: `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-3host-sharded-20260714`
+- Sharding: same 3-host split as above.
+
+Relaunch PIDs:
+
+| Host | Shard | PID | GPUs |
+|---|---:|---:|---|
+| qjy000 / qjhs-sh-lab-01 | 0/3 | 2044936 | 0,1,2,3,4,5,6,7 |
+| qjy001 / qjhs-sh-lab-02 | 1/3 | 4191780 | 0,1,2,3,4,5,6,7 |
+| qjy003 / qjhs-sh-lab-04 | 2/3 | 3638052 | 0,1,2,3,4,5,6,7 |
+
+Relaunch commands used the same launcher with environment overrides:
+
+```bash
+FUSIONRAG_QWEN3_RATE_SWEEP_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate_sweep_reusecache_20260714
+FUSIONRAG_QWEN3_RATE_SWEEP_CACHE_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-3host-sharded-20260714
+FUSIONRAG_QWEN3_RATE_SWEEP_SHARD_COUNT=3
+FUSIONRAG_QWEN3_RATE_SWEEP_SHARD_INDEX={0,1,2}
+FUSIONRAG_QWEN3_RATE_SWEEP_GPUS=0,1,2,3,4,5,6,7
+```
+
+Initial health check:
+- 24 segment `run.log` files created in the reusecache output root.
+- All three hosts entered `run` state after old orphan workers were cleared.
+- No `Traceback`, `CUDA out of memory`, `Killed`, or `FileNotFoundError` found in the reusecache output root at the initial check.
+- `On-demand: Generating cache` still appears, so the reused cache is not perfectly complete for this task allocation, but it is now running rather than blocked by the old empty-cache workers.
+- No complete segment yet at the initial health check (`FINAL RESULTS`: 0). Metrics should be summarized only after segments complete.
+
+Summarizer:
+- `scripts/summarize_musique_qk_draft_rate_sweep.py`
+- It ignores unfinished segment CSVs by requiring the sibling `run.log` to contain `FINAL RESULTS`.
+

## 2026-07-14 MuSiQue Shared Cache Consolidation

Old MuSiQue cache roots were worker-local and duplicated data by host/GPU. A consolidated shared cache was built with hard links from the existing cache files, so no full data copy was made.

New shared cache root:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-shared-cache-20260714
```

Layout:

```text
fusionrag-qwen3-musique-shared-cache-20260714/
  Qwen3-32B/
    musique/
      kv_cache/
      preprocess_kv_cache_global_topk10_bge/
      merge_report.json
```

Source priority:

1. `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-3host-sharded-20260714`
2. `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-qjy000-lab03-v2-20260714`
3. `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-qjy001-lab03-v2-20260714`
4. `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-qjy003-lab03-v2-20260714`
5. `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-rate-sweep-cache-20260714`

Merge result:

| Subdir | Unique files | Key files | Value files |
|---|---:|---:|---:|
| `kv_cache` | 6836 | 3418 | 3418 |
| `preprocess_kv_cache_global_topk10_bge` | 4466 | 2233 | 2233 |

Consolidated cache size:

```text
235G
```

No `worker_gpu*` or `qjhs-sh-lab-*` directories exist under the new shared cache root. The merge report found one duplicate size mismatch, but the conflicting old copy was a zero-byte file; the nonzero file from the `3host-sharded` source was selected.

Launcher update:

- `scripts/run_musique_qk_draft_rate_sweep.py` now defaults to this shared cache root.
- It supports `FUSIONRAG_QWEN3_RATE_SWEEP_CACHE_LAYOUT=shared` by default.
- The old worker-local layout is only for debug via `FUSIONRAG_QWEN3_RATE_SWEEP_CACHE_LAYOUT=worker`.
- Formal evaluation should use `FUSIONRAG_QWEN3_RATE_SWEEP_CACHE_READONLY=1`.

Recommended formal run settings:

```bash
FUSIONRAG_QWEN3_RATE_SWEEP_CACHE_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-shared-cache-20260714
FUSIONRAG_QWEN3_RATE_SWEEP_CACHE_LAYOUT=shared
FUSIONRAG_QWEN3_RATE_SWEEP_CACHE_READONLY=1
```

After validating that this shared cache works for the next MuSiQue run, the old worker-local MuSiQue cache roots can be deleted to reclaim space while keeping the consolidated hard-linked data alive.

## 2026-07-14 Cleanup After MuSiQue Shared Cache Consolidation

After building and validating the consolidated MuSiQue shared cache, the old worker-local MuSiQue cache roots were removed.

Kept:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-shared-cache-20260714  235G
```

Removed:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-3host-sharded-20260714
/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-qjy000-lab03-v2-20260714
/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-qjy001-lab03-v2-20260714
/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-qjy003-lab03-v2-20260714
/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-rate-sweep-cache-20260714
```

Cleanup command:

```bash
rm -rf -- \
  /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-3host-sharded-20260714 \
  /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-qjy000-lab03-v2-20260714 \
  /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-qjy001-lab03-v2-20260714 \
  /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-qjy003-lab03-v2-20260714 \
  /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-rate-sweep-cache-20260714
```

Disk status:

```text
before: /mnt/qjhs-sh-lab-03 14T total, 7.7T used, 6.4T available, 55% used
after:  /mnt/qjhs-sh-lab-03 14T total, 4.9T used, 9.1T available, 35% used
```

The retained shared cache still has the expected layout and no worker-local directories:

```text
Qwen3-32B/musique/kv_cache
Qwen3-32B/musique/preprocess_kv_cache_global_topk10_bge
```

Missing cache entries, if any, should be generated by the pipeline under this shared root in future non-readonly cache-build runs. Formal evaluation should still use readonly mode.

## 2026-07-14 Canonical Readonly Rate 0.15 Add-On

Purpose: fill the missing `rate=0.15` point for MuSiQue under both online methods.

Commit at launch: `1afba6f`

Output root:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate015_canonical_20260714
```

Cache root:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache
```

Launch settings:

```bash
FUSIONRAG_QWEN3_RATE_SWEEP_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate015_canonical_20260714
FUSIONRAG_QWEN3_RATE_SWEEP_CACHE_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache
FUSIONRAG_QWEN3_RATE_SWEEP_CACHE_LAYOUT=shared
FUSIONRAG_QWEN3_RATE_SWEEP_CACHE_READONLY=1
FUSIONRAG_QWEN3_RATE_SWEEP_METHODS=online_qk,online_draft
FUSIONRAG_QWEN3_RATE_SWEEP_RATES=0.15
FUSIONRAG_QWEN3_RATE_SWEEP_GPUS=0,1,2,3,4,5,6,7
```

Planned allocation: qjy000, shared canonical cache, readonly. Expected tasks: `2 methods x 1 rate x 8 segments = 16`.

Actual launch:

| Host | Dataset | PID | GPUs |
|---|---|---:|---|
| qjy000 | `musique` | `3724486` | `0,2,3,4,7` |

Initial health check:

- MuSiQue output root exists and has `5` run logs immediately after startup.
- Worker logs report `cache_root=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache`, `cache_layout=shared`, `cache_readonly=1`.
- No early `Traceback`, CUDA OOM, no-space, `PytorchStreamReader`, `.pt cannot be opened`, or readonly cache miss was found.

