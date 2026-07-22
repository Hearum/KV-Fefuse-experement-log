# Qwen3 Cross-Dataset Online QK / Draft Rate Sweep

## Goal

Run Qwen3-32B online recompute rate sweep for `online_qk` and `online_draft` on the three cross-dataset RAG tasks:

- `2wikimqa`
- `hotpotqa`
- `triviaqa`

This extends the previous MuSiQue-only rate sweep to the datasets in `cross_dataset_offline_generalization`.

## Settings

- Model: `/mnt/qjhs-sh-lab-01/models/Qwen3-32B`
- Draft model: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`
- BGE: `/mnt/qjhs-sh-lab-01/models/bge-m3`
- Data root: `MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/`
- Methods: `online_qk`, `online_draft`
- Rates: `0.0, 0.1, 0.3, 0.5, 0.8, 0.9`
- Segment size: 25 examples
- TopK docs: 10
- Preprocess: `true`
- Recall: `bge`
- Scope: `global`
- Judge: `GLM-5.2`
- Main script: `scripts/run_cross_dataset_qk_draft_rate_sweep.py`
- Summarizer: `scripts/summarize_cross_dataset_qk_draft_rate_sweep.py`

Total tasks:

- 2WikiMQA: 8 segments x 2 methods x 6 rates = 96 segment runs
- HotpotQA: 11 segments x 2 methods x 6 rates = 132 segment runs
- TriviaQA: 11 segments x 2 methods x 6 rates = 132 segment runs
- Total: 360 segment runs

## Output

Output root:

```bash
/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_20260714
```

Cache root:

```bash
/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-cross-dataset-rate-sweep-cache-20260714
```

Each worker uses a host/GPU/dataset-specific cache path to avoid cross-worker write conflicts.

## Launch Plan

Initial launch uses qjy001 and qjy003 idle GPUs. qjy000 is not used initially because it is already running the offline32b_top2 TriviaQA missing-segment rerun plus other active jobs.

Commands:

```bash
ssh qjy001 'cd /home/hming/FusionRAG-pca-analysis && FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_GPUS=0,1,2,3,4 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_SHARD_COUNT=2 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_SHARD_INDEX=0 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/scripts/run_cross_dataset_qk_draft_rate_sweep.py > /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_20260714/logs/launcher_qjy001_shard0.log 2>&1 < /dev/null &'

ssh qjy003 'cd /home/hming/FusionRAG-pca-analysis && FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_GPUS=0,1,2,3,4,7 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_SHARD_COUNT=2 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_SHARD_INDEX=1 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/scripts/run_cross_dataset_qk_draft_rate_sweep.py > /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_20260714/logs/launcher_qjy003_shard1.log 2>&1 < /dev/null &'
```

Summarize current finished segments:

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_20260714 \
python3 MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/scripts/summarize_cross_dataset_qk_draft_rate_sweep.py
```

## Status

Created on 2026-07-14. Not launched at document creation time.

## 2026-07-14 Launch Status

Commits:

- Initial launcher commit: `6455118`
- Quoting fix commit before successful launch: `281a651`

First launch attempt immediately failed before running segment inference because shell patching had stripped quotes in several `task[...]` expressions. No valid segment results were produced by that failed attempt.

Retry launch PIDs:

| Host | Shard | PID | GPUs |
|---|---:|---:|---|
| qjy001 / qjhs-sh-lab-02 | 0/2 | 207028 | 0,1,2,3,4 |
| qjy003 / qjhs-sh-lab-04 | 1/2 | 3830420 | 0,1,2,3,4,7 |

Retry commands:

```bash
ssh qjy001 'cd /home/hming/FusionRAG-pca-analysis && FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_GPUS=0,1,2,3,4 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_SHARD_COUNT=2 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_SHARD_INDEX=0 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/scripts/run_cross_dataset_qk_draft_rate_sweep.py > /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_20260714/logs/launcher_qjy001_shard0_retry1.log 2>&1 < /dev/null &'

ssh qjy003 'cd /home/hming/FusionRAG-pca-analysis && FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_GPUS=0,1,2,3,4,7 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_SHARD_COUNT=2 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_SHARD_INDEX=1 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/scripts/run_cross_dataset_qk_draft_rate_sweep.py > /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_20260714/logs/launcher_qjy003_shard1_retry1.log 2>&1 < /dev/null &'
```

Initial health check after retry:

- Worker logs created for qjy001 GPUs 0-4 and qjy003 GPUs 0-4/7.
- First tasks are `2wikimqa` `rate=0.0` for `online_qk` and `online_draft`.
- 11 run logs were created immediately after retry; no immediate Python traceback in the retry launcher logs.

## 2026-07-14 Static Extra Workers

Issue found: the first retry launch used only qjy001 GPUs 0-4 and qjy003 GPUs 0-4/7. Several GPUs stayed idle: qjy000 GPUs 3/4/5/7, qjy001 GPUs 5/6/7, and qjy003 GPUs 5/6. The original two-shard split was therefore under-utilizing available cards.

Fix: add `scripts/run_cross_dataset_qk_draft_rate_sweep_static_extra.py`, a simple static extra launcher. It accepts `FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_STRIDE` and `FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_OFFSET`, so each extra GPU gets a fixed slice of the global task list. To avoid duplicating the already-running first-wave segments, this extra launcher skips any task whose `run.log` already exists. Failed or incomplete segments will be handled by a later explicit rerun after summarization.

Extra launcher commit: `320aaa0`

Static extra workers launched:

| Host | GPU | offset/stride |
|---|---:|---:|
| qjy000 | 3 | 0/9 |
| qjy000 | 4 | 1/9 |
| qjy000 | 5 | 2/9 |
| qjy000 | 7 | 3/9 |
| qjy001 | 5 | 4/9 |
| qjy001 | 6 | 5/9 |
| qjy001 | 7 | 6/9 |
| qjy003 | 5 | 7/9 |
| qjy003 | 6 | 8/9 |

Initial health after extra launch:

- All 9 extra workers started.
- Output run logs increased from 11 to 20.
- Extra workers skipped the already-started first-wave segments and began fixed-slice tasks such as `2wikimqa online_qk rate=0.1` and remaining `2wikimqa online_draft rate=0.0` segments.

## 2026-07-14 Resume With Original Two-Shard Plan

After the later dataset-balanced launcher attempts were found to be unreliable, the cross-dataset rate sweep was resumed with the original two-shard launcher plan recorded above. The resumed run intentionally uses the stable `run_cross_dataset_qk_draft_rate_sweep.py` interface and the original output/cache roots, so completed segment logs can be reused and incomplete segments can be overwritten by the normal launcher.

Commit at launch time:

- `5bacddf`

Launch commands:

```bash
ssh qjy001 'cd /home/hming/FusionRAG-pca-analysis && mkdir -p /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_20260714/logs && FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_GPUS=0,1,2,3,4 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_SHARD_COUNT=2 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_SHARD_INDEX=0 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/scripts/run_cross_dataset_qk_draft_rate_sweep.py > /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_20260714/logs/launcher_qjy001_shard0_resume_oldplan_20260714.log 2>&1 < /dev/null &'

ssh qjy003 'cd /home/hming/FusionRAG-pca-analysis && mkdir -p /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_20260714/logs && FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_GPUS=0,1,2,3,4,7 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_SHARD_COUNT=2 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_SHARD_INDEX=1 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/scripts/run_cross_dataset_qk_draft_rate_sweep.py > /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_20260714/logs/launcher_qjy003_shard1_resume_oldplan_20260714.log 2>&1 < /dev/null &'
```

Initial health check:

- qjy001 launcher PID: `366587`; active GPUs: `0,1,2,3,4`.
- qjy003 launcher PID: `3977655`; active GPUs: `0,1,2,3,4,7`.
- First live tasks are `2wikimqa online_qk/online_draft rate=0.0` segments.
- No immediate traceback, CUDA OOM, or killed error was found in the new launcher logs at startup.

Note: older aborted static-extra worker logs remain under the same log directory and include `rc=-9` entries from intentionally stopped workers. For final summarization, use completed segment run logs and CSVs under the output root rather than stale worker tails.

## 2026-07-14 Resume Static Extra Workers After Old Plan

The original two-shard plan only occupied qjy001 GPUs `0-4` and qjy003 GPUs `0-4,7`, leaving many cards idle. Static extra workers were therefore relaunched on the idle GPUs using the previously added `run_cross_dataset_qk_draft_rate_sweep_static_extra.py` script. This script skips any task that already has a `run.log`, so it should not duplicate currently active two-shard segments.

Commit at launch time:

- `a041aab`

Extra worker launch allocation:

| Host | GPUs | stride/offsets |
|---|---|---|
| qjy000 | `2,3,4,5,6,7` | `11 / 0..5` |
| qjy001 | `5,6,7` | `11 / 6..8` |
| qjy003 | `5,6` | `11 / 9..10` |

Representative launch command pattern:

```bash
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_GPUS=<gpu> \
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_STRIDE=11 \
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_OFFSET=<offset> \
nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/scripts/run_cross_dataset_qk_draft_rate_sweep_static_extra.py \
  > /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_20260714/logs/launcher_<host>_gpu<gpu>_static_extra_resume2.log 2>&1 < /dev/null &
```

Initial health check after this launch:

- Static extra launcher logs: `11`.
- Output root run logs: `59`.
- CSV count: `27`.
- No immediate traceback, CUDA OOM, or killed error was found in the new `resume2` launcher logs.
- qjy000 GPUs `2-7`, qjy001 GPUs `0-7`, and qjy003 GPUs `0-7` had active memory allocation after launch. qjy000 GPU `1` remained occupied by the earlier offline32b_top2 TriviaQA missing-segment rerun.

## 2026-07-14 Abort Current Cross-Dataset Sweep Due To Full Disk

Status update after querying partial results:

- Output root: `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_20260714`
- Cache root: `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-cross-dataset-rate-sweep-cache-20260714`
- `/mnt/qjhs-sh-lab-03` was full: `14T used, 11G available, 100%`.
- Multiple run logs contained `OSError: [Errno 28] No space left on device`.
- Several cache tensor reads failed with messages such as `*.pt cannot be opened` and `PytorchStreamReader failed reading zip archive`, consistent with partial/corrupted cache writes after the disk filled.
- The current run was stopped to avoid generating more corrupted cache/results.

Partial 2WikiMQA summary before abort, not valid as a final result because some high-rate segments failed or were incomplete:

| Dataset | Method | Rate | CSV files | Unique rows | Main Acc | Avg F1 | Avg EM | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 2wikimqa | online_qk | 0.0 | 8 | 200 | 0.390 | 0.302 | 0.185 | complete-looking |
| 2wikimqa | online_qk | 0.1 | 8 | 200 | 0.455 | 0.330 | 0.190 | complete-looking |
| 2wikimqa | online_qk | 0.3 | 4 | 100 | 0.490 | 0.366 | 0.230 | incomplete |
| 2wikimqa | online_qk | 0.5 | 8 | 200 | 0.495 | 0.393 | 0.230 | complete-looking |
| 2wikimqa | online_qk | 0.8 | 7 | 175 | 0.514 | 0.403 | 0.269 | incomplete |
| 2wikimqa | online_qk | 0.9 | 2 | 50 | 0.540 | 0.407 | 0.240 | incomplete |
| 2wikimqa | online_draft | 0.0 | 8 | 200 | 0.390 | 0.302 | 0.185 | complete-looking |
| 2wikimqa | online_draft | 0.1 | 7 | 175 | 0.469 | 0.345 | 0.206 | incomplete |
| 2wikimqa | online_draft | 0.3 | 4 | 100 | 0.480 | 0.336 | 0.170 | incomplete |
| 2wikimqa | online_draft | 0.5 | 8 | 200 | 0.510 | 0.372 | 0.215 | complete-looking, raw rows include duplicates |
| 2wikimqa | online_draft | 0.8 | 3 | 75 | 0.507 | 0.341 | 0.187 | incomplete |
| 2wikimqa | online_draft | 0.9 | 1 | 25 | 0.520 | 0.407 | 0.320 | incomplete |

HotpotQA and TriviaQA had no usable summarized rows from this run before the disk/corruption issue. Some HotpotQA segment logs existed, but they hit the same cache open / no-space errors and should not be used as results.

Action required before rerun:

1. Do not continue using `/mnt/qjhs-sh-lab-03` for new cache/output until space is freed.
2. Move the next rerun to a filesystem with enough free space, for example `/mnt/qjhs-sh-lab-01` which had about `2.5T` free at the time of this check.
3. Use fresh cache/output roots for the rerun, because the current cache root contains partially written tensor files.
4. Prefer dataset-specific scheduling so HotpotQA and TriviaQA start immediately instead of waiting behind all 2WikiMQA tasks.

## Cache Organization Rule For Future Runs

The cache policy is now documented in [`CACHE_POLICY.md`](CACHE_POLICY.md). The core rule is that KV cache must be keyed by semantic reuse dimensions, not by worker identity.

Correct formal-run cache layout:

```text
CACHE_ROOT/Qwen3-32B/<dataset>/kv_cache
CACHE_ROOT/Qwen3-32B/<dataset>/preprocess_kv_cache_global_topk10_bge
```

Incorrect formal-run cache layout:

```text
CACHE_ROOT/<host>/worker_gpu<gpu>/<dataset>/Qwen3-32B/<dataset>/...
```

The previous worker-local layout caused each GPU worker to build a separate copy of the same raw KV and preprocess KV. This is what expanded the cross-dataset sweep cache to `7.3T`.

Launcher changes after this finding:

- `run_cross_dataset_qk_draft_rate_sweep.py` now defaults to `FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_LAYOUT=shared`.
- `run_cross_dataset_qk_draft_rate_sweep_static_extra.py` now defaults to `FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_LAYOUT=shared`.
- The old per-worker layout is still available only for isolated debugging via `FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_LAYOUT=worker`.
- Formal evaluation should set `FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_READONLY=1` after cache build, so rate/method workers cannot write or corrupt cache files.

Formal rerun should be organized in two phases:

1. Build the shared cache once per `(model, dataset, topk, recall_method, preprocess_scope)`.
2. Run all `rate` and `method` evaluations against that cache in readonly mode.

Do not reuse `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-cross-dataset-rate-sweep-cache-20260714` as a formal cache root; it contains partial/corrupted `.pt` files from the full-disk failure.

## 2026-07-14 Shared-Cache Rerun Relaunch

The corrupt worker-local cache root was removed as requested:

```bash
rm -rf -- /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-cross-dataset-rate-sweep-cache-20260714
```

Disk status after cleanup:

```text
/mnt/qjhs-sh-lab-03: 14T total, 6.8T used, 7.3T available, 49% used
```

Shared-cache rerun paths:

```text
Output root: /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_sharedcache_20260714
Cache root:  /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-cross-dataset-shared-cache-20260714
```

Phase 1 is cache build only. It runs `online_qk rate=0.0` because raw KV and preprocess KV are reusable across later rate/method sweeps. The three datasets are launched simultaneously, one dataset per host:

| Host | Dataset | PID | GPUs | Cache layout | Readonly |
|---|---|---:|---|---|---|
| qjy000 | 2wikimqa | `2348126` | `0-7` | shared | `0` |
| qjy001 | hotpotqa | `1965498` | `0-7` | shared | `0` |
| qjy003 | triviaqa | `1199231` | `0-7` | shared | `0` |

Common launch settings:

```bash
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_LAYOUT=shared
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_READONLY=0
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_METHODS=online_qk
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_RATES=0.0
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_GPUS=0,1,2,3,4,5,6,7
```

Initial health check:

- qjy000, qjy001, and qjy003 all had active GPU memory allocation on all eight GPUs.
- Run logs by dataset after startup: `2wikimqa=8`, `hotpotqa=8`, `triviaqa=8`.
- Shared cache size after startup: `6.1G`; later quick check showed `82G`.
- No immediate traceback, CUDA OOM, no-space, `PytorchStreamReader`, or `.pt cannot be opened` error was found in the new shared-cache logs.

Next step after Phase 1 completes: launch the full `online_qk,online_draft` x `0.0,0.1,0.3,0.5,0.8,0.9` sweep with `FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_READONLY=1` against the same shared cache root.

## 2026-07-14 Canonical Readonly Full Rate Sweep

After the Qwen3-32B canonical cache root was finalized, the full cross-dataset rate sweep was launched in readonly mode.

Output root:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_canonical_20260714
```

Cache root:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache
```

Settings:

```bash
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_LAYOUT=shared
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_READONLY=1
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_METHODS=online_qk,online_draft
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_RATES=0.0,0.1,0.3,0.5,0.8,0.9
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_GPUS=0,1,2,3,4,5,6,7
```

Dataset-to-host allocation:

| Host | Dataset | PID | GPUs |
|---|---|---:|---|
| qjy000 | 2wikimqa | `2872184` | `0-7` |
| qjy001 | hotpotqa | `2195232` | `0-7` |
| qjy003 | triviaqa | `1402604` | `0-7` |

Initial health check:

- All three launcher logs were created under the output root.
- Worker logs report `cache_root=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache`, `cache_layout=shared`, `cache_readonly=1`.
- Startup run logs by dataset: `2wikimqa=8`, `hotpotqa=8`, `triviaqa=8`.
- No immediate `Traceback`, `RuntimeError`, readonly cache miss, no-space, `PytorchStreamReader`, `.pt cannot be opened`, or CUDA OOM error was found at startup.

MuSiQue note:

The earlier MuSiQue rate sweep at `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate_sweep_reusecache_20260714` is already complete with `96/96` segment logs containing `FINAL RESULTS`.

## 2026-07-14 Canonical Readonly Rate 0.15 Add-On

Purpose: fill the missing `rate=0.15` point for both formal online methods so it can be compared with the existing `0.0,0.1,0.3,0.5,0.8,0.9` sweep.

Commit at launch: `1afba6f`

Output root:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate015_canonical_20260714
```

Cache root:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache
```

Launch settings:

```bash
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate015_canonical_20260714
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_LAYOUT=shared
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_READONLY=1
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_METHODS=online_qk,online_draft
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_RATES=0.15
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_GPUS=0,1,2,3,4,5,6,7
```

Planned allocation:

| Host | Dataset(s) | Notes |
|---|---|---|
| qjy001 | `2wikimqa,hotpotqa` | shared canonical cache, readonly |
| qjy003 | `triviaqa` | shared canonical cache, readonly |

Expected tasks: `2 methods x 1 rate x (8+11+11) segments = 60`.

Actual launch:

| Host | Dataset(s) | PID | GPUs |
|---|---|---:|---|
| qjy001 | `2wikimqa,hotpotqa` | `2601147` | `0-7` |
| qjy003 | `triviaqa` | `1774723` | `0-7` |

Initial health check:

- Cross output root exists and has `16` run logs immediately after startup.
- Worker logs report `cache_root=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache`, `cache_layout=shared`, `cache_readonly=1`.
- No early `Traceback`, CUDA OOM, no-space, `PytorchStreamReader`, `.pt cannot be opened`, or readonly cache miss was found.

## 2026-07-14 Rate Sweep Figures By Dataset

Request: plot the Qwen3-32B Online QK / Online Draft rate sweep on each dataset.

Source result roots:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_canonical_20260714
/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate015_canonical_20260714
/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate_sweep_reusecache_20260714
/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate015_canonical_20260714
```

Rates included: `0.0, 0.1, 0.15, 0.3, 0.5, 0.8, 0.9`.

Plot command:

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/scripts/plot_qwen3_rate_sweep_by_dataset.py
```

Generated table:

```text
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/qwen3_online_qk_draft_rate_sweep_by_dataset_with_rate015.csv
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/qwen3_online_qk_draft_rate_sweep_by_dataset_with_rate015.json
```

Generated figures:

```text
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/figures/qwen3_rate_sweep_2wikimqa.png
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/figures/qwen3_rate_sweep_hotpotqa.png
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/figures/qwen3_rate_sweep_triviaqa.png
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/figures/qwen3_rate_sweep_musique.png
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/figures/qwen3_rate_sweep_main_acc_all_datasets.png
```

Each per-dataset figure has four panels: Main Acc, Sub Acc, F1, and EM. The summary grid plots Main Acc for all datasets.

Quick observations from the plotted table:

- 2WikiMQA improves gradually with rate but remains the weakest dataset; best Main Acc is around 52.5% for Online Draft at rate 0.9.
- HotpotQA is already high at rate 0 and has small/non-monotonic gains; the best Main Acc is 86.54% for Online QK rate 0.9 or Online Draft rates 0.1/0.5/0.8.
- TriviaQA benefits strongly from modest recompute; Online Draft peaks at rate 0.15 for Main Acc (91.11%), while F1/EM peak at higher rates.
- MuSiQue benefits most from higher rate; Online Draft reaches 83.70% Main Acc at rate 0.9.

## 2026-07-14 Final Record: Qwen3-32B Cross-Dataset Rate Sweep Rerun

This section is the final index for the completed Qwen3-32B cross-dataset Online QK / Online Draft rate-sweep rerun.

### Scope

Datasets:

```text
2wikimqa, hotpotqa, triviaqa
```

Methods:

```text
online_qk, online_draft
```

Rates:

```text
0.0, 0.1, 0.15, 0.3, 0.5, 0.8, 0.9
```

Model and cache:

```text
Model: /mnt/qjhs-sh-lab-01/models/Qwen3-32B
Draft model: /mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct
Canonical cache root: /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache
Cache mode: shared, readonly for formal evaluation
```

### Result Roots

Main sweep, rates `0.0,0.1,0.3,0.5,0.8,0.9`:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_canonical_20260714
```

Rate `0.15` add-on:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate015_canonical_20260714
```

### Completion

| Result root | Expected tasks | FINAL RESULTS | CSV files | Error scan |
|---|---:|---:|---:|---|
| `qwen3_cross_dataset_online_qk_draft_rate_sweep_canonical_20260714` | 360 | 360 | 360 | no Traceback/OOM/no-space/cache-miss found |
| `qwen3_cross_dataset_online_qk_draft_rate015_canonical_20260714` | 60 | 60 | 60 | no Traceback/OOM/no-space/cache-miss found |

Expected task counts:

```text
Main sweep: 3 datasets x 2 methods x 6 rates x dataset segments = 360
Rate 0.15: 3 datasets x 2 methods x 1 rate x dataset segments = 60
```

### Launch Commands

Main sweep launch, one dataset per host:

```bash
# qjy000: 2WikiMQA
cd /raid/home/hming/FusionRAG-pca-analysis
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_canonical_20260714 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_LAYOUT=shared FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_READONLY=1 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_DATASETS=2wikimqa FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_METHODS=online_qk,online_draft FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_RATES=0.0,0.1,0.3,0.5,0.8,0.9 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_GPUS=0,1,2,3,4,5,6,7 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/scripts/run_cross_dataset_qk_draft_rate_sweep.py   > /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate_sweep_canonical_20260714/launcher_qjy000.nohup.log 2>&1 &

# qjy001: HotpotQA, same env except FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_DATASETS=hotpotqa
# qjy003: TriviaQA, same env except FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_DATASETS=triviaqa
```

Rate `0.15` add-on launch:

```bash
# qjy001: 2WikiMQA + HotpotQA
cd /home/hming/FusionRAG-pca-analysis
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate015_canonical_20260714 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_LAYOUT=shared FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_READONLY=1 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_DATASETS=2wikimqa,hotpotqa FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_METHODS=online_qk,online_draft FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_RATES=0.15 FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_GPUS=0,1,2,3,4,5,6,7 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/scripts/run_cross_dataset_qk_draft_rate_sweep.py   > /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate015_canonical_20260714/launcher_qjy001.nohup.log 2>&1 &

# qjy003: TriviaQA, same env except FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_DATASETS=triviaqa
```

### Summary Files And Figures

Combined table with `rate=0.15` merged into the sweep:

```text
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/qwen3_online_qk_draft_rate_sweep_by_dataset_with_rate015.csv
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/qwen3_online_qk_draft_rate_sweep_by_dataset_with_rate015.json
```

Plot script:

```text
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/scripts/plot_qwen3_rate_sweep_by_dataset.py
```

Figures:

```text
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/figures/qwen3_rate_sweep_2wikimqa.png
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/figures/qwen3_rate_sweep_hotpotqa.png
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/figures/qwen3_rate_sweep_triviaqa.png
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/figures/qwen3_rate_sweep_main_acc_all_datasets.png
```

### Final Main Accuracy Table

| Dataset | Method | r=0.0 | r=0.1 | r=0.15 | r=0.3 | r=0.5 | r=0.8 | r=0.9 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2WikiMQA | online_qk | 39.00 | 45.50 | 47.50 | 49.50 | 49.50 | 52.00 | 51.00 |
| 2WikiMQA | online_draft | 39.00 | 46.00 | 47.00 | 50.00 | 51.00 | 51.00 | 52.50 |
| HotpotQA | online_qk | 83.08 | 84.62 | 84.62 | 86.15 | 84.23 | 85.77 | 86.54 |
| HotpotQA | online_draft | 82.31 | 86.54 | 85.77 | 85.77 | 86.54 | 86.54 | 85.38 |
| TriviaQA | online_qk | 87.04 | 86.67 | 88.15 | 90.00 | 90.74 | 91.11 | 90.00 |
| TriviaQA | online_draft | 86.67 | 87.78 | 91.11 | 89.63 | 90.37 | 89.26 | 89.26 |

### Notes

- This folder records only the cross-dataset part: 2WikiMQA, HotpotQA, TriviaQA.
- The companion MuSiQue sweep is recorded under `MOTIVATION_EXPERIMENTS/qwen3_rate_sweep_online_qk_draft/` and is merged only for the all-dataset plotting script.
- Batch provenance for the different Online QK/Draft rate=0.15 datasets is separately indexed in `MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/online_qk_draft_rate015_data_registry.md`.

