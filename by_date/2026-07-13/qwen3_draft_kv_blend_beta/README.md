# Qwen3 DraftModel KV Blend Beta Probe

## Goal

Test whether selected-token recompute must fully overwrite cached document KV, or whether partial update is better.

Selection is unchanged: native online DraftModel, rate=0.15. Only the writeback of recomputed selected document KV is changed.

Formula:

```text
KV_after = beta * KV_cached + (1 - beta) * KV_recomputed
```

Definitions:

- `beta=0`: native full selected-token KV update, equivalent to current DraftModel recompute behavior.
- `beta=1`: selected tokens are recomputed but their document KV is restored to cached KV, equivalent to no document KV update for selected tokens.
- `0<beta<1`: partial update.

Environment interface:

```bash
FUSIONRAG_REPROCESS_KV_BLEND_BETA=<0..1>
FUSIONRAG_REPROCESS_KV_BLEND_MODE=kv|key|value
```

Default is `beta=0`, `mode=kv`, so existing experiments are unchanged.

## 2026-07-13 Implementation

Modified:

- `ktransformers/util/utils.py`

Implementation point:

- `load_kv_and_generate(...)` after selected document tokens are recomputed and before query/decode uses the updated `past_key_values`.
- The code saves the original selected doc-token K/V, runs recompute, then writes back the beta blend for selected document positions only.
- Existing `FUSIONRAG_REPROCESS_UPDATE_MODE=k_only|v_only|none` restore logic remains compatible; restore semantics take precedence for the side that is explicitly disabled.

Sanity checks planned:

1. MuSiQue small subset `0-25`, mode `kv`, beta in `0, 0.25, 0.5, 0.75, 1.0`.
2. If sane, expand to full MuSiQue and then cross-dataset.
3. If `kv` shows signal, split `key` vs `value` because Value updates are expected to matter more.


## 2026-07-13 Full-Sample Parallel Sweep

User request: run the KV blend beta experiment on all samples, not only the 0-50 Hotpot probe, and parallelize on idle qjy machines.

Experiment definition:
- Baseline pipeline: native online `DraftModel`, `rate=0.15`, `preprocess=true`, `recall_method=bge`, `preprocess_scope=global`, Qwen3-32B main model, Qwen2.5-3B-Instruct draft model.
- Selection logic is unchanged.
- Only selected doc-token KV writeback is changed:
  - `K_after = beta * K_cached + (1 - beta) * K_recomputed`
  - `V_after = beta * V_cached + (1 - beta) * V_recomputed`
- `beta=0` is native full selected-token KV update.
- `beta=1` means selected tokens are recomputed but doc K/V writeback is restored to cached K/V, so it approximates no doc-KV update for selected tokens.

Full sweep grid:
- betas: `0, 0.25, 0.5, 0.75, 1.0`
- datasets: `2wikimqa` 0-200, `hotpotqa` 0-260, `triviaqa` 0-270, `musique` 0-135
- segment size: 25 samples
- mode: `kv` only for this first full sweep. Key-only/value-only should wait until the KV beta curve is known.

Parallel launch:
- qjy000 / qjhs-sh-lab-01: GPU1-7, shard 0/2. GPU0 was left for the already-running MuSiQue beta=0 cache warmup.
- qjy003 / qjhs-sh-lab-04: GPU0-7, shard 1/2.
- qjy002 / qjhs-sh-lab-03 was skipped because all GPUs were already high-memory occupied.
- qjy001 was not touched.

Commands launched:

```bash
# qjy000
cd /raid/home/hming/FusionRAG-pca-analysis
FUSIONRAG_KV_BLEND_GPUS=1,2,3,4,5,6,7 FUSIONRAG_KV_BLEND_SHARD_ID=0 FUSIONRAG_KV_BLEND_SHARD_COUNT=2 setsid MOTIVATION_EXPERIMENTS/qwen3_draft_kv_blend_beta/scripts/full_beta_sweep_workers.py   > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_full_sweep/logs/launcher_qjy000.log 2>&1 < /dev/null &

# qjy003
cd /home/hming/FusionRAG-pca-analysis
FUSIONRAG_KV_BLEND_GPUS=0,1,2,3,4,5,6,7 FUSIONRAG_KV_BLEND_SHARD_ID=1 FUSIONRAG_KV_BLEND_SHARD_COUNT=2 setsid MOTIVATION_EXPERIMENTS/qwen3_draft_kv_blend_beta/scripts/full_beta_sweep_workers.py   > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_full_sweep/logs/launcher_qjy003.log 2>&1 < /dev/null &
```

Output paths:
- Full results: `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_full_sweep/results_full/`
- Full logs: `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_full_sweep/logs/`
- Manifests:
  - `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_full_sweep/manifest_qjhs-sh-lab-01_shard0_of_2.csv`
  - `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_full_sweep/manifest_qjhs-sh-lab-04_shard1_of_2.csv`
- Summary command:

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
MOTIVATION_EXPERIMENTS/qwen3_draft_kv_blend_beta/scripts/summarize_full_beta_sweep.py
```

Current interpretation target:
- If `beta=0` is best: selected-token KV recompute should be fully written back; replacing it with a lightweight adapter must approximate the full delta.
- If `beta=0.25` or `beta=0.5` is best: the recomputed delta is useful but may need damping; a future adapter could predict a scaled/regularized delta rather than full delta.
- If `beta=1` is close to `beta=0`: selected doc-token KV update contributes less than expected; effort should move toward selection/query-side effects rather than doc KV delta modeling.

Status at launch:
- qjy000/qjy003 workers started successfully and began beta=0 2Wiki/Hotpot segments.
- No final full-sweep conclusions yet; wait for all segments and then compare `beta × dataset` in `full_beta_summary.csv`.


## 2026-07-13 Initial Results Snapshot

This snapshot separates completed probe results from incomplete full-sweep partials.

### Completed HotpotQA 0-50 probe

Path: `MOTIVATION_EXPERIMENTS/qwen3_draft_kv_blend_beta/results/hotpot_beta_summary.csv`

| beta | finished | main acc | main correct / total | avg F1 | avg EM | interpretation |
|---:|:---:|---:|---:|---:|---:|---|
| 0 | yes | 88.00% | 44 / 50 | 0.6012 | 0.4400 | native selected-token full KV writeback |
| 0.25 | yes | 88.00% | 44 / 50 | 0.5996 | 0.4400 | similar to native |
| 0.5 | yes | 90.00% | 45 / 50 | 0.6113 | 0.4400 | best main acc in this small probe |
| 0.75 | yes | 86.00% | 43 / 50 | 0.6154 | 0.4800 | lower main acc, higher EM/F1 |
| 1.0 | yes | 86.00% | 43 / 50 | 0.6119 | 0.4800 | no doc-KV writeback is worse than beta=0/0.5 on main acc |

Preliminary reading: on HotpotQA 0-50, full recompute writeback is not strictly optimal; `beta=0.5` gives +1/50 main accuracy over native beta=0. However this is only 50 samples, so it should be treated as a directional probe, not a final conclusion.

### Full-sweep partial status

Path: `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_full_sweep/full_beta_summary.csv`

Current partial rows after launching qjy000/qjy003 workers:

| beta | dataset | finished segments | observed samples | main acc | avg F1 | avg EM |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 2wikimqa | 0 / 8 | 49 | 44.90% | 0.4462 | 0.2245 |
| 0 | hotpotqa | 0 / 7 | 62 | 88.71% | 0.6109 | 0.4355 |

These full-sweep rows are incomplete because `finished_segments=0`; they only prove the queue is producing CSVs. Do not compare them against completed beta values yet.

Current active workers:
- qjy000: full beta sweep on GPU1-7 plus old MuSiQue beta=0 on GPU0.
- qjy003: full beta sweep on GPU0-7.
- qjy002 skipped due high GPU memory occupancy; qjy001 untouched.


## 2026-07-13 Cache Concurrency Fix

Problem found after launching qjy001:
- MuSiQue did not have a reusable Qwen3-32B full global-topk10 BGE cache under the paths used by this experiment.
- The initial full-sweep worker used per-host/per-GPU MuSiQue cache paths, e.g. `fusionrag-qwen3-kv-blend-beta-full-cache/qjhs-sh-lab-02_gpu*/musique`.
- This caused two problems:
  - storage blow-up from duplicated MuSiQue KV/preprocess cache;
  - unsafe cache concurrency if multiple workers were changed to a single shared cache without coordination.

Fix applied:
- Added `FUSIONRAG_KV_BLEND_SHARED_MUSIQUE_CACHE` support to `scripts/full_beta_sweep_workers.py`.
- Stopped the old full-sweep workers and orphan child jobs from qjy000/qjy001/qjy003.
- Restarted workers with explicit separation:
  - qjy000: non-MuSiQue only, `FUSIONRAG_KV_BLEND_DATASETS=2wikimqa,hotpotqa,triviaqa`, shard 0/2.
  - qjy003: non-MuSiQue only, `FUSIONRAG_KV_BLEND_DATASETS=2wikimqa,hotpotqa,triviaqa`, shard 1/2.
  - qjy001: MuSiQue only, one GPU only, `FUSIONRAG_KV_BLEND_GPUS=0`, single writer to shared cache.
- Shared MuSiQue cache path:
  - `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-kv-blend-beta-shared-cache/musique`

Important concurrency rule:
- MuSiQue shared cache must have only one writer. Do not run multiple MuSiQue workers against the same shared cache unless file locking or atomic cache writes are verified.
- 2Wiki/Hotpot/Trivia continue to use existing per-GPU cross-dataset cache because those caches already exist and are being read/reused.

Restart commands used:

```bash
# qjy000: non-MuSiQue only
FUSIONRAG_KV_BLEND_DATASETS=2wikimqa,hotpotqa,triviaqa FUSIONRAG_KV_BLEND_GPUS=0,1,2,3,4,5,6,7 FUSIONRAG_KV_BLEND_SHARD_ID=0 FUSIONRAG_KV_BLEND_SHARD_COUNT=2 setsid MOTIVATION_EXPERIMENTS/qwen3_draft_kv_blend_beta/scripts/full_beta_sweep_workers.py   > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_full_sweep/logs/launcher_qjy000_non_musique.log 2>&1 < /dev/null &

# qjy003: non-MuSiQue only
FUSIONRAG_KV_BLEND_DATASETS=2wikimqa,hotpotqa,triviaqa FUSIONRAG_KV_BLEND_GPUS=0,1,2,3,4,5,6,7 FUSIONRAG_KV_BLEND_SHARD_ID=1 FUSIONRAG_KV_BLEND_SHARD_COUNT=2 setsid MOTIVATION_EXPERIMENTS/qwen3_draft_kv_blend_beta/scripts/full_beta_sweep_workers.py   > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_full_sweep/logs/launcher_qjy003_non_musique.log 2>&1 < /dev/null &

# qjy001: MuSiQue single writer only
FUSIONRAG_KV_BLEND_DATASETS=musique FUSIONRAG_KV_BLEND_GPUS=0 FUSIONRAG_KV_BLEND_SHARD_ID=0 FUSIONRAG_KV_BLEND_SHARD_COUNT=1 FUSIONRAG_KV_BLEND_SHARED_MUSIQUE_CACHE=/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-kv-blend-beta-shared-cache/musique setsid MOTIVATION_EXPERIMENTS/qwen3_draft_kv_blend_beta/scripts/full_beta_sweep_workers.py   > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_full_sweep/logs/launcher_qjy001_musique_single_writer.log 2>&1 < /dev/null &
```

Current verification:
- qjy000/qjy003 active result paths are only `2wikimqa`, `hotpotqa`, `triviaqa`.
- qjy001 has one active MuSiQue process and its `--cache_path` is the shared path above.


### Additional qjy001 Non-MuSiQue Workers

After making qjy001 a single-writer machine for MuSiQue on GPU0, GPUs 1-7 were idle. They were added back for non-MuSiQue datasets only:

```bash
cd /home/hming/FusionRAG-pca-analysis
FUSIONRAG_KV_BLEND_DATASETS=2wikimqa,hotpotqa,triviaqa FUSIONRAG_KV_BLEND_GPUS=1,2,3,4,5,6,7 FUSIONRAG_KV_BLEND_SHARD_ID=0 FUSIONRAG_KV_BLEND_SHARD_COUNT=1 setsid MOTIVATION_EXPERIMENTS/qwen3_draft_kv_blend_beta/scripts/full_beta_sweep_workers.py   > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_full_sweep/logs/launcher_qjy001_non_musique.log 2>&1 < /dev/null &
```

This does not touch the MuSiQue shared cache. qjy001 now runs:
- GPU0: MuSiQue only, single writer to shared cache.
- GPU1-7: 2Wiki/Hotpot/Trivia only, using existing per-GPU cross-dataset caches.


## 2026-07-13 Current Results Snapshot After Cache Fix

Summary generated from `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_full_sweep/full_beta_summary.csv` after restarting workers with the MuSiQue single-writer cache rule.

### Complete / near-complete non-MuSiQue results

| Dataset | beta | finished seg | Main Acc | Main Correct / Total | Sub Acc | Avg F1 | Avg EM |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2WikiMQA | 0 | 8/8 | 47.00% | 94/200 | 47.00% | 0.3695 | 0.2050 |
| 2WikiMQA | 0.25 | 8/8 | 47.00% | 94/200 | 47.00% | 0.3640 | 0.2100 |
| 2WikiMQA | 0.5 | 8/8 | 46.00% | 92/200 | 46.45% | 0.3681 | 0.2227 |
| 2WikiMQA | 0.75 | 8/8 | 46.00% | 92/200 | 48.31% | 0.3590 | 0.2246 |
| 2WikiMQA | 1.0 | 8/8 | 44.50% | 89/200 | 46.64% | 0.3741 | 0.2292 |
| HotpotQA | 0 | 11/11 | 86.54% | 225/260 | 86.54% | 0.6227 | 0.4923 |
| HotpotQA | 0.25 | 11/11 | 86.15% | 224/260 | 86.21% | 0.6261 | 0.4943 |
| HotpotQA | 0.5 | 11/11 | 85.77% | 223/260 | 86.00% | 0.6297 | 0.4967 |
| HotpotQA | 0.75 | 11/11 | 85.77% | 223/260 | 85.85% | 0.6257 | 0.4952 |
| HotpotQA | 1.0 | 10/11 | 85.77% | 223/260 | 86.31% | 0.6094 | 0.4777 |
| TriviaQA | 0 | 11/11 | 91.11% | 246/270 | 91.11% | 0.6513 | 0.5519 |
| TriviaQA | 0.25 | 11/11 | 90.37% | 244/270 | 90.38% | 0.6452 | 0.5395 |
| TriviaQA | 0.5 | 11/11 | 92.22% | 249/270 | 92.31% | 0.6429 | 0.5354 |
| TriviaQA | 0.75 | 11/11 | 90.74% | 245/270 | 89.94% | 0.6200 | 0.5061 |
| TriviaQA | 1.0 | 11/11 | 89.63% | 242/270 | 90.00% | 0.6284 | 0.5167 |

### MuSiQue status

MuSiQue is still running under the single-writer rule on qjy001 GPU0. Current partial rows:

| Dataset | beta | finished seg | Main Acc | Main Correct / Total | Sub Acc | Avg F1 | Avg EM |
|---|---:|---:|---:|---:|---:|---:|---:|
| MuSiQue | 0 | 2/6 | 81.69% | 58/71 | 87.69% | 0.5669 | 0.2000 |
| MuSiQue | 0.25 | 0/4 | 85.00% | 17/20 | 84.85% | 0.5741 | 0.2121 |
| MuSiQue | 0.5 | 0/1 | 100.00% | 2/2 | 100.00% | 0.5806 | 0.0000 |

Do not draw final conclusions from MuSiQue yet; only beta=0 has meaningful partial coverage so far.

### Current interpretation

- For 2WikiMQA and HotpotQA, beta=0 or beta=0.25 is best on main accuracy. Larger beta weakens main accuracy. This means selected-token KV recompute writeback is useful; fully suppressing update (`beta=1`) is not a good replacement.
- TriviaQA is different: beta=0.5 gives the best main accuracy, 92.22% vs beta=0 at 91.11%. This supports the hypothesis that partial/damped KV update can help on some extraction-style workloads.
- Overall, the beta curve is dataset-dependent. A fixed global beta may not be enough; future adapter should likely predict update strength or gate by dataset/query/context features.
- The current evidence is stronger for a lightweight adapter that predicts a scaled/regularized Delta-KV than for simply disabling recompute KV writeback.

## Relation to earlier rate=0 / rate=0.15 baselines

This beta sweep starts from native online `DraftModel`, `rate=0.15`; the selector and the reprocess call are unchanged. Only the document KV writeback after recompute is blended.

- `beta=0` is intended to match native online `DraftModel rate=0.15`: selected document tokens receive the fully recomputed K/V. It is directly comparable to earlier `online_draft_rate015` only when model, dataset slice, preprocess cache, judge, and code path are identical.
- `beta=1` is not the same as a true `rate=0` run. It still runs `rate=0.15` selection and the reprocess forward, then restores selected document-token K/V to the cached values. It should be read as "suppress selected doc-token KV writeback", not as "no reprocess pipeline".
- Therefore the strict correspondence is: `beta=0` <-> native `rate=0.15` DraftModel; `beta=1` is only an approximation to "no document KV update" and should not be numerically expected to equal the old `rate=0` baseline.

Earlier MuSiQue `online_draft_rate015` summary used 135 main examples and reported 99/135 main accuracy = 73.33%. The current MuSiQue beta sweep is not yet a matched comparison because it is incomplete and uses the new shared-cache run after the cache-concurrency fix.

### Why current beta=0 does not numerically match earlier rate=0.15 records

A follow-up alignment check found that the current MuSique `beta=0` row should not be treated as a clean reproduction of the earlier `online_draft_rate015` table yet.

Observed differences:

- Earlier `online_draft_rate015` used cache root `/raid/home/hming/fusionrag-reflect-qwen3-full-cache`; the current beta MuSique run uses `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-kv-blend-beta-shared-cache/musique` after the cache-concurrency fix. The old cache root is not currently visible on qjy000, so beta=0 regenerated/uses a different cache.
- The current MuSique beta table is incomplete/mixed: only `seg_0_25` and `seg_125_135` have `FINAL RESULTS`; other MuSique segment directories contain partial CSVs or old interrupted metadata from the earlier multi-writer phase. The summarizer currently reads CSV rows even when the segment is not finished, while separately reporting `finished_segments`.
- Same-slice `seg_0_25` already differs: old `online_draft_rate015` has 15/21 main correct, current beta=0 has 17/21. All 38 predictions differ textually. Three rows differ in semantic `Correct`.
- F1/EM are especially not comparable across the two records because old predictions often include `<think>...</think>` wrappers, while current predictions are cleaner answer strings. Example: old predicted `<think>

</think>

National Cycle Network`, current predicted `National Cycle Network`; both are semantically correct, but lexical F1 differs.

Conclusion: `beta=0` is still the intended code-path equivalent of native DraftModel `rate=0.15`, but the existing numeric tables are not a strict A/B comparison. A clean equivalence test should rerun one fixed slice with the same cache root, same output cleaning, same judge cache policy, and compare per-row predictions.

### Important correction: beta=1 is not equal to rate=0 in the default code path

A code-path inspection shows why `beta=1` does not strictly correspond to `rate=0` in the current beta sweep.

In `load_kv_and_generate`:

- `rate=0` sets `k_need_index` to query tokens only, so the query prefill attends to cached document KV.
- `rate=0.15, beta=1` first selects document tokens, then appends query tokens to the same `k_need_index`. The model forward runs over `[selected_doc_tokens + query_tokens]` in one reprocess prefill. The selected document tokens are recomputed before the query tokens inside that forward.
- The beta blend/restore is applied after the forward returns. Therefore the final stored selected document KV is restored to cached KV when `beta=1`, but the query logits and query KV for that forward may already have used the transient recomputed document KV.

So default `beta=1` means: transiently recompute selected doc tokens for the query prefill, then restore selected doc KV afterward. It is not the same as true `rate=0`, where selected doc tokens are never recomputed and the query sees only cached doc KV.

To test a stricter equivalence to `rate=0`, the experiment should either run true `rate=0`, or run a strict two-stage variant where selected doc tokens are recomputed, restored/blended, and only then query tokens are prefetched. In the current code this is closer to using `FUSIONRAG_STRICT_REPROCESS_ABLATION=1` with `beta=1`, but it still needs a direct per-row sanity check against true `rate=0`.

### Status of the non-strict beta sweep

The current non-strict beta sweep should not be used as evidence for the effect of different beta values on recomputed document KV.

Reason: in the default code path, `k_need_index` contains `[selected_doc_tokens + query_tokens]`, and the model runs them in a single forward. The beta blend is applied only after that forward returns. Therefore the query prefill / first-token logits have already seen the fully recomputed selected document KV, independent of whether beta is 0.25, 0.5, 0.75, or 1.0.

What the current table can still reflect:

- `beta=0`: native full writeback path.
- `beta>0`: first token is still produced from the full-recompute query prefill, but later decode steps may see the blended/restored document KV because the cache is modified after the prefill forward.

Therefore the current beta sweep is not a clean test of `V = beta * cached + (1-beta) * recomputed` for the RAG query prefill. The existing results should be treated as a diagnostic/invalid non-strict run, not as the final beta conclusion.

Required correction: rerun a strict two-stage beta experiment:

1. select document tokens as before;
2. run recompute only for selected document tokens;
3. immediately apply beta blend to selected document K/V;
4. then run query prefill as a separate forward so the query sees the blended KV.

Only that strict version can support the intended comparison: `beta=0` ~= native `rate=0.15`, and `beta=1` ~= no selected document KV update / closer to `rate=0`.

## 2026-07-13 Strict Beta Query-Prefill Interface

The earlier non-strict beta sweep is invalid for measuring query-prefill exposure to blended KV, because selected doc tokens and query tokens were run in one forward and beta blend happened only after that forward.

A separate export interface was added so the original pipeline remains unchanged by default:

```bash
export FUSIONRAG_STRICT_BETA_QUERY_PREFILL=1
```

Semantics when enabled:

1. select document tokens exactly as before;
2. run recompute only for selected document tokens;
3. immediately apply `FUSIONRAG_REPROCESS_KV_BLEND_BETA` / `FUSIONRAG_REPROCESS_KV_BLEND_MODE` to selected document K/V;
4. run query prefill in a separate forward, so query sees the blended/restored document KV.

Default behavior is unchanged because `FUSIONRAG_STRICT_BETA_QUERY_PREFILL` defaults to `0`.

Launcher support:

```bash
export FUSIONRAG_KV_BLEND_STRICT_QUERY_PREFILL=1
```

The beta worker script forwards this to child jobs as `FUSIONRAG_STRICT_BETA_QUERY_PREFILL=1` and records `strict_query_prefill` in `task_meta.csv`.

### Strict smoke reproduction

Smoke result path:

```text
/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_strict_smoke
```

Reusable smoke script:

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
GPU=0 MOTIVATION_EXPERIMENTS/qwen3_draft_kv_blend_beta/scripts/run_strict_beta_smoke.sh
```

Common settings:

- dataset: HotpotQA reflect, sample `0:1`
- model: Qwen3-32B
- selector: DraftModel, `rate=0.15` for beta runs
- cache: `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-attn-alpha-cross-cache-v2/worker_gpu2/hotpotqa`
- preprocess: true, BGE global topk10

Commands:

```bash
# Strict beta=1: selected doc tokens are recomputed, restored to cached KV, then query prefill runs.
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 FUSIONRAG_STRICT_BETA_QUERY_PREFILL=1 FUSIONRAG_REPROCESS_KV_BLEND_BETA=1 FUSIONRAG_REPROCESS_KV_BLEND_MODE=kv /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python test_fusionrag_reflect_preprocess_exp.py   --model_type qwen3   --model_path /mnt/qjhs-sh-lab-01/models/Qwen3-32B   --model_name Qwen3-32B   --data_path MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/hotpotqa_reflect.json   --dataset_name hotpotqa   --cache_path /mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-attn-alpha-cross-cache-v2/worker_gpu2/hotpotqa   --result_path /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_strict_smoke/beta1_hotpot_0_1   --start_sample 0 --end_sample 1   --rate 0.15 --topk 10 --preprocess true --recall_method bge   --reprocess_method DraftModel   --draft_model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct   --revert_rope true --preprocess_scope global   --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3   --device cuda:0 --use_multi_gpu false   --openai_base_url http://36.150.226.221:32355/v1   --openai_api_key api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS   --openai_model GLM-5.2
```

For strict beta=0, use the same command with `FUSIONRAG_REPROCESS_KV_BLEND_BETA=0` and result path `strict_beta0_hotpot_0_1`.

For true rate=0, use the same base command without strict/beta envs and set `--rate 0`, result path `true_rate0_hotpot_0_1`.

### Strict smoke results

| Method | Rate | Strict query prefill | Beta | Prompt eval tokens | Correct | F1 | EM | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| strict beta=0 | 0.15 | yes | 0 | 254 | 1/1 | 0.2162 | 0.0000 | full selected-doc KV writeback before query prefill |
| strict beta=1 | 0.15 | yes | 1 | 254 | 1/1 | 0.1231 | 0.0000 | selected-doc KV restored before query prefill |
| true rate=0 | 0 | no | n/a | 31 | 1/1 | 0.1231 | 0.0000 | no selected-doc recompute |

Key sanity observation: on this sample, strict `beta=1` matches true `rate=0` in prediction text and F1/EM, while strict `beta=0` remains correct but generates a different answer and higher lexical F1. This supports that the new strict interface is measuring the intended mechanism: query prefill sees the beta-applied document KV.

A rate=0 bug was also fixed while testing: `reprocess_attention_ablation` is now initialized before the `rate != 0` branch, so true `rate=0` no longer raises `UnboundLocalError`.

## 2026-07-13 Strict HotpotQA Sweep Launch

After identifying the non-strict beta sweep issue, a clean strict sweep was launched on one dataset first, before expanding to other datasets.

Dataset: HotpotQA reflect, samples `0:260`.

Reason for choosing HotpotQA first:

- existing cross-dataset preprocess KV cache is already available;
- avoids the MuSiQue cache inconsistency/multi-writer issue from the earlier run;
- sample count is large enough for a first full-dataset sanity curve.

Strict interface:

```bash
FUSIONRAG_KV_BLEND_STRICT_QUERY_PREFILL=1
```

Result root:

```text
/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_strict_hotpot
```

Cache policy:

- Use existing HotpotQA cross cache only:
  `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-attn-alpha-cross-cache-v2/worker_gpu{0..7}/hotpotqa`
- Do not generate or share a new MuSiQue preprocess cache in this run.
- Assign globally unique GPU indices across machines to avoid concurrent access to the same `worker_gpuX/hotpotqa` cache path:
  - qjy000/qjhs-sh-lab-01: GPUs `0,1,2`, shard `0/3`
  - qjy001/qjhs-sh-lab-02: GPUs `3,4,5`, shard `1/3`
  - qjy003/qjhs-sh-lab-04: GPUs `6,7`, shard `2/3`

Launch commands:

```bash
# qjy000
cd /raid/home/hming/FusionRAG-pca-analysis
FUSIONRAG_KV_BLEND_FULL_EXP_ROOT=/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_strict_hotpot FUSIONRAG_KV_BLEND_STRICT_QUERY_PREFILL=1 FUSIONRAG_KV_BLEND_DATASETS=hotpotqa FUSIONRAG_KV_BLEND_BETAS=0,0.25,0.5,0.75,1.0 FUSIONRAG_KV_BLEND_GPUS=0,1,2 FUSIONRAG_KV_BLEND_SHARD_ID=0 FUSIONRAG_KV_BLEND_SHARD_COUNT=3 setsid MOTIVATION_EXPERIMENTS/qwen3_draft_kv_blend_beta/scripts/full_beta_sweep_workers.py   > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_strict_hotpot/logs/launcher_qjy000.log 2>&1 < /dev/null &

# qjy001
cd /home/hming/FusionRAG-pca-analysis
FUSIONRAG_KV_BLEND_FULL_EXP_ROOT=/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_strict_hotpot FUSIONRAG_KV_BLEND_STRICT_QUERY_PREFILL=1 FUSIONRAG_KV_BLEND_DATASETS=hotpotqa FUSIONRAG_KV_BLEND_BETAS=0,0.25,0.5,0.75,1.0 FUSIONRAG_KV_BLEND_GPUS=3,4,5 FUSIONRAG_KV_BLEND_SHARD_ID=1 FUSIONRAG_KV_BLEND_SHARD_COUNT=3 setsid MOTIVATION_EXPERIMENTS/qwen3_draft_kv_blend_beta/scripts/full_beta_sweep_workers.py   > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_strict_hotpot/logs/launcher_qjy001.log 2>&1 < /dev/null &

# qjy003
cd /home/hming/FusionRAG-pca-analysis
FUSIONRAG_KV_BLEND_FULL_EXP_ROOT=/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_strict_hotpot FUSIONRAG_KV_BLEND_STRICT_QUERY_PREFILL=1 FUSIONRAG_KV_BLEND_DATASETS=hotpotqa FUSIONRAG_KV_BLEND_BETAS=0,0.25,0.5,0.75,1.0 FUSIONRAG_KV_BLEND_GPUS=6,7 FUSIONRAG_KV_BLEND_SHARD_ID=2 FUSIONRAG_KV_BLEND_SHARD_COUNT=3 setsid MOTIVATION_EXPERIMENTS/qwen3_draft_kv_blend_beta/scripts/full_beta_sweep_workers.py   > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_draft_kv_blend_beta_strict_hotpot/logs/launcher_qjy003.log 2>&1 < /dev/null &
```

Status at launch: all workers started and `task_meta.csv` records `strict_query_prefill=1` for first HotpotQA segments.

Summary script:

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
MOTIVATION_EXPERIMENTS/qwen3_draft_kv_blend_beta/scripts/summarize_strict_hotpot.py
```
