# Qwen3 MuSiQue Fresh Preprocess KV Rerun on qjy000

## Goal

Run MuSiQue on qjy000 with freshly regenerated raw/preprocess KV, isolated from all previous alpha/blend/shared-cache experiments.

## Experiment Scope

- Dataset: MuSiQue, `data/result_reflect.json`, samples `0-200`
- Model: Qwen3-32B
- Methods:
  - `online_qk_rate015`
  - `online_draft_rate015`
- Rate: `0.15`
- Preprocess: `true`
- Recall: `bge`
- Top-k: `10`
- Preprocess scope: `global`
- Judge: `GLM-5.2`

## Cache Isolation

Fresh result root:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_fresh_preprocess_qjy000_lab03_v2_20260714
```

Fresh cache root:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-qjy000-lab03-v2-20260714
```

Rules:

- `FUSIONRAG_PREPROCESS_CACHE_READONLY` is not set.
- The launcher writes per-GPU cache subtrees under the fresh cache root.
- No previous shared alpha/blend cache root is used.

## Launch Script

```text
MOTIVATION_EXPERIMENTS/qwen3_musique_fresh_preprocess_qjy000/scripts/launch_musique_fresh_preprocess_qjy000.sh
```

## Launch Command

To be launched after committing this folder:

```bash
ssh qjy000 'cd /raid/home/hming/FusionRAG-pca-analysis && MOTIVATION_EXPERIMENTS/qwen3_musique_fresh_preprocess_qjy000/scripts/launch_musique_fresh_preprocess_qjy000.sh'
```

## 2026-07-14 storage adjustment before launch

-  was full, so this run was moved to , which had about 11T free.
- Removed incomplete cache from the aborted strict fresh cross-dataset run:  (~135G).
- qjy002 waiting strict fresh workers were stopped before deleting that incomplete cache.


Corrected storage note:
- `/mnt/qjhs-sh-lab-04` was full, so this run was moved to `/mnt/qjhs-sh-lab-03`, which had about 11T free.
- Removed incomplete cache from the aborted strict fresh cross-dataset run: `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-strict-fresh-cache-20260713` (~135G).
- qjy002 waiting strict fresh workers were stopped before deleting that incomplete cache.

## 2026-07-14 pause and cleanup for storage target change

User requested moving this run to `/mnt/qjhs-sh-1ab-02`.

Observed on qjy000:

```text
Available mounts: /mnt/qjhs-sh-lab-01, /mnt/qjhs-sh-lab-03, /mnt/qjhs-sh-lab-04
Missing mounts: /mnt/qjhs-sh-1ab-02, /mnt/qjhs-sh-lab-02
```

Actions taken before relaunch:

- Stopped the qjy000 MuSiQue fresh-preprocess launcher and child `test_fusionrag_reflect_preprocess_exp.py` processes.
- Removed incomplete partial cache at `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-qjy000-lab03-v2-20260714`.
- Kept the small result/log directory at `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_fresh_preprocess_qjy000_lab03_v2_20260714` for provenance.
- No fresh MuSiQue job is currently running after this pause.

Current disk status after cleanup:

```text
/mnt/qjhs-sh-lab-01: about 2.5T free
/mnt/qjhs-sh-lab-03: about 11T free
/mnt/qjhs-sh-lab-04: about 135G free, still effectively full
```

Relaunch is blocked until the intended writable target is clarified, because `/mnt/qjhs-sh-1ab-02` does not exist on qjy000.

## 2026-07-14 relaunch on lab03 v2

User confirmed to write under `/mnt/qjhs-sh-lab-03`.

To avoid mixing with the interrupted lab03 attempt, this relaunch uses new isolated roots:

```text
Result root: /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_fresh_preprocess_qjy000_lab03_v2_20260714
Cache root:  /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-qjy000-lab03-v2-20260714
```


Launch command:
ssh qjy000 "cd /raid/home/hming/FusionRAG-pca-analysis && MOTIVATION_EXPERIMENTS/qwen3_musique_fresh_preprocess_qjy000/scripts/launch_musique_fresh_preprocess_qjy000.sh"

Launch status:
- Runtime commit: bf73f44
- Host: qjy000 / qjhs-sh-lab-01
- Launcher PID: 677229
- Status at launch check: 8 worker logs created; Online QK rate=0.15 MuSiQue segments 0-200 started on GPUs 0-7.
- Verified cache root is under /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-qjy000-lab03-v2-20260714.

## 2026-07-14 qjy001/qjy003 lab03 replicas

User pointed out qjy001 and qjy003 were not launched. They are launched as independent replicas with separate result/cache roots to avoid overwriting qjy000 segment logs or cache files.

qjy001 roots:
- Result root: /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_fresh_preprocess_qjy001_lab03_v2_20260714
- Cache root: /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-qjy001-lab03-v2-20260714

qjy003 roots:
- Result root: /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_fresh_preprocess_qjy003_lab03_v2_20260714
- Cache root: /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-qjy003-lab03-v2-20260714

Launch commands use commit 0dc82d4 and the same MuSiQue fresh-preprocess launcher with host-specific EXP/CACHE roots.

Launch status for qjy001/qjy003 replicas:
- Record commit before launch: c73e9b0
- qjy001 launcher PID: 3635958; host qjhs-sh-lab-02; 8 worker logs created and Online QK MuSiQue segments 0-200 started.
- qjy003 launcher PID: 3139526; host qjhs-sh-lab-04; 8 worker logs created and Online QK MuSiQue segments 0-200 started.
- qjy001/qjy003 use separate lab03 result/cache roots, so they do not overwrite qjy000 outputs.

## 2026-07-14 fix duplicate multi-host launch

The previous qjy000/qjy001/qjy003 launches were wrong: each host built the full MuSiQue task list and therefore ran the same examples 0-200. Those duplicate jobs were stopped.

Fix:
- Added launcher sharding env vars: FUSIONRAG_STRICT_FRESH_SHARD_INDEX and FUSIONRAG_STRICT_FRESH_SHARD_COUNT.
- Relaunch will use SHARD_COUNT=3 with qjy000 index 0, qjy001 index 1, qjy003 index 2.
- Relaunch will use a new shared result/cache root with host-specific cache subtrees, so old partial replica outputs are not mixed.

Sharded relaunch status:
- Sharding commit: 9f825bb
- Shared result root: /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_fresh_preprocess_3host_sharded_20260714
- Shared cache root: /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-musique-fresh-preprocess-3host-sharded-20260714
- qjy000 shard 0/3 PID 897569: online_qk 0-25,75-100,150-175; online_draft 25-50,100-125,175-200.
- qjy001 shard 1/3 PID 3706984: online_qk 25-50,100-125,175-200; online_draft 50-75,125-150.
- qjy003 shard 2/3 PID 3207574: online_qk 50-75,125-150; online_draft 0-25,75-100,150-175.
- This covers all 16 MuSiQue tasks exactly once; previous duplicate full-replica jobs were stopped.

## 2026-07-14 pipeline audit: why rate=0.15 looked suspiciously close to rate=1

User concern: current `rate=0.15` recompute appearing close to `rate=1` full recompute is suspicious. I reviewed the active sharded pipeline and found several confounders.

Runtime commit for the active sharded jobs: `9f825bb`.

Record commit before launch: `3b2d1de`.

Active command template, as emitted by `MOTIVATION_EXPERIMENTS/qwen3_baseline_provenance_audit/scripts/run_strict_fresh_cache_qk_draft_rate015.py`:

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python test_fusionrag_reflect_preprocess_exp.py \
  --model_type qwen3 \
  --model_path /mnt/qjhs-sh-lab-01/models/Qwen3-32B \
  --model_name Qwen3-32B \
  --data_path data/result_reflect.json \
  --dataset_name musique \
  --rate 0.15 \
  --topk 10 \
  --preprocess true \
  --recall_method bge \
  --reprocess_method FusionRAG-or-DraftModel \
  --revert_rope true \
  --preprocess_scope global \
  --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3 \
  --openai_model GLM-5.2
```

Audit findings:

1. The active `rate=0.15` run is not a raw-cache online recompute baseline. It uses `--preprocess true --recall_method bge --topk 10 --preprocess_scope global`, so document KV is first built from a global BGE top-k preprocess cache and then only 15% online recompute is applied.

2. The preprocess is global over the dataset document corpus, not per-example only. Logs show examples retrieving docs like `Q136-Doc15`, `Q111-Doc13`, etc. while processing `Q1`, so this is a transductive global-corpus preprocess setting. It may be a valid method variant, but it is not a clean measurement of online recompute alone.

3. The current fresh run has no same-directory `rate=1` baseline. Every checked segment log reports `No rate=1 files found ... Continuing without rate=1 baseline comparison...`, and the written CSV `Rate1_Correct` column is `N/A`. Any current claim that `rate=0.15` is close to `rate=1` must therefore come from an external or historical baseline table, not from this fresh run embedded rate-1 comparison.

4. Historical baseline tables are mixed across configurations. Some older full/rate sweep records are for `Qwen2.5-7B-Instruct`, while this run is `Qwen3-32B`. These cannot be used as strict evidence for `Qwen3-32B rate=0.15 ~= rate=1`.

5. Code review of `ktransformers/util/utils.py` confirms the normal rate path still selects only document tokens before appending all query tokens for prefill. For nonzero rate, `doc_reprocess_index = list(k_need_index)` and then query token positions are appended to `k_need_index`; for zero rate, `doc_reprocess_index=[]` and only query tokens are prefetched. Therefore profile fields named `reprocess_prefill_tokens` can include query tokens and should not be interpreted as document recompute rate.

6. `FUSIONRAG_FORCE_ALL_REPROCESS` exists in code and would force all doc tokens, but inspected active launcher/test process environments did not show this variable set. No evidence so far that the active `rate=0.15` run was silently converted to full doc recompute.

7. Draft selection default `profile` mode allows a small expansion of high-attention contiguous components up to roughly `target_count * 1.1`, and QK group mode can overshoot by group size 16. This can make the realized selected count slightly above 15%, but it cannot explain behavior equivalent to full recompute.

Partial result status at audit time:

| Method | CSV files | Rows so far | Correct | Correct rate | Avg F1 | Avg EM | Rate1 columns |
|---|---:|---:|---:|---:|---:|---:|---|
| online_qk_rate015 | 8 | 90 | 80 | 0.8889 | 0.6275 | 0.2556 | `N/A` |
| online_draft_rate015 | 8 | 90 | 81 | 0.9000 | 0.6509 | 0.2222 | `N/A` |

These are partial rows from the still-running MuSiQue shard jobs and should not be treated as final full-dataset metrics.

Interpretation:

- The suspiciously high `rate=0.15` performance is most plausibly explained by a strong `global topk10 BGE preprocess` cache and/or by comparing against non-identical historical `rate=1` baselines.
- It is not yet evidence that 15% online recompute alone recovers full attention.
- The next strict check should be same model, same data split, same judge, same fresh cache root, same preprocess mode, and same prompt/code version for `rate=0`, `rate=0.15`, and `rate=1`.

Required follow-up controls:

1. Run `Qwen3-32B MuSiQue` with the same fresh global BGE preprocess root for `rate=0`, `rate=0.15`, and `rate=1` in one result family.
2. Run the same three rates with `preprocess=false` or raw KV to isolate online recompute from offline/global preprocess.
3. Add lightweight logging for `doc_tokens`, `selected_doc_tokens`, `query_tokens`, and `selected_doc_tokens/doc_tokens` per sample so selected recompute ratio is explicit in CSV/profile output.
4. If global preprocess remains the best setting, report it as a separate method: `global-BGE-preprocess + online-recompute`, not as native online recompute.
