# Shared KV Cache Policy

This experiment must use semantic cache roots, not worker-local cache roots.

## Problem Found

The aborted 2026-07-14 cross-dataset rate sweep used cache paths of the form:

```text
CACHE_ROOT/<host>/worker_gpu<gpu>/<dataset>/Qwen3-32B/<dataset>/...
```

This made each GPU worker build its own raw KV cache and preprocess KV cache for the same model, dataset, recall method, top-k, and preprocess scope. One sampled 2WikiMQA worker used about `402G`:

```text
kv_cache                              272G
preprocess_kv_cache_global_topk10_bge 130G
```

With many workers this duplicated cache grew to `7.3T` and filled `/mnt/qjhs-sh-lab-03`, which then produced partial `.pt` files and errors such as `*.pt cannot be opened`.

## Required Layout

`test_fusionrag_reflect_preprocess_exp.py` automatically appends `model_name/dataset_name` below `--cache_path`. Therefore the launcher should pass one shared semantic root:

```text
CACHE_ROOT/Qwen3-32B/<dataset>/kv_cache
CACHE_ROOT/Qwen3-32B/<dataset>/preprocess_kv_cache_global_topk10_bge
```

Do not add `host`, `worker_gpu`, or another explicit `dataset` component to `--cache_path` unless running an intentionally isolated debug job.

## Execution Rule

Use two phases for large sweeps:

1. Cache build phase
   - Build raw KV and preprocess KV once per `(model, dataset, topk, recall_method, preprocess_scope)`.
   - If sharded, each sample/chunk should have exactly one writer.
   - Do not run multiple rates/methods that write the same cache files concurrently.

2. Evaluation phase
   - All `rate` and `method` workers reuse the same shared cache root.
   - Set `FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_READONLY=1` so evaluation cannot create or overwrite cache files.

## Launcher Controls

The cross-dataset rate sweep launchers now support:

```bash
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_LAYOUT=shared   # default, required for formal runs
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_LAYOUT=worker   # debug only
FUSIONRAG_QWEN3_CROSS_RATE_SWEEP_CACHE_READONLY=1      # formal evaluation phase
```

Formal reruns should use a fresh cache root on a filesystem with enough space, for example `/mnt/qjhs-sh-lab-01`, because the old `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-cross-dataset-rate-sweep-cache-20260714` root contains corrupted partial tensor files from the full-disk failure.

## Non-Negotiable Rules

- Never use `CACHE_ROOT/<host>/worker_gpu<gpu>/<dataset>` for a formal multi-GPU sweep.
- Never reuse a cache root after `No space left on device`, `PytorchStreamReader failed reading zip archive`, or `*.pt cannot be opened` unless the corrupt files have been audited and rebuilt.
- Result CSVs and logs are small; cache tensors are the storage risk.
- Record cache root, cache layout, readonly setting, command, and commit in `README.md` for every new run.
