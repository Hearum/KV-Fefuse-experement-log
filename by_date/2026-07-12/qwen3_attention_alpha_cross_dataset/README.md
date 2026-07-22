# Qwen3 Attention Alpha Cross-Dataset Queue

Goal: validate attention-output alpha blend settings that beat the musique baseline on additional datasets.

Datasets queued: 2wikimqa, hotpotqa, triviaqa.
Combinations queued: uniform alpha=0.1/0.25, random alpha=0.05/0.1/0.25.

Queues:
- qjy000: `local_cross_dataset_queue_qjy000.sh`
- qjy003: `local_cross_dataset_queue_qjy003.sh`

## Fixed-GPU Execution Plan

The earlier dynamic queue has been replaced by fixed-GPU workers. Each GPU receives a deterministic task list and runs it serially. A worker checks whether a task is already complete; if a previous process is still running the same segment, it waits; otherwise it waits for its assigned GPU to become free and runs the next task in the foreground.

Script:

`fixed_gpu_cross_dataset_workers.py`

Task split:

- qjy000: all 2wikimqa segments and hotpotqa 0-125.
- qjy003: hotpotqa 125-260 and all triviaqa segments.

Combinations:

- uniform alpha=0.1
- uniform alpha=0.25
- random alpha=0.05
- random alpha=0.1
- random alpha=0.25

This avoids opportunistic scheduling gaps and makes every GPU follow a fixed sequential queue.

## 2026-07-12 Restart Note

Cross-dataset data must use the homogenized reflect files from the earlier
`cross_dataset_offline_generalization` experiment:

- `MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/2wikimqa_reflect.json`
- `MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/hotpotqa_reflect.json`
- `MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/triviaqa_reflect.json`

The earlier launch accidentally used raw dataset files such as
`data/2wikimqa-200.jsonl`, `data/hotpotqa-260-100-10-doc.jsonl`, and
`data/triviaqa-270-100-10-doc.jsonl`. Those are not directly compatible with
`test_fusionrag_reflect_preprocess_exp.py` and caused `JSONDecodeError` /
`KeyError: question`. The fixed worker now follows the old cross-dataset
supervisor convention and uses the reflect JSON files.

Additional fix: cache paths are now per worker GPU,
`fusionrag-qwen3-attn-alpha-cross-cache/worker_gpu{gpu}/{dataset}`, matching
the old supervisor pattern and avoiding concurrent writes to the same cache
directory.

Because qjy000 and qjy003 share the experiment directory, worker logs now
include the hostname:

- `logs/fixed_worker_qjhs-sh-lab-01_gpu*.log` for qjy000
- `logs/fixed_worker_qjhs-sh-lab-04_gpu*.log` for qjy003

Current launched processes:

- qjy000 fixed worker parent PID: `588582`
- qjy003 fixed worker parent PID: `2359472`

Current status at launch: no immediate `rc=1` failures after the reflect-data
fix. Some workers report `wait existing ...` because previously started
reflect-data child jobs are still running for the same output segment; the
worker will skip those segments after `FINAL RESULTS` appears, or continue to
the next task when the previous process exits.

## 2026-07-12 V2 Restart After Disk/CUDA Failures

The first cross-dataset launch after the reflect-data fix was invalid. Most segments failed with one of:

- `CUDA out of memory` from constructing full `[heads, selected_doc_tokens, kv_len]` uniform/random attention matrices on longer cross-dataset documents;
- `PytorchStreamWriter failed writing file` / `unexpected pos`;
- `No space left on device`;
- corrupted cache files that could not be opened.

Root cause: output/cache were on `/raid/home`, and the old cross cache grew to about 1.4T, filling the shared `/raid` filesystem. Those failed partial segments should not be used as valid cross-dataset results.

Actions taken:

- Stopped the broken cross-dataset workers on qjy000 and qjy003.
- Deleted only the old corrupted cache directory:
  - `/raid/home/hming/fusionrag-qwen3-attn-alpha-cross-cache`
- This freed about 1.4T; `/raid` moved from 100% full to about 91% used.
- New cache root:
  - `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-attn-alpha-cross-cache-v2`
- New output/log root:
  - `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_alpha_cross_dataset_v2`
- Worker count reduced from 8 GPUs per host to 4 GPUs per host via `FUSIONRAG_CROSS_WORKER_GPUS=0,1,2,3`.
- Attention ablation output computation was changed to chunk selected document tokens using `FUSIONRAG_REPROCESS_ATTENTION_CHUNK=64` by default. This avoids allocating the full ablation weight tensor at once.

Current v2 launch:

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
PYTHONDONTWRITEBYTECODE=1 FUSIONRAG_CROSS_WORKER_GPUS=0,1,2,3 \
  nohup ./MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/fixed_gpu_cross_dataset_workers.py \
  > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_alpha_cross_dataset_v2/logs/fixed_gpu_workers_qjy000.outer.log 2>&1 < /dev/null &
```

qjy003 uses the same script from `/home/hming/FusionRAG-pca-analysis` and writes to the same `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_alpha_cross_dataset_v2` output root.

Initial v2 status: qjy000 is running 2WikiMQA `uniform_alpha0p1` segments 0-100 across GPU0-3; qjy003 is running HotpotQA `uniform_alpha0p1` segments 125-225 across GPU0-3. No immediate `No space` / old-cache path issue observed after restart.

## 2026-07-13 Cache Cleanup And 8-GPU Resume

User request: clean unused KV cache and finish the remaining cross-dataset validation.

Cleanup performed conservatively:

- Removed the old broken cross-dataset cache on `/raid/home`:
  - `/raid/home/hming/fusionrag-qwen3-attn-alpha-cross-cache`
- Removed old non-235B KV/cache directories that were not part of the active v2 run.
- Did not remove model directories, result CSVs, README/log files, or any 235B cache directory.
- Did not remove the active v2 cache:
  - `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-attn-alpha-cross-cache-v2`
- Attempted to remove `/raid/home/hming/fusionrag-pca-topk-cache-5`, but it remains because files inside it return `Permission denied`; size is about 9.5G, so it is not the current bottleneck.

Disk status after cleanup:

- qjy000 `/raid/home`: about 2.6T free.
- `/mnt/qjhs-sh-lab-04`: about 8.9T free.

The v2 fixed workers were relaunched with all available qjy000 and qjy003 GPUs:

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
PYTHONDONTWRITEBYTECODE=1 FUSIONRAG_CROSS_WORKER_GPUS=0,1,2,3,4,5,6,7 \
  nohup ./MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/fixed_gpu_cross_dataset_workers.py \
  > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_alpha_cross_dataset_v2/logs/fixed_gpu_workers_qjy000.outer.log 2>&1 < /dev/null &
```

qjy003 was launched with the same `FUSIONRAG_CROSS_WORKER_GPUS=0,1,2,3,4,5,6,7` setting from `/home/hming/FusionRAG-pca-analysis`.

Current status at 2026-07-13 00:12 CST:

- qjy000 worker parent PID: `1748691`; 8 child workers are active.
- qjy003 worker parent PID: `2813361`; 8 child workers are active.
- Completed valid v2 segments: 8/24.
- Running segments: the remaining `uniform_alpha0p1` cross-dataset segments for 2WikiMQA, HotpotQA, and TriviaQA.
- Error scan over v2 `run.log` files found no current `OutOfMemory`, `No space`, corrupted-cache, or `PytorchStreamWriter` failures after moving cache/output to `/mnt` and enabling chunked attention ablation.

Valid completed v2 partial results so far:

| Dataset | Setting | Completed range | Main/Sub acc | Avg F1 | Avg EM |
|---|---|---:|---:|---:|---:|
| 2wikimqa | uniform alpha=0.1 | 0-100 | 44/100 = 44.00% | 0.3480 | 0.2200 |
| hotpotqa | uniform alpha=0.1 | 125-225 | 87/100 = 87.00% | 0.6355 | 0.5300 |

Interpretation at this checkpoint: these are partial segment results only and should not be used as final cross-dataset conclusions. The important operational result is that the v2 run is now progressing on `/mnt` without the earlier disk/OOM/cache-corruption failure mode.

## 2026-07-13 00:57 CST Partial Result Checkpoint

Current v2 result root:

- `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_alpha_cross_dataset_v2`

Current completion:

- Completed valid segments: 13/25.
- qjy000 worker parent PID `1748691` is still running with 8 child workers.
- qjy003 worker parent PID `2813361` is still running with 8 child workers.
- Error scan found no current `Traceback`, `OutOfMemory`, `No space`, corrupted-cache, or `PytorchStreamWriter` failures.
- One earlier grep hit for `Killed` was a false positive from a 2WikiMQA question string (`A Woman Has Killed`), not an OS/process kill.

Completed segment results:

| Dataset | Segment | Main/Sub acc | Avg F1 | Avg EM |
|---|---:|---:|---:|---:|
| 2wikimqa | 0-25 | 9/25 = 36.00% | 0.3775 | 0.2400 |
| 2wikimqa | 25-50 | 12/25 = 48.00% | 0.3419 | 0.1200 |
| 2wikimqa | 50-75 | 14/25 = 56.00% | 0.3905 | 0.3600 |
| 2wikimqa | 75-100 | 9/25 = 36.00% | 0.2821 | 0.1600 |
| hotpotqa | 0-25 | 24/25 = 96.00% | 0.5573 | 0.4800 |
| hotpotqa | 25-50 | 21/25 = 84.00% | 0.5977 | 0.3600 |
| hotpotqa | 50-75 | 20/25 = 80.00% | 0.6482 | 0.4800 |
| hotpotqa | 75-100 | 22/25 = 88.00% | 0.6215 | 0.5200 |
| hotpotqa | 125-150 | 21/25 = 84.00% | 0.5816 | 0.4800 |
| hotpotqa | 150-175 | 24/25 = 96.00% | 0.6865 | 0.5600 |
| hotpotqa | 175-200 | 20/25 = 80.00% | 0.5569 | 0.4800 |
| hotpotqa | 200-225 | 22/25 = 88.00% | 0.7171 | 0.6000 |
| hotpotqa | 250-260 | 8/10 = 80.00% | 0.7380 | 0.5000 |

Weighted partial aggregate:

| Dataset | Completed examples | Main/Sub acc | Avg F1 | Avg EM |
|---|---:|---:|---:|---:|
| 2wikimqa | 100 | 44/100 = 44.00% | 0.3480 | 0.2200 |
| hotpotqa | 210 | 182/210 = 86.67% | 0.6264 | 0.4952 |

Still running:

- 2wikimqa: 100-200.
- hotpotqa: 225-250.
- triviaqa: all observed segments are still running at this checkpoint.

Interpretation: the current numbers are still partial. HotpotQA uniform alpha=0.1 remains strong on completed segments; 2WikiMQA is much weaker and cannot be judged until the remaining half finishes. TriviaQA has no completed segment yet in v2.

## 2026-07-13 Final Result For Uniform Alpha 0.1 Cross-Dataset V2

Current v2 result root:

- `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_alpha_cross_dataset_v2`

The `uniform alpha=0.1` cross-dataset validation finished all observed segments.

Completion:

- Completed valid segments: 30/30.
- Error scan found no current `Traceback`, `OutOfMemory`, `No space`, corrupted-cache, or `PytorchStreamWriter` failures.
- `/mnt/qjhs-sh-lab-04` has about 2.8T free at the end of this check.

Final weighted aggregate:

| Dataset | Examples | Main/Sub acc | Avg F1 | Avg EM |
|---|---:|---:|---:|---:|
| 2wikimqa | 200 | 96/200 = 48.00% | 0.3669 | 0.2050 |
| hotpotqa | 260 | 226/260 = 86.92% | 0.6133 | 0.4846 |
| triviaqa | 270 | 242/270 = 89.63% | 0.6398 | 0.5333 |

Completed segment details:

| Dataset | Segment | Main/Sub acc | Avg F1 | Avg EM |
|---|---:|---:|---:|---:|
| 2wikimqa | 0-25 | 9/25 = 36.00% | 0.3775 | 0.2400 |
| 2wikimqa | 25-50 | 12/25 = 48.00% | 0.3419 | 0.1200 |
| 2wikimqa | 50-75 | 14/25 = 56.00% | 0.3905 | 0.3600 |
| 2wikimqa | 75-100 | 9/25 = 36.00% | 0.2821 | 0.1600 |
| 2wikimqa | 100-125 | 14/25 = 56.00% | 0.5263 | 0.3200 |
| 2wikimqa | 125-150 | 12/25 = 48.00% | 0.3177 | 0.1200 |
| 2wikimqa | 150-175 | 12/25 = 48.00% | 0.3795 | 0.1200 |
| 2wikimqa | 175-200 | 14/25 = 56.00% | 0.3198 | 0.2000 |
| hotpotqa | 0-25 | 24/25 = 96.00% | 0.5573 | 0.4800 |
| hotpotqa | 25-50 | 21/25 = 84.00% | 0.5977 | 0.3600 |
| hotpotqa | 50-75 | 20/25 = 80.00% | 0.6482 | 0.4800 |
| hotpotqa | 75-100 | 22/25 = 88.00% | 0.6215 | 0.5200 |
| hotpotqa | 100-125 | 21/25 = 84.00% | 0.5575 | 0.4400 |
| hotpotqa | 125-150 | 21/25 = 84.00% | 0.5816 | 0.4800 |
| hotpotqa | 150-175 | 24/25 = 96.00% | 0.6865 | 0.5600 |
| hotpotqa | 175-200 | 20/25 = 80.00% | 0.5569 | 0.4800 |
| hotpotqa | 200-225 | 22/25 = 88.00% | 0.7171 | 0.6000 |
| hotpotqa | 225-250 | 23/25 = 92.00% | 0.5589 | 0.4400 |
| hotpotqa | 250-260 | 8/10 = 80.00% | 0.7380 | 0.5000 |
| triviaqa | 0-25 | 20/25 = 80.00% | 0.5253 | 0.3600 |
| triviaqa | 25-50 | 24/25 = 96.00% | 0.7807 | 0.6000 |
| triviaqa | 50-75 | 24/25 = 96.00% | 0.8671 | 0.7600 |
| triviaqa | 75-100 | 23/25 = 92.00% | 0.5519 | 0.5200 |
| triviaqa | 100-125 | 24/25 = 96.00% | 0.7348 | 0.6800 |
| triviaqa | 125-150 | 20/25 = 80.00% | 0.6021 | 0.5200 |
| triviaqa | 150-175 | 24/25 = 96.00% | 0.6916 | 0.6000 |
| triviaqa | 175-200 | 25/25 = 100.00% | 0.6154 | 0.4800 |
| triviaqa | 200-225 | 23/25 = 92.00% | 0.5266 | 0.4000 |
| triviaqa | 225-250 | 17/25 = 68.00% | 0.3995 | 0.3200 |
| triviaqa | 250-270 | 18/20 = 90.00% | 0.7682 | 0.6500 |

Interpretation:

- The small attention smoothing setting (`uniform alpha=0.1`) generalizes clearly on HotpotQA and TriviaQA in this cross-dataset run, at least in absolute completed-run accuracy.
- 2WikiMQA remains much weaker at 48.00%, so this intervention is not uniformly robust across multi-hop styles.
- This result supports the earlier MuSiQue finding that a small perturbation to selected-token recompute attention can improve behavior, while large perturbations were destructive. The current run does not by itself prove the method beats each dataset baseline; it establishes that the setting is stable enough to complete full cross-dataset evaluation without the earlier v1 cache/OOM failure mode.

## 2026-07-13 Comparison Against Native Online DraftModel Rate 0.15

Baseline source:

- `MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/cross_dataset_summary.csv`
- Baseline row: `online_draft_rate015`
- Meaning in that table: online DraftModel selector, rate=0.15, same cross-dataset reflect format.

Compared method:

- `uniform_alpha0p1` from `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_alpha_cross_dataset_v2`
- Same selection logic as DraftModel rate=0.15, but during selected-token recompute, attention output is mixed with uniform attention using `alpha=0.1`.

Accuracy comparison:

| Dataset | Native DraftModel rate=0.15 | Uniform alpha=0.1 | Delta |
|---|---:|---:|---:|
| 2wikimqa | 101/200 = 50.50% | 96/200 = 48.00% | -2.50 pp |
| hotpotqa | 207/260 = 79.62% | 226/260 = 86.92% | +7.31 pp |
| triviaqa | 214/270 = 79.26% | 242/270 = 89.63% | +10.37 pp |
| micro total | 522/730 = 71.51% | 564/730 = 77.26% | +5.75 pp |

F1 / EM comparison:

| Dataset | Native F1 | Uniform F1 | Delta F1 | Native EM | Uniform EM | Delta EM |
|---|---:|---:|---:|---:|---:|---:|
| 2wikimqa | 0.2687 | 0.3669 | +0.0982 | 0.0150 | 0.2050 | +0.1900 |
| hotpotqa | 0.3245 | 0.6133 | +0.2888 | 0.0385 | 0.4846 | +0.4461 |
| triviaqa | 0.2137 | 0.6398 | +0.4261 | 0.0556 | 0.5333 | +0.4777 |

Interpretation:

- Against native online DraftModel rate=0.15, `uniform alpha=0.1` improves HotpotQA and TriviaQA substantially in main/sub accuracy.
- It hurts 2WikiMQA accuracy by 2.50 points, although F1/EM increase there. This means the smoothing changes answer form/content in a way that improves lexical overlap but slightly reduces binary judged correctness on 2WikiMQA.
- Micro-averaged over the three cross-dataset sets, `uniform alpha=0.1` improves from 522/730 to 564/730, a +42 example gain and +5.75 percentage points.
- This suggests the small uniform component is not just harmless noise; it is acting like a recompute-time attention regularizer. However, because 2WikiMQA regresses, this should not be treated as a universal replacement yet. The next high-value check is per-example flip analysis against native DraftModel to see whether improvements are concentrated in answerability/calibration cases or whether the judge/prompt format changed the comparison.

## 2026-07-13 Full Alpha Sweep Results

Correction to the earlier checkpoint: the fixed worker queue did run all planned combinations. The earlier report only summarized `uniform alpha=0.1`, which was incomplete. The completed combinations are:

- `uniform alpha=0.1`
- `uniform alpha=0.25`
- `random alpha=0.05`
- `random alpha=0.1`
- `random alpha=0.25`

All five combinations completed all 30 cross-dataset segments with no missing segment.

Final weighted aggregate by combination:

| Method | 2WikiMQA acc | HotpotQA acc | TriviaQA acc | Micro acc | Avg F1 by dataset |
|---|---:|---:|---:|---:|---|
| native online DraftModel rate=0.15 | 101/200 = 50.50% | 207/260 = 79.62% | 214/270 = 79.26% | 522/730 = 71.51% | 0.2687 / 0.3245 / 0.2137 |
| uniform alpha=0.1 | 96/200 = 48.00% | 226/260 = 86.92% | 242/270 = 89.63% | 564/730 = 77.26% | 0.3669 / 0.6133 / 0.6398 |
| uniform alpha=0.25 | 99/200 = 49.50% | 223/260 = 85.77% | 237/270 = 87.78% | 559/730 = 76.58% | 0.3419 / 0.5893 / 0.6043 |
| random alpha=0.05 | 94/200 = 47.00% | 231/260 = 88.85% | 241/270 = 89.26% | 566/730 = 77.53% | 0.3745 / 0.6184 / 0.6398 |
| random alpha=0.1 | 95/200 = 47.50% | 228/260 = 87.69% | 242/270 = 89.63% | 565/730 = 77.40% | 0.3652 / 0.6129 / 0.6367 |
| random alpha=0.25 | 100/200 = 50.00% | 224/260 = 86.15% | 238/270 = 88.15% | 562/730 = 76.99% | 0.3422 / 0.5990 / 0.6063 |

Delta versus native online DraftModel rate=0.15:

| Method | 2WikiMQA delta | HotpotQA delta | TriviaQA delta | Micro delta |
|---|---:|---:|---:|---:|
| uniform alpha=0.1 | -2.50 pp | +7.31 pp | +10.37 pp | +5.75 pp |
| uniform alpha=0.25 | -1.00 pp | +6.15 pp | +8.52 pp | +5.07 pp |
| random alpha=0.05 | -3.50 pp | +9.23 pp | +10.00 pp | +6.03 pp |
| random alpha=0.1 | -3.00 pp | +8.08 pp | +10.37 pp | +5.89 pp |
| random alpha=0.25 | -0.50 pp | +6.54 pp | +8.89 pp | +5.48 pp |

Interpretation:

- The best micro accuracy in this sweep is `random alpha=0.05`: 566/730 = 77.53%, +6.03 pp over native online DraftModel rate=0.15.
- `uniform alpha=0.1` is close: 564/730 = 77.26%, +5.75 pp.
- All tested perturbation settings improve HotpotQA and TriviaQA substantially over native DraftModel rate=0.15.
- All tested perturbation settings still underperform native DraftModel on 2WikiMQA accuracy, although F1/EM are higher than native on 2WikiMQA. This points to dataset-specific answer/judge behavior or a real multi-hop reasoning degradation that is not captured by overlap metrics alone.
- The sweep suggests a small stochastic or uniform smoothing term during selected-token recompute is consistently useful on HotpotQA/TriviaQA, but the correct alpha is small. The current evidence favors `random alpha=0.05` as the best cross-dataset setting, with `uniform alpha=0.1` as the best simple deterministic setting.

## 2026-07-13 Full Attention Baseline Added To Alpha Sweep Table

Full attention baseline source:

- `MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/cross_dataset_summary.csv`
- Baseline row: `full_rate1`
- Meaning: rate=1.0 full document-token recompute baseline, no selector.

Full attention / native Draft / alpha sweep comparison:

| Method | Rate | 2WikiMQA acc | HotpotQA acc | TriviaQA acc | Micro acc | Avg F1 by dataset |
|---|---:|---:|---:|---:|---:|---|
| full attention / full recompute | 1.00 | 113/200 = 56.50% | 207/260 = 79.62% | 211/270 = 78.15% | 531/730 = 72.74% | 0.3319 / 0.3076 / 0.2036 |
| native online DraftModel | 0.15 | 101/200 = 50.50% | 207/260 = 79.62% | 214/270 = 79.26% | 522/730 = 71.51% | 0.2687 / 0.3245 / 0.2137 |
| uniform alpha=0.1 | 0.15 | 96/200 = 48.00% | 226/260 = 86.92% | 242/270 = 89.63% | 564/730 = 77.26% | 0.3669 / 0.6133 / 0.6398 |
| uniform alpha=0.25 | 0.15 | 99/200 = 49.50% | 223/260 = 85.77% | 237/270 = 87.78% | 559/730 = 76.58% | 0.3419 / 0.5893 / 0.6043 |
| random alpha=0.05 | 0.15 | 94/200 = 47.00% | 231/260 = 88.85% | 241/270 = 89.26% | 566/730 = 77.53% | 0.3745 / 0.6184 / 0.6398 |
| random alpha=0.1 | 0.15 | 95/200 = 47.50% | 228/260 = 87.69% | 242/270 = 89.63% | 565/730 = 77.40% | 0.3652 / 0.6129 / 0.6367 |
| random alpha=0.25 | 0.15 | 100/200 = 50.00% | 224/260 = 86.15% | 238/270 = 88.15% | 562/730 = 76.99% | 0.3422 / 0.5990 / 0.6063 |

Delta versus full attention / full recompute:

| Method | 2WikiMQA delta | HotpotQA delta | TriviaQA delta | Micro delta |
|---|---:|---:|---:|---:|
| native online DraftModel | -6.00 pp | +0.00 pp | +1.11 pp | -1.23 pp |
| uniform alpha=0.1 | -8.50 pp | +7.31 pp | +11.48 pp | +4.52 pp |
| uniform alpha=0.25 | -7.00 pp | +6.15 pp | +9.63 pp | +3.84 pp |
| random alpha=0.05 | -9.50 pp | +9.23 pp | +11.11 pp | +4.79 pp |
| random alpha=0.1 | -9.00 pp | +8.08 pp | +11.48 pp | +4.66 pp |
| random alpha=0.25 | -6.50 pp | +6.54 pp | +10.00 pp | +4.25 pp |

Interpretation with full attention included:

- Full attention is not the best aggregate baseline on these cross-dataset reflect conversions: micro 72.74%, while the best alpha method (`random alpha=0.05`) reaches 77.53%.
- Full attention is still clearly best on 2WikiMQA: 56.50%, while all rate=0.15 Draft/alpha variants are below it.
- The alpha methods beat full attention on HotpotQA and TriviaQA by large margins. This likely means the full recompute path is not a clean universal oracle in this reflect evaluation setting; selected-token recompute plus mild attention perturbation can act as a useful regularizer for answer style/calibration.
- Therefore, the practical target should not be simply “approximate full attention exactly.” For HotpotQA/TriviaQA, exact full attention is not empirically optimal here. For 2WikiMQA, however, the full attention behavior remains a better target and the smoothing variants currently lose reasoning accuracy.

## 2026-07-13 Dataset-Difference Diagnosis: Why 2WikiMQA Is Weak

Question: why do small uniform/random attention perturbations improve HotpotQA / TriviaQA / MuSiQue but underperform on 2WikiMQA? Is it because RAG retrieves more documents, longer context, or task type?

### Dataset structure statistics

All three cross-dataset reflect sets use one intermediate query and 10 retrieved documents per example. Therefore, the 2WikiMQA regression is **not** caused by retrieving more documents in the cross-dataset setup.

| Dataset | examples | intermediate contexts | retrieved docs | words/doc | total retrieved words | question words |
|---|---:|---:|---:|---:|---:|---:|
| 2WikiMQA | 200 | mean 1.0 | mean 10.0 | mean 340.0, p90 670, max 817 | mean 3398.7, p90 4572, max 7146 | mean 11.7 |
| HotpotQA | 260 | mean 1.0 | mean 10.0 | mean 106.5, p90 160, max 411 | mean 1064.7, p90 1245, max 1556 | mean 18.1 |
| TriviaQA | 270 | mean 1.0 | mean 10.0 | mean 108.2, p90 162, max 558 | mean 1081.7, p90 1275, max 1722 | mean 12.4 |
| MuSiQue | 200 | mean 2.4 | mean 26.6 total across intermediate contexts | mean 88.6, p90 156, max 298 | mean 2361.8, p90 4030, max 6105 | mean 12.5 |

Important distinction:

- 2WikiMQA has the same document count as HotpotQA/TriviaQA, but each document is about 3.2x longer.
- MuSiQue has more total retrieved documents, but they are split across multiple intermediate contexts and are much shorter per document. The effective sub-query retrieval unit is closer to short-snippet retrieval, not 10 long Wikipedia pages in one context.

### Question-type differences

Rough regex buckets from the converted reflect examples:

| Dataset | yes/no | which | comparison-like | date/year/born/died | WH/entity |
|---|---:|---:|---:|---:|---:|
| 2WikiMQA | 10.0% | 29.0% | 25.5% | 31.0% | 87.0% |
| HotpotQA | 6.9% | 34.6% | 11.9% | 17.3% | 88.1% |
| TriviaQA | 0.4% | 44.1% | 3.0% | 5.6% | 91.5% |
| MuSiQue | 0.0% | 13.0% | 1.5% | 14.5% | 97.5% |

2WikiMQA is much more concentrated in comparison and temporal/entity-disambiguation questions. These are exactly the cases where small attention smoothing can hurt: the model often needs to preserve a sharp link between two entities and their attributes, not just improve answer style or calibrate extraction.

### Paired flip analysis against native DraftModel

Native baseline: `online_draft_rate015` from `cross_dataset_offline_generalization`.
Alpha methods compared on the same question keys.

For `uniform alpha=0.1`:

| Dataset | wrong->right | right->wrong | net | native acc | alpha acc |
|---|---:|---:|---:|---:|---:|
| 2WikiMQA | 21 | 26 | -5 | 50.5% | 48.0% |
| HotpotQA | 30 | 11 | +19 | 79.6% | 86.9% |
| TriviaQA | 34 | 6 | +28 | 79.3% | 89.6% |

For `random alpha=0.05`:

| Dataset | wrong->right | right->wrong | net | native acc | alpha acc |
|---|---:|---:|---:|---:|---:|
| 2WikiMQA | 19 | 26 | -7 | 50.5% | 47.0% |
| HotpotQA | 33 | 9 | +24 | 79.6% | 88.8% |
| TriviaQA | 34 | 7 | +27 | 79.3% | 89.3% |

2WikiMQA is not failing because the perturbation never helps; it still fixes 19-21 examples. The problem is that it breaks slightly more originally correct examples, especially in long-context and comparison/which categories.

### Where 2WikiMQA breaks

For `uniform alpha=0.1` on 2WikiMQA:

| Bucket | n | wrong->right | right->wrong | net | native acc | alpha acc |
|---|---:|---:|---:|---:|---:|---:|
| all | 200 | 21 | 26 | -5 | 50.5% | 48.0% |
| comparison-like | 33 | 2 | 10 | -8 | 78.8% | 54.5% |
| which-other | 14 | 1 | 5 | -4 | 57.1% | 28.6% |
| date/year | 46 | 3 | 4 | -1 | 58.7% | 56.5% |
| other | 87 | 8 | 5 | +3 | 36.8% | 40.2% |
| long context, >2500 retrieved words | 172 | 14 | 24 | -10 | 49.4% | 43.6% |
| short context, <1300 retrieved words | 9 | 5 | 0 | +5 | 33.3% | 88.9% |

For `random alpha=0.05` on 2WikiMQA:

| Bucket | n | wrong->right | right->wrong | net | native acc | alpha acc |
|---|---:|---:|---:|---:|---:|---:|
| all | 200 | 19 | 26 | -7 | 50.5% | 47.0% |
| comparison-like | 33 | 2 | 9 | -7 | 78.8% | 57.6% |
| date/year | 46 | 2 | 4 | -2 | 58.7% | 54.3% |
| long context, >2500 retrieved words | 172 | 12 | 24 | -12 | 49.4% | 42.4% |
| short context, <1300 retrieved words | 9 | 5 | 0 | +5 | 33.3% | 88.9% |

This is the strongest evidence so far. Small attention smoothing helps short-context 2WikiMQA cases, but most 2WikiMQA examples are long-context cases, and those regress.

### Working hypothesis

The method appears to work best when the retrieved context is short-snippet style and the model benefits from regularized recompute attention:

- HotpotQA: 10 docs, but short snippets, about 1065 total retrieved words.
- TriviaQA: 10 docs, also short snippets, about 1082 total retrieved words.
- MuSiQue: many retrieved docs in total, but decomposed into multiple intermediate contexts with short docs; the sub-query unit is short and focused.

The method is weak when the task requires sharp entity-attribute binding over long retrieved pages:

- 2WikiMQA: 10 docs but about 3399 retrieved words on average, with p90 above 4572 words.
- High comparison and temporal-disambiguation rates.
- Long-context examples dominate the dataset and show negative net flips.

Interpretation for future method design:

- A global fixed alpha is too crude. The adapter/attention perturbation should be context-aware.
- For short retrieved contexts or extraction-style tasks, small smoothing/noise can improve calibration and answer formatting.
- For long-document comparison/temporal reasoning, recompute needs sharper attention preservation; alpha should be reduced or disabled, or the method should selectively avoid smoothing comparison-critical heads/layers/tokens.
- This supports a gated adapter design rather than a universal perturbation: use features such as total retrieved length, question type, and possibly attention entropy / selected-token concentration to choose whether to apply smoothing.


## 2026-07-13 Better-than-Baseline Alpha整理图表

本节把已经完成、可比较的 attention alpha 实验统一整理到表和图中。Baseline 统一定义为各数据集的 `native online DraftModel, rate=0.15`。

新增文件：

- `alpha_vs_native_baseline_summary.csv`: 四个数据集 + cross-dataset micro 的全量 delta 表。
- `better_than_native_baseline_alpha.csv`: 只保留 main accuracy 高于 native baseline 的 rows。
- `figures/alpha_vs_native_baseline_delta_heatmap.png`: 所有 alpha 组合相对 baseline 的 accuracy delta heatmap。
- `figures/better_than_native_baseline_alpha_bars.png`: 只画正收益数据集的 bar plot。
- `scripts/plot_better_than_baseline_alpha.py`: 复现整理表和图的脚本。

核心结果：

| Dataset | Native Draft baseline | Best alpha setting | Best alpha acc | Delta |
|---|---:|---|---:|---:|
| 2WikiMQA | 101/200 = 50.50% | none beats baseline | 100/200 = 50.00% (`random alpha=0.25`) | -0.50 pp |
| HotpotQA | 207/260 = 79.62% | `random alpha=0.05` | 231/260 = 88.85% | +9.23 pp |
| TriviaQA | 214/270 = 79.26% | `uniform alpha=0.1` / `random alpha=0.1` | 242/270 = 89.63% | +10.37 pp |
| MuSiQue | 99/135 = 73.33% | `uniform alpha=0.1` / `uniform alpha=0.25` | 107/135 = 79.26% | +5.93 pp |
| Micro over 2WikiMQA+HotpotQA+TriviaQA | 522/730 = 71.51% | `random alpha=0.05` | 566/730 = 77.53% | +6.03 pp |

结论：

- 这组 alpha smoothing 不是所有数据集都稳定提升。它在 HotpotQA、TriviaQA、MuSiQue 上明显超过 native DraftModel baseline，但在 2WikiMQA 上没有超过 baseline。
- 最稳的 cross-dataset 设置是 `random alpha=0.05`，micro accuracy 从 71.51% 提到 77.53%。
- 最简单的确定性设置是 `uniform alpha=0.1`，micro accuracy 77.26%，并且在 MuSiQue 上达到当前最好 main accuracy。
- 2WikiMQA 的负收益和前面的诊断一致：长上下文、comparison/temporal/entity-binding 更多，过度 smoothing 容易破坏精确绑定。因此后续如果继续这条线，应做 gated alpha，而不是全局固定 alpha。

复现命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/plot_better_than_baseline_alpha.py
```


### 2026-07-13 Replot with DraftModel baseline and all datasets

根据反馈，重新绘制了更接近 `selection_cost_figures/qk_ttft_breakdown_by_rate.png` 风格的图：低饱和 teal/coral/periwinkle 配色，保留浅网格，不再使用红绿 heatmap 作为主图。

新增图：

- `figures/alpha_accuracy_by_dataset_with_draft_baseline.png`: 每个数据集均画出 `native DraftModel` baseline 和 5 个 alpha 组合的 main accuracy。
- `figures/alpha_delta_by_dataset_with_all_methods.png`: 每个数据集均画出 5 个 alpha 组合相对 `native DraftModel` baseline 的 delta，包含 2WikiMQA 的负收益。

这两张图比前一版更适合读结论：

- `alpha_accuracy_by_dataset_with_draft_baseline.png` 用于看绝对性能和 baseline 差距。
- `alpha_delta_by_dataset_with_all_methods.png` 用于看每个方法在哪些数据集提升、在哪些数据集下降。

关键视觉结论不变：HotpotQA / TriviaQA / MuSiQue 上 alpha smoothing 明显高于 native DraftModel；2WikiMQA 上所有 alpha 组合都没有超过 native DraftModel。


### 2026-07-13 Reference-style replot with full/QK/Draft baselines

按 `selection_cost_figures/update_selection_cost_figures.py` 的绘图风格重画：无主标题、浅 grid、无 bar 数值标注、同时输出 PNG/PDF。

新增文件：

- `alpha_with_full_qk_draft_accuracy_summary.csv`: 每个数据集下 `Full attention r=1`、`Online QK r=0.15`、`Online Draft r=0.15` 和 alpha 组合的 accuracy / delta 表。
- `figures/alpha_accuracy_with_full_qk_draft_reference_style.png/pdf`: 绝对 main accuracy，包含 full attention、online QK、online Draft 和所有 alpha 组合。
- `figures/alpha_delta_with_full_qk_draft_reference_style.png/pdf`: 相对 `Online Draft r=0.15` 的 delta，包含 full attention、online QK 和 alpha 组合。
- `scripts/plot_alpha_reference_style.py`: 新版绘图脚本。

数据源：

- 2WikiMQA / HotpotQA / TriviaQA 的 full/QK/Draft: `cross_dataset_offline_generalization/cross_dataset_summary.csv`。
- MuSiQue 的 QK/Draft: `qwen3_rate015_online_offline/summary.csv`。
- MuSiQue 的 full attention: `qwen3_hybrid70_online_baselines/summary.csv`。
- Alpha 方法: `alpha_vs_native_baseline_summary.csv` 和 MuSiQue alpha summary。


### 2026-07-13 Micro aggregate single-figure plot

根据反馈，不再按 dataset 分块，而是把 2WikiMQA、HotpotQA、TriviaQA、MuSiQue 拼成一个总集来画。横轴为方法，纵轴为 micro main accuracy。

新增文件：

- `micro_all_datasets_method_accuracy.csv`: 四个数据集合并后的 correct / total / accuracy / delta 表。
- `figures/micro_all_datasets_method_accuracy.png/pdf`: 单图方法对比。baseline 使用灰阶，alpha 方法统一使用蓝绿色，避免多色混乱。
- `scripts/plot_micro_method_accuracy.py`: 复现脚本。

Micro 总集大小：`865` examples。

| Method | Correct / Total | Main Acc | Delta vs Online Draft |
|---|---:|---:|---:|
| Full attention r=1 | 636/865 | 73.53% | +1.73 pp |
| Online QK r=0.15 | 609/865 | 70.40% | -1.39 pp |
| Online Draft r=0.15 | 621/865 | 71.79% | +0.00 pp |
| Uniform a=0.1 | 671/865 | 77.57% | +5.78 pp |
| Uniform a=0.25 | 666/865 | 76.99% | +5.20 pp |
| Random a=0.05 | 671/865 | 77.57% | +5.78 pp |
| Random a=0.1 | 670/865 | 77.46% | +5.66 pp |
| Random a=0.25 | 668/865 | 77.23% | +5.43 pp |

这张图用于展示总体趋势：小 alpha smoothing 在四数据集合并后稳定超过 Online Draft、Online QK 和 full attention。


## 2026-07-13 Rate=1 Baseline Source Audit

Question: are the `Full attention r=1` numbers used in the alpha comparison truly same-run baselines?

Short answer: **No. They are valid historical baseline rows, but not same-directory / same-run baselines for the alpha sweep.** Therefore any statement that alpha is strictly better than full attention should be treated as provisional until a same-pipeline rate=1 rerun is completed.

Audit evidence:

- Alpha result CSV files contain `Rate1_Predicted`, `Rate1_Correct`, `Rate1_F1`, `Rate1_EM` columns.
- In the original alpha result shards, these columns are `N/A`; re-aggregation over `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_alpha_cross_dataset_v2` found `rate1_non_na=0` for all 15 dataset/method combinations.
- The code in `test_fusionrag_reflect_preprocess_exp.py` only fills `Rate1_*` by finding `rate_1*.csv` under the **same output directory**. Since alpha output directories do not contain same-run rate=1 CSVs, no automatic aligned baseline was used.
- The figure used `Full attention r=1` from historical summaries:
  - 2WikiMQA / HotpotQA / TriviaQA: `cross_dataset_offline_generalization/cross_dataset_summary.csv`.
  - MuSiQue: `qwen3_hybrid70_online_baselines/summary.csv`.

Current status of the previous plot:

- The arithmetic is correct for the historical baseline: `636/865 = 73.53%`.
- The alpha numbers are also correctly re-aggregated from raw shards.
- The comparison is **not strict same-run evidence** that alpha beats full attention.
- The correct wording is: alpha beats the currently recorded historical rate=1 baselines under the current aggregation, but this must be verified with same-pipeline rate=1 runs.

Implementation-side check:

- The alpha ablation is passed through `FUSIONRAG_REPROCESS_ATTENTION_ABLATION` and `FUSIONRAG_REPROCESS_ATTENTION_ABLATION_ALPHA`.
- In `models/modeling_qwen3.py`, the ablation applies only to the first `reprocess_doc_token_count` query positions in the reprocess forward. In the default non-strict path, those are selected document tokens; query tokens are appended after them and are not directly uniform/random ablated.
- No direct evidence was found that alpha accidentally uses rate=1 outputs or loads a same-run rate=1 baseline.

Required follow-up to close this:

1. Run rate=1 baselines in the same experiment family / same summarizer as the alpha runs.
2. Prefer one output root containing both `rate_1.0_revert_rope.csv` and alpha CSVs, so `Rate1_*` fields are populated automatically.
3. Regenerate the micro plot with the same-run rate=1 numbers only.

New audit file: `rate1_baseline_source_audit.csv`.

## 2026-07-13 Same-pipeline rate=1 rerun launched

Purpose: close the baseline-source gap found in `rate1_baseline_source_audit.csv` by rerunning `rate=1.0` with the same alpha-experiment family settings.

Run root:

- `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_alpha_rate1_same_pipeline`

Script:

- `scripts/run_same_pipeline_rate1.py`
- `scripts/summarize_same_pipeline_rate1.py`

Launch command:

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
EXP=/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_alpha_rate1_same_pipeline
mkdir -p "$EXP/logs"
nohup MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/run_same_pipeline_rate1.py \
  > "$EXP/logs/launcher_qjy000.log" 2>&1 < /dev/null &
echo $! > "$EXP/logs/launcher_qjy000.pid"
```

Configuration:

- Model: Qwen3-32B
- `--rate 1.0`
- `--preprocess true`
- `--recall_method bge`
- `--preprocess_scope global`
- `--topk 10`
- `--revert_rope true`
- `--reprocess_method DraftModel`
- `--draft_model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`
- Same GLM judge endpoint/model as alpha runs: `GLM-5.2`
- Result layout: `full_rate1_draft_layout/{dataset}/seg_*`
- Cache root: `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-attn-alpha-cross-cache-v2/worker_gpu*/{dataset}`

Datasets queued:

- 2WikiMQA: 0-200
- HotpotQA: 0-260
- TriviaQA: 0-270
- MuSiQue: 0-200 raw reflect samples; final main/sub denominators should be summarized with the same MuSiQue grouping logic as previous alpha summaries.

Progress command:

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/summarize_same_pipeline_rate1.py
```

Status at launch: queue started on qjy000 GPUs 0-7. No finished segment yet at the first immediate summary check.

Additional launch on qjy003:

```bash
cd /home/hming/FusionRAG-pca-analysis
EXP=/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_alpha_rate1_same_pipeline
mkdir -p "$EXP/logs"
FUSIONRAG_RATE1_DATASETS=hotpotqa,triviaqa,musique \
FUSIONRAG_RATE1_WORKER_GPUS=0,1,2,3,4,5,6,7 \
nohup MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/run_same_pipeline_rate1.py \
  > "$EXP/logs/launcher_qjy003_hotpot_trivia_musique.log" 2>&1 < /dev/null &
echo $! > "$EXP/logs/launcher_qjy003_hotpot_trivia_musique.pid"
```

Immediate status after launch:

- qjy000 GPUs 0-7: running 2WikiMQA rate=1 shards.
- qjy003 GPUs 0-7: running HotpotQA rate=1 shards first, then queued TriviaQA and MuSiQue.
- qjy001 was not used for this rerun.
- `summarize_same_pipeline_rate1.py` was fixed to ignore partial CSV files and count only shards whose `run.log` contains `FINAL RESULTS`.

## 2026-07-13 Uniform alpha=0.1 rate sweep launched on qjy001

Purpose: collect `uniform alpha=0.1` attention-ablation results at higher update rates.

Run root:

- `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_uniform_a01_rate_sweep`

Scripts:

- `scripts/run_uniform_a01_rate_sweep.py`
- `scripts/summarize_uniform_a01_rate_sweep.py`

Configuration:

- Model: Qwen3-32B
- Draft model: Qwen2.5-3B-Instruct
- `FUSIONRAG_REPROCESS_ATTENTION_ABLATION=uniform`
- `FUSIONRAG_REPROCESS_ATTENTION_ABLATION_ALPHA=0.1`
- Rates: `0.3, 0.5, 0.8, 1.0`
- Datasets: 2WikiMQA, HotpotQA, TriviaQA, MuSiQue
- `--preprocess true`, `--recall_method bge`, `--preprocess_scope global`, `--topk 10`, `--revert_rope true`
- Judge: GLM-5.2, same endpoint as previous alpha runs
- Cache root: `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-uniform-a01-rate-sweep-cache`

Launch command:

```bash
cd /home/hming/FusionRAG-pca-analysis
EXP=/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_uniform_a01_rate_sweep
mkdir -p "$EXP/logs"
FUSIONRAG_UNIFORM_A01_WORKER_GPUS=1,2,3,4,5,6,7 \
nohup MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/run_uniform_a01_rate_sweep.py \
  > "$EXP/logs/launcher_qjy001.log" 2>&1 < /dev/null &
echo $! > "$EXP/logs/launcher_qjy001.pid"
```

qjy001 status at launch:

- GPU0 already occupied by existing KV blend beta job and was not used.
- GPUs 1-7 were idle and are now running this rate sweep.
- No 235B process was launched.

Progress command:

```bash
cd /home/hming/FusionRAG-pca-analysis
MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/summarize_uniform_a01_rate_sweep.py
```

Immediate status: queue launched; first tasks are `2wikimqa rate=0.3` shards; no completed segment at first summary check.

## 2026-07-13 Uniform alpha=0.1 rate sweep rerun after cache cleanup

Cleanup:

- Removed failed/corrupt cache directory: `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-uniform-a01-rate-sweep-cache`.
- This directory was created by the failed qjy001 sweep and contained partially written KV cache files after `No space left on device` errors.
- No model files, result CSVs, README/log files, or unrelated experiment caches were removed.

Rerun location:

- Result root: `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_uniform_a01_rate_sweep_rerun`
- Cache root: `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-uniform-a01-rate-sweep-cache-rerun`
- Reason: `/mnt/qjhs-sh-lab-04` remained nearly full after cleanup, while `/mnt/qjhs-sh-lab-03` had about 11T free.

Launch command on qjy001:

```bash
cd /home/hming/FusionRAG-pca-analysis
EXP=/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_uniform_a01_rate_sweep_rerun
CACHE=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-uniform-a01-rate-sweep-cache-rerun
mkdir -p "$EXP/logs" "$CACHE"
FUSIONRAG_UNIFORM_A01_RATE_SWEEP_ROOT="$EXP" \
FUSIONRAG_UNIFORM_A01_CACHE_ROOT="$CACHE" \
FUSIONRAG_UNIFORM_A01_WORKER_GPUS=1,2,3,4,5,6,7 \
FUSIONRAG_UNIFORM_A01_RATES=0.3,0.5,0.8,1.0 \
nohup MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/run_uniform_a01_rate_sweep.py \
  > "$EXP/logs/launcher_qjy001_rerun.log" 2>&1 < /dev/null &
echo $! > "$EXP/logs/launcher_qjy001_rerun.pid"
```

Current status at launch verification:

- qjy001 GPU0 is occupied by the existing KV blend beta job and was not used.
- qjy001 GPUs 1-7 loaded Qwen3-32B and started `rate=0.3` 2WikiMQA shards.
- First observed task: `uniform_alpha0p1_rate0p3/2wikimqa/seg_0_25`, already generating system/document KV cache.
- No 235B job was launched.

## 2026-07-13 Stop duplicated rate sweep and reorganize cache plan

The `uniform alpha=0.1` rate sweep rerun on qjy001 was stopped before completion because the cache layout was wrong.

What happened:

- The script used `--cache_path $CACHE_ROOT/worker_gpu{gpu}/{dataset}`.
- This did not create a separate cache per rate, but it did create a separate cache per GPU worker.
- As a result, the same dataset raw/preprocess KV was generated repeatedly across workers.
- Example: 2WikiMQA had seven duplicated worker caches, each around 211-239G.
- This made the sweep much slower than expected and consumed about 1.7T on `/mnt/qjhs-sh-lab-03`.

Cleanup performed:

- Stopped qjy001 `run_uniform_a01_rate_sweep.py` launcher/workers and the child `test_fusionrag_reflect_preprocess_exp.py` jobs that used:
  - `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_uniform_a01_rate_sweep_rerun`
  - `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-uniform-a01-rate-sweep-cache-rerun`
- Removed only the duplicated cache directory:
  - `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-qwen3-uniform-a01-rate-sweep-cache-rerun`
- Kept result/log directory for traceability:
  - `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_uniform_a01_rate_sweep_rerun`
- No model files, README files, result CSVs, or unrelated caches were removed.

Observed partial result before stopping:

| Rate | Dataset | Finished rows | Correct | Main acc | Avg F1 | Avg EM |
|---:|---|---:|---:|---:|---:|---:|
| 0.3 | 2WikiMQA | 175 | 83 | 47.43 | 34.95 | 19.43 |

This partial result is not a final dataset result and should not be used as a completed benchmark.

Reorganization plan:

1. Use a canonical preprocess cache rather than experiment-private worker caches.
2. Cache key should include only content-changing preprocess parameters:
   - model/tokenizer
   - dataset/data file
   - preprocess scope
   - recall method
   - topk
   - BGE model/version
3. Cache key should not include downstream experiment parameters:
   - alpha/random/uniform attention ablation
   - beta blend value
   - recompute rate
   - DraftModel vs QK selection after cache loading
4. Generate canonical cache in a separate single-writer stage.
5. Run experiments in read-only cache mode. If cache is missing, fail early instead of regenerating in multiple workers.
6. Keep result paths separate from cache paths.
7. Treat cross-dataset caches and MuSiQue caches separately for now:
   - `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-attn-alpha-cross-cache-v2` contains large Qwen3 cache coverage for 2WikiMQA/HotpotQA/TriviaQA, but not useful MuSiQue coverage.
   - MuSiQue cache coverage currently appears mostly in beta-related caches and is fragmented by worker/host.

Next implementation step:

- Add a cache audit script that reports, per candidate cache root and dataset:
  - raw KV key/value file counts
  - preprocess KV key/value file counts
  - missing/corrupt `.pt` files sampled or fully checked
  - whether the root is suitable as canonical read-only cache
- Then update alpha/blend/rate sweep launchers so that all workers point to the same canonical cache root and do not create `worker_gpu*` cache copies.

## 2026-07-13 MuSiQue uniform alpha=0.1 rate sweep using existing shared cache

Purpose: rerun MuSiQue rate sweep without regenerating preprocess KV.

Existing cache selected:

- `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-kv-blend-beta-shared-cache/musique`
- Coverage observed before launch:
  - raw key/value files: `3122 / 3122`
  - preprocess key/value files: `1584 / 1584`
- This is the most complete existing MuSiQue preprocess cache found. Other beta-full MuSiQue caches are fragmented by host/GPU worker.

Code guard added:

- `test_fusionrag_reflect_preprocess_exp.py` now supports:
  - `FUSIONRAG_PREPROCESS_CACHE_READONLY=1`
- In read-only mode, missing raw/preprocess KV raises `FileNotFoundError` instead of generating new `.pt` cache files.
- Default behavior is unchanged when the env var is not set.

Launched on qjy001:

```bash
cd /home/hming/FusionRAG-pca-analysis
EXP=/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_uniform_a01_musique_rate_sweep_sharedcache
CACHE=/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-kv-blend-beta-shared-cache/musique

# One GPU per rate, shared read-only cache.
# rate=0.3 -> GPU0
# rate=0.5 -> GPU1
# rate=0.8 -> GPU2
# rate=1.0 -> GPU3
```

Common command template:

```bash
CUDA_VISIBLE_DEVICES=$GPU \
PYTHONUNBUFFERED=1 \
FUSIONRAG_PREPROCESS_CACHE_READONLY=1 \
FUSIONRAG_REPROCESS_ATTENTION_ABLATION=uniform \
FUSIONRAG_REPROCESS_ATTENTION_ABLATION_ALPHA=0.1 \
nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python test_fusionrag_reflect_preprocess_exp.py \
  --model_type qwen3 \
  --model_path /mnt/qjhs-sh-lab-01/models/Qwen3-32B \
  --model_name Qwen3-32B \
  --data_path data/result_reflect.json \
  --dataset_name musique \
  --cache_path "$CACHE" \
  --result_path "$OUT" \
  --start_sample 0 \
  --end_sample 135 \
  --rate "$RATE" \
  --topk 10 \
  --preprocess true \
  --recall_method bge \
  --reprocess_method DraftModel \
  --draft_model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct \
  --revert_rope true \
  --preprocess_scope global \
  --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3 \
  --device cuda:0 \
  --use_multi_gpu false \
  --openai_base_url http://36.150.226.221:32355/v1 \
  --openai_api_key api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS \
  --openai_model GLM-5.2
```

Initial status:

- qjy001 GPU0-3 loaded Qwen3-32B and started running rates `0.3, 0.5, 0.8, 1.0`.
- Logs show `Answering sub-questions with on-demand KV cache generation` and progress through MuSiQue main questions.
- Logs do not show `Generating system KV cache`, `Generated KV cache`, `Preprocessing document`, `Read-only cache mode`, or `Traceback` at the first check, so the run is using existing cache rather than regenerating preprocess KV.

Result root:

- `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_uniform_a01_musique_rate_sweep_sharedcache`

## 2026-07-13 Cross-dataset uniform alpha=0.1 rate sweep using existing cache

Purpose: run the remaining datasets with shared/read-only preprocess KV, including `rate=0.15` as a validation point.

Code guard:

- All runs use `FUSIONRAG_PREPROCESS_CACHE_READONLY=1`.
- Missing raw/preprocess KV should fail with `FileNotFoundError`; no new `.pt` cache should be generated.

Result root:

- `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_uniform_a01_cross_rate_sweep_sharedcache`

Cache roots used:

- Cross datasets: `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-attn-alpha-cross-cache-v2`
- MuSiQue: `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-kv-blend-beta-shared-cache/musique`

Launched tasks:

| Host | GPU | Dataset | Rate | Sample range | Cache path |
|---|---:|---|---:|---|---|
| qjy000 | 0 | 2WikiMQA | 0.15 | 0-200 | `.../worker_gpu1/2wikimqa` |
| qjy000 | 1 | 2WikiMQA | 0.3 | 0-200 | `.../worker_gpu1/2wikimqa` |
| qjy000 | 2 | 2WikiMQA | 0.5 | 0-200 | `.../worker_gpu1/2wikimqa` |
| qjy000 | 3 | 2WikiMQA | 0.8 | 0-200 | `.../worker_gpu6/2wikimqa` |
| qjy000 | 4 | 2WikiMQA | 1.0 | 0-200 | `.../worker_gpu6/2wikimqa` |
| qjy000 | 5 | HotpotQA | 0.15 | 0-260 | `.../worker_gpu3/hotpotqa` |
| qjy000 | 6 | HotpotQA | 0.3 | 0-260 | `.../worker_gpu4/hotpotqa` |
| qjy000 | 7 | HotpotQA | 0.5 | 0-260 | `.../worker_gpu6/hotpotqa` |
| qjy003 | 0 | HotpotQA | 0.8 | 0-260 | `.../worker_gpu7/hotpotqa` |
| qjy003 | 1 | HotpotQA | 1.0 | 0-260 | `.../worker_gpu7/hotpotqa` |
| qjy003 | 2 | TriviaQA | 0.15 | 0-270 | `.../worker_gpu0/triviaqa` |
| qjy003 | 3 | TriviaQA | 0.3 | 0-270 | `.../worker_gpu1/triviaqa` |
| qjy003 | 4 | TriviaQA | 0.5 | 0-270 | `.../worker_gpu2/triviaqa` |
| qjy003 | 5 | TriviaQA | 0.8 | 0-270 | `.../worker_gpu0/triviaqa` |
| qjy003 | 6 | TriviaQA | 1.0 | 0-270 | `.../worker_gpu1/triviaqa` |
| qjy003 | 7 | MuSiQue | 0.15 | 0-135 | `fusionrag-qwen3-kv-blend-beta-shared-cache/musique` |

Common settings:

- `FUSIONRAG_REPROCESS_ATTENTION_ABLATION=uniform`
- `FUSIONRAG_REPROCESS_ATTENTION_ABLATION_ALPHA=0.1`
- `FUSIONRAG_PREPROCESS_CACHE_READONLY=1`
- Qwen3-32B main model
- Qwen2.5-3B-Instruct DraftModel
- `--preprocess true --recall_method bge --preprocess_scope global --topk 10 --revert_rope true`
- Judge: GLM-5.2

Initial verification:

- qjy000 and qjy003 processes launched successfully.
- Logs did not show `Generating system KV cache`, `Generated KV cache`, `Preprocessing document`, `Read-only cache mode`, or `Traceback` at the first check.
- Therefore the runs appear to be reading existing preprocess KV rather than generating new cache.

### 2026-07-13 current results snapshot for shared-cache rate sweep

All runs below use existing preprocess KV with `FUSIONRAG_PREPROCESS_CACHE_READONLY=1` and `uniform alpha=0.1`.

Completed cross-dataset rows at this snapshot:

| Dataset | Rate | Rows | Correct | Acc | Avg F1 | Avg EM |
|---|---:|---:|---:|---:|---:|---:|
| 2WikiMQA | 0.15 | 200 | 96 | 48.00 | 36.69 | 20.50 |
| 2WikiMQA | 0.3 | 200 | 97 | 48.50 | 35.20 | 19.50 |
| 2WikiMQA | 1.0 | 200 | 97 | 48.50 | 36.40 | 21.00 |
| HotpotQA | 0.15 | 260 | 228 | 87.69 | 61.33 | 48.46 |
| HotpotQA | 0.3 | 260 | 224 | 86.15 | 62.26 | 49.62 |
| HotpotQA | 0.5 | 260 | 225 | 86.54 | 61.76 | 50.00 |
| HotpotQA | 1.0 | 260 | 228 | 87.69 | 61.42 | 48.46 |
| TriviaQA | 0.15 | 270 | 242 | 89.63 | 63.98 | 53.33 |
| TriviaQA | 0.3 | 270 | 242 | 89.63 | 65.26 | 54.81 |
| TriviaQA | 0.5 | 270 | 238 | 88.15 | 64.97 | 54.81 |
| TriviaQA | 0.8 | 270 | 240 | 88.89 | 66.05 | 55.56 |
| TriviaQA | 1.0 | 270 | 241 | 89.26 | 67.78 | 57.41 |

Still running / partial at this snapshot:

| Dataset | Rate | Rows so far | Correct so far | Partial acc |
|---|---:|---:|---:|---:|
| 2WikiMQA | 0.5 | 176 | 88 | 50.00 |
| 2WikiMQA | 0.8 | 136 | 66 | 48.53 |
| HotpotQA | 0.8 | 240 | 210 | 87.50 |

MuSiQue requires grouped main-question accuracy because each row is a sub-question:

| Rate | Sub rows | Sub correct | Sub acc | Main total | Main correct | Main acc |
|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 180 | 156 | 86.67 | 97 | 76 | 78.35 |
| 0.3 | 180 | 160 | 88.89 | 97 | 78 | 80.41 |
| 0.5 | 180 | 159 | 88.33 | 97 | 79 | 81.44 |
| 0.8 | 180 | 167 | 92.78 | 97 | 84 | 86.60 |
| 1.0 | 180 | 163 | 90.56 | 97 | 80 | 82.47 |

Process status at snapshot:

- qjy000 still running: 2WikiMQA `rate=0.5` and `rate=0.8`.
- qjy003 still running: HotpotQA `rate=0.8`.
- qjy001 MuSiQue `rate=0.3/0.5/0.8/1.0` completed; qjy003 MuSiQue `rate=0.15` completed.
- No `Read-only cache mode`, `Traceback`, `No space left`, `Generating system KV cache`, or `Preprocessing document` messages were found in logs at this snapshot.

## 2026-07-13 corrected summary plot and native QK/Draft rerun

Corrected plot:

- Replaced the old historical full-attention baseline `636/865 = 73.53%` with the same-pipeline corrected full-attention rerun `680/865 = 78.61%`.
- New table:
  - `MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/micro_all_datasets_method_accuracy_corrected_full.csv`
- New figures:
  - `MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/figures/micro_all_datasets_method_accuracy_corrected_full.png`
  - `MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/figures/micro_all_datasets_method_accuracy_corrected_full.pdf`

Corrected micro summary currently shown in the figure:

| Method | Correct / Total | Acc | Delta vs corrected full |
|---|---:|---:|---:|
| Full attention r=1 corrected | 680 / 865 | 78.61 | 0.00 pp |
| Online QK r=0.15 historical | 609 / 865 | 70.40 | -8.21 pp |
| Online Draft r=0.15 historical | 621 / 865 | 71.79 | -6.82 pp |
| Uniform a=0.1 r=0.15 | 671 / 865 | 77.57 | -1.04 pp |
| Uniform a=0.25 r=0.15 | 666 / 865 | 76.99 | -1.62 pp |
| Random a=0.05 r=0.15 | 671 / 865 | 77.57 | -1.04 pp |
| Random a=0.1 r=0.15 | 670 / 865 | 77.46 | -1.16 pp |
| Random a=0.25 r=0.15 | 668 / 865 | 77.23 | -1.39 pp |

Native Online QK / DraftModel rerun:

Purpose: rerun `online_qk_rate015` and `online_draft_rate015` under the current shared-cache/read-only setup to see whether the historical native baselines change.

Result root:

- `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_native_qk_draft_rate015_sharedcache_rerun`

Common settings:

- `FUSIONRAG_PREPROCESS_CACHE_READONLY=1`
- `rate=0.15`
- `--preprocess true --recall_method bge --preprocess_scope global --topk 10 --revert_rope true`
- Qwen3-32B main model
- Qwen2.5-3B-Instruct draft model for DraftModel runs
- Judge: GLM-5.2

Definitions:

- `online_qk_rate015`: `--reprocess_method FusionRAG`
- `online_draft_rate015`: `--reprocess_method DraftModel`

Launched on qjy003:

| GPU | Method | Dataset | Range | Cache |
|---:|---|---|---|---|
| 0 | online_qk_rate015 | 2WikiMQA | 0-200 | alpha cross cache `worker_gpu1/2wikimqa` |
| 1 | online_qk_rate015 | HotpotQA | 0-260 | alpha cross cache `worker_gpu3/hotpotqa` |
| 2 | online_qk_rate015 | TriviaQA | 0-270 | alpha cross cache `worker_gpu0/triviaqa` |
| 3 | online_qk_rate015 | MuSiQue | 0-135 | beta shared MuSiQue cache |
| 4 | online_draft_rate015 | 2WikiMQA | 0-200 | alpha cross cache `worker_gpu6/2wikimqa` |
| 5 | online_draft_rate015 | HotpotQA | 0-260 | alpha cross cache `worker_gpu4/hotpotqa` |
| 6 | online_draft_rate015 | TriviaQA | 0-270 | alpha cross cache `worker_gpu1/triviaqa` |
| 7 | online_draft_rate015 | MuSiQue | 0-135 | beta shared MuSiQue cache |

Initial status:

- All 8 qjy003 GPUs loaded jobs.
- No completed CSV yet at first check.
- No `Read-only cache mode`, `Traceback`, `No space left`, `Generating system KV cache`, or `Preprocessing document` messages were found at first check.

Additional native Online QK / DraftModel replica on qjy001:

Purpose: use idle qjy001 GPUs to run an independent replica of the same native `online_qk_rate015` / `online_draft_rate015` shared-cache rerun. This does not overwrite the qjy003 result root; whichever run completes first can be used, and the duplicate can be retained for reproducibility check.

Replica result root:

- `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_native_qk_draft_rate015_sharedcache_rerun_qjy001_replica`

Launched on qjy001:

| GPU | Method | Dataset | Range |
|---:|---|---|---|
| 0 | online_qk_rate015 | 2WikiMQA | 0-200 |
| 1 | online_qk_rate015 | HotpotQA | 0-260 |
| 2 | online_qk_rate015 | TriviaQA | 0-270 |
| 3 | online_qk_rate015 | MuSiQue | 0-135 |
| 4 | online_draft_rate015 | 2WikiMQA | 0-200 |
| 5 | online_draft_rate015 | HotpotQA | 0-260 |
| 6 | online_draft_rate015 | TriviaQA | 0-270 |
| 7 | online_draft_rate015 | MuSiQue | 0-135 |

Initial status:

- qjy001 GPU0-7 started loading jobs.
- No completed CSV yet at first check.
- No `Read-only cache mode`, `Traceback`, `No space left`, `Generating system KV cache`, or `Preprocessing document` messages were found at first check.


## 2026-07-13 rate=1 baseline old-vs-new generation audit

User question: compare the two `rate=1` generation-result versions, because the corrected full baseline may have been increased by optimization/code pollution.

### Files compared

Old historical `rate=1`:
- Cross datasets: `MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/results/full_rate1/{2wikimqa,hotpotqa,triviaqa}`
- MuSiQue: `MOTIVATION_EXPERIMENTS/qwen3_hybrid70_online_baselines/full_rate1`

New same-pipeline corrected `rate=1`:
- `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_alpha_rate1_same_pipeline/full_rate1_draft_layout/{dataset}`

Generated audit tables:
- `rate1_old_vs_new_dedup_strict_diff.csv`
- `rate1_old_vs_new_dedup_flip_examples.csv`

### Strict deduplicated comparison

The comparison uses `(Main Question, Sub Question, Ground Truth)` as key and drops duplicate keys before computing accuracy on common rows.

| Dataset | Common rows | Old correct | New correct | Old acc | New acc | Old true -> new false | Old false -> new true | Prediction changed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2WikiMQA | 200 | 113 | 96 | 56.50% | 48.00% | 30 | 13 | 196 / 200 |
| HotpotQA | 260 | 207 | 230 | 79.62% | 88.46% | 12 | 35 | 252 / 260 |
| TriviaQA | 270 | 211 | 242 | 78.15% | 89.63% | 9 | 40 | 253 / 270 |
| MuSiQue sub rows | 248 | 215 | 224 | 86.69% | 90.32% | 11 | 20 | 243 / 248 |

### Interpretation

This is not a small random fluctuation: 94%-98% of common-row predictions changed. The corrected full baseline is higher on HotpotQA/TriviaQA/MuSiQue but lower on 2WikiMQA, so the evidence does not support a simple story that the best alpha method polluted the full baseline upward everywhere.

The more likely explanation is baseline drift between historical and current runs. The old CSVs often contain generated text with `</think>` / repeated `Answer:` artifacts, while the current Qwen3 path uses the explicit no-thinking prompt and cleaner decoded answer. A sample 2Wiki shard also shows prompt eval count differs by 3 tokens between old and new runs. Therefore these two `rate=1` sets are not strict same-input/same-code reproductions.

Important code check: in `test_fusionrag_reflect_preprocess_exp.py`, `rate == 1` directly calls `prefill_and_generate` on the full prompt and does not call `load_kv_and_generate`; thus it should not use FusionRAG/DraftModel token selection. However, the new run still used a `DraftModel` experiment layout and loaded a draft model before generation. To rule out any remaining method-flag side effect under the current code, a same-code same-data A/B was launched.

### Current-code same-source A/B launched

Purpose: verify whether current `rate=1` is invariant to `--reprocess_method` when all other settings are held fixed.

Command:

```bash
ssh qjy000 'cd /raid/home/hming/FusionRAG-pca-analysis && nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/run_rate1_current_method_ab_2wikimqa.py > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_rate1_current_method_ab/launcher.log 2>&1 &'
```

Script:
- `MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/run_rate1_current_method_ab_2wikimqa.py`

Result root:
- `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_rate1_current_method_ab`

Settings:
- Code/record commit for this already-launched A/B script and audit tables: `2fd5a08` (`Record rate1 baseline audit`).
- Base HEAD observed before launching the A/B, before this post-hoc commit rule was added: `fc6bf5c`.
- Dataset: 2WikiMQA, samples 0-200, 8 shards.
- Methods: `FusionRAG` and `DraftModel`.
- Rate: `1.0`.
- Model: Qwen3-32B.
- Judge: GLM-5.2.
- Cache mode: `FUSIONRAG_PREPROCESS_CACHE_READONLY=1`.
- qjy000 GPUs used: `0,1,2,4,5,6,7`; GPU3 was already occupied.

Decision rule:
- If current `FusionRAG rate=1` and current `DraftModel rate=1` produce identical or near-identical outputs, then the old-vs-new gap is historical prompt/code/judge drift, not method pollution.
- If they differ materially under the current code, then `rate=1` is not method-neutral and all full-attention baselines must be rerun with one fixed canonical method/path.

### 2026-07-13 current-code rate=1 A/B result

Result summary file:
- `rate1_current_method_ab_2wikimqa_summary.csv`

Experiment launch command:

```bash
ssh qjy000 'cd /raid/home/hming/FusionRAG-pca-analysis && nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/run_rate1_current_method_ab_2wikimqa.py > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_rate1_current_method_ab/launcher.log 2>&1 &'
```

Relevant commits:
- Script/audit-table commit: `2fd5a08` (`Record rate1 baseline audit`).
- Documentation hash added before result summarization: `13c7ef4` (`Document rate1 audit commit hash`).
- Base HEAD observed before the A/B launch: `fc6bf5c`.

Result root:
- `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_attention_rate1_current_method_ab`

Strict 2WikiMQA result, current code:

| Method | Rows | Correct | Accuracy |
|---|---:|---:|---:|
| `rate=1`, `--reprocess_method FusionRAG` | 200 | 97 | 48.50% |
| `rate=1`, `--reprocess_method DraftModel` | 200 | 96 | 48.00% |

Pairwise output comparison:
- Common keys: 200 / 200.
- `Predicted` text changed between current FusionRAG and current DraftModel: 0 / 200.
- Correctness flips: FusionRAG true -> DraftModel false = 1; FusionRAG false -> DraftModel true = 0.

Interpretation:
- Under the current code, `rate=1` generation is effectively method-neutral: `FusionRAG` and `DraftModel` produce identical decoded answers on 2WikiMQA.
- The 1-sample accuracy difference is from judge/labeling instability, not model generation change.
- Therefore the old-vs-new `rate=1` gap is not caused by current `DraftModel` method flag polluting full attention generation.
- The earlier historical `2WikiMQA` rate=1 result (`113/200 = 56.50%`) differs from current same-code output (`96-97/200 = 48.00%-48.50%`) because historical prompt/code/postprocess changed, not because the alpha method made the new baseline artificially high.

### 2026-07-13 native QK/Draft rerun partial status

Launch roots:
- qjy003: `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_native_qk_draft_rate015_sharedcache_rerun`
- qjy001 replica: `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_native_qk_draft_rate015_sharedcache_rerun_qjy001_replica`

Launch commands are recorded above in the `corrected summary plot and native QK/Draft rerun` section. These runs were launched before the explicit commit-hash rule was added; the relevant documentation/script commit after the fact is `13c7ef4` and the current result-recording pre-commit HEAD is `13c7ef4`.

Current status: not final. CSVs exist for all datasets/methods, but qjy001/qjy003 still have long-running processes and row counts are incomplete for some datasets.

Partial qjy003 CSV snapshot:

| Method | Dataset | Current rows | Current correct | Current acc |
|---|---|---:|---:|---:|
| Online QK r=0.15 | 2WikiMQA | 180 | 84 | 46.67% |
| Online QK r=0.15 | HotpotQA | 260 | 224 | 86.15% |
| Online QK r=0.15 | TriviaQA | 270 | 241 | 89.26% |
| Online QK r=0.15 | MuSiQue sub | 180 | 147 | 81.67% |
| Online Draft r=0.15 | 2WikiMQA | 176 | 80 | 45.45% |
| Online Draft r=0.15 | HotpotQA | 247 | 214 | 86.64% |
| Online Draft r=0.15 | TriviaQA | 270 | 245 | 90.74% |
| Online Draft r=0.15 | MuSiQue sub | 180 | 155 | 86.11% |

Partial qjy001 replica snapshot:

| Method | Dataset | Current rows | Current correct | Current acc |
|---|---|---:|---:|---:|
| Online QK r=0.15 | 2WikiMQA | 172 | 78 | 45.35% |
| Online QK r=0.15 | HotpotQA | 237 | 205 | 86.50% |
| Online QK r=0.15 | TriviaQA | 270 | 240 | 88.89% |
| Online QK r=0.15 | MuSiQue sub | 180 | 147 | 81.67% |
| Online Draft r=0.15 | 2WikiMQA | 170 | 80 | 47.06% |
| Online Draft r=0.15 | HotpotQA | 229 | 199 | 86.90% |
| Online Draft r=0.15 | TriviaQA | 270 | 245 | 90.74% |
| Online Draft r=0.15 | MuSiQue sub | 180 | 155 | 86.11% |

Do not use the native rerun snapshot as final baseline until the remaining processes finish and row counts reach the intended sample sizes.

### 2026-07-13 native QK/Draft rerun final result

Result summary file:
- `native_qk_draft_rate015_rerun_final_summary.csv`

Result roots:
- qjy003: `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_native_qk_draft_rate015_sharedcache_rerun`
- qjy001 replica: `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_native_qk_draft_rate015_sharedcache_rerun_qjy001_replica`

Experiment launch commands:
- The per-GPU commands are recorded in the earlier `2026-07-13 corrected summary plot and native QK/Draft rerun` section.
- These jobs were launched before the explicit commit-before-experiment rule was added. Relevant record commits are `13c7ef4` and `5acd34b`; this final-result record is committed after the results are complete.

Completion status:
- qjy003: 8 CSVs, 8 `FINAL RESULTS` logs, no real crash/OOM error found.
- qjy001 replica: 8 CSVs, 8 `FINAL RESULTS` logs, no real crash/OOM error found.
- Earlier `Killed` grep hits were false positives from the question text `A Woman Has Killed`, not process failures.

Final qjy003 results:

| Method | Dataset | Correct / Total | Acc |
|---|---|---:|---:|
| Online QK r=0.15 | 2WikiMQA | 94 / 200 | 47.00% |
| Online QK r=0.15 | HotpotQA | 224 / 260 | 86.15% |
| Online QK r=0.15 | TriviaQA | 241 / 270 | 89.26% |
| Online QK r=0.15 | MuSiQue sub | 147 / 180 | 81.67% |
| Online QK r=0.15 | MuSiQue main | 68 / 97 | 70.10% |
| Online Draft r=0.15 | 2WikiMQA | 94 / 200 | 47.00% |
| Online Draft r=0.15 | HotpotQA | 225 / 260 | 86.54% |
| Online Draft r=0.15 | TriviaQA | 245 / 270 | 90.74% |
| Online Draft r=0.15 | MuSiQue sub | 155 / 180 | 86.11% |
| Online Draft r=0.15 | MuSiQue main | 75 / 97 | 77.32% |

Final qjy001 replica results:

| Method | Dataset | Correct / Total | Acc |
|---|---|---:|---:|
| Online QK r=0.15 | 2WikiMQA | 93 / 200 | 46.50% |
| Online QK r=0.15 | HotpotQA | 224 / 260 | 86.15% |
| Online QK r=0.15 | TriviaQA | 240 / 270 | 88.89% |
| Online QK r=0.15 | MuSiQue sub | 147 / 180 | 81.67% |
| Online QK r=0.15 | MuSiQue main | 68 / 97 | 70.10% |
| Online Draft r=0.15 | 2WikiMQA | 95 / 200 | 47.50% |
| Online Draft r=0.15 | HotpotQA | 226 / 260 | 86.92% |
| Online Draft r=0.15 | TriviaQA | 245 / 270 | 90.74% |
| Online Draft r=0.15 | MuSiQue sub | 155 / 180 | 86.11% |
| Online Draft r=0.15 | MuSiQue main | 75 / 97 | 77.32% |

Micro summary:

| Host | Method | Sub-row micro | Main-grouped micro |
|---|---|---:|---:|
| qjy003 | Online QK r=0.15 | 706 / 910 = 77.58% | 627 / 827 = 75.82% |
| qjy003 | Online Draft r=0.15 | 719 / 910 = 79.01% | 639 / 827 = 77.27% |
| qjy001 | Online QK r=0.15 | 704 / 910 = 77.36% | 625 / 827 = 75.57% |
| qjy001 | Online Draft r=0.15 | 721 / 910 = 79.23% | 641 / 827 = 77.51% |

Interpretation:
- Current-code rerun makes Online Draft r=0.15 consistently better than Online QK r=0.15 by about 1.4-2.0 micro points.
- Replica variance is small: qjy001 vs qjy003 differs by at most 2 samples on most dataset/method cells, consistent with judge/API nondeterminism rather than implementation failure.
- These current native baselines are much closer to the current corrected `rate=1` baseline family than the older historical records. Future plots should use current-code rerun baselines, not mix in older historical baseline rows.

## 2026-07-14 GLM714 Clean Rejudge For Alpha Cross-Dataset Table

Request: preserve the existing full-attention / Online QK / Online Draft / uniform-alpha / random-alpha cross-dataset table, then rerun answer judging with the current GLM LLM judge after removing `<think>` tags and leading `Answer:` markers.

Record commit at setup time: `775c62e85afd61dad7be47831bcfcbfd70975b0d`.

Rejudge script:
- `scripts/rejudge_alpha_cross_dataset_glm714_clean.py`

Launch command:

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
PYTHONDONTWRITEBYTECODE=1 nohup python3 \
  MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/rejudge_alpha_cross_dataset_glm714_clean.py \
  > MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/rejudge_glm714_clean_20260714/rejudge.log \
  2>&1 < /dev/null &
```

Original table before GLM714 rejudge, from `alpha_with_full_qk_draft_accuracy_summary.csv`:

| Dataset | Method | Correct / Total | Acc |
|---|---|---:|---:|
| 2WikiMQA | Full attention r=1 | 113 / 200 | 56.50% |
| 2WikiMQA | Online QK r=0.15 | 107 / 200 | 53.50% |
| 2WikiMQA | Online Draft r=0.15 | 101 / 200 | 50.50% |
| 2WikiMQA | Uniform a=0.1 | 96 / 200 | 48.00% |
| 2WikiMQA | Uniform a=0.25 | 99 / 200 | 49.50% |
| 2WikiMQA | Random a=0.05 | 94 / 200 | 47.00% |
| 2WikiMQA | Random a=0.1 | 95 / 200 | 47.50% |
| 2WikiMQA | Random a=0.25 | 100 / 200 | 50.00% |
| HotpotQA | Full attention r=1 | 207 / 260 | 79.62% |
| HotpotQA | Online QK r=0.15 | 206 / 260 | 79.23% |
| HotpotQA | Online Draft r=0.15 | 207 / 260 | 79.62% |
| HotpotQA | Uniform a=0.1 | 226 / 260 | 86.92% |
| HotpotQA | Uniform a=0.25 | 223 / 260 | 85.77% |
| HotpotQA | Random a=0.05 | 231 / 260 | 88.85% |
| HotpotQA | Random a=0.1 | 228 / 260 | 87.69% |
| HotpotQA | Random a=0.25 | 224 / 260 | 86.15% |
| TriviaQA | Full attention r=1 | 211 / 270 | 78.15% |
| TriviaQA | Online QK r=0.15 | 212 / 270 | 78.52% |
| TriviaQA | Online Draft r=0.15 | 214 / 270 | 79.26% |
| TriviaQA | Uniform a=0.1 | 242 / 270 | 89.63% |
| TriviaQA | Uniform a=0.25 | 237 / 270 | 87.78% |
| TriviaQA | Random a=0.05 | 241 / 270 | 89.26% |
| TriviaQA | Random a=0.1 | 242 / 270 | 89.63% |
| TriviaQA | Random a=0.25 | 238 / 270 | 88.15% |
| MuSiQue | Full attention r=1 | 105 / 135 | 77.78% |
| MuSiQue | Online QK r=0.15 | 84 / 135 | 62.22% |
| MuSiQue | Online Draft r=0.15 | 99 / 135 | 73.33% |
| MuSiQue | Uniform a=0.1 | 107 / 135 | 79.26% |
| MuSiQue | Uniform a=0.25 | 107 / 135 | 79.26% |
| MuSiQue | Random a=0.05 | 105 / 135 | 77.78% |
| MuSiQue | Random a=0.1 | 105 / 135 | 77.78% |
| MuSiQue | Random a=0.25 | 106 / 135 | 78.52% |

Important source-status note before rejudge:
- Full attention / Online QK / Online Draft detailed CSVs are available for all four datasets.
- MuSiQue alpha detailed CSVs are available for the listed alpha settings.
- The repository currently lacks complete detailed CSVs for several 2WikiMQA / HotpotQA / TriviaQA alpha settings even though the old aggregate table exists. The rejudge script records missing segments in `rejudged_summary.json`; GLM714 results are valid only for rows whose detailed CSV exists.

## 2026-07-14 Alpha Config Rate Sweep From Corrected Micro Figure

Request: take the five alpha-ablation configurations shown in `figures/micro_all_datasets_method_accuracy_corrected_full.png` and run a rate sweep for each configuration.

Configurations:

- `uniform alpha=0.1`
- `uniform alpha=0.25`
- `random alpha=0.05`
- `random alpha=0.1`
- `random alpha=0.25`

Important setup decisions:

- The figure already contains `rate=0.15`; this add-on sweep runs `rate=0.3,0.5,0.8,1.0` and later summaries should merge the existing `0.15` rows with these new rows.
- Method is `DraftModel` with attention ablation controlled by `FUSIONRAG_REPROCESS_ATTENTION_ABLATION` and `FUSIONRAG_REPROCESS_ATTENTION_ABLATION_ALPHA`.
- Cache root is the canonical shared Qwen3-32B root: `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache`.
- Cache is readonly via `FUSIONRAG_PREPROCESS_CACHE_READONLY=1`; this run must not create per-worker or per-experiment cache copies.
- Attention ablation chunking uses `FUSIONRAG_REPROCESS_ATTENTION_CHUNK=64` to avoid the old full attention-matrix OOM mode.
- To avoid occupying every available GPU for a follow-up sweep, launch allocation is limited to 8 total GPUs: 4 on qjy001 and 4 on qjy003.

Launcher script:

```text
MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/run_alpha_config_rate_sweep.py
```

Output root:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_alpha_config_rate_sweep_canonical_20260714
```

Launch commands:

```bash
# qjy001: 2WikiMQA + HotpotQA
cd /home/hming/FusionRAG-pca-analysis
FUSIONRAG_ALPHA_RATE_SWEEP_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_alpha_config_rate_sweep_canonical_20260714 FUSIONRAG_ALPHA_RATE_SWEEP_CACHE_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache FUSIONRAG_ALPHA_RATE_SWEEP_CACHE_READONLY=1 FUSIONRAG_ALPHA_RATE_SWEEP_DATASETS=2wikimqa,hotpotqa FUSIONRAG_ALPHA_RATE_SWEEP_RATES=0.3,0.5,0.8,1.0 FUSIONRAG_ALPHA_RATE_SWEEP_COMBOS=uniform:0.1,uniform:0.25,random:0.05,random:0.1,random:0.25 FUSIONRAG_ALPHA_RATE_SWEEP_GPUS=0,1,2,3 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/run_alpha_config_rate_sweep.py   > /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_alpha_config_rate_sweep_canonical_20260714/launcher_qjy001.nohup.log 2>&1 &

# qjy003: TriviaQA + MuSiQue
cd /home/hming/FusionRAG-pca-analysis
FUSIONRAG_ALPHA_RATE_SWEEP_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_alpha_config_rate_sweep_canonical_20260714 FUSIONRAG_ALPHA_RATE_SWEEP_CACHE_ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache FUSIONRAG_ALPHA_RATE_SWEEP_CACHE_READONLY=1 FUSIONRAG_ALPHA_RATE_SWEEP_DATASETS=triviaqa,musique FUSIONRAG_ALPHA_RATE_SWEEP_RATES=0.3,0.5,0.8,1.0 FUSIONRAG_ALPHA_RATE_SWEEP_COMBOS=uniform:0.1,uniform:0.25,random:0.05,random:0.1,random:0.25 FUSIONRAG_ALPHA_RATE_SWEEP_GPUS=0,1,2,3 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/run_alpha_config_rate_sweep.py   > /mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_alpha_config_rate_sweep_canonical_20260714/launcher_qjy003.nohup.log 2>&1 &
```

Expected new segment tasks: `5 configs x 4 rates x 38 dataset segments = 760`.

Actual launch after commit `7cb8d3d`:

| Host | Dataset(s) | PID | GPUs | Status at initial health check |
|---|---|---:|---|---|
| qjy001 | `2wikimqa,hotpotqa` | `2705114` | `0,1,2,3` | workers started, waiting for GPUs because the earlier `rate=0.15` add-on is still using them |
| qjy003 | `triviaqa,musique` | `1863323` | `0,1,2,3` | workers started and running `triviaqa uniform_alpha0p1 rate=0.3` segments |

Initial health check:

- Worker logs: `8`.
- Run logs: `4` immediately after startup.
- Completed segments: `0` at startup.
- Worker logs confirm `cache_root=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache`, `cache_readonly=True`, `attn_chunk=64`.
- Error scan found no early `Traceback`, CUDA OOM, no-space, `PytorchStreamReader`, `.pt cannot be opened`, or readonly cache miss.

### 2026-07-14 Full Idle-GPU Expansion

User request: many GPUs were idle, so expand the alpha-config rate sweep to use all currently idle qjy-series GPUs.

Implementation changes:

- Commit 402d93f added a running-segment guard to avoid launching a segment whose output directory is already being processed.
- Commit e5f8171 changed the guard from wait-existing to skip-running, so extra workers can move on to later unclaimed segments instead of idling behind the first running segment.

Additional launch PIDs:

| Host | Dataset(s) | PID | GPUs | Note |
|---|---|---:|---|---|
| qjy000 | musique | 3997366 | 0,1,2,3,4,5,6,7 | fills all previously idle qjy000 GPUs |
| qjy001 | 2wikimqa,hotpotqa | 2731894 | 4,6,7 | first expansion on idle qjy001 GPUs |
| qjy003 | triviaqa,musique | 1888391 | 4,5,6,7 | first expansion on idle qjy003 GPUs |
| qjy001 | 2wikimqa,hotpotqa | 2740990 | 5 | fills remaining idle qjy001 GPU |
| qjy003 | triviaqa,musique | 1897337 | 2 | fills remaining idle qjy003 GPU |

Health check after expansion:

- qjy000: GPUs 0-7 all allocated.
- qjy001: GPUs 0-7 all allocated.
- qjy003: GPUs 0-7 all allocated.
- Alpha sweep status shortly after expansion: 24 worker logs, 28 run logs, 8 completed segments, 27 CSV files.
- Error scan found no Traceback, CUDA OOM, no-space, PytorchStreamReader, cannot-open-cache, or readonly-cache-miss errors.
- All runs still use canonical cache root /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache in readonly mode.

## 2026-07-14 Current Rate 0.15 Vs Old Micro Figure Baseline Check

Question: whether the newly rerun cross-dataset rate=0.15 Online QK / Online Draft numbers match the old figure figures/micro_all_datasets_method_accuracy.png.

Short answer: they do not match.

Comparison table:

MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/current_rate015_vs_micro_figure_baseline_comparison.csv

Key differences:

| Dataset | Method | Old figure | Current rerun | Delta |
|---|---|---:|---:|---:|
| 2WikiMQA | Online QK r=0.15 | 107/200 = 53.50% | 95/200 = 47.50% | -6.00 pp |
| 2WikiMQA | Online Draft r=0.15 | 101/200 = 50.50% | 94/200 = 47.00% | -3.50 pp |
| HotpotQA | Online QK r=0.15 | 206/260 = 79.23% | 220/260 = 84.62% | +5.38 pp |
| HotpotQA | Online Draft r=0.15 | 207/260 = 79.62% | 223/260 = 85.77% | +6.15 pp |
| TriviaQA | Online QK r=0.15 | 212/270 = 78.52% | 238/270 = 88.15% | +9.63 pp |
| TriviaQA | Online Draft r=0.15 | 214/270 = 79.26% | 246/270 = 91.11% | +11.85 pp |
| MuSiQue | Online QK r=0.15 | 84/135 = 62.22% | 100/135 = 74.07% | +11.85 pp |
| MuSiQue | Online Draft r=0.15 | 99/135 = 73.33% | 108/135 = 80.00% | +6.67 pp |
| Micro all | Online QK r=0.15 | 609/865 = 70.40% | 653/865 = 75.49% | +5.09 pp |
| Micro all | Online Draft r=0.15 | 621/865 = 71.79% | 671/865 = 77.57% | +5.78 pp |

Interpretation:

- The new current-code rerun is not numerically compatible with figures/micro_all_datasets_method_accuracy.png for the native Online QK/Draft baselines.
- The difference is directionally mixed by dataset: 2WikiMQA is lower in the rerun, while HotpotQA, TriviaQA, and MuSiQue are substantially higher.
- Therefore future alpha plots should not mix the old figure's native baseline rows with current rerun rows. A clean comparison needs either rerunning all alpha settings under the current pipeline/judge/cache setup, or explicitly labeling the old alpha rows as historical.

## 2026-07-14 Corrected-Full Micro Figure With Current Online Baselines

Request: create a new figure following `figures/micro_all_datasets_method_accuracy_corrected_full.png`, but replace the Online QK / Online Draft rows with the newly rerun `rate=0.15` online data. Do not overwrite the original figure.

New current-online rows:

| Method | Correct / Total | Main Acc |
|---|---:|---:|
| Online QK r=0.15 | 653 / 865 | 75.49% |
| Online Draft r=0.15 | 671 / 865 | 77.57% |

Full attention and alpha rows are unchanged from the corrected-full figure, so alpha rows are still historical alpha sweep rows until the current alpha reruns finish.

Generated files:

```text
MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/micro_all_datasets_method_accuracy_corrected_full_current_online.csv
MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/figures/micro_all_datasets_method_accuracy_corrected_full_current_online.png
MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/figures/micro_all_datasets_method_accuracy_corrected_full_current_online.pdf
```

Original figure preserved:

```text
MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/figures/micro_all_datasets_method_accuracy_corrected_full.png
```

Plot command:

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/plot_micro_method_accuracy_current_online.py
```

## 2026-07-14 Alpha Rate Sweep Scope Reduction

User changed scope: only keep `uniform_alpha0p1` and `random_alpha0p1`; stop other alpha configs to reduce runtime. The previous broad sweep had included `uniform_alpha0p25`, `random_alpha0p05`, and `random_alpha0p25`, but those are no longer part of the active queue.

Code commit for this relaunch: `a79b7e7 fix: shard alpha rate sweep workers`.

Launcher/output root:
`/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_alpha_config_rate_sweep_canonical_20260714`

Canonical cache root, readonly:
`/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache`

Active configs:
- methods: `uniform:0.1`, `random:0.1`
- rates: `0.3,0.5,0.8,1.0`
- datasets: `2wikimqa,hotpotqa,triviaqa,musique`
- segment size: 25 examples
- GPUs: 12 total, 4 per host on qjy000/qjy001/qjy003
- sharding: static shard by task index modulo 3, so the three hosts do not race the same segment.

Launch commands:

```bash
# qjy000, shard 0/3
cd /raid/home/hming/FusionRAG-pca-analysis
ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_alpha_config_rate_sweep_canonical_20260714
FUSIONRAG_ALPHA_RATE_SWEEP_COMBOS="uniform:0.1,random:0.1" FUSIONRAG_ALPHA_RATE_SWEEP_RATES="0.3,0.5,0.8,1.0" FUSIONRAG_ALPHA_RATE_SWEEP_GPUS="0,1,2,3" FUSIONRAG_ALPHA_RATE_SWEEP_SHARD_COUNT=3 FUSIONRAG_ALPHA_RATE_SWEEP_SHARD_INDEX=0 FUSIONRAG_ALPHA_RATE_SWEEP_FREE_MEM_MB=10000 FUSIONRAG_ALPHA_RATE_SWEEP_CACHE_READONLY=1 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/run_alpha_config_rate_sweep.py   > "$ROOT"/launcher_qjy000_2combo_12gpu_shard0.nohup.log 2>&1 &

# qjy001, shard 1/3
cd /home/hming/FusionRAG-pca-analysis
ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_alpha_config_rate_sweep_canonical_20260714
FUSIONRAG_ALPHA_RATE_SWEEP_COMBOS="uniform:0.1,random:0.1" FUSIONRAG_ALPHA_RATE_SWEEP_RATES="0.3,0.5,0.8,1.0" FUSIONRAG_ALPHA_RATE_SWEEP_GPUS="0,1,2,3" FUSIONRAG_ALPHA_RATE_SWEEP_SHARD_COUNT=3 FUSIONRAG_ALPHA_RATE_SWEEP_SHARD_INDEX=1 FUSIONRAG_ALPHA_RATE_SWEEP_FREE_MEM_MB=10000 FUSIONRAG_ALPHA_RATE_SWEEP_CACHE_READONLY=1 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/run_alpha_config_rate_sweep.py   > "$ROOT"/launcher_qjy001_2combo_12gpu_shard1.nohup.log 2>&1 &

# qjy003, shard 2/3
cd /home/hming/FusionRAG-pca-analysis
ROOT=/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_alpha_config_rate_sweep_canonical_20260714
FUSIONRAG_ALPHA_RATE_SWEEP_COMBOS="uniform:0.1,random:0.1" FUSIONRAG_ALPHA_RATE_SWEEP_RATES="0.3,0.5,0.8,1.0" FUSIONRAG_ALPHA_RATE_SWEEP_GPUS="0,1,2,3" FUSIONRAG_ALPHA_RATE_SWEEP_SHARD_COUNT=3 FUSIONRAG_ALPHA_RATE_SWEEP_SHARD_INDEX=2 FUSIONRAG_ALPHA_RATE_SWEEP_FREE_MEM_MB=10000 FUSIONRAG_ALPHA_RATE_SWEEP_CACHE_READONLY=1 nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/run_alpha_config_rate_sweep.py   > "$ROOT"/launcher_qjy003_2combo_12gpu_shard2.nohup.log 2>&1 &
```

Current progress snapshot at 2026-07-14 16:50 CST:

| Config | Rate | Done segments | Notes |
|---|---:|---:|---|
| uniform_alpha0p1 | 0.3 | 37/38 | only musique 25-50 still running/logged |
| uniform_alpha0p1 | 0.5 | 36/38 | triviaqa and musique each have 1 remaining/logged segment |
| uniform_alpha0p1 | 0.8 | 19/38 | active on 2WikiMQA/HotpotQA; old OOM logs remain on 2 triviaqa segments |
| uniform_alpha0p1 | 1.0 | 17/38 | 2WikiMQA/HotpotQA mostly not started yet; old OOM logs remain on triviaqa/musique |
| random_alpha0p1 | 0.3 | 1/38 | queue just started; one musique segment done, one running/logged |
| random_alpha0p1 | 0.5 | 1/38 | queue just started |
| random_alpha0p1 | 0.8 | 1/38 | queue just started |
| random_alpha0p1 | 1.0 | 1/38 | queue just started |

Interpretation/status:
- The active queue is now the reduced 2-config sweep only; previous `0.05/0.25` configs were stopped.
- Existing completed segments for the two retained configs are reused; missing segments continue from the same root.
- Some OOM strings remain in old `run.log` files from the pre-reduction aggressive 24-card run. The relaunch uses 12 cards and static sharding to reduce duplicate work and memory contention.


## 2026-07-14 MuSiQue Rate Sweep Figure: QK / Draft / Uniform / Random

User request: plot the rate sweep curves on MuSiQue with native online QK, native online DraftModel, `uniform alpha=0.1`, and `random alpha=0.1` on the same figure.

Script:

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/plot_musique_rate_sweep_qk_draft_alpha.py
```

Outputs:

- `MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/figures/musique_rate_sweep_qk_draft_uniform_random.png`
- `MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/figures/musique_rate_sweep_qk_draft_uniform_random.pdf`
- `MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/musique_rate_sweep_qk_draft_uniform_random.csv`

Data sources:

- Online QK / Online DraftModel: `MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/qwen3_online_qk_draft_rate_sweep_by_dataset_with_rate015.csv`, dataset=`musique`.
- `uniform alpha=0.1` rate=0.15: historical full MuSiQue alpha sweep from `qwen3_draft_attention_ablation_rate015/README.md`.
- `random alpha=0.1` rate=0.15: historical full MuSiQue alpha sweep from `qwen3_draft_attention_ablation_rate015/README.md`.
- `uniform/random alpha=0.1` rates 0.3/0.5/0.8/1.0: `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_attention_alpha_config_rate_sweep_canonical_20260714`.

Important caveat:

- `uniform alpha=0.1` is complete for MuSiQue at rates 0.3/0.5/0.8/1.0: 8/8 segments each. The plotting script deduplicates rows by `(Main Question, Sub Question, Ground Truth)` before aggregating, because earlier interrupted/retried segments could leave duplicate CSV rows.
- `random alpha=0.1` is not complete yet for rates 0.3/0.5/0.8/1.0: only 1/8 MuSiQue segments per rate were present when the figure was generated. The figure draws those points with dashed/open markers and the CSV marks them as `partial`. Do not use the random curve as a final result until the remaining segments finish.

Current complete MuSiQue uniform points after deduplication:

| Method | Rate | Main Acc | Sub Acc | Status |
|---|---:|---:|---:|---|
| uniform alpha=0.1 | 0.15 | 107/135 = 79.26% | 217/248 = 87.50% | complete historical |
| uniform alpha=0.1 | 0.3 | 111/135 = 82.22% | 223/248 = 89.92% | complete |
| uniform alpha=0.1 | 0.5 | 109/135 = 80.74% | 221/248 = 89.11% | complete |
| uniform alpha=0.1 | 0.8 | 112/135 = 82.96% | 224/248 = 90.32% | complete |
| uniform alpha=0.1 | 1.0 | 109/135 = 80.74% | 222/248 = 89.52% | complete |

Reading:

- On MuSiQue, the complete `uniform alpha=0.1` curve remains competitive with or above native DraftModel at comparable rates.
- The curve is not monotonic: rate 0.8 is the best complete uniform point by main/sub accuracy, while rate 1.0 drops back. This supports the earlier observation that increasing recompute/update rate does not automatically improve answer accuracy.
- The random curve is currently only a visual placeholder for partial progress; wait for full random completion before comparing random vs uniform across rates.

### Correction: Current-Only MuSiQue Rate Sweep Figure

The first version of `musique_rate_sweep_qk_draft_uniform_random.png` mixed two incompatible sources:

- native online QK/Draft from the current rate-sweep rerun;
- alpha rate=0.15 points from the older MuSiQue attention-ablation table.

That was not a valid same-source comparison. The historical table has `native online DraftModel rate=0.15 = 99/135 main, 209/248 sub`, while the current rerun/source used for the rate-sweep plot has a different native DraftModel curve. Therefore the user-observed mismatch was real.

The figure and CSV have been regenerated as a current-only rate-sweep plot:

- Native online QK/Draft source is now `MOTIVATION_EXPERIMENTS/qwen3_rate_sweep_online_qk_draft/rate_sweep_current_summary.csv`, which uses the MuSiQue 248-sub-question deduplicated口径.
- Historical alpha=0.15 points are excluded from the rate-sweep plot.
- Current `uniform alpha=0.1` points are included only for rates 0.3/0.5/0.8/1.0, all complete at 8/8 segments.
- Current `random alpha=0.1` points remain partial and are drawn with dashed/open markers.

Corrected outputs overwrite the previous figure/table paths:

- `figures/musique_rate_sweep_qk_draft_uniform_random.png`
- `figures/musique_rate_sweep_qk_draft_uniform_random.pdf`
- `musique_rate_sweep_qk_draft_uniform_random.csv`

Current-only complete uniform points:

| Method | Rate | Main Acc | Sub Acc | Status |
|---|---:|---:|---:|---|
| uniform alpha=0.1 | 0.3 | 111/135 = 82.22% | 223/248 = 89.92% | complete |
| uniform alpha=0.1 | 0.5 | 109/135 = 80.74% | 221/248 = 89.11% | complete |
| uniform alpha=0.1 | 0.8 | 112/135 = 82.96% | 224/248 = 90.32% | complete |
| uniform alpha=0.1 | 1.0 | 109/135 = 80.74% | 222/248 = 89.52% | complete |

The older rate=0.15 alpha table is still valid as a historical standalone result, but it should not be placed on the same curve as the current rerun unless native baselines are also taken from the same historical run.

## 2026-07-14 Per-Dataset Version Of Current-Online Micro Figure

User request: take the data behind `figures/micro_all_datasets_method_accuracy_corrected_full_current_online.png` and plot it per dataset.

Generated with:

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/scripts/plot_dataset_method_accuracy_current_online.py
```

Outputs:

- `MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/figures/dataset_method_accuracy_corrected_full_current_online.png`
- `MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/figures/dataset_method_accuracy_corrected_full_current_online.pdf`
- `MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/dataset_method_accuracy_corrected_full_current_online.csv`

Data construction matches the current-online micro figure:

- Full attention and alpha rows come from `alpha_with_full_qk_draft_accuracy_summary.csv`.
- Online QK/Draft rows are replaced by current rerun values from `current_rate015_vs_micro_figure_baseline_comparison.csv`.
- Therefore this is not a fully same-run table; it is the per-dataset decomposition of the existing current-online micro figure.

Key per-dataset observations from this figure:

- 2WikiMQA: all online/alpha methods are below the historical full-attention row. Best alpha among plotted methods is `Random a=0.25` at 100/200 = 50.00%, still below full attention 113/200 = 56.50%.
- HotpotQA: alpha methods are above current Online Draft; best is `Random a=0.05` at 231/260 = 88.85%.
- TriviaQA: current Online Draft is strongest among plotted methods at 246/270 = 91.11%; alpha methods are close but lower.
- MuSiQue: current Online Draft is 108/135 = 80.00%, slightly above historical `Uniform a=0.1` and `Uniform a=0.25` at 107/135 = 79.26%.

This plot makes the earlier micro aggregate easier to interpret: the apparent aggregate gain is not uniform across datasets. HotpotQA benefits most from alpha smoothing; TriviaQA and MuSiQue current Online Draft are already strong; 2WikiMQA remains the failure case.
