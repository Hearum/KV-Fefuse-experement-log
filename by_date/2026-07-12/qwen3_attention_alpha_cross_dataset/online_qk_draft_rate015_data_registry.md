# Online QK / DraftModel Rate 0.15 Data Registry

This file records the known Qwen3-32B `online_qk` / `online_draft` rate=0.15 result batches so later plots do not mix incompatible sources.

Last updated: 2026-07-14.

## Summary

| Batch | What it is | Primary folder / file | Coverage | Main-use caveat |
|---|---|---|---|---|
| A. Earliest offline/online experiment | Original cross-dataset Qwen3-32B offline-vs-online run | `MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/` | 2WikiMQA, HotpotQA, TriviaQA | Historical generation and old judging. No MuSiQue. |
| A2. GLM-clean rejudge of Batch A | Rejudge Batch A predictions after removing think/answer labels | `MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rejudge_glm_clean_20260714/` | 2WikiMQA, HotpotQA, TriviaQA | Rejudges old predictions only; does not rerun generation. |
| B. 2026-07-13 current-code native rerun | Native Online QK/Draft rerun on shared cache, with qjy003 and qjy001 replica | `MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/native_qk_draft_rate015_rerun_final_summary.csv` | 2WikiMQA, HotpotQA, TriviaQA, MuSiQue | Uses 910 sub-row / 827 main-grouped micro; not the same as the 865-main plot口径. |
| C. 2026-07-14 rate=0.15 add-on for rate sweep | Current rate-sweep-compatible rerun | `MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/qwen3_online_qk_draft_rate_sweep_by_dataset_with_rate015.csv` | 2WikiMQA, HotpotQA, TriviaQA, MuSiQue | This is the source used by `micro_all_datasets_method_accuracy_corrected_full_current_online.png`. |

## Batch A: Earliest Offline/Online Qwen3-32B Experiment

Experiment folder:

```text
MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/
```

Important files:

```text
MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/README.md
MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/cross_dataset_summary.csv
MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/results/
MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/alpha_with_full_qk_draft_accuracy_summary.csv
MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/micro_all_datasets_method_accuracy.csv
MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/micro_all_datasets_method_accuracy_corrected_full.csv
```

Launch evidence:

```text
MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/logs/supervisor.log
MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/logs/accuracy_tasks.tsv
```

Numbers used by older micro figures:

| Dataset | Method | Correct / Total | Accuracy |
|---|---|---:|---:|
| 2WikiMQA | online_qk_rate015 | 107 / 200 | 53.50% |
| 2WikiMQA | online_draft_rate015 | 101 / 200 | 50.50% |
| HotpotQA | online_qk_rate015 | 206 / 260 | 79.23% |
| HotpotQA | online_draft_rate015 | 207 / 260 | 79.62% |
| TriviaQA | online_qk_rate015 | 212 / 270 | 78.52% |
| TriviaQA | online_draft_rate015 | 214 / 270 | 79.26% |
| Micro, 3 datasets | online_qk_rate015 | 525 / 730 | 71.92% |
| Micro, 3 datasets | online_draft_rate015 | 522 / 730 | 71.51% |

The alpha cross-dataset figure also combined these cross-dataset rows with a historical MuSiQue row:

| Dataset | Method | Correct / Total | Accuracy |
|---|---|---:|---:|
| MuSiQue | Online QK r=0.15 | 84 / 135 | 62.22% |
| MuSiQue | Online Draft r=0.15 | 99 / 135 | 73.33% |
| Micro, 4 datasets | Online QK r=0.15 | 609 / 865 | 70.40% |
| Micro, 4 datasets | Online Draft r=0.15 | 621 / 865 | 71.79% |

Figures using this source for Online QK / Draft:

```text
MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/figures/micro_all_datasets_method_accuracy.png
MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/figures/micro_all_datasets_method_accuracy_corrected_full.png
```

## Batch A2: GLM-Clean Rejudge Of Batch A

Experiment folder:

```text
MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rejudge_glm_clean_20260714/
```

Important files:

```text
MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rejudge_glm_clean_20260714/README.md
MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rejudge_glm_clean_20260714/rejudge_cross_dataset_glm_clean.py
MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rejudge_glm_clean_20260714/rejudge.log
MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rejudge_glm_clean_20260714/rejudged_rows.csv
MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rejudge_glm_clean_20260714/rejudged_summary.csv
MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rejudge_glm_clean_20260714/rejudged_summary.json
```

Completion evidence:

```text
loaded_rows=4310
missing_segments=0
judged=4310/4310
```

This pass removes `<think>...</think>`, residual think tags, and leading answer labels before GLM judging. It does not rerun generation.

Online QK / Draft GLM-clean results:

| Dataset | Method | Old Correct / Total | GLM-clean Correct / Total | GLM-clean Accuracy |
|---|---|---:|---:|---:|
| 2WikiMQA | online_qk_rate015 | 107 / 200 | 112 / 200 | 56.00% |
| 2WikiMQA | online_draft_rate015 | 101 / 200 | 106 / 200 | 53.00% |
| HotpotQA | online_qk_rate015 | 206 / 260 | 226 / 260 | 86.92% |
| HotpotQA | online_draft_rate015 | 207 / 260 | 228 / 260 | 87.69% |
| TriviaQA | online_qk_rate015 | 212 / 270 | 241 / 270 | 89.26% |
| TriviaQA | online_draft_rate015 | 214 / 270 | 239 / 270 | 88.52% |
| Micro, 3 datasets | online_qk_rate015 | 525 / 730 | 579 / 730 | 79.32% |
| Micro, 3 datasets | online_draft_rate015 | 522 / 730 | 573 / 730 | 78.49% |

Do not use this as a current-pipeline generation baseline. It only answers how the old predictions score under a cleaner GLM judging pass.

## Batch B: 2026-07-13 Current-Code Native QK/Draft Rerun

Experiment folders:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_native_qk_draft_rate015_sharedcache_rerun
/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_native_qk_draft_rate015_sharedcache_rerun_qjy001_replica
```

Project summary file:

```text
MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/native_qk_draft_rate015_rerun_final_summary.csv
```

Documentation section:

```text
MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/README.md
Section: 2026-07-13 native QK/Draft rerun final result
```

Final qjy003 results:

| Dataset | Method | Correct / Total | Accuracy |
|---|---|---:|---:|
| 2WikiMQA | Online QK r=0.15 | 94 / 200 | 47.00% |
| 2WikiMQA | Online Draft r=0.15 | 94 / 200 | 47.00% |
| HotpotQA | Online QK r=0.15 | 224 / 260 | 86.15% |
| HotpotQA | Online Draft r=0.15 | 225 / 260 | 86.54% |
| TriviaQA | Online QK r=0.15 | 241 / 270 | 89.26% |
| TriviaQA | Online Draft r=0.15 | 245 / 270 | 90.74% |
| MuSiQue sub | Online QK r=0.15 | 147 / 180 | 81.67% |
| MuSiQue sub | Online Draft r=0.15 | 155 / 180 | 86.11% |
| MuSiQue main | Online QK r=0.15 | 68 / 97 | 70.10% |
| MuSiQue main | Online Draft r=0.15 | 75 / 97 | 77.32% |

Final qjy001 replica results:

| Dataset | Method | Correct / Total | Accuracy |
|---|---|---:|---:|
| 2WikiMQA | Online QK r=0.15 | 93 / 200 | 46.50% |
| 2WikiMQA | Online Draft r=0.15 | 95 / 200 | 47.50% |
| HotpotQA | Online QK r=0.15 | 224 / 260 | 86.15% |
| HotpotQA | Online Draft r=0.15 | 226 / 260 | 86.92% |
| TriviaQA | Online QK r=0.15 | 240 / 270 | 88.89% |
| TriviaQA | Online Draft r=0.15 | 245 / 270 | 90.74% |
| MuSiQue sub | Online QK r=0.15 | 147 / 180 | 81.67% |
| MuSiQue sub | Online Draft r=0.15 | 155 / 180 | 86.11% |
| MuSiQue main | Online QK r=0.15 | 68 / 97 | 70.10% |
| MuSiQue main | Online Draft r=0.15 | 75 / 97 | 77.32% |

Micro summary in this file:

| Host | Method | Sub-row micro | Main-grouped micro |
|---|---|---:|---:|
| qjy003 | Online QK r=0.15 | 706 / 910 = 77.58% | 627 / 827 = 75.82% |
| qjy003 | Online Draft r=0.15 | 719 / 910 = 79.01% | 639 / 827 = 77.27% |
| qjy001 | Online QK r=0.15 | 704 / 910 = 77.36% | 625 / 827 = 75.57% |
| qjy001 | Online Draft r=0.15 | 721 / 910 = 79.23% | 641 / 827 = 77.51% |

Caveat: this batch should not be directly plotted against 865-main-question alpha figures because the MuSiQue grouping differs and the documented micro totals are 910 sub rows / 827 main groups.

## Batch C: 2026-07-14 Rate=0.15 Add-On For Rate Sweep

Experiment folders:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_cross_dataset_online_qk_draft_rate015_canonical_20260714
/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_online_qk_draft_rate015_canonical_20260714
```

Project summary file:

```text
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/qwen3_online_qk_draft_rate_sweep_by_dataset_with_rate015.csv
```

Documentation sections:

```text
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/README.md
MOTIVATION_EXPERIMENTS/qwen3_rate_sweep_online_qk_draft/README.md
```

Completion status:

```text
Cross-dataset: 60 / 60 FINAL RESULTS, bad=0
MuSiQue:       16 / 16 FINAL RESULTS, bad=0
```

Results:

| Dataset | Method | Correct / Total | Accuracy |
|---|---|---:|---:|
| 2WikiMQA | online_qk | 95 / 200 | 47.50% |
| 2WikiMQA | online_draft | 94 / 200 | 47.00% |
| HotpotQA | online_qk | 220 / 260 | 84.62% |
| HotpotQA | online_draft | 223 / 260 | 85.77% |
| TriviaQA | online_qk | 238 / 270 | 88.15% |
| TriviaQA | online_draft | 246 / 270 | 91.11% |
| MuSiQue | online_qk | 100 / 135 | 74.07% |
| MuSiQue | online_draft | 108 / 135 | 80.00% |
| Micro, 4 datasets | online_qk | 653 / 865 | 75.49% |
| Micro, 4 datasets | online_draft | 671 / 865 | 77.57% |

Figures using this source for Online QK / Draft:

```text
MOTIVATION_EXPERIMENTS/qwen3_attention_alpha_cross_dataset/figures/micro_all_datasets_method_accuracy_corrected_full_current_online.png
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/figures/qwen3_rate_sweep_main_acc_all_datasets.png
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/figures/qwen3_rate_sweep_2wikimqa.png
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/figures/qwen3_rate_sweep_hotpotqa.png
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/figures/qwen3_rate_sweep_triviaqa.png
MOTIVATION_EXPERIMENTS/qwen3_cross_dataset_rate_sweep_online_qk_draft/figures/qwen3_rate_sweep_musique.png
```

## Recommendation

Use Batch C for current 865-main-question micro figures and native rate-sweep plots.

Use Batch A2 only when the question is about rejudging old generated predictions.

Use Batch B only when comparing current-code rerun replicas under the 910-sub-row / 827-main-grouped native rerun protocol.
