# Cross-Dataset GLM Clean Rejudge - 2026-07-14

## Purpose

Rejudge the existing Qwen3-32B cross-dataset offline/online prediction CSV files with the current GLM judge, without rerunning the generation pipeline.

This is a clean rejudge pass: before sending a prediction to GLM, the script removes Qwen thinking blocks and leading answer labels:

- `<think>...</think>`
- residual `<think>` / `</think>`
- leading `Answer:`, `Final Answer:`, or `答案:`

No other semantic rewriting is applied.

## Reproduction

- Repo commit: `3ec6deda9df1afc38f070fae199fdcff910c1a9c`
- Host: `qjy000`
- Working directory: `/raid/home/hming/FusionRAG-pca-analysis`
- Judge endpoint: `http://36.150.226.221:32355/v1`
- Judge model: `GLM-5.2`
- Judge setting: `temperature=0`, `thinking={"type":"disabled"}`
- Input root: `MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/results/`
- Output root: `MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rejudge_glm_clean_20260714/`

Command:

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
python3 MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rejudge_glm_clean_20260714/rejudge_cross_dataset_glm_clean.py \
  > MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rejudge_glm_clean_20260714/rejudge.log 2>&1
```

Actual launch used `nohup` with PID file:

```bash
nohup python3 MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rejudge_glm_clean_20260714/rejudge_cross_dataset_glm_clean.py \
  > MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rejudge_glm_clean_20260714/rejudge.log 2>&1 &
```

## Output Files

- `rejudge_cross_dataset_glm_clean.py`: rejudge script.
- `judge_cache_glm52_clean.jsonl`: per-question/prediction GLM cache.
- `rejudged_rows.csv`: row-level old and GLM-clean labels.
- `rejudged_summary.csv`: aggregated accuracy table.
- `rejudged_summary.json`: summary plus missing segment list.
- `rejudge.log`: run progress.

## Coverage

- Loaded rows: 4310
- Missing segments: 0
- Datasets: `2wikimqa`, `hotpotqa`, `triviaqa`
- Methods: `full_rate1`, `online_qk_rate015`, `online_draft_rate015`, `offline3b_mean`, `offline3b_freq_boundary2`, `offline32b_top2`

Note: `triviaqa/offline32b_top2` has only 200 rows in the old generated CSVs, so it remains incomplete and should not be compared as a full 270-example TriviaQA run.

## New Accuracy Table

| dataset | method | rows | old Main Acc | GLM-clean Main Acc | old Sub Acc | GLM-clean Sub Acc | old->GLM false->true | old->GLM true->false |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimqa | full_rate1 | 200 | 113/200 (56.50%) | 117/200 (58.50%) | 113/200 (56.50%) | 117/200 (58.50%) | 5 | 1 |
| 2wikimqa | online_qk_rate015 | 200 | 107/200 (53.50%) | 112/200 (56.00%) | 107/200 (53.50%) | 112/200 (56.00%) | 6 | 1 |
| 2wikimqa | online_draft_rate015 | 200 | 101/200 (50.50%) | 106/200 (53.00%) | 101/200 (50.50%) | 106/200 (53.00%) | 8 | 3 |
| 2wikimqa | offline3b_mean | 200 | 96/200 (48.00%) | 104/200 (52.00%) | 96/200 (48.00%) | 104/200 (52.00%) | 10 | 2 |
| 2wikimqa | offline3b_freq_boundary2 | 200 | 99/200 (49.50%) | 105/200 (52.50%) | 99/200 (49.50%) | 105/200 (52.50%) | 7 | 1 |
| 2wikimqa | offline32b_top2 | 200 | 103/200 (51.50%) | 106/200 (53.00%) | 103/200 (51.50%) | 106/200 (53.00%) | 6 | 3 |
| hotpotqa | full_rate1 | 260 | 207/260 (79.62%) | 236/260 (90.77%) | 207/260 (79.62%) | 236/260 (90.77%) | 30 | 1 |
| hotpotqa | online_qk_rate015 | 260 | 206/260 (79.23%) | 226/260 (86.92%) | 206/260 (79.23%) | 226/260 (86.92%) | 21 | 1 |
| hotpotqa | online_draft_rate015 | 260 | 207/260 (79.62%) | 228/260 (87.69%) | 207/260 (79.62%) | 228/260 (87.69%) | 22 | 1 |
| hotpotqa | offline3b_mean | 260 | 204/260 (78.46%) | 229/260 (88.08%) | 204/260 (78.46%) | 229/260 (88.08%) | 27 | 2 |
| hotpotqa | offline3b_freq_boundary2 | 260 | 201/260 (77.31%) | 225/260 (86.54%) | 201/260 (77.31%) | 225/260 (86.54%) | 25 | 1 |
| hotpotqa | offline32b_top2 | 260 | 197/260 (75.77%) | 227/260 (87.31%) | 197/260 (75.77%) | 227/260 (87.31%) | 31 | 1 |
| triviaqa | full_rate1 | 270 | 211/270 (78.15%) | 246/270 (91.11%) | 211/270 (78.15%) | 246/270 (91.11%) | 35 | 0 |
| triviaqa | online_qk_rate015 | 270 | 212/270 (78.52%) | 241/270 (89.26%) | 212/270 (78.52%) | 241/270 (89.26%) | 32 | 3 |
| triviaqa | online_draft_rate015 | 270 | 214/270 (79.26%) | 239/270 (88.52%) | 214/270 (79.26%) | 239/270 (88.52%) | 27 | 2 |
| triviaqa | offline3b_mean | 270 | 215/270 (79.63%) | 241/270 (89.26%) | 215/270 (79.63%) | 241/270 (89.26%) | 26 | 0 |
| triviaqa | offline3b_freq_boundary2 | 270 | 218/270 (80.74%) | 243/270 (90.00%) | 218/270 (80.74%) | 243/270 (90.00%) | 26 | 1 |
| triviaqa | offline32b_top2 | 200 | 156/200 (78.00%) | 180/200 (90.00%) | 156/200 (78.00%) | 180/200 (90.00%) | 25 | 1 |

## Observations

1. GLM-clean rejudge raises all methods because the old CSV judge under-counted many semantically correct answers, especially on HotpotQA and TriviaQA.
2. The relative ordering still does not show rate=0.15 matching full recompute after clean judging. Full rate=1 remains strongest on HotpotQA and TriviaQA under GLM-clean.
3. On 2WikiMQA, all methods remain much lower than HotpotQA/TriviaQA, so the dataset difficulty gap is not only a judge artifact.
4. Because this pass reuses old generated predictions, it answers only the evaluation-label question. It does not prove whether the current pipeline generation behavior has changed.
