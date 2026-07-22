# Qwen3 Online QK/Draft Baseline Provenance Audit

Goal: test whether historical Online QK / Online Draft `rate=0.15` baselines are comparable to current-code reruns, or whether prompt/code/postprocess changes can explain the metric shift.

## 2026-07-13 controlled replay launch plan

Question from user: the current rerun only proves current code can reproduce; it does not prove that old Online QK/Draft metrics were not affected by later optimizations. We therefore run a controlled replay with the old launch parameters.

Old launch script used as source of truth:
- `MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/run_cross_dataset_supervisor.sh`

Old launch parameter subset to reproduce:
- model: `/mnt/qjhs-sh-lab-01/models/Qwen3-32B`
- data: `MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/2wikimqa_reflect.json`
- dataset: `2wikimqa`
- range: `0-25`
- rate: `0.15`
- preprocess: `true`
- recall: `bge`
- topk: `10`
- rope: `--revert_rope true`
- judge: `GLM-5.2`
- QK method: `--reprocess_method FusionRAG`
- Draft method: `--reprocess_method DraftModel --draft_model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`

Compared code versions:
- current HEAD before launcher commit: `b7264dd`
- old code: `5b62705`, before current Qwen3 no-think prompt change `1049cbb`

Experiment launcher:
- `scripts/run_2wiki_0_25_current_vs_5b62705.py`

Result root:
- `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_baseline_provenance_audit/2wiki_0_25_current_vs_5b62705`

Expected interpretation:
- If old commit and current commit produce different decoded answers on the same 25 examples, then historical baseline drift is real and old metrics should not be mixed with current alpha/full results.
- If they match closely, then the old/current metric gap likely comes from dataset/cache/judge randomness or later summarization differences rather than generation-path code.

Launch command after committing this plan/script:

```bash
ssh qjy000 'cd /raid/home/hming/FusionRAG-pca-analysis && nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_baseline_provenance_audit/scripts/run_2wiki_0_25_current_vs_5b62705.py > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_baseline_provenance_audit/2wiki_0_25_current_vs_5b62705/launcher.log 2>&1 &'
```

## 2026-07-13 old cross-dataset config replay

User request: reproduce the old Online QK / Online Draft performance and check whether the old preprocess KV still exists.

Old source script:
- `MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/run_cross_dataset_supervisor.sh`

Old original cache root from the script:
- `/raid/home/hming/fusionrag-crossdataset-qwen3-cache`

Cache status:
- The original `/raid/home/hming/fusionrag-crossdataset-qwen3-cache` directory is no longer present.
- An equivalent worker/dataset structured cache exists at `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-attn-alpha-cross-cache-v2`.
- This cache contains raw KV and `preprocess_kv_cache_global_topk10_bge` for `2wikimqa`, `hotpotqa`, and `triviaqa` under `worker_gpu*` directories.

Replay launcher:
- `scripts/run_cross_dataset_old_config_replay.py`

Replay result root:
- `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_baseline_provenance_audit/cross_dataset_old_config_replay_current_head`

Replay scope:
- datasets: `2wikimqa`, `hotpotqa`, `triviaqa`
- methods: `online_qk_rate015`, `online_draft_rate015`
- segmenting: same old 25-example segments
- model: Qwen3-32B
- rate: `0.15`
- topk: `10`
- preprocess: `true`
- recall: `bge`
- preprocess scope: `global`
- read-only cache: `FUSIONRAG_PREPROCESS_CACHE_READONLY=1`

Launch command:

```bash
ssh qjy000 'cd /raid/home/hming/FusionRAG-pca-analysis && nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_baseline_provenance_audit/scripts/run_cross_dataset_old_config_replay.py > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_baseline_provenance_audit/cross_dataset_old_config_replay_current_head/launcher.log 2>&1 &'
```

Interpretation rule:
- If this old-config/current-code replay returns close to old records (`QK: 107/206/212`, `Draft: 101/207/214`), the old performance is reproducible with the surviving cache/config.
- If it returns close to the recent current rerun, then the old records likely depend on older code/prompt/postprocess/judge behavior, not merely cache segmentation.
- If read-only cache misses occur, record the missing cache paths and do not silently regenerate preprocess KV.

## 2026-07-13 MuSiQue old rate=0.15 replay

User request: first reproduce the old MuSiQue performance, then reproduce the other cross-dataset results.

Old source script:
- `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/launch_qwen3_rate015_segments.sh`

Old recorded MuSiQue results:
- Online QK r=0.15: `84/135 main = 62.22%`, `189/248 sub = 76.21%`
- Online Draft r=0.15: `99/135 main = 73.33%`, `209/248 sub = 84.27%`

Old original cache root from the script:
- `/raid/home/hming/fusionrag-reflect-qwen3-full-cache` (not present now)

Available replacement cache:
- `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-kv-blend-beta-shared-cache/musique`
- Contains `Qwen3-32B/musique/kv_cache` and `Qwen3-32B/musique/preprocess_kv_cache_global_topk10_bge`.

Replay launcher:
- `scripts/run_musique_old_rate015_replay.py`

Replay result root:
- `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_baseline_provenance_audit/musique_old_rate015_replay_current_head`

Settings copied from old script:
- data: `./data/result_reflect.json`
- dataset: `musique`
- range: `0-200` as eight 25-example segments
- methods: `online_qk_rate015`, `online_draft_rate015`
- rate: `0.15`
- topk: `10`
- preprocess: `true`
- recall: `bge`
- preprocess scope: `global`
- judge: `GLM-5.2`
- read-only cache: `FUSIONRAG_PREPROCESS_CACHE_READONLY=1`

Launch command:

```bash
ssh qjy000 'cd /raid/home/hming/FusionRAG-pca-analysis && nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_baseline_provenance_audit/scripts/run_musique_old_rate015_replay.py > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_baseline_provenance_audit/musique_old_rate015_replay_current_head/launcher.log 2>&1 &'
```

## 2026-07-14 current-vs-old code audit for suspicious rate=0.15 behavior

User concern: current `rate=0.15` recompute performance being close to `rate=1` may come from code drift rather than a real method effect. I compared current code against two old baselines:

- `e8c2b34` from 2026-07-09 01:48, before the recent alpha/blend/fresh-cache work.
- `5b62705`, the old baseline commit already used by `scripts/run_2wiki_0_25_current_vs_5b62705.py`.

Commands used:

```bash
git diff --stat e8c2b34 -- ktransformers/util/utils.py test_fusionrag_reflect_preprocess_exp.py
git diff e8c2b34 -- ktransformers/util/utils.py
git diff e8c2b34 -- test_fusionrag_reflect_preprocess_exp.py
git diff 5b62705 -- test_fusionrag_reflect_preprocess_exp.py
git diff 5b62705 -- ktransformers/util/utils.py
diff -u /mnt/qjhs-sh-lab-01/wjh/FusionRAG/ktransformers/models/modeling_qwen3.py models/modeling_qwen3.py
```

Main differences found:

1. Qwen3 model implementation source changed. Old code imported `Qwen3ForCausalLM` from `ktransformers.models.modeling_qwen3`. Current code prepends the repo root to `sys.path`, appends the upstream ktransformers path, and imports `Qwen2ForCausalLM as Qwen3ForCausalLM` from local `models/modeling_qwen3.py`. This local file was introduced by `c9aa866` and contains the selected-token attention ablation hooks. Even when ablation env vars are unset, this is a different Python module path and should be treated as a possible generation/recompute drift source.

2. Qwen3 prompt format changed. Old non-long-decode prompt:

```text
<|im_start|>user
/no_think
Question: ...
<|im_start|>assistant
Answer:
```

Current prompt:

```text
<|im_start|>user
Question: ...
/no_think
<|im_start|>assistant
<think>

</think>

Answer:
```

This can directly change output format, empty `<think>` behavior, prompt token count, and GLM judge accuracy. It also affects `rate=1`, so old `rate=1` and current `rate=1` should not be mixed.

3. Recompute path gained optional experimental controls after the old baseline:

- `FUSIONRAG_REPROCESS_ATTENTION_ABLATION={uniform,random}`
- `FUSIONRAG_REPROCESS_ATTENTION_ABLATION_ALPHA`
- `FUSIONRAG_REPROCESS_KV_BLEND_BETA`
- `FUSIONRAG_REPROCESS_KV_BLEND_MODE`
- `FUSIONRAG_FORCE_ALL_REPROCESS`
- `FUSIONRAG_REPROCESS_UPDATE_MODE`
- static Key/Value bias and linear adapter paths

These are mostly default-off in normal runs. I checked active fresh-rate015 process environments and did not see `FUSIONRAG_FORCE_ALL_REPROCESS` or attention/blend variables set. So they are less likely to explain the current normal rerun unless a launcher exports them.

4. Selection accounting did not become full recompute by default. Current `load_kv_and_generate` still does:

```text
doc_reprocess_index = selected doc tokens
k_need_index = selected doc tokens + query tokens
```

Therefore `reprocess_prefill_tokens` includes query tokens and is not the document recompute rate. The better metric is `selected_doc_tokens / doc_tokens`.

5. Preprocess/cache code gained read-only cache checks and `rag_docs` recall mode. These do not change default `recall_method=bge`, but they do change failure behavior under `FUSIONRAG_PREPROCESS_CACHE_READONLY=1`.

Interpretation:

- The largest real drift relative to old records is not the rate selection formula. It is the Qwen3 generation path: local `models/modeling_qwen3.py` plus the prompt template change.
- A strict comparison must run four cells on the same small slice: old commit + old prompt/modeling, current commit + current prompt/modeling, current commit forced to old prompt, and current commit forced to upstream modeling. Without this, `rate=0.15 ~= rate=1` could just be prompt/model-path drift.
- If the user wants to know whether recompute itself is wrong, the next instrumentation should log per sample: `doc_tokens`, `selected_doc_tokens`, `query_tokens`, imported module file path for `modeling_qwen3`, prompt token ids length, and all `FUSIONRAG_*` env vars.



## 2026-07-14 correction: old baseline after Qwen3.5/235B MoE support

User clarification: the relevant previous experiments were likely run after adding Qwen3.5 / 235B MoE support. I checked the commit timeline:

- `b7ec8af` - `add qwen3 moe fusionrag adapter`; adds `ktransformers/models/modeling_qwen3_moe.py` and updates `test_fusionrag_reflect_preprocess_exp.py` / `utils.py`.
- `e8c2b34` - `align static cache with sharded qwen3 moe layers`; further aligns StaticCache with sharded Qwen3 MoE layers.
- `1049cbb` - later checkpoint before attention ablation; adds many static-bias/cache/read-only and prompt changes.
- `c9aa866` - later selected-token attention ablation for Qwen3; adds local `models/modeling_qwen3.py`.

Correction to the earlier audit:

- If the old result being compared was produced after `b7ec8af` or `e8c2b34`, then Qwen3.5/235B MoE support itself is not the suspicious drift.
- For those results, the more relevant comparison base is `e8c2b34`, not `5b62705`.
- Relative to `e8c2b34`, the main suspicious changes are still:
  1. current local `models/modeling_qwen3.py` introduced by `c9aa866`;
  2. Qwen3 prompt template change around `1049cbb`;
  3. optional recompute attention/blend/bias hooks, default-off unless exported.

Revised interpretation:

- The old-vs-current drift should be split into two questions:
  1. post-MoE-support baseline (`e8c2b34`) vs current HEAD: likely prompt/modeling_qwen3/attention-hook drift;
  2. pre-MoE baseline (`5b62705`) vs current HEAD: includes additional Qwen3/MoE infrastructure drift and is too broad for this specific suspicion.

Next strict test should therefore replay a small slice at `e8c2b34` and current HEAD with the same cache and judge, then optionally isolate current HEAD with old prompt and upstream Qwen3 modeling.



## 2026-07-14 e8c2b34 vs current MuSiQue 0-25 replay

User request: use idle GPUs to cut an old branch and run a direct comparison. I created a detached worktree at `e8c2b34` and ran MuSiQue samples `0-25` on qjy000 GPUs 3-6.

Result root:

```text
/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_e8c2b34_vs_current_musique_0_25_20260714
```

Worktrees / commits:

- current repo: `/raid/home/hming/FusionRAG-pca-analysis`, HEAD `3ec6ded` with two pre-existing uncommitted pipeline changes in `utils.py` and `test_fusionrag_reflect_preprocess_exp.py`.
- old repo: `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag_worktrees/fusionrag_e8c2b34`, HEAD `e8c2b34`.

Common settings:

- model: `/mnt/qjhs-sh-lab-01/models/Qwen3-32B`
- dataset: `data/result_reflect.json`, `musique`, `start_sample=0`, `end_sample=25`
- rate: `0.15`
- preprocess: `true`
- recall: `bge`, `topk=10`, `preprocess_scope=global`
- cache path: `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-kv-blend-beta-shared-cache/musique`
- judge: `GLM-5.2`

Launch mapping:

- current QK: GPU 3, PID 1319970
- current Draft: GPU 4, PID 1321716
- e8 QK: GPU 5, PID 1323009
- e8 Draft: GPU 6, PID 1324656

Each run has its exact `command.txt` under the result root.

Results:

| Method | Rows | Correct | Correct rate | Avg F1 | Avg EM | Predictions containing `<think>` | Predictions containing literal `Answer:` |
|---|---:|---:|---:|---:|---:|---:|---:|
| current QK | 38 | 33 | 86.84% | 0.5206 | 0.2632 | 0 | 0 |
| current Draft | 38 | 33 | 86.84% | 0.5401 | 0.2368 | 0 | 0 |
| e8c2b34 QK | 38 | 33 | 86.84% | 0.1531 | 0.0000 | 29 | 15 |
| e8c2b34 Draft | 38 | 34 | 89.47% | 0.1457 | 0.0263 | 32 | 12 |

Row-wise correct flips:

- current QK vs e8 QK: 38 common rows, 4 correct-label flips.
- current Draft vs e8 Draft: 38 common rows, 3 correct-label flips.

Interpretation:

- Correct-rate judged by GLM is almost unchanged between `e8c2b34` and current on this slice.
- F1/EM changed massively because `e8c2b34` outputs often include `<think>`, `</think>`, and literal `Answer:` in the predicted text, while current outputs do not.
- This supports the prompt/output-format drift hypothesis. The suspicious old-vs-current metric mismatch is not primarily explained by rate selection becoming full recompute on this slice.
- For future comparisons, GLM Correct and token-level F1/EM must be separated. F1/EM is strongly contaminated by Qwen3 thinking-format differences unless prediction cleaning is made identical.

## 2026-07-13 strict fresh-cache QK/Draft rerun

User request:
- Re-run Online QK and Online DraftModel with newly generated preprocess KV folders.
- Launch on qjy001 and qjy002 across datasets.
- This is a cache-provenance control for the concern that the shared preprocess cache used by alpha/blend experiments may have better quality or mixed provenance.

Launcher:
- `scripts/run_strict_fresh_cache_qk_draft_rate015.py`

Launcher commit:
- `8986018`

Result root:
- `/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_strict_fresh_cache_qk_draft_rate015_20260713`

Fresh cache root:
- `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-strict-fresh-cache-20260713`

Important cache rules:
- `FUSIONRAG_PREPROCESS_CACHE_READONLY` is intentionally unset.
- Each host/GPU writes to a separate cache subtree: `${CACHE_ROOT}/${host}/worker_gpu${gpu}/${dataset}`.
- This avoids different workers writing the same cache files concurrently.
- Existing shared alpha/blend cache roots are not used.

Settings:
- model: Qwen3-32B
- methods: `online_qk_rate015`, `online_draft_rate015`
- rate: `0.15`
- topk: `10`
- preprocess: `true`
- recall: `bge`
- preprocess scope: `global`
- reprocess: `FusionRAG` for QK, `DraftModel` with Qwen2.5-3B-Instruct for DraftModel
- judge: `GLM-5.2`

Launch commands:

```bash
ssh qjy001 'cd /home/hming/FusionRAG-pca-analysis && FUSIONRAG_STRICT_FRESH_DATASETS=2wikimqa,hotpotqa nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_baseline_provenance_audit/scripts/run_strict_fresh_cache_qk_draft_rate015.py > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_strict_fresh_cache_qk_draft_rate015_20260713/logs/launcher_qjy001.log 2>&1 < /dev/null &'

ssh qjy002 'cd /home/hming/FusionRAG-pca-analysis && FUSIONRAG_STRICT_FRESH_DATASETS=triviaqa,musique nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_baseline_provenance_audit/scripts/run_strict_fresh_cache_qk_draft_rate015.py > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_strict_fresh_cache_qk_draft_rate015_20260713/logs/launcher_qjy002.log 2>&1 < /dev/null &'
```

Interpretation rule:
- Compare these fresh-cache reruns against the current shared-cache QK/Draft reruns.
- If fresh-cache and shared-cache metrics match within judge variance, cache provenance is unlikely to explain the current baseline shift.
- If they differ materially, then current shared-cache results are cache-dependent and should not be mixed with old or fresh-cache baselines.

Path correction before launch:
- qjy001/qjy002 use `/home/hming/FusionRAG-pca-analysis`; launcher now auto-detects `/raid/home/hming/...` vs `/home/hming/...`.

Launch status after starting jobs:
- Launch/runtime commit: `d70af29`
- qjy001 launcher PID: `3506053`; datasets: 2wikimqa, hotpotqa; status: running on qjhs-sh-lab-02 GPUs 0-7.
- qjy002 launcher PID: `1406446`; datasets: triviaqa, musique; status: queued/waiting on qjhs-sh-lab-03 GPUs because all GPUs were occupied at launch.
- Initial check: qjy001 worker logs show Online QK 2WikiMQA segments running and fresh cache directories under `fusionrag-qwen3-strict-fresh-cache-20260713/qjhs-sh-lab-02/worker_gpu*/2wikimqa`.

## 2026-07-14 old 99/135 Draft rate=0.15 lookup

User concern:
- Current MuSiQue DraftModel `rate=0.15` fresh rerun looks too high; old result was expected to be around `99/135` main questions.

Old result source:
- Directory: `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/`
- Summary file: `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/summary.csv`
- Launch script: `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/launch_qwen3_rate015_segments.sh`
- Old online Draft result: `99/135` main, `209/248` unique sub-questions, F1 `0.1905`, EM `0.0081`.

Current comparison source:
- Directory: `/mnt/qjhs-sh-lab-03/lpl/hming/qwen3_musique_fresh_preprocess_3host_sharded_20260714/online_draft_rate015`
- Current online Draft result on the same 248 unique `(Main Question, Sub Question)` keys: `219/248` sub-questions.
- Main aggregation is duplicate-sensitive because one duplicated key changes label in the current run: raw/all-row aggregation gives `107/135`, last-row unique aggregation gives `108/135`.

Alignment result:

| Check | Old | Current |
|---|---:|---:|
| CSV files | 8 | 8 |
| Raw rows | 250 | 250 |
| Unique `(Main, Sub)` keys | 248 | 248 |
| Keys only in old | 0 | - |
| Keys only in current | - | 0 |
| Unique sub correct | 209 | 219 |
| Sub correct-label flips | - | 32 total: 21 old wrong -> current correct, 11 old correct -> current wrong |
| Main correct-label flips | - | 27 total: 18 old wrong -> current correct, 9 old correct -> current wrong |

Important observation:
- Many old rows have `Predicted` values such as `<think>\n\n</think>\n\nAnswer: Kenton County`, but the GLM judge `Reason` says the predicted answer is empty.
- The corresponding current row often has plain answer text such as `Kenton County` and is judged correct.
- Therefore the old `99/135` Draft number is a real historical record, but part of its gap to current reruns is caused by prompt/output-format and judge parsing behavior, not by a different sample set.

Examples:
- `Which county is Fort Mitchell, Kentucky located in?`: old prediction `<think>...</think> Answer: Kenton County` judged wrong as empty; current `Kenton County` judged correct.
- `Where was Mark Dismore born?`: old prediction `<think>...</think> Greenfield, Indiana` judged wrong as empty; current `Greenfield, Indiana` judged correct.
- `What record label was Joni James signed to...`: old prediction `<think>...</think> MGM Records` judged wrong as empty; current full sentence containing `MGM Records` judged correct.

Conclusion for baseline comparison:
- The old expected number is confirmed: `online_draft_rate015 = 99/135` in `qwen3_rate015_online_offline`.
- It should not be compared directly against current fresh reruns unless answer cleaning / judge input formatting is made identical.
- For a strict reproduction, rerun old and current code with the same prompt template and the same post-processing before GLM judge, or rejudge both old/current CSVs after applying the same prediction cleaner.

## 2026-07-14 rejudge old online CSVs with current GLM

User request:
- Re-run GLM judge on the old online CSVs to check whether the historical `99/135` Draft number is affected by judge parsing / output-format drift.

Input CSVs:
- Old Draft: `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/online_draft_rate015/seg_*/Qwen3-32B/musique/DraftModel_global_topk10_bge/rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv`
- Old QK: `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/online_qk_rate015/seg_*/Qwen3-32B/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv`

Judge setup:
- Judge model: `GLM-5.2`
- Base URL: `http://36.150.226.221:32355/v1`
- Evaluated the raw old CSV `Predicted` string against `Ground Truth` and `Sub Question`.
- No model generation or KV recompute was rerun.
- GLM thinking was disabled in the judge request.

Commands:

```bash
ssh qjy000 'cd /raid/home/hming/FusionRAG-pca-analysis && nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/rejudge_old_online_glm/rejudge_old_online_draft_glm.py > MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/rejudge_old_online_glm/rejudge.log 2>&1 < /dev/null &'

ssh qjy000 'cd /raid/home/hming/FusionRAG-pca-analysis && nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/rejudge_old_online_qk_glm/rejudge_old_online_qk_glm.py > MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/rejudge_old_online_qk_glm/rejudge.log 2>&1 < /dev/null &'
```

Outputs:
- Draft rejudged CSV: `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/rejudge_old_online_glm/old_online_draft_rate015_rejudged_glm52.csv`
- Draft summary: `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/rejudge_old_online_glm/summary.json`
- QK rejudged CSV: `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/rejudge_old_online_qk_glm/old_online_qk_rate015_rejudged_glm52.csv`
- QK summary: `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/rejudge_old_online_qk_glm/summary.json`

Results:

| Method | Historical main | Rejudged main | Historical sub | Rejudged sub | Raw-row label flips |
|---|---:|---:|---:|---:|---:|
| Online Draft rate=0.15 | 99/135 (73.33%) | 105/135 (77.78%) | 209/248 (84.27%) | 216/248 (87.10%) | 13: 10 old wrong -> rejudge correct, 3 old correct -> rejudge wrong |
| Online QK rate=0.15 | 84/135 (62.22%) | 90/135 (66.67%) | 189/248 (76.21%) | 193/248 (77.82%) | 12: 8 old wrong -> rejudge correct, 4 old correct -> rejudge wrong |

Interpretation:
- The historical Draft `99/135` is not stable under current GLM judge on the same old predictions; it becomes `105/135`.
- The shift is large enough to confirm that old-vs-current baseline comparisons are partly contaminated by judge/output-format drift.
- Rejudging does not fully explain the current fresh Draft result around `107/135`; after rejudge, the old CSV is still lower by about 2 main questions.
- For strict method comparison, use either rejudged historical CSVs or rerun all methods with identical current prompt, answer cleaner, and judge.

## 2026-07-14 rejudge old online CSVs after cleaning Qwen3 wrappers

User request:
- Remove Qwen3 thinking wrappers and leading answer labels from old online CSV predictions, then re-run GLM judge.

Prediction cleaner:
- Remove `<think>...</think>` blocks.
- Remove any residual `<think>` / `</think>` tags.
- Remove leading `Answer:`, `Final Answer:`, or `答案:` labels.
- No other semantic rewriting is applied.

Commands:

```bash
ssh qjy000 'cd /raid/home/hming/FusionRAG-pca-analysis && nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/rejudge_old_online_draft_clean_glm/rejudge_old_online_draft_clean_glm.py > MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/rejudge_old_online_draft_clean_glm/rejudge.log 2>&1 < /dev/null &'

ssh qjy000 'cd /raid/home/hming/FusionRAG-pca-analysis && nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/rejudge_old_online_qk_clean_glm/rejudge_old_online_qk_clean_glm.py > MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/rejudge_old_online_qk_clean_glm/rejudge.log 2>&1 < /dev/null &'
```

Outputs:
- Draft clean rejudged CSV: `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/rejudge_old_online_draft_clean_glm/old_online_draft_rate015_clean_rejudged_glm52.csv`
- Draft summary: `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/rejudge_old_online_draft_clean_glm/summary.json`
- QK clean rejudged CSV: `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/rejudge_old_online_qk_clean_glm/old_online_qk_rate015_clean_rejudged_glm52.csv`
- QK summary: `MOTIVATION_EXPERIMENTS/qwen3_rate015_online_offline/rejudge_old_online_qk_clean_glm/summary.json`

Results:

| Method | Historical main | Raw rejudge main | Clean rejudge main | Historical sub | Raw rejudge sub | Clean rejudge sub |
|---|---:|---:|---:|---:|---:|---:|
| Online Draft rate=0.15 | 99/135 (73.33%) | 105/135 (77.78%) | 113/135 (83.70%) | 209/248 (84.27%) | 216/248 (87.10%) | 224/248 (90.32%) |
| Online QK rate=0.15 | 84/135 (62.22%) | 90/135 (66.67%) | 101/135 (74.81%) | 189/248 (76.21%) | 193/248 (77.82%) | 207/248 (83.47%) |

Flip counts versus historical labels:
- Draft clean rejudge: 17 raw-row flips; 16 old wrong -> clean correct, 1 old correct -> clean wrong.
- QK clean rejudge: 20 raw-row flips; 19 old wrong -> clean correct, 1 old correct -> clean wrong.

Interpretation:
- Removing `<think>` / `Answer:` wrappers changes the old Online Draft result from `99/135` to `113/135`, which is higher than the current fresh Draft run around `107/135`.
- Removing the wrappers also changes old Online QK from `84/135` to `101/135`.
- Therefore the historical Qwen3 online baseline table is heavily contaminated by answer-format/judge parsing artifacts. It should not be used as an accuracy baseline unless predictions are cleaned and rejudged consistently.
- The method ordering after clean rejudge still favors Draft over QK (`113/135` vs `101/135`), but both absolute accuracies move substantially upward.
