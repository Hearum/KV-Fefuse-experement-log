# Experiment Log

## 2026-07-14

- 新建 `MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/`。
- 目标：按 setup 原始数据格式重跑 Qwen3-32B cross-dataset，数据集命名为 `musique-v2/2wikimqa-v2/hotpotqa-v2/triviaqa-v2`。
- 关键约束：raw/preprocess KV cache 只按 `model/dataset-v2` 共用，不按 worker/GPU/实验复制。
- 当前 commit baseline：待补正式 commit；当前工作树已有大量历史未跟踪实验文件，本目录先独立记录。
- 暂未启动正式评测；先完成脚本和 smoke。



### Smoke: musique-v2 online_qk rate=0.15 example 0:1

命令见 README。结果：runner 成功完成并写出 1 行 CSV；预测 `Salma Hayek`，gold `Maria Bello`，EM/F1 均为 0。shared cache 位于 `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2`，当前约 31G，0 字节 `.pt` 为 0。

修复项：

- `ktransformers/unified_process_cache.py` 增加 `model_type=qwen3` 分支。
- qwen3 输出清理 `<think>...</think>`。
- qwen3 RoPE 路径走 layer self-attn rotary embedding。
- RoPE cos/sin 显式迁移到 `chunk_key_cache.device`，避免 CPU/GPU mismatch。

结论：setup-v2 online runner 已通过单样本 smoke，但首次 preprocess cache 生成非常慢；正式并行前必须先做 cache warmup。


### 2026-07-14 正式 v2 任务启动：online_qk@0.15 + full_rate1

启动 commit：`d120ab4`。

第一批使用 qjy000 8 张卡：

| GPU | dataset | method | rate | PID | log |
|---:|---|---|---:|---:|---|
| 0 | `musique-v2` | `online_qk` | 0.15 | 2021125 | `logs/warmup_online_qk_rate015/musique-v2.log` |
| 1 | `2wikimqa-v2` | `online_qk` | 0.15 | 2021127 | `logs/warmup_online_qk_rate015/2wikimqa-v2.log` |
| 2 | `hotpotqa-v2` | `online_qk` | 0.15 | 2021129 | `logs/warmup_online_qk_rate015/hotpotqa-v2.log` |
| 3 | `triviaqa-v2` | `online_qk` | 0.15 | 2021131 | `logs/warmup_online_qk_rate015/triviaqa-v2.log` |
| 4 | `musique-v2` | `full_rate1` | 1.0 | 2026409 | `logs/full_rate1/musique-v2.log` |
| 5 | `2wikimqa-v2` | `full_rate1` | 1.0 | 2026411 | `logs/full_rate1/2wikimqa-v2.log` |
| 6 | `hotpotqa-v2` | `full_rate1` | 1.0 | 2026413 | `logs/full_rate1/hotpotqa-v2.log` |
| 7 | `triviaqa-v2` | `full_rate1` | 1.0 | 2026415 | `logs/full_rate1/triviaqa-v2.log` |

`online_qk@0.15` 同时承担各 dataset 的 shared raw/preprocess cache warmup；各 dataset 目录独立，因此没有同 dataset 多 worker 抢写。`full_rate1` 不写 preprocess cache，可与 warmup 并行。

### 2026-07-14 20:24 partial summary

这些是运行中 partial 结果，不是最终表；`complete=False` 表示该 dataset/method 还没跑满 expected rows。

| dataset | method | rate | rows | expected | complete | EM | F1 |
|---|---|---:|---:|---:|---|---:|---:|
| `2wikimqa-v2` | `full_rate1` | 1.0 | 37 | 200 | False | 0.4324 | 0.6030 |
| `2wikimqa-v2` | `online_qk` | 0.15 | 1 | 200 | False | 0.0000 | 0.0000 |
| `hotpotqa-v2` | `full_rate1` | 1.0 | 151 | 260 | False | 0.6821 | 0.8007 |
| `hotpotqa-v2` | `online_qk` | 0.15 | 6 | 260 | False | 0.6667 | 0.7619 |
| `musique-v2` | `full_rate1` | 1.0 | 14 | 200 | False | 0.2857 | 0.4139 |
| `musique-v2` | `online_qk` | 0.15 | 2 | 200 | False | 0.0000 | 0.1667 |
| `triviaqa-v2` | `full_rate1` | 1.0 | 153 | 270 | False | 0.6797 | 0.7896 |
| `triviaqa-v2` | `online_qk` | 0.15 | 5 | 270 | False | 0.6000 | 0.6500 |

运行状态：qjy000 GPU0-3 跑 `online_qk@0.15` 并预热 shared cache；GPU4-7 跑 `full_rate1`。shared cache 当前约 120G，0 字节 `.pt` 为 0。

### 2026-07-14 20:33 summary update

`full_rate1` 的 HotpotQA-v2 和 TriviaQA-v2 已完整；其余仍在运行。

| dataset | method | rate | rows | expected | complete | EM | F1 |
|---|---|---:|---:|---:|---|---:|---:|
| `2wikimqa-v2` | `full_rate1` | 1.0 | 85 | 200 | False | 0.4941 | 0.5924 |
| `2wikimqa-v2` | `online_qk` | 0.15 | 4 | 200 | False | 0.0000 | 0.2500 |
| `hotpotqa-v2` | `full_rate1` | 1.0 | 260 | 260 | True | 0.7038 | 0.8078 |
| `hotpotqa-v2` | `online_qk` | 0.15 | 14 | 260 | False | 0.6429 | 0.7166 |
| `musique-v2` | `full_rate1` | 1.0 | 31 | 200 | False | 0.2581 | 0.4179 |
| `musique-v2` | `online_qk` | 0.15 | 4 | 200 | False | 0.2500 | 0.3333 |
| `triviaqa-v2` | `full_rate1` | 1.0 | 270 | 270 | True | 0.6630 | 0.7693 |
| `triviaqa-v2` | `online_qk` | 0.15 | 13 | 270 | False | 0.6154 | 0.7115 |

进度说明：`hotpotqa-v2/full_rate1` 与 `triviaqa-v2/full_rate1` 可作为当前已完成 baseline；online_qk 仍在预热各自 dataset cache，当前结果不能作为最终 accuracy。shared cache 当前约 236G，0 字节 `.pt` 为 0。


### 2026-07-14 full_rate1 分片加速

为加速尚未完成的 `full_rate1`：

- qjy001 GPU0-5：`musique-v2` 分片 `50-75,75-100,100-125,125-150,150-175,175-200`，PID `3787540,3787542,3787544,3787546,3787548,3787550`。
- qjy003 GPU0-2：`2wikimqa-v2` 分片 `132-160,160-180,180-200`，PID `2870771,2870773,2870775`。

这些结果写在各自机器的 `/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/results/`，跑完后需要汇回 qjy000 主目录。`scripts/summarize_setup_v2.py` 已改为按 `(dataset, method, rate, Question)` 去重，允许和 qjy000 monolithic run 有重叠而不重复计数。

### 2026-07-14 full_rate1 完整结果

四个 setup-v2 数据集的 full attention / full recompute baseline 已完整。

| dataset | method | rows | EM | F1 | raw rows | complete |
|---|---|---:|---:|---:|---:|---|
| `2wikimqa-v2` | `full_rate1` | 200/200 | 0.4800 | 0.5899 | 268 | True |
| `hotpotqa-v2` | `full_rate1` | 260/260 | 0.7038 | 0.8078 | 260 | True |
| `musique-v2` | `full_rate1` | 200/200 | 0.2550 | 0.4003 | 230 | True |
| `triviaqa-v2` | `full_rate1` | 270/270 | 0.6630 | 0.7693 | 270 | True |

说明：`raw_rows > rows` 来自 qjy001/qjy003 分片和 qjy000 monolithic 的重叠补跑；`scripts/summarize_setup_v2.py` 按 question 去重后计入最终 rows。

当前仍在运行：qjy000 GPU0-3 的 `online_qk@0.15`，同时继续预热 shared preprocess cache。当前 cache 约 704G，0 字节 `.pt` 为 0。


## 2026-07-14 21:10 - Shared cache lock before multi-GPU sharding

Reason: setup-v2 first-time preprocess cache is shared by model/dataset. Running one worker per GPU without locking can make multiple workers write the same `*_key.pt` / `*_value.pt` concurrently. The previous launch therefore used only four monolithic workers, one dataset per GPU, which was safe but slow.

Code change:
- Added file-level cache write locks in `ktransformers/unified_process_cache.py`.
- Locked raw KV generation, system chunk copy into preprocess cache, preprocess KV generation, and on-the-fly corpus raw KV generation.
- Cache readiness now requires both key and value files, plus Cache-Craft attention files when applicable.

Current policy:
- Same model/dataset must share one cache directory under `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2`.
- Experiments should shard examples across GPUs but not create per-worker cache pools.
- Existing four unsharded online_qk workers will be stopped after this commit, then relaunched as sharded workers using the locked cache writer.

Reproduction commands for the check:
```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python -m py_compile ktransformers/unified_process_cache.py
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/summarize_setup_v2.py
```


## 2026-07-14 21:13 - Relaunch online_qk rate=0.15 with 24 sharded workers

Commit: `b62947c`

Stopped old unsharded workers:
```bash
kill 2021125 2021127 2021129 2021131
```

Generated a focused task file with only `online_qk` at `rate=0.15`:
```bash
# output: MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/logs/online_qk_rate015_tasks.tsv
# datasets: musique-v2=200, 2wikimqa-v2=200, hotpotqa-v2=260, triviaqa-v2=270
# segment size: 25 examples; total tasks: 38
```

Launch policy:
- Hosts: `qjy000`, `qjy001`, `qjy003`.
- GPUs: 8 per host, 24 workers total.
- Worker command template:
```bash
nohup /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/run_setup_v2_worker.py \
  --slot SLOT --total-slots 24 --gpu GPU \
  --tasks MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/logs/online_qk_rate015_tasks.tsv \
  --cache-root /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2 \
  > logs/online_qk_rate015_HOST_gpuGPU.outer.log 2>&1 < /dev/null &
```

Rationale: use all available cards while keeping one shared cache tree, relying on `b62947c` file-level locks to prevent concurrent writes of the same KV files.


## DraftModel launch bug fixed

Issue: setup-v2 `online_draft` / `uniform_alpha0p1_draft` failed with:
```text
ValueError: draft_model must be provided when reprocess_method='DraftModel'
```

Cause: `unified_process_cache.py` only loaded `draft_model_path` when `reprocess_method == "speculative_prefill"`, but this experiment uses `reprocess_method == "DraftModel"`.

Fix: load the draft model when `reprocess_method in ("speculative_prefill", "DraftModel")`.

Failed partial draft/uniform logs before this fix are kept as anomaly records and should not be counted as valid results.


## 2026-07-14 21:43 - Manual qk tail acceleration

Commit: `e8785bd`

Reason: the first qk modulo queue assigned all MuSiQue segments to qjy000. MuSiQue first-time preprocess cache generation is slow, so qk tail tasks for HotpotQA and early TriviaQA would wait behind MuSiQue despite idle GPUs on qjy001/qjy003.

Manual tail tasks launched with the same shared cache and same result layout:
```bash
# qjy003 gpu4: hotpotqa-v2 online_qk rate=0.15 seg_200_225, then mark task0024 done for original queue
# qjy003 gpu5: hotpotqa-v2 online_qk rate=0.15 seg_225_250, then mark task0025 done for original queue
# qjy001 gpu0: hotpotqa-v2 online_qk rate=0.15 seg_250_260, then mark task0026 done for original queue
# qjy001 gpu6: triviaqa-v2 online_qk rate=0.15 seg_0_25, then mark task0027 done for original queue
```

This does not change the method under test; it only moves queued segments to idle GPUs and prevents the original worker queue from rerunning them after success.


## 2026-07-14 23:57 - Missing Items Plan and Current Coverage

Current setup-v2 generation coverage:

| Method | Rate | Coverage |
|---|---:|---|
| `full_rate1` | 1.0 | complete on `musique-v2`, `2wikimqa-v2`, `hotpotqa-v2`, `triviaqa-v2` |
| `online_qk` | 0.15 | complete on all four datasets |
| `online_draft` | 0.15 | complete on all four datasets |
| `uniform_alpha0p1_draft` | 0.15 | complete on 2Wiki/MuSiQue/TriviaQA; HotpotQA was missing 7 rows in `seg_150_175` and is being fixed by a dedicated run on `qjy003 GPU0` |

Known missing items before final report:

1. **Offline setup-v2 methods are not run yet.**
   - Existing `cross_dataset_offline_generalization` offline results are for the older reflect-format pipeline.
   - setup-v2 uses the setup-standard direct QA format, so old fixed-set artifacts cannot be reported as setup-v2 results without either converting the fixed-set indexing or adding setup-v2 fixed-set support.
   - Target offline methods to reproduce on setup-v2: `offline3b_mean`, `offline3b_freq_boundary2`, `offline32b_top2` at `rate=0.15`; optionally `offline_qk_mean` / `offline_qk_mean_boundary2` remain pending because old cross-dataset also marks them pending.

2. **GLM-clean judging is not done for setup-v2 yet.**
   - Current `setup_v2_summary.csv` is local exact/F1 only.
   - Added `scripts/rejudge_setup_v2_glm_clean.py` to reuse the same GLM-5.2 clean judge style as `cross_dataset_offline_generalization/rejudge_glm_clean_20260714`.
   - Required output: `rejudge_glm_clean_20260714/rejudged_rows.csv`, `rejudged_summary.csv`, `rejudged_summary.json`.

3. **Path portability bug fixed.**
   - `scripts/summarize_setup_v2.py` previously hardcoded `/raid/home/hming/FusionRAG-pca-analysis`, which failed on qjy001/qjy003.
   - It now uses `/raid/...` or `/home/...` whichever exists.

4. **Scheduling correction.**
   - Queue/modulo scheduling caused idle GPUs and duplicate segment runs.
   - Cleanup stage now uses fixed explicit `(host, GPU, dataset, method, rate, segment)` assignments only.

Commands added/used:

```bash
# local exact/F1 summary
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/summarize_setup_v2.py

# setup-v2 GLM clean rejudge
GLM_REJUDGE_WORKERS=10 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/rejudge_setup_v2_glm_clean.py   > MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/rejudge_glm_clean_20260714/rejudge.log 2>&1
```


## 2026-07-15 00:30 setup-v2 offline / GLM 状态修正

用户指出两个缺口：offline 组合尚未跑、setup-v2 结果尚未走 GLM clean rejudge。当前结论如下：

1. 当前 setup-v2 online 表只包含 `full_rate1`、`online_qk`、`online_draft`、`uniform_alpha0p1_draft`。这些是本目录同源 direct-QA pipeline 的自动 EM/F1 结果，不是 GLM rejudge 结果。
2. 旧 `cross_dataset_offline_generalization` 的 offline 结果来自较早 reflect-format pipeline，不能直接和 setup-v2 direct-QA 表混用。
3. 旧 offline fixed-set artifact 可直接对齐的 setup-v2 数据集只有 `hotpotqa-v2` 与 `triviaqa-v2`：二者均为 10 documents/example，fixed-set 也是 10 chunk/example。
4. `2wikimqa-v2` 当前 setup-v2 文件本身每例 document 数不固定，审计为 9-24 passages/example；旧 fixed-set 多为 10 chunks/example，不能直接复用，否则 selection support 不完整。
5. `musique-v2` 当前 setup-v2 文件为 19-31 passages/example，且旧 cross-dataset offline 目录没有 musique fixed-set，因此也不能直接复用旧 offline。
6. 已添加 setup-v2 runner 的 offline fixed-set 注入口：`offline3b_mean`、`offline3b_freq_boundary2`、`offline32b_top2`。该入口只应先用于 HotpotQA/TriviaQA；2Wiki/MuSiQue 后续需要重建 setup-v2 自己的 fixed-set。
7. 已添加但尚未完成运行的 GLM clean rejudge 脚本：`scripts/rejudge_setup_v2_glm_clean.py`。它会清理 `<think>...</think>` 和 `Answer:` 后统一调用 GLM judge。

本轮 smoke：

```bash
# 误跑：传参缺口导致实际仍是 online FusionRAG selection，不能计为 offline
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py   --dataset 2wikimqa-v2 --method offline3b_mean --rate 0.15 --start 0 --end 5 --gpu 0   --cache-root /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2

# 正确进入 offline fixed-set 分支，但因 2Wiki setup-v2 有 >10 passages 而旧 fixed-set 缺 chunk10，失败退出
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py   --dataset 2wikimqa-v2 --method offline3b_mean --rate 0.15 --start 5 --end 10 --gpu 0   --cache-root /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2
```

下一步执行：

1. 先在 `hotpotqa-v2`、`triviaqa-v2` 跑 setup-v2 同源 offline 三组：`offline3b_mean`、`offline3b_freq_boundary2`、`offline32b_top2`，rate=0.15。
2. 启动 setup-v2 当前结果的 GLM clean rejudge，得到可和旧 GLM 表一致口径的结果。
3. 单独规划 `2wikimqa-v2` / `musique-v2` 的 setup-v2 fixed-set 生成，不复用旧 10-chunk fixed-set。


## 2026-07-15 00:55 offline 固定分配启动记录

Code commit: `e49e897`。

GLM clean rejudge 已启动，当前只过滤在线四组，避免把 2Wiki offline smoke 误纳入：

```bash
nohup env SETUP_V2_REJUDGE_METHODS=full_rate1,online_qk,online_draft,uniform_alpha0p1_draft \
  GLM_REJUDGE_WORKERS=10 \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/rejudge_setup_v2_glm_clean.py \
  > MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/rejudge_glm_clean_20260714/rejudge_online4.log 2>&1 &
# pid: 3963091 on qjy000
```

Offline fixed-set 同源 setup-v2 实验只启动 HotpotQA/TriviaQA，原因是这两个数据集和旧 fixed-set 都是 10 doc/chunk；2Wiki/MuSiQue 需要先重建 setup-v2 fixed-set。

第一次尝试使用 zsh `set -- $item`，因为 zsh 不按 bash 方式拆分字符串，导致 `--method` 为空，所有任务秒退。失败日志在 `logs/offline_launch_20260715/`，不计入结果。

已用 `bash -lc` 和冒号分隔任务表重新固定分配，日志目录：`logs/offline_launch_fixed_20260715/`。

固定分配如下：

| host | gpu | dataset | method | segment |
|---|---:|---|---|---|
| qjy000 | 0-3 | hotpotqa-v2 | offline3b_mean | 0-65,65-130,130-195,195-260 |
| qjy000 | 4-7 | hotpotqa-v2 | offline3b_freq_boundary2 | 0-65,65-130,130-195,195-260 |
| qjy001 | 0-3 | hotpotqa-v2 | offline32b_top2 | 0-65,65-130,130-195,195-260 |
| qjy001 | 4-7 | triviaqa-v2 | offline3b_mean | 0-68,68-136,136-203,203-270 |
| qjy003 | 0-3 | triviaqa-v2 | offline3b_freq_boundary2 | 0-68,68-136,136-203,203-270 |
| qjy003 | 4-7 | triviaqa-v2 | offline32b_top2 | 0-68,68-136,136-203,203-270 |

启动后检查：三台机器 GPU 均出现显存占用，说明 fixed launch 已经实际运行；后续需要用 `summarize_setup_v2.py` 汇总完成 rows。


## 2026-07-15 01:10 当前结果汇总

Code commits:

- `f5bbefb`：setup-v2 offline fixed-set runner。
- `e49e897`：GLM rejudge method filter。
- `72e07dd`：offline fixed launch 文档记录。

### 自动 EM/F1 汇总

文件：`setup_v2_summary.csv`。

注意：`2wikimqa-v2,offline3b_mean` 的 5 行来自 2Wiki smoke，不是有效 offline 结果；2Wiki/MuSiQue 需要重建 setup-v2 fixed-set 后再跑 offline。

```csv
dataset,method,rate,rows,raw_rows,expected_rows,complete,em,f1
2wikimqa-v2,full_rate1,1.0,200,268,200,True,0.48,0.589860795174611
2wikimqa-v2,offline3b_mean,0.15,5,5,200,False,0.0,0.3
2wikimqa-v2,online_draft,0.15,200,200,200,True,0.43,0.5768699160889564
2wikimqa-v2,online_qk,0.15,200,229,200,True,0.42,0.5312623607005759
2wikimqa-v2,uniform_alpha0p1_draft,0.15,200,200,200,True,0.43,0.5632340420406782
hotpotqa-v2,full_rate1,1.0,260,260,260,True,0.7038461538461539,0.8077728227225129
hotpotqa-v2,offline32b_top2,0.15,260,260,260,True,0.6807692307692308,0.7842189428879711
hotpotqa-v2,offline3b_freq_boundary2,0.15,260,260,260,True,0.6846153846153846,0.7832825288746338
hotpotqa-v2,offline3b_mean,0.15,260,265,260,True,0.6923076923076923,0.7894721970495961
hotpotqa-v2,online_draft,0.15,260,277,260,True,0.7076923076923077,0.8111911248504927
hotpotqa-v2,online_qk,0.15,260,346,260,True,0.6653846153846154,0.7708671120884641
hotpotqa-v2,uniform_alpha0p1_draft,0.15,260,289,260,True,0.7038461538461539,0.8093841604473745
musique-v2,full_rate1,1.0,200,230,200,True,0.255,0.4002775677376809
musique-v2,online_draft,0.15,200,200,200,True,0.24,0.37878427248645763
musique-v2,online_qk,0.15,200,214,200,True,0.215,0.3556672976357179
musique-v2,uniform_alpha0p1_draft,0.15,200,285,200,True,0.25,0.39276670073242526
triviaqa-v2,full_rate1,1.0,270,270,270,True,0.662962962962963,0.7693137726471061
triviaqa-v2,offline32b_top2,0.15,270,270,270,True,0.6481481481481481,0.7557791300928555
triviaqa-v2,offline3b_freq_boundary2,0.15,270,270,270,True,0.6407407407407407,0.7515247381914048
triviaqa-v2,offline3b_mean,0.15,270,270,270,True,0.6481481481481481,0.7569181349298307
triviaqa-v2,online_draft,0.15,270,270,270,True,0.6666666666666666,0.7711656244989576
triviaqa-v2,online_qk,0.15,270,358,270,True,0.6481481481481481,0.7513650546983879
triviaqa-v2,uniform_alpha0p1_draft,0.15,270,270,270,True,0.6592592592592592,0.7645787870349272
```

关键观察：

1. HotpotQA-v2：offline 三组全量完成，但都低于 online DraftModel / uniform alpha=0.1 / full。`offline3b_mean` 最好，EM/F1 = 0.6923/0.7895；online DraftModel 是 0.7077/0.8112，full 是 0.7038/0.8078。
2. TriviaQA-v2：offline 三组全量完成，和 online_qk 接近，但低于 online DraftModel 与 full。`offline3b_mean` 和 `offline32b_top2` EM 都是 0.6481，online DraftModel/full 是 0.6667/0.6630。
3. 这说明旧 offline fixed-set 在 setup-v2 direct-QA pipeline 里不能替代 online DraftModel；它可作为 lower-cost baseline，但不是当前最优方案。
4. Hotpot/Trivia 的 offline fixed-set 至少证明接口和同源 pipeline 可运行；下一步更关键的是为 2Wiki/MuSiQue 重新生成与 setup-v2 document 数一致的 fixed-set，而不是复用旧 10-chunk artifact。

### GLM clean rejudge 汇总

本次 GLM 只重判在线四组：`full_rate1,online_qk,online_draft,uniform_alpha0p1_draft`，共 3720 rows。文件：`rejudge_glm_clean_20260714/rejudged_summary.csv`。

```csv
dataset,method,rate,rows,expected_rows,complete,glm_correct,glm_acc
2wikimqa-v2,full_rate1,1.0,200,200,True,123,0.615
2wikimqa-v2,online_draft,0.15,200,200,True,120,0.6
2wikimqa-v2,online_qk,0.15,200,200,True,106,0.53
2wikimqa-v2,uniform_alpha0p1_draft,0.15,200,200,True,116,0.58
hotpotqa-v2,full_rate1,1.0,260,260,True,237,0.9115384615384615
hotpotqa-v2,online_draft,0.15,260,260,True,231,0.8884615384615384
hotpotqa-v2,online_qk,0.15,260,260,True,223,0.8576923076923076
hotpotqa-v2,uniform_alpha0p1_draft,0.15,260,260,True,232,0.8923076923076924
musique-v2,full_rate1,1.0,200,200,True,83,0.415
musique-v2,online_draft,0.15,200,200,True,81,0.405
musique-v2,online_qk,0.15,200,200,True,75,0.375
musique-v2,uniform_alpha0p1_draft,0.15,200,200,True,83,0.415
triviaqa-v2,full_rate1,1.0,270,270,True,243,0.9
triviaqa-v2,online_draft,0.15,270,270,True,243,0.9
triviaqa-v2,online_qk,0.15,270,270,True,236,0.8740740740740741
triviaqa-v2,uniform_alpha0p1_draft,0.15,270,270,True,242,0.8962962962962963
```

GLM 结论：

1. Full attention 仍是最强或并列最强：2Wiki 61.5%，Hotpot 91.15%，MuSiQue 41.5%，Trivia 90.0%。
2. online DraftModel rate=0.15 在 Trivia 与 full 持平，Hotpot/2Wiki/MuSiQue 略低于 full。
3. uniform alpha=0.1 并没有稳定超过 full：MuSiQue 与 full 持平，Hotpot/Trivia 接近 full，但 2Wiki 低于 online DraftModel。
4. online_qk 是四组里最弱，尤其 2Wiki 和 MuSiQue 降幅明显。

### 对用户问题的直接回答

- offline 组合之前确实没有在 setup-v2 同源 pipeline 上跑；现在 HotpotQA-v2 / TriviaQA-v2 已补齐三组，2Wiki/MuSiQue 因 document/chunk 数和旧 fixed-set 不一致，不能直接补。
- 当前 setup-v2 pipeline 和旧 `cross_dataset_offline_generalization` 的 reflect-format pipeline 不同，因此旧 offline 结果不能直接并入当前表。
- 现在已经有 GLM clean rejudge：在线四组完成，结果如上。offline 的 GLM rejudge 还没跑；建议先确认是否需要对 Hotpot/Trivia offline 也走 GLM，然后再启动过滤方法为 `offline3b_mean,offline3b_freq_boundary2,offline32b_top2`。


## 2026-07-15 00:40 offline GLM 与 setup-v2 fixed-set 兼容性收尾

新增提交：

- `2cabd65`：`rejudge_setup_v2_glm_clean.py` 增加 `SETUP_V2_REJUDGE_DATASETS`，避免 offline GLM 混入 2Wiki smoke。
- 新增 `scripts/audit_setup_v2_fixed_set_compat.py`：审计 setup-v2 文档数与旧 fixed-set chunk 数是否一致。
- 新增 `scripts/derive_setup_v2_fixed_sets_from_scores.py`：从 setup-v2 score cache 派生 chunk-local fixed-set；该脚本不跑模型 forward，只消费已有 score tensor。
- 新增 `scripts/rejudge_setup_v2_glm_clean_offline_hotpot_trivia.py`：offline Hotpot/Trivia GLM clean rejudge 独立输出入口。

### offline GLM clean rejudge

启动命令：

```bash
cd /home/hming/FusionRAG-pca-analysis
nohup env   SETUP_V2_REJUDGE_METHODS=offline3b_mean,offline3b_freq_boundary2,offline32b_top2   SETUP_V2_REJUDGE_DATASETS=hotpotqa-v2,triviaqa-v2   GLM_REJUDGE_WORKERS=10   /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/rejudge_setup_v2_glm_clean_offline_hotpot_trivia.py   > MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/rejudge_glm_clean_offline_hotpot_trivia_20260715/rejudge.log 2>&1 &
```

输出文件：`rejudge_glm_clean_offline_hotpot_trivia_20260715/rejudged_summary.csv`。

```csv
dataset,method,rate,rows,expected_rows,complete,glm_correct,glm_acc
hotpotqa-v2,offline32b_top2,0.15,260,260,True,225,0.8653846153846154
hotpotqa-v2,offline3b_freq_boundary2,0.15,260,260,True,224,0.8615384615384616
hotpotqa-v2,offline3b_mean,0.15,260,260,True,227,0.8730769230769231
triviaqa-v2,offline32b_top2,0.15,270,270,True,236,0.8740740740740741
triviaqa-v2,offline3b_freq_boundary2,0.15,270,270,True,234,0.8666666666666667
triviaqa-v2,offline3b_mean,0.15,270,270,True,237,0.8777777777777778
```

观察：

1. HotpotQA-v2 offline GLM：`offline3b_mean` 最好，227/260 = 87.31%；低于 full 237/260 = 91.15%，也低于 online DraftModel 231/260 = 88.85%。
2. TriviaQA-v2 offline GLM：`offline3b_mean` 最好，237/270 = 87.78%；低于 full / online DraftModel 的 243/270 = 90.00%，接近 online_qk 236/270 = 87.41%。
3. 因此在 setup-v2 direct-QA 口径下，旧 offline fixed-set 仍是可用低成本 baseline，但不是能替代 online DraftModel/full recompute 的方案。

### 在线四组 GLM 修正

由于第一次 offline GLM 误写入了在线目录，已将 offline 输出复制到独立目录，并用在线四组 filter 从 cache 重建在线 summary。当前在线 summary 以此文件为准：`rejudge_glm_clean_20260714/rejudged_summary.csv`。

```csv
dataset,method,rate,rows,expected_rows,complete,glm_correct,glm_acc
2wikimqa-v2,full_rate1,1.0,200,200,True,123,0.615
2wikimqa-v2,online_draft,0.15,200,200,True,120,0.6
2wikimqa-v2,online_qk,0.15,200,200,True,106,0.53
2wikimqa-v2,uniform_alpha0p1_draft,0.15,200,200,True,116,0.58
hotpotqa-v2,full_rate1,1.0,260,260,True,237,0.9115384615384615
hotpotqa-v2,online_draft,0.15,260,260,True,231,0.8884615384615384
hotpotqa-v2,online_qk,0.15,260,260,True,223,0.8576923076923076
hotpotqa-v2,uniform_alpha0p1_draft,0.15,260,260,True,232,0.8923076923076924
musique-v2,full_rate1,1.0,200,200,True,83,0.415
musique-v2,online_draft,0.15,200,200,True,82,0.41
musique-v2,online_qk,0.15,200,200,True,75,0.375
musique-v2,uniform_alpha0p1_draft,0.15,200,200,True,83,0.415
triviaqa-v2,full_rate1,1.0,270,270,True,243,0.9
triviaqa-v2,online_draft,0.15,270,270,True,243,0.9
triviaqa-v2,online_qk,0.15,270,270,True,236,0.8740740740740741
triviaqa-v2,uniform_alpha0p1_draft,0.15,270,270,True,242,0.8962962962962963
```

注意：重建后 `musique-v2/online_draft` 为 82/200 = 41.00%，替代上一版文档里记录的 81/200 = 40.50%。

### fixed-set 兼容性审计

命令：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/audit_setup_v2_fixed_set_compat.py
```

输出文件：`setup_v2_fixed_set_compat.csv`。

```csv
dataset,examples,setup_doc_count_min,setup_doc_count_max,setup_doc_count_hist,old_fixed_available,old_fixed_doc_count_min,old_fixed_doc_count_max,old_fixed_compatible,action
2wikimqa-v2,200,9,24,"{""9"": 1, ""10"": 21, ""11"": 31, ""12"": 30, ""13"": 43, ""14"": 18, ""15"": 11, ""16"": 7, ""17"": 5, ""18"": 5, ""19"": 5, ""20"": 7, ""21"": 8, ""22"": 5, ""23"": 2, ""24"": 1}",True,7,10,False,rebuild setup-v2 fixed-set
hotpotqa-v2,260,10,10,"{""10"": 260}",True,10,10,True,reuse old fixed-set
triviaqa-v2,270,10,10,"{""10"": 270}",True,10,10,True,reuse old fixed-set
musique-v2,200,19,31,"{""19"": 6, ""20"": 13, ""21"": 32, ""22"": 26, ""23"": 19, ""24"": 30, ""25"": 23, ""26"": 19, ""27"": 11, ""28"": 7, ""29"": 10, ""30"": 3, ""31"": 1}",False,,,False,rebuild setup-v2 fixed-set
```

结论：

- `hotpotqa-v2`、`triviaqa-v2`：setup-v2 每例都是 10 doc，旧 fixed-set 也是 10 chunk，可以复用。
- `2wikimqa-v2`：setup-v2 是 9-24 passages/example，而旧 fixed-set 是 7-10 chunk/example，不兼容。
- `musique-v2`：setup-v2 是 19-31 passages/example，旧 cross-dataset offline 没有 musique fixed-set，不兼容。

### 2Wiki/MuSiQue 后续执行入口

不能复用旧 fixed-set。下一步应先生成 setup-v2 score cache，要求每个 `.npz` 至少包含：

- `scores`: shape `[calibration_queries, total_doc_tokens]`
- `starts`: system + each document + query 的 cumulative starts
- `system_len`: system prompt token length

然后用新脚本派生 fixed-set：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/derive_setup_v2_fixed_sets_from_scores.py   --score-cache-dir <setup-v2-score-cache-dir>   --out-dir MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/fixed_sets_<dataset>_3b   --prefix offline3b   --rate 0.15
```

临时 fake score cache 已验证该 derive 脚本能生成 `chunk_fixed_sets_npz/example000_rate0p15_chunk_local_sets.npz` 与 `fixed_set_manifest.csv`。


## 2026-07-15 01:25 setup-v2 2Wiki/MuSiQue offline fixed-set 重建入口

背景：HotpotQA-v2 / TriviaQA-v2 可以复用旧 10-chunk offline fixed-set；但 2WikiMQA-v2 与 MuSiQue-v2 在 setup-v2 direct-QA 格式下的 document 数不固定，旧 fixed-set 不兼容。因此本轮补的是“先生成 setup-v2 同源 score cache，再派生 fixed-set”的入口，而不是直接跑全量 offline。

新增脚本：

- `scripts/build_setup_v2_draft_score_cache.py`
  - 输入：setup-v2 原始 dataset、Qwen3-32B tokenizer、DraftModel。
  - 数据入口：直接调用当前 `prepare_data`，保证 Passage/chunk 边界和 `run_setup_v2_task.py` 一致。
  - control query：来自其他 setup-v2 examples 的原始 question，避免用当前 example query 学自己的 fixed-set。
  - 输出：`scores`, `starts`, `context_lengths`, `system_len`, `queries`, `labels` 等 npz 字段。
- `scripts/derive_setup_v2_fixed_sets_from_scores.py`
  - 已有脚本，本轮用 smoke score cache 验证可派生 chunk-local fixed-set。
- `scripts/run_setup_v2_task.py`
  - 新增 `--offline-fixed-set-dir` 覆盖参数。不传时保持旧 Hotpot/Trivia fixed-set 逻辑；传入时可读取 setup-v2 新 fixed-set。

### Smoke 1: 2WikiMQA-v2 score cache

命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/build_setup_v2_draft_score_cache.py \
  --dataset 2wikimqa-v2 \
  --example-start 0 \
  --example-end 2 \
  --control-count 2 \
  --score-layers last \
  --device cuda:0 \
  --out-dir MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/score_cache_smoke_2wikimqa_3b \
  > MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/logs/score_cache_smoke/2wikimqa_0_2_3b.log 2>&1
```

结果：

- 生成 `score_cache_smoke_2wikimqa_3b/score_cache_npz/setup_v2_2wikimqa-v2_example000_scores.npz`
- 生成 `score_cache_smoke_2wikimqa_3b/score_cache_npz/setup_v2_2wikimqa-v2_example001_scores.npz`
- 每个 example 使用 2 个 other-example control queries。

### Smoke 2: 2WikiMQA-v2 derive fixed-set

命令：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/derive_setup_v2_fixed_sets_from_scores.py \
  --score-cache-dir MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/score_cache_smoke_2wikimqa_3b/score_cache_npz \
  --out-dir MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/fixed_sets_smoke_2wikimqa_3b \
  --prefix offline3b \
  --rate 0.15
```

结果：

- 生成 2 个 `exampleXXX_rate0p15_chunk_local_sets.npz`。
- example000 有 13 chunks，因此 npz 内有 39 个 key（三种方法 × 13 chunks）。
- example001 有 14 chunks，因此 npz 内有 42 个 key（三种方法 × 14 chunks）。
- 派生方法名包括：`offline3b_mean_score_global`, `offline3b_top2_mean_global`, `offline3b_freq_boundary0p02_global`。

### Smoke 3: 2WikiMQA-v2 runner 读取新 fixed-set

命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py \
  --dataset 2wikimqa-v2 \
  --method offline3b_mean \
  --rate 0.15 \
  --start 0 \
  --end 2 \
  --gpu 0 \
  --offline-fixed-set-dir MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/fixed_sets_smoke_2wikimqa_3b/chunk_fixed_sets_npz \
  --result-root MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/smoke_results \
  > MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/logs/fixed_set_smoke/2wikimqa_offline3b_mean_0_2.log 2>&1
```

日志确认：

```text
[setup-v2] offline_fixed_set_dir=MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/fixed_sets_smoke_2wikimqa_3b/chunk_fixed_sets_npz
[setup-v2] offline_fixed_set_method=offline3b_mean_score_global
>>> [runner] ONLY_EXAMPLES active: [1, 2]
```

输出 CSV：

- `smoke_results/offline3b_mean/2wikimqa-v2/rate_0p15/seg_0_2/csv/reprocess_method_FusionRAG_rate_0.15_revert_rope_True_topk_10.csv`

两条 smoke 的答案都未命中，因此该 smoke 只证明接口可运行，不作为 accuracy 结果。

### Smoke 4: MuSiQue-v2 score cache + derive

命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/build_setup_v2_draft_score_cache.py \
  --dataset musique-v2 \
  --example-start 0 \
  --example-end 1 \
  --control-count 2 \
  --score-layers last \
  --device cuda:0 \
  --out-dir MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/score_cache_smoke_musique_3b \
  > MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/logs/score_cache_smoke/musique_0_1_3b.log 2>&1

/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/derive_setup_v2_fixed_sets_from_scores.py \
  --score-cache-dir MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/score_cache_smoke_musique_3b/score_cache_npz \
  --out-dir MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/fixed_sets_smoke_musique_3b \
  --prefix offline3b \
  --rate 0.15
```

结果：

- example000 有 24 chunks，doc token 总长 16598，context_len 17390。
- fixed-set npz 有 72 个 key（三种方法 × 24 chunks）。
- manifest 中存在个别 chunk 的 `selected_count=0`，这是 global top-15% selection 可出现的结果，不是格式错误。如果要避免某个 document 完全不更新，下一步需要设计 per-chunk quota / hybrid global+quota 方法。

### 当前结论

1. 2Wiki/MuSiQue 的 setup-v2 offline 缺口不是 GLM 或 runner 问题，而是 fixed-set 与 setup-v2 chunk 边界不兼容。
2. 新的 score-cache builder 已经能按 setup-v2 原始数据、当前 `prepare_data` chunk 边界生成 DraftModel score tensor。
3. 新 fixed-set 可以被 `run_setup_v2_task.py` 读取并跑通 offline generation。
4. 下一步可以扩大到全量 score cache：先 2Wiki/MuSiQue 的 3B score cache；派生 `offline3b_mean`、`offline3b_freq_boundary2`、`offline3b_top2`；再用 24 卡跑 setup-v2 同源 offline@0.15，最后走 GLM clean rejudge。
5. 不建议立刻跑 32B teacher score cache；先验证 3B fixed-set 全量是否接近 Hotpot/Trivia 的表现上限，再决定是否上 32B。


## 2026-07-15 01:35 setup-v2 3B score-cache 全量启动

Code commit: `e148bbd`。

目标：为 `2wikimqa-v2` 和 `musique-v2` 重建 setup-v2 同源 offline fixed-set 所需的 DraftModel score cache。当前只做 3B DraftModel，不启动 32B teacher。

统一参数：

- `draft_model_path=/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`
- `model_path=/mnt/qjhs-sh-lab-01/models/Qwen3-32B`
- `control_count=8`
- `score_layers=None`，即脚本默认使用 draft model 后半层做 RRF score，和旧 control-query builder 的默认口径一致。
- 输出根目录：`MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/score_cache_full_3b_20260715/`

固定分配：

| host | GPU | dataset | segment | log |
|---|---:|---|---|---|
| qjy000 | 0 | `2wikimqa-v2` | 0-25 | `logs/score_cache_full_3b_20260715/qjy000_2wikimqa-v2_0_25_gpu0.log` |
| qjy000 | 1 | `2wikimqa-v2` | 25-50 | `logs/score_cache_full_3b_20260715/qjy000_2wikimqa-v2_25_50_gpu1.log` |
| qjy000 | 2 | `2wikimqa-v2` | 50-75 | `logs/score_cache_full_3b_20260715/qjy000_2wikimqa-v2_50_75_gpu2.log` |
| qjy000 | 3 | `2wikimqa-v2` | 75-100 | `logs/score_cache_full_3b_20260715/qjy000_2wikimqa-v2_75_100_gpu3.log` |
| qjy000 | 4 | `musique-v2` | 0-25 | `logs/score_cache_full_3b_20260715/qjy000_musique-v2_0_25_gpu4.log` |
| qjy000 | 5 | `musique-v2` | 25-50 | `logs/score_cache_full_3b_20260715/qjy000_musique-v2_25_50_gpu5.log` |
| qjy000 | 6 | `musique-v2` | 50-75 | `logs/score_cache_full_3b_20260715/qjy000_musique-v2_50_75_gpu6.log` |
| qjy000 | 7 | `musique-v2` | 75-100 | `logs/score_cache_full_3b_20260715/qjy000_musique-v2_75_100_gpu7.log` |
| qjy001 | 0 | `2wikimqa-v2` | 100-125 | `logs/score_cache_full_3b_20260715/qjy001_2wikimqa-v2_100_125_gpu0.log` |
| qjy001 | 1 | `2wikimqa-v2` | 125-150 | `logs/score_cache_full_3b_20260715/qjy001_2wikimqa-v2_125_150_gpu1.log` |
| qjy001 | 2 | `2wikimqa-v2` | 150-175 | `logs/score_cache_full_3b_20260715/qjy001_2wikimqa-v2_150_175_gpu2.log` |
| qjy001 | 3 | `2wikimqa-v2` | 175-200 | `logs/score_cache_full_3b_20260715/qjy001_2wikimqa-v2_175_200_gpu3.log` |
| qjy001 | 4 | `musique-v2` | 100-125 | `logs/score_cache_full_3b_20260715/qjy001_musique-v2_100_125_gpu4.log` |
| qjy001 | 5 | `musique-v2` | 125-150 | `logs/score_cache_full_3b_20260715/qjy001_musique-v2_125_150_gpu5.log` |
| qjy001 | 6 | `musique-v2` | 150-175 | `logs/score_cache_full_3b_20260715/qjy001_musique-v2_150_175_gpu6.log` |
| qjy001 | 7 | `musique-v2` | 175-200 | `logs/score_cache_full_3b_20260715/qjy001_musique-v2_175_200_gpu7.log` |

启动后检查：qjy000 8 张卡均有 7-9GB 显存占用并在输出 `example=X scored_query=Y/8`；qjy001 8 个日志也都进入 scoring，未见启动失败。

下一步：

1. 等待 400 个 score npz 生成完成。
2. 用 `derive_setup_v2_fixed_sets_from_scores.py` 分别派生：
   - `fixed_sets_2wikimqa_v2_3b_20260715`
   - `fixed_sets_musique_v2_3b_20260715`
3. 用 `--offline-fixed-set-dir` 跑 setup-v2 offline@0.15。
4. 汇总自动 EM/F1，再走 GLM clean rejudge。


## 2026-07-15 01:45 2WikiMQA-v2 setup-v2 offline@0.15 启动

Code commits:

- `e148bbd`: setup-v2 score-cache builder + `--offline-fixed-set-dir`。
- `60bb27e`: 增加 `offline3b_top2` 方法标签，对应 `offline3b_top2_mean_global`。

前置完成：`2wikimqa-v2` 的 3B score cache 已生成 200/200 个 npz，并派生 fixed-set：

```text
fixed_sets_2wikimqa_v2_3b_20260715/chunk_fixed_sets_npz/*.npz = 200 files
fixed_set_manifest.csv rows = 8298
zero selected chunk rows = 0
methods = offline3b_mean_score_global, offline3b_top2_mean_global, offline3b_freq_boundary0p02_global
```

启动三组 setup-v2 同源 offline@0.15：

| host | GPU | method | segment | fixed-set method |
|---|---:|---|---|---|
| qjy003 | 0 | `offline3b_mean` | 0-50 | `offline3b_mean_score_global` |
| qjy003 | 1 | `offline3b_mean` | 50-100 | `offline3b_mean_score_global` |
| qjy003 | 2 | `offline3b_mean` | 100-150 | `offline3b_mean_score_global` |
| qjy003 | 3 | `offline3b_mean` | 150-200 | `offline3b_mean_score_global` |
| qjy003 | 4 | `offline3b_freq_boundary2` | 0-50 | `offline3b_freq_boundary0p02_global` |
| qjy003 | 5 | `offline3b_freq_boundary2` | 50-100 | `offline3b_freq_boundary0p02_global` |
| qjy003 | 6 | `offline3b_freq_boundary2` | 100-150 | `offline3b_freq_boundary0p02_global` |
| qjy003 | 7 | `offline3b_freq_boundary2` | 150-200 | `offline3b_freq_boundary0p02_global` |
| qjy000 | 0 | `offline3b_top2` | 0-50 | `offline3b_top2_mean_global` |
| qjy000 | 1 | `offline3b_top2` | 50-100 | `offline3b_top2_mean_global` |
| qjy000 | 2 | `offline3b_top2` | 100-150 | `offline3b_top2_mean_global` |
| qjy000 | 3 | `offline3b_top2` | 150-200 | `offline3b_top2_mean_global` |

固定 fixed-set 目录：

```text
MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/fixed_sets_2wikimqa_v2_3b_20260715/chunk_fixed_sets_npz
```

启动后检查：12 个分片都进入 `run_setup_v2_task.py`，日志均打印了对应 `offline_fixed_set_method`，说明没有回退到 online selection。

当前并行状态：

- qjy000 GPU0-3: 2Wiki `offline3b_top2` generation。
- qjy000 GPU4-7: MuSiQue score-cache 0-100。
- qjy001 GPU4-7: MuSiQue score-cache 100-200。
- qjy003 GPU0-7: 2Wiki `offline3b_mean` / `offline3b_freq_boundary2` generation。

下一步：2Wiki 三组完成后运行 `summarize_setup_v2.py` 得到自动 EM/F1；MuSiQue score-cache 达到 200/200 后派生 fixed-set 并启动 MuSiQue offline。


## 2026-07-15 01:55 MuSiQue-v2 setup-v2 offline@0.15 启动

前置完成：`musique-v2` 的 3B score cache 已生成 200/200 个 npz，并派生 fixed-set：

```text
fixed_sets_musique_v2_3b_20260715/chunk_fixed_sets_npz/*.npz = 200 files
fixed_set_manifest.csv rows = 14244
zero selected chunk rows = 0
methods = offline3b_mean_score_global, offline3b_top2_mean_global, offline3b_freq_boundary0p02_global
```

启动三组 setup-v2 同源 offline@0.15：

| host | GPU | method | segment | fixed-set method |
|---|---:|---|---|---|
| qjy001 | 0 | `offline3b_mean` | 0-50 | `offline3b_mean_score_global` |
| qjy001 | 1 | `offline3b_mean` | 50-100 | `offline3b_mean_score_global` |
| qjy001 | 2 | `offline3b_mean` | 100-150 | `offline3b_mean_score_global` |
| qjy001 | 3 | `offline3b_mean` | 150-200 | `offline3b_mean_score_global` |
| qjy001 | 4 | `offline3b_freq_boundary2` | 0-50 | `offline3b_freq_boundary0p02_global` |
| qjy001 | 5 | `offline3b_freq_boundary2` | 50-100 | `offline3b_freq_boundary0p02_global` |
| qjy001 | 6 | `offline3b_freq_boundary2` | 100-150 | `offline3b_freq_boundary0p02_global` |
| qjy001 | 7 | `offline3b_freq_boundary2` | 150-200 | `offline3b_freq_boundary0p02_global` |
| qjy000 | 4 | `offline3b_top2` | 0-50 | `offline3b_top2_mean_global` |
| qjy000 | 5 | `offline3b_top2` | 50-100 | `offline3b_top2_mean_global` |
| qjy000 | 6 | `offline3b_top2` | 100-150 | `offline3b_top2_mean_global` |
| qjy000 | 7 | `offline3b_top2` | 150-200 | `offline3b_top2_mean_global` |

固定 fixed-set 目录：

```text
MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/fixed_sets_musique_v2_3b_20260715/chunk_fixed_sets_npz
```

当前并行状态：

- qjy003: 2Wiki `offline3b_mean` / `offline3b_freq_boundary2` still running。
- qjy000 GPU0-3: 2Wiki `offline3b_top2` still running。
- qjy001 GPU0-7: MuSiQue `offline3b_mean` / `offline3b_freq_boundary2` running。
- qjy000 GPU4-7: MuSiQue `offline3b_top2` running。

下一步：等待 2Wiki/MuSiQue offline 完成后，运行 `summarize_setup_v2.py`，再对新增 6 组启动 GLM clean rejudge。


## 2026-07-15 02:25 setup-v2 rebuilt offline 结果汇总

本节是 2WikiMQA-v2 / MuSiQue-v2 重新按 setup-v2 chunk 边界生成 3B DraftModel score cache 后的 offline fixed-set 结果。它不是旧 reflect-format offline fixed-set 的原样复现。

相关 commits：

- `e148bbd`: 新增 setup-v2 score-cache builder 与 `--offline-fixed-set-dir`。
- `60bb27e`: 新增 `offline3b_top2` 方法标签。
- `1ae8027`: GLM rejudge 支持 `SETUP_V2_REJUDGE_OUT_DIR`，避免覆盖旧在线四组结果。
- `694892e`: 记录 2Wiki rebuilt offline 启动。
- `ff4a004`: 记录 MuSiQue rebuilt offline 启动。

### 自动 EM/F1

文件：`setup_v2_summary.csv`。

| dataset | method | rows | EM | F1 | 对比 |
|---|---|---:|---:|---:|---|
| `2wikimqa-v2` | `full_rate1` | 200 | 0.4800 | 0.5899 | full baseline |
| `2wikimqa-v2` | `online_draft` | 200 | 0.4300 | 0.5769 | online DraftModel@0.15 |
| `2wikimqa-v2` | `online_qk` | 200 | 0.4200 | 0.5313 | online QK@0.15 |
| `2wikimqa-v2` | `offline3b_mean` | 200 | 0.4500 | 0.5684 | rebuilt fixed-set；EM 高于 online Draft，F1 略低 |
| `2wikimqa-v2` | `offline3b_freq_boundary2` | 200 | 0.4350 | 0.5687 | rebuilt fixed-set |
| `2wikimqa-v2` | `offline3b_top2` | 200 | 0.4350 | 0.5592 | rebuilt fixed-set |
| `musique-v2` | `full_rate1` | 200 | 0.2550 | 0.4003 | full baseline |
| `musique-v2` | `online_draft` | 200 | 0.2400 | 0.3788 | online DraftModel@0.15 |
| `musique-v2` | `online_qk` | 200 | 0.2150 | 0.3557 | online QK@0.15 |
| `musique-v2` | `offline3b_mean` | 200 | 0.2450 | 0.3879 | rebuilt fixed-set；介于 online Draft 和 full 之间 |
| `musique-v2` | `offline3b_freq_boundary2` | 200 | 0.2300 | 0.3675 | rebuilt fixed-set；低于 online Draft |
| `musique-v2` | `offline3b_top2` | 200 | 0.2450 | 0.3847 | rebuilt fixed-set；介于 online Draft 和 full 之间 |

### GLM clean rejudge

2Wiki 输出目录：`rejudge_glm_clean_offline_2wiki_rebuilt_20260715/`。

```csv
dataset,method,rate,rows,expected_rows,complete,glm_correct,glm_acc
2wikimqa-v2,offline3b_freq_boundary2,0.15,200,200,True,114,0.57
2wikimqa-v2,offline3b_mean,0.15,200,200,True,114,0.57
2wikimqa-v2,offline3b_top2,0.15,200,200,True,113,0.565
```

MuSiQue 输出目录：`rejudge_glm_clean_offline_musique_rebuilt_20260715/`。

```csv
dataset,method,rate,rows,expected_rows,complete,glm_correct,glm_acc
musique-v2,offline3b_freq_boundary2,0.15,200,200,True,77,0.385
musique-v2,offline3b_mean,0.15,200,200,True,81,0.405
musique-v2,offline3b_top2,0.15,200,200,True,81,0.405
```

和在线四组 GLM 对比：

| dataset | full | online Draft | online QK | uniform a=0.1 | best rebuilt offline |
|---|---:|---:|---:|---:|---:|
| `2wikimqa-v2` | 123/200 = 61.5% | 120/200 = 60.0% | 106/200 = 53.0% | 116/200 = 58.0% | 114/200 = 57.0% |
| `musique-v2` | 83/200 = 41.5% | 82/200 = 41.0% | 75/200 = 37.5% | 83/200 = 41.5% | 81/200 = 40.5% |

### 结论

1. 重新按 setup-v2 chunk 边界构建 offline fixed-set 后，2Wiki/MuSiQue 的 offline 方法能跑通且不再有 chunk 不兼容问题。
2. 但在 GLM 口径下，3B rebuilt offline 仍不能超过 online DraftModel：
   - 2Wiki best rebuilt offline = 57.0%，低于 online Draft 60.0%，低于 full 61.5%。
   - MuSiQue best rebuilt offline = 40.5%，低于 online Draft 41.0%，低于 full 41.5%。
3. 自动 EM/F1 和 GLM 的趋势不完全一致：2Wiki 的 `offline3b_mean` 自动 EM 高于 online Draft，但 GLM 低于 online Draft，说明 exact/ROUGE 对 2Wiki 的判断不够可靠，应以 GLM clean 为主。
4. `offline3b_freq_boundary2` 在 rebuilt setup-v2 上没有带来稳定收益；2Wiki 与 mean 持平或略低，MuSiQue 明显低于 mean/top2。
5. 当前最合理的下一步不是继续扩同类 3B offline selector，而是：
   - 若坚持 offline 路线，补 `offline32b_top2` 的 setup-v2 score cache，看更强 teacher 是否能弥合 2Wiki/MuSiQue gap。
   - 或者转向 hybrid：offline fixed-set 作为低成本初筛，再用少量 online query-conditioned token 补充，因为纯 offline support 在 GLM 上没有达到 online DraftModel。

### Reproduce commands

2Wiki GLM：

```bash
SETUP_V2_REJUDGE_OUT_DIR=MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/rejudge_glm_clean_offline_2wiki_rebuilt_20260715 \
SETUP_V2_REJUDGE_METHODS=offline3b_mean,offline3b_freq_boundary2,offline3b_top2 \
SETUP_V2_REJUDGE_DATASETS=2wikimqa-v2 \
GLM_REJUDGE_WORKERS=10 \
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/rejudge_setup_v2_glm_clean.py
```

MuSiQue GLM：

```bash
SETUP_V2_REJUDGE_OUT_DIR=MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/rejudge_glm_clean_offline_musique_rebuilt_20260715 \
SETUP_V2_REJUDGE_METHODS=offline3b_mean,offline3b_freq_boundary2,offline3b_top2 \
SETUP_V2_REJUDGE_DATASETS=musique-v2 \
GLM_REJUDGE_WORKERS=10 \
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/rejudge_setup_v2_glm_clean.py
```


## 2026-07-15 02:40 自动 GLM 汇总入口

用户要求：以后不要把 EM/F1 汇总和 GLM clean rejudge 拆成两个手动步骤；每次汇总 EM/F1 时，顺带跑 GLM 测评。

实现：

- `scripts/summarize_setup_v2.py` 现在默认流程为：
  1. 扫描 `results/*/*/rate_*/*/csv/reprocess_method_*.csv`。
  2. 按 `(dataset, method, rate, question)` 去重。
  3. 写 `setup_v2_summary.csv/json`，包含 EM/F1。
  4. 自动调用 `scripts/rejudge_setup_v2_glm_clean.py`。
  5. 只对 `setup_v2_summary.csv` 中 `complete=True` 的 group 做 GLM rejudge，避免半截 segment 混入。
  6. 写 `setup_v2_summary_with_glm.csv/json`，把 EM/F1 和 GLM correct/acc 拼在同一张表。
- `scripts/rejudge_setup_v2_glm_clean.py` 新增：
  - `SETUP_V2_REJUDGE_COMPLETE_ONLY=1`：只判完整 group。
  - `SETUP_V2_REJUDGE_OUT_DIR=<dir>`：指定输出目录，避免覆盖历史 online/offline rejudge 目录。

默认命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/summarize_setup_v2.py
```

默认输出：

```text
setup_v2_summary.csv
setup_v2_summary.json
setup_v2_summary_with_glm.csv
setup_v2_summary_with_glm.json
rejudge_glm_clean_auto/rejudged_rows.csv
rejudge_glm_clean_auto/rejudged_summary.csv
rejudge_glm_clean_auto/rejudged_summary.json
```

调试时如果只想快速刷新 EM/F1，不跑 GLM：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/summarize_setup_v2.py \
  --skip-glm

# 或者
SETUP_V2_SUMMARY_SKIP_GLM=1 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/summarize_setup_v2.py
```

验证：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/setup_standard_v2_cross_dataset/scripts/summarize_setup_v2.py \
  --skip-glm
```

该命令已通过，确认 EM/F1 summary 正常生成。没有在验证阶段触发全量 GLM，避免重复判几千条；后续正式汇总直接不带 `--skip-glm` 即可自动跑 GLM。


## 2026-07-15 02:50 当前完整性能表

本表为 setup-v2 direct-QA 口径，GLM clean 为主，EM/F1 为辅助。来源：

- `setup_v2_summary.csv`
- `rejudge_glm_clean_20260714/rejudged_summary.csv`
- `rejudge_glm_clean_offline_hotpot_trivia_20260715/rejudged_summary.csv`
- `rejudge_glm_clean_offline_2wiki_rebuilt_20260715/rejudged_summary.csv`
- `rejudge_glm_clean_offline_musique_rebuilt_20260715/rejudged_summary.csv`

### 2WikiMQA-v2

| Method | Rate | EM | F1 | GLM Acc |
|---|---:|---:|---:|---:|
| `full_rate1` | 1.0 | 48.00 | 58.99 | 123/200 = 61.50 |
| `online_draft` | 0.15 | 43.00 | 57.69 | 120/200 = 60.00 |
| `uniform_alpha0p1_draft` | 0.15 | 43.00 | 56.32 | 116/200 = 58.00 |
| `offline3b_mean` | 0.15 | 45.00 | 56.84 | 114/200 = 57.00 |
| `offline3b_freq_boundary2` | 0.15 | 43.50 | 56.87 | 114/200 = 57.00 |
| `offline3b_top2` | 0.15 | 43.50 | 55.92 | 113/200 = 56.50 |
| `online_qk` | 0.15 | 42.00 | 53.13 | 106/200 = 53.00 |
| `preprocess_rate0` | 0.0 | 35.00 | 47.82 | 91/200 = 45.50 |
| `raw_rate0` | 0.0 | 32.50 | 42.96 | 84/200 = 42.00 |

### HotpotQA-v2

| Method | Rate | EM | F1 | GLM Acc |
|---|---:|---:|---:|---:|
| `full_rate1` | 1.0 | 70.38 | 80.78 | 237/260 = 91.15 |
| `uniform_alpha0p1_draft` | 0.15 | 70.38 | 80.94 | 232/260 = 89.23 |
| `online_draft` | 0.15 | 70.77 | 81.12 | 231/260 = 88.85 |
| `offline3b_mean` | 0.15 | 69.23 | 78.95 | 227/260 = 87.31 |
| `offline32b_top2` | 0.15 | 68.08 | 78.42 | 225/260 = 86.54 |
| `offline3b_freq_boundary2` | 0.15 | 68.46 | 78.33 | 224/260 = 86.15 |
| `online_qk` | 0.15 | 66.54 | 77.09 | 223/260 = 85.77 |
| `preprocess_rate0` | 0.0 | 68.08 | 77.33 | 222/260 = 85.38 |
| `raw_rate0` | 0.0 | 62.69 | 74.27 | 205/260 = 78.85 |

### MuSiQue-v2

| Method | Rate | EM | F1 | GLM Acc |
|---|---:|---:|---:|---:|
| `full_rate1` | 1.0 | 25.50 | 40.03 | 83/200 = 41.50 |
| `uniform_alpha0p1_draft` | 0.15 | 25.00 | 39.28 | 83/200 = 41.50 |
| `online_draft` | 0.15 | 24.00 | 37.88 | 82/200 = 41.00 |
| `offline3b_mean` | 0.15 | 24.50 | 38.79 | 81/200 = 40.50 |
| `offline3b_top2` | 0.15 | 24.50 | 38.47 | 81/200 = 40.50 |
| `offline3b_freq_boundary2` | 0.15 | 23.00 | 36.75 | 77/200 = 38.50 |
| `online_qk` | 0.15 | 21.50 | 35.57 | 75/200 = 37.50 |
| `preprocess_rate0` | 0.0 | 20.50 | 34.01 | 65/200 = 32.50 |
| `raw_rate0` | 0.0 | 15.00 | 27.77 | 59/200 = 29.50 |

### TriviaQA-v2

| Method | Rate | EM | F1 | GLM Acc |
|---|---:|---:|---:|---:|
| `full_rate1` | 1.0 | 66.30 | 76.93 | 243/270 = 90.00 |
| `online_draft` | 0.15 | 66.67 | 77.12 | 243/270 = 90.00 |
| `uniform_alpha0p1_draft` | 0.15 | 65.93 | 76.46 | 242/270 = 89.63 |
| `offline3b_mean` | 0.15 | 64.81 | 75.69 | 237/270 = 87.78 |
| `preprocess_rate0` | 0.0 | 64.44 | 74.59 | 237/270 = 87.78 |
| `online_qk` | 0.15 | 64.81 | 75.14 | 236/270 = 87.41 |
| `offline32b_top2` | 0.15 | 64.81 | 75.58 | 236/270 = 87.41 |
| `offline3b_freq_boundary2` | 0.15 | 64.07 | 75.15 | 234/270 = 86.67 |
| `raw_rate0` | 0.0 | 57.41 | 67.65 | 218/270 = 80.74 |

### 结论

1. `full_rate1` 仍是最稳 baseline。
2. `online_draft@0.15` 是当前低重算方案里最强或接近最强：2Wiki/Hotpot/MuSiQue/Trivia 都强于或接近 offline。
3. rebuilt offline 3B fixed-set 在 2Wiki/MuSiQue 能跑通，但 GLM 上没有超过 online DraftModel。
4. `uniform_alpha0p1_draft` 在 MuSiQue 与 full 持平，在 Hotpot/Trivia 接近 full，但 2Wiki 明显低于 full/online Draft。
5. `online_qk@0.15` 整体最弱，尤其在 2Wiki 和 MuSiQue 上落后明显。
6. 下一步如果继续 offline 路线，应优先补 setup-v2 的 `offline32b_top2` score cache；如果目标是性能，建议转向 hybrid：offline fixed-set 初筛 + 少量 online query-conditioned 补充。


## 2026-07-15 03:05 rate=0 raw/preprocess 空白对照启动

新增 method：

- `raw_rate0`: `reprocess_method=FusionRAG`, `rate=0.0`, `preprocess=False`。含义：加载 raw KV，不重算任何 doc token。
- `preprocess_rate0`: `reprocess_method=FusionRAG`, `rate=0.0`, `preprocess=True`。含义：加载 preprocess KV，不重算任何 doc token。

Code commit: `af59146`。

第一批每个 dataset/method 一个 full-range worker：

| host | GPU | dataset | method | segment |
|---|---:|---|---|---|
| qjy000 | 0 | `2wikimqa-v2` | `raw_rate0` | 0-200 |
| qjy000 | 1 | `2wikimqa-v2` | `preprocess_rate0` | 0-200 |
| qjy000 | 2 | `hotpotqa-v2` | `raw_rate0` | 0-260 |
| qjy000 | 3 | `hotpotqa-v2` | `preprocess_rate0` | 0-260 |
| qjy001 | 0 | `musique-v2` | `raw_rate0` | 0-200 |
| qjy001 | 1 | `musique-v2` | `preprocess_rate0` | 0-200 |
| qjy001 | 2 | `triviaqa-v2` | `raw_rate0` | 0-270 |
| qjy001 | 3 | `triviaqa-v2` | `preprocess_rate0` | 0-270 |

加速分片：

| host | GPU | dataset | method | segment |
|---|---:|---|---|---|
| qjy000 | 4 | `2wikimqa-v2` | `raw_rate0` | 50-125 |
| qjy000 | 5 | `2wikimqa-v2` | `preprocess_rate0` | 50-125 |
| qjy000 | 6 | `2wikimqa-v2` | `raw_rate0` | 125-200 |
| qjy000 | 7 | `2wikimqa-v2` | `preprocess_rate0` | 125-200 |
| qjy001 | 4 | `triviaqa-v2` | `raw_rate0` | 70-170 |
| qjy001 | 5 | `triviaqa-v2` | `preprocess_rate0` | 70-170 |
| qjy001 | 6 | `triviaqa-v2` | `raw_rate0` | 170-270 |
| qjy001 | 7 | `triviaqa-v2` | `preprocess_rate0` | 170-270 |
| qjy003 | 0 | `hotpotqa-v2` | `raw_rate0` | 70-165 |
| qjy003 | 1 | `hotpotqa-v2` | `preprocess_rate0` | 70-165 |
| qjy003 | 2 | `hotpotqa-v2` | `raw_rate0` | 165-260 |
| qjy003 | 3 | `hotpotqa-v2` | `preprocess_rate0` | 165-260 |
| qjy003 | 4 | `musique-v2` | `raw_rate0` | 50-125 |
| qjy003 | 5 | `musique-v2` | `preprocess_rate0` | 50-125 |
| qjy003 | 6 | `musique-v2` | `raw_rate0` | 125-200 |
| qjy003 | 7 | `musique-v2` | `preprocess_rate0` | 125-200 |

说明：加速分片和 full-range worker 有重叠；`summarize_setup_v2.py` 按 `(dataset, method, rate, question)` 去重，因此重叠不会重复计入最终性能表。


## 2026-07-15 03:25 rate=0 raw/preprocess 空白对照结果

本节补齐每个数据集两组 rate=0 空白对照：

- `raw_rate0`: raw KV，不重算任何 doc token。
- `preprocess_rate0`: preprocess KV，不重算任何 doc token。

自动 EM/F1 来源：`setup_v2_summary.csv`。
GLM 来源：`rejudge_glm_clean_rate0_controls_20260715/rejudged_summary.csv`。

| dataset | method | rows | EM | F1 | GLM Acc |
|---|---|---:|---:|---:|---:|
| `2wikimqa-v2` | `raw_rate0` | 200 | 32.50 | 42.96 | 84/200 = 42.00 |
| `2wikimqa-v2` | `preprocess_rate0` | 200 | 35.00 | 47.82 | 91/200 = 45.50 |
| `hotpotqa-v2` | `raw_rate0` | 260 | 62.69 | 74.27 | 205/260 = 78.85 |
| `hotpotqa-v2` | `preprocess_rate0` | 260 | 68.08 | 77.33 | 222/260 = 85.38 |
| `musique-v2` | `raw_rate0` | 200 | 15.00 | 27.77 | 59/200 = 29.50 |
| `musique-v2` | `preprocess_rate0` | 200 | 20.50 | 34.01 | 65/200 = 32.50 |
| `triviaqa-v2` | `raw_rate0` | 270 | 57.41 | 67.65 | 218/270 = 80.74 |
| `triviaqa-v2` | `preprocess_rate0` | 270 | 64.44 | 74.59 | 237/270 = 87.78 |

### 放入当前性能表后的关键对比

| dataset | full | online Draft@0.15 | raw rate=0 | preprocess rate=0 | best offline fixed-set |
|---|---:|---:|---:|---:|---:|
| `2wikimqa-v2` | 61.50 | 60.00 | 42.00 | 45.50 | 57.00 |
| `hotpotqa-v2` | 91.15 | 88.85 | 78.85 | 85.38 | 87.31 |
| `musique-v2` | 41.50 | 41.00 | 29.50 | 32.50 | 40.50 |
| `triviaqa-v2` | 90.00 | 90.00 | 80.74 | 87.78 | 87.78 |

### 结论

1. `preprocess_rate0` 在四个数据集上都显著优于 `raw_rate0`，说明老的 preprocess KV 确实不是无效缓存；它已经离 full attention 更近。
2. rate=0 仍明显低于 online Draft/full，尤其 2Wiki 和 MuSiQue：不做任何重算会丢掉大量跨文档/上下文更新。
3. Hotpot/Trivia 上 `preprocess_rate0` 已经接近 offline fixed-set，Trivia 上 GLM 与 `offline3b_mean` 持平，说明这些较短/较稳定任务里 preprocess KV 已经覆盖了相当一部分收益。
4. MuSiQue 上 rate=0 差距最大：`preprocess_rate0` 只有 32.5% GLM，而 online Draft 是 41.0%，说明多跳长上下文更依赖 online update。
5. 后续表格比较必须把 `raw_rate0` 和 `preprocess_rate0` 作为空白下界，否则无法判断 offline fixed-set 或 DraftModel 的真实增益。

## 2026-07-15 MuSiQue-v2 rate=0.05 补充实验启动

目的：补充 MuSiQue-v2 表中 rate=0.05 的 online/offline/uniform 方法结果。

Code commit: 83a6c94.

Fixed-set 派生命令：
```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/derive_setup_v2_fixed_sets_from_scores.py --score-cache-dir MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/score_cache_full_3b_20260715/musique-v2/score_cache_npz --out-dir MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/fixed_sets_musique_v2_3b_rate0p05_20260715 --prefix offline3b --rate 0.05 --boundary-rate 0.02
```

Fixed-set 输出：`MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/fixed_sets_musique_v2_3b_rate0p05_20260715`，npz 数量：200。

启动命令：
```bash
FUSIONRAG_PREPROCESS_CACHE_READONLY=1 CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py --dataset musique-v2 --method online_qk --rate 0.05 --start 0 --end 200 --gpu 0  > MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/logs/musique_rate0p05_20260715/online_qk_rate0p05.log 2>&1 &
FUSIONRAG_PREPROCESS_CACHE_READONLY=1 CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py --dataset musique-v2 --method online_draft --rate 0.05 --start 0 --end 200 --gpu 1  > MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/logs/musique_rate0p05_20260715/online_draft_rate0p05.log 2>&1 &
FUSIONRAG_PREPROCESS_CACHE_READONLY=1 CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py --dataset musique-v2 --method uniform_alpha0p1_draft --rate 0.05 --start 0 --end 200 --gpu 2  > MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/logs/musique_rate0p05_20260715/uniform_alpha0p1_draft_rate0p05.log 2>&1 &
FUSIONRAG_PREPROCESS_CACHE_READONLY=1 CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py --dataset musique-v2 --method offline3b_mean --rate 0.05 --start 0 --end 200 --gpu 3 --offline-fixed-set-dir MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/fixed_sets_musique_v2_3b_rate0p05_20260715/chunk_fixed_sets_npz > MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/logs/musique_rate0p05_20260715/offline3b_mean_rate0p05.log 2>&1 &
FUSIONRAG_PREPROCESS_CACHE_READONLY=1 CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py --dataset musique-v2 --method offline3b_top2 --rate 0.05 --start 0 --end 200 --gpu 4 --offline-fixed-set-dir MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/fixed_sets_musique_v2_3b_rate0p05_20260715/chunk_fixed_sets_npz > MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/logs/musique_rate0p05_20260715/offline3b_top2_rate0p05.log 2>&1 &
FUSIONRAG_PREPROCESS_CACHE_READONLY=1 CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py --dataset musique-v2 --method offline3b_freq_boundary2 --rate 0.05 --start 0 --end 200 --gpu 5 --offline-fixed-set-dir MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/fixed_sets_musique_v2_3b_rate0p05_20260715/chunk_fixed_sets_npz > MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/logs/musique_rate0p05_20260715/offline3b_freq_boundary2_rate0p05.log 2>&1 &
```

加速分片启动：
```text
qjy001: methods online_qk/online_draft/uniform_alpha0p1_draft/offline3b_mean/offline3b_top2/offline3b_freq_boundary2, segment 50-125, GPUs 0-5
qjy003: same methods, segment 125-200, GPUs 0-5
```

加速分片实际启动：qjy001 segment 50-125，qjy003 segment 125-200，每台 GPUs 0-5。

加速分片重新启动：qjy001/qjy003 使用 /home/hming/FusionRAG-pca-analysis，qjy001 segment 50-125，qjy003 segment 125-200，每台 GPUs 0-5。

## 2026-07-15 MuSiQue-v2 rate=0.05 结果与 GLM 同步汇总接口

本次补充 MuSiQue-v2 中 `rate=0.05` 的结果。`full_rate1` 是 `rate=1.0` 的 full recompute baseline，不单独设置 `rate=0.05`。

### 为什么之前 GLM judge 是分开跑的

EM/F1 是本地 deterministic 指标，读 CSV 后可以立即计算；GLM judge 是外部 LLM 调用，需要 endpoint、重试、cache、并发控制和去重。此前为了避免每次 summarize 都触发全量重判，GLM judge 被单独脚本管理。

现在已经把两者接到同一个 summarize pipeline：`summarize_setup_v2.py` 会先计算 EM/F1，再按指定 dataset/method/rate 调用 `rejudge_setup_v2_glm_clean.py`，最后生成 `setup_v2_summary_with_glm.csv/json`。对于新增实验，推荐直接使用这一条命令，而不是手动分两步汇总。

同步汇总命令：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/summarize_setup_v2.py \
  --glm-output-dir MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/rejudge_glm_clean_musique_rate0p05_20260715 \
  --glm-datasets musique-v2 \
  --glm-methods online_qk,online_draft,uniform_alpha0p1_draft,offline3b_mean,offline3b_top2,offline3b_freq_boundary2 \
  --glm-rates 0.05 \
  --glm-workers 12
```

对应代码 commit：`83a6c94` 开始修正 offline fixed-set rate，当前工作区在该 commit 之后又补充了 summarize 的 GLM filter 参数。

结果来源：

- EM/F1：`setup_v2_summary.csv`
- GLM：`rejudge_glm_clean_musique_rate0p05_20260715/rejudged_summary.csv`
- 合并表：`setup_v2_summary_with_glm.csv`

### MuSiQue-v2 当前表格补齐

| Method | Rate | EM | F1 | GLM Acc |
|---|---:|---:|---:|---:|
| `full_rate1` | 1.0 | 25.50 | 40.03 | 83/200 = 41.50 |
| `uniform_alpha0p1_draft` | 0.15 | 25.00 | 39.28 | 83/200 = 41.50 |
| `online_draft` | 0.15 | 24.00 | 37.88 | 82/200 = 41.00 |
| `offline3b_mean` | 0.15 | 24.50 | 38.79 | 81/200 = 40.50 |
| `offline3b_top2` | 0.15 | 24.50 | 38.47 | 81/200 = 40.50 |
| `offline3b_freq_boundary2` | 0.15 | 23.00 | 36.75 | 77/200 = 38.50 |
| `online_qk` | 0.15 | 21.50 | 35.57 | 75/200 = 37.50 |
| `uniform_alpha0p1_draft` | 0.05 | 22.00 | 35.94 | 76/200 = 38.00 |
| `offline3b_mean` | 0.05 | 21.50 | 34.96 | 74/200 = 37.00 |
| `online_draft` | 0.05 | 20.00 | 33.80 | 72/200 = 36.00 |
| `offline3b_top2` | 0.05 | 20.50 | 34.31 | 70/200 = 35.00 |
| `offline3b_freq_boundary2` | 0.05 | 19.50 | 33.28 | 69/200 = 34.50 |
| `online_qk` | 0.05 | 20.50 | 33.90 | 67/200 = 33.50 |
| `preprocess_rate0` | 0.0 | 20.50 | 34.01 | 65/200 = 32.50 |
| `raw_rate0` | 0.0 | 15.00 | 27.77 | 59/200 = 29.50 |

### 观察

1. `rate=0.05` 明显低于 `rate=0.15`，说明 MuSiQue-v2 对少量 token 重算比较敏感；5% 的更新量还不能稳定接近 full attention。
2. `uniform_alpha0p1_draft@0.05` 是本组最强，GLM 38.00%，高于普通 `online_draft@0.05` 的 36.00%。这说明轻微 attention 平滑在低 rate 下仍有收益，但不能替代足够的重算 token 数量。
3. `offline3b_mean@0.05` 达到 GLM 37.00%，比 `preprocess_rate0` 的 32.50% 有明显提升，说明 offline fixed-set 即使只更新 5% token 也能补一部分 gap。
4. `online_qk@0.05` 只有 GLM 33.50%，接近 `preprocess_rate0`，说明在 MuSiQue-v2 低 rate 下，QK selection 找到的少量 token 不如 Draft/uniform/offline mean 稳定。
5. 这张表支持后续重点看 `rate=0.05 -> 0.15` 的增益曲线，而不是只比较单点；真正可用的轻量方法至少需要在低 rate 下显著超过 `preprocess_rate0`，并向 `full_rate1` 靠近。

## 2026-07-15 Cross-dataset rate=0 raw/preprocess 对照补齐

本节把四个 setup-v2 数据集的 `raw_rate0` 和 `preprocess_rate0` 统一补进 baseline 表。两者均为 `rate=0.0`，即不做 online doc token 重算：

- `raw_rate0`: 加载 raw KV cache。
- `preprocess_rate0`: 加载 preprocess KV cache。

结果来源：

- EM/F1：`setup_v2_summary.csv`
- GLM：`rejudge_glm_clean_rate0_controls_20260715/rejudged_summary.csv`

| Dataset | Method | Rows | EM | F1 | GLM Acc |
|---|---|---:|---:|---:|---:|
| `2wikimqa-v2` | `raw_rate0` | 200 | 32.50 | 42.96 | 84/200 = 42.00 |
| `2wikimqa-v2` | `preprocess_rate0` | 200 | 35.00 | 47.82 | 91/200 = 45.50 |
| `hotpotqa-v2` | `raw_rate0` | 260 | 62.69 | 74.27 | 205/260 = 78.85 |
| `hotpotqa-v2` | `preprocess_rate0` | 260 | 68.08 | 77.33 | 222/260 = 85.38 |
| `musique-v2` | `raw_rate0` | 200 | 15.00 | 27.77 | 59/200 = 29.50 |
| `musique-v2` | `preprocess_rate0` | 200 | 20.50 | 34.01 | 65/200 = 32.50 |
| `triviaqa-v2` | `raw_rate0` | 270 | 57.41 | 67.65 | 218/270 = 80.74 |
| `triviaqa-v2` | `preprocess_rate0` | 270 | 64.44 | 74.59 | 237/270 = 87.78 |

### 直接观察

1. `preprocess_rate0` 在四个数据集上都优于 `raw_rate0`，说明 preprocess KV 本身已经提供了稳定收益，不是无效缓存。
2. 提升幅度按 GLM Acc 计算：2Wiki +3.50，Hotpot +6.54，MuSiQue +3.00，Trivia +7.04。
3. HotpotQA/TriviaQA 的 `preprocess_rate0` 已经接近一些 low-rate 方法，说明这些数据集对 online recompute 的依赖较弱；MuSiQue 和 2Wiki 仍需要显著 online update 才能靠近 full attention。
4. 后续所有 online/offline fixed-set 方法比较都应该同时报告这两条 rate=0 下界，否则无法判断方法相对 cache-only baseline 的真实增益。

## 2026-07-15 Cross-dataset rate=0.05 online/offline 补跑

目的：补齐 setup-v2 四个数据集在 `rate=0.05` 下的 online 与 offline baseline。MuSiQue-v2 已完成，本轮新增/补跑 `2wikimqa-v2`、`hotpotqa-v2`、`triviaqa-v2`。

覆盖方法：

- `online_draft`
- `online_qk`
- `offline3b_mean`
- `offline3b_top2`
- `offline3b_freq_boundary2`
- `offline32b_top2`：历史方法名保留为 `offline32b_top2`，当前 runner 实际使用 fixed-set key `offline32b_mean_score_global`。

代码 commit：`38e0b0c`。该 commit 修复了共享 cache 并发启动时 `prepare_data` 目录创建的 race：`os.makedirs(..., exist_ok=True)`。

### Fixed-set 派生

本轮不复用 `rate=0.15` 的 fixed-set mask，而是从 score cache 严格派生 `rate=0.05` fixed-set。

```bash
# 2Wiki setup-v2 3B
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/derive_setup_v2_fixed_sets_from_scores.py \
  --score-cache-dir MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/score_cache_full_3b_20260715/2wikimqa-v2/score_cache_npz \
  --out-dir MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/fixed_sets_2wikimqa_v2_3b_rate0p05_20260715 \
  --prefix offline3b --rate 0.05 --boundary-rate 0.02

# 2Wiki/Hotpot/Trivia 32B，以及 Hotpot/Trivia 3B 使用 cross_dataset_offline_generalization 的 score cache 派生。
```

Fixed-set 输出目录：

- `fixed_sets_2wikimqa_v2_3b_rate0p05_20260715/`
- `fixed_sets_2wikimqa_v2_32b_rate0p05_20260715/`
- `fixed_sets_hotpotqa-v2_3b_rate0p05_20260715/`
- `fixed_sets_hotpotqa-v2_32b_rate0p05_20260715/`
- `fixed_sets_triviaqa-v2_3b_rate0p05_20260715/`
- `fixed_sets_triviaqa-v2_32b_rate0p05_20260715/`

### 启动分配

所有任务使用共享 preprocess cache：`/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2`，并设置 `FUSIONRAG_PREPROCESS_CACHE_READONLY=1`。

| host | dataset | methods | GPUs | segment |
|---|---|---|---|---|
| qjy000 | `2wikimqa-v2` | 6 methods above | 0-5 | 0-200 |
| qjy001 | `hotpotqa-v2` | 6 methods above | 0-5 | 0-260 |
| qjy003 | `triviaqa-v2` | 6 methods above | 0-5 | 0-270 |

第一次同时启动时，2Wiki 和 Hotpot 部分 worker 触发共享 cache 目录创建 race。修复并提交 `38e0b0c` 后，已重启失败 worker：

- qjy000：2Wiki 六组全部重启，日志在 `logs/rate0p05_all_baselines_20260715/relaunch/qjy000_2wiki_*.log`
- qjy001：Hotpot 除仍在运行的 `offline3b_top2` 外，其余五组重启，日志在 `logs/rate0p05_all_baselines_20260715/relaunch/qjy001_hotpot_*.log`
- qjy003：Trivia 六组首次启动后持续运行，日志在 `logs/rate0p05_all_baselines_20260715/qjy003_trivia_*.log`

结果目录统一为：`results/<method>/<dataset>/rate_0p05/seg_*`。

后续完成后使用一条命令同步汇总 EM/F1 和 GLM：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/summarize_setup_v2.py \
  --glm-output-dir MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/rejudge_glm_clean_rate0p05_all_baselines_20260715 \
  --glm-datasets 2wikimqa-v2,hotpotqa-v2,musique-v2,triviaqa-v2 \
  --glm-methods online_qk,online_draft,offline3b_mean,offline3b_top2,offline3b_freq_boundary2,offline32b_top2 \
  --glm-rates 0.05 \
  --glm-workers 16
```

## 2026-07-15 Cross-dataset rate=0.05 online/offline 结果

本节补齐 `rate=0.05` 下 online 与 offline 方法在四个 setup-v2 数据集上的结果。MuSiQue-v2 使用此前已完成的 GLM 结果；2Wiki/Hotpot/Trivia 使用本轮 `rejudge_glm_clean_rate0p05_new_datasets_20260715` 的 GLM 结果。

汇总命令：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/summarize_setup_v2.py \
  --glm-output-dir MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/rejudge_glm_clean_rate0p05_new_datasets_20260715 \
  --glm-datasets 2wikimqa-v2,hotpotqa-v2,triviaqa-v2 \
  --glm-methods online_qk,online_draft,offline3b_mean,offline3b_top2,offline3b_freq_boundary2,offline32b_top2 \
  --glm-rates 0.05 \
  --glm-workers 16
```

结果来源：

- EM/F1：`setup_v2_summary.csv`
- 新 GLM：`rejudge_glm_clean_rate0p05_new_datasets_20260715/rejudged_summary.csv`
- MuSiQue-v2 GLM：`rejudge_glm_clean_musique_rate0p05_20260715/rejudged_summary.csv`

注意：`2wikimqa-v2/offline32b_top2@0.05` 未报告有效结果。原因是可用的旧 32B score/fixed-set cache 与 setup-v2 2Wiki 的 document chunk schema 不一致，加载时报缺失 `chunk10_offline32b_mean_score_global`。因此这里不使用伪结果；若后续需要该点，必须先生成 setup-v2 2Wiki 专用 32B score cache。

### 2wikimqa-v2

| Method | Rate | Rows | EM | F1 | GLM Acc |
|---|---:|---:|---:|---:|---:|
| `online_draft` | 0.05 | 200 | 41.50 | 53.90 | 103/200 = 51.50 |
| `online_qk` | 0.05 | 200 | 39.50 | 51.52 | 102/200 = 51.00 |
| `offline3b_mean` | 0.05 | 200 | 35.50 | 47.86 | 92/200 = 46.00 |
| `offline3b_top2` | 0.05 | 200 | 35.50 | 48.39 | 97/200 = 48.50 |
| `offline3b_freq_boundary2` | 0.05 | 200 | 38.00 | 48.94 | 98/200 = 49.00 |
| `offline32b_top2` | 0.05 | N/A | N/A | N/A | N/A |

### hotpotqa-v2

| Method | Rate | Rows | EM | F1 | GLM Acc |
|---|---:|---:|---:|---:|---:|
| `online_draft` | 0.05 | 260 | 69.62 | 80.42 | 226/260 = 86.92 |
| `online_qk` | 0.05 | 260 | 69.62 | 78.58 | 229/260 = 88.08 |
| `offline3b_mean` | 0.05 | 260 | 68.85 | 78.43 | 223/260 = 85.77 |
| `offline3b_top2` | 0.05 | 260 | 68.08 | 77.44 | 225/260 = 86.54 |
| `offline3b_freq_boundary2` | 0.05 | 260 | 68.46 | 77.55 | 224/260 = 86.15 |
| `offline32b_top2` | 0.05 | 260 | 68.85 | 78.30 | 224/260 = 86.15 |

### musique-v2

| Method | Rate | Rows | EM | F1 | GLM Acc |
|---|---:|---:|---:|---:|---:|
| `online_draft` | 0.05 | 200 | 20.00 | 33.80 | 72/200 = 36.00 |
| `online_qk` | 0.05 | 200 | 20.50 | 33.90 | 67/200 = 33.50 |
| `offline3b_mean` | 0.05 | 200 | 21.50 | 34.96 | 74/200 = 37.00 |
| `offline3b_top2` | 0.05 | 200 | 20.50 | 34.31 | 70/200 = 35.00 |
| `offline3b_freq_boundary2` | 0.05 | 200 | 19.50 | 33.28 | 69/200 = 34.50 |
| `offline32b_top2` | 0.05 | N/A | N/A | N/A | N/A |

### triviaqa-v2

| Method | Rate | Rows | EM | F1 | GLM Acc |
|---|---:|---:|---:|---:|---:|
| `online_draft` | 0.05 | 270 | 65.56 | 76.57 | 239/270 = 88.52 |
| `online_qk` | 0.05 | 270 | 64.07 | 74.43 | 235/270 = 87.04 |
| `offline3b_mean` | 0.05 | 270 | 64.44 | 75.21 | 237/270 = 87.78 |
| `offline3b_top2` | 0.05 | 270 | 64.07 | 74.83 | 237/270 = 87.78 |
| `offline3b_freq_boundary2` | 0.05 | 270 | 64.44 | 75.58 | 237/270 = 87.78 |
| `offline32b_top2` | 0.05 | 270 | 64.44 | 74.54 | 238/270 = 88.15 |

### 观察

1. 低 rate 下 online 方法仍通常强于 offline fixed-set：2Wiki 上 `online_draft/qk` 分别为 51.50/51.00 GLM，高于 3B offline 的 46.00-49.00；Hotpot 上 `online_qk` 最高 88.08；Trivia 上 `online_draft` 最高 88.52。
2. offline fixed-set 在 Hotpot/Trivia 上接近 online，说明这些数据集的低 rate token 选择更稳定；但 2Wiki 和 MuSiQue 更依赖 online signal。
3. 与 `preprocess_rate0` 比较：2Wiki 从 45.50 提升到最高 51.50，MuSiQue 从 32.50 提升到最高 38.00，Hotpot/Trivia 的提升较小。这说明低 rate 更新的收益主要集中在更强多跳/组合推理任务。
4. `rate=0.05` 仍明显低于多数 `rate=0.15` 结果，后续如果关注低成本可用点，应重点比较 `0.05 -> 0.15` 的边际收益和 TTFT 成本，而不是只看 accuracy。

