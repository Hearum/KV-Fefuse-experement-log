# Cross-Dataset Offline Selection Generalization

# 跨数据集 Offline Selection 泛化实验计划

## 目的

验证当前在 MuSiQue 上表现较好的纯 offline token selection 方法，是否能泛化到其他 RAG 数据集。所有方法都使用同一套 FusionRAG reflect pipeline 重新跑 accuracy，不复用旧答案。

## 数据集

本轮使用仓库中已有数据，统一转换成 reflect 格式：

| 数据集 | 输入文件 | 样本数 | 说明 |
|---|---|---:|---|
| 2WikiMQA | `data/2wikimqa-200.jsonl` -> `data/2wikimqa_reflect.json` | 200 | 每条样本转成一个 main question 和一个 sub-question |
| HotpotQA | `data/hotpotqa-260-100-10-doc.jsonl` -> `data/hotpotqa_reflect.json` | 260 | 同上 |
| TriviaQA | `data/triviaqa-270-100-10-doc.jsonl` -> `data/triviaqa_reflect.json` | 270 | 同上 |

注意：这次转换是为了复用现有 reflect pipeline，因此 Main Acc 与 Sub Acc 在这些单 sub-question 转换数据上通常接近或相同。

## 对比方法

| 实验名 | rate | selection 来源 | 含义 |
|---|---:|---|---|
| `full_rate1` | 1.00 | 不选择，全部重算 | full attention / full recompute 空白上界 |
| `online_qk_rate015` | 0.15 | online FusionRAG-QK | 当前 FusionRAG 的 query-conditioned QK selector |
| `online_draft_rate015` | 0.15 | online DraftModel 3B | 使用 Qwen2.5-3B-Instruct 作为 online draft selector |
| `offline3b_mean` | 0.15 | offline 3B teacher score cache | 每个 chunk 预先聚合 draft score，按 mean score 选 15% token |
| `offline3b_freq_boundary2` | 0.15 | offline 3B teacher score cache + boundary | 3B frequency baseline 中用 2% 文档边界 token 替换，保持总 rate=15% |
| `offline32b_top2` | 0.15 | offline 32B teacher score cache | 使用 32B teacher 的 top2-mean score 聚合，选 15% token |

### 待追加方法：Offline QK

当前 180-task 主实验先不打断。主实验完成后追加以下 QK fixed-set 方法：

| 实验名 | rate | selection 来源 | 含义 |
|---|---:|---|---|
| `offline_qk_mean` | 0.15 | offline QK calibration score | 使用主模型/兼容主模型的 QK attention importance，对 calibration queries 做 mean aggregation 后固定选择每个 chunk 的 15% token |
| `offline_qk_mean_boundary2` | 0.15 | offline QK + boundary | 总 rate 仍为 15%，其中 2% budget 用 chunk boundary token 替换，其余来自 QK mean score |

具体计划见 `PENDING_QK_ADDENDUM.md`。

## 执行流程

1. 等待 3B/32B score cache 完整生成，并等待所有 score cache 写入进程退出。
2. 从 score cache 派生每个数据集的 offline fixed set：
   - `fixed_sets_<dataset>_3b`
   - `fixed_sets_<dataset>_32b`
3. 按数据集、方法、segment 并行运行 accuracy pipeline。
4. 汇总每个方法在每个数据集上的 Main Acc / Sub Acc / F1 / EM / 平均 prefill 时间 / 平均 selection 时间。
5. 将结果写入本目录 `README.md` 和 `cross_dataset_summary.csv/json`。

实现细节：accuracy 阶段每个 `worker GPU + dataset` 使用独立 `cache_path`。同一 GPU worker 内部任务串行执行，因此不会并发写同一批 `.pt` cache 文件；同时又能让同一 worker 后续方法/segment 复用已生成 KV，避免每个 segment 完全重复生成。

## 复现入口

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
tmux new -s cross_dataset_supervisor \
  'bash MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/run_cross_dataset_supervisor.sh'
```

查看日志：

```bash
tail -f /raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/logs/supervisor.log
```


## 当前结果汇总

| dataset | method | rate | Main Acc | Sub Acc | F1 | EM | prefill(s) | selection(s) | rows | finished seg | missing seg | traceback/killed | 含义 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2wikimqa | full_rate1 | 1.00 | 113/200 (56.50%) | 113/200 (56.50%) | 0.3319 | 0.0250 | 3.8673 | 0.0000 | 200 | 8 | 0 | 0/2 | rate=1.0 full recompute baseline; no selector. |
| 2wikimqa | online_qk_rate015 | 0.15 | 107/200 (53.50%) | 107/200 (53.50%) | 0.2767 | 0.0150 | 2.2925 | 0.3244 | 200 | 8 | 0 | 0/4 | online FusionRAG-QK selector, rate=0.15. |
| 2wikimqa | online_draft_rate015 | 0.15 | 101/200 (50.50%) | 101/200 (50.50%) | 0.2687 | 0.0150 | 2.4394 | 0.4012 | 200 | 8 | 0 | 0/2 | online DraftModel selector with Qwen2.5-3B-Instruct, rate=0.15. |
| 2wikimqa | offline3b_mean | 0.15 | 96/200 (48.00%) | 96/200 (48.00%) | 0.2806 | 0.0250 | 2.0429 | 0.0000 | 200 | 8 | 0 | 0/2 | offline fixed set from 3B teacher mean-score aggregation, rate=0.15. |
| 2wikimqa | offline3b_freq_boundary2 | 0.15 | 99/200 (49.50%) | 99/200 (49.50%) | 0.2801 | 0.0200 | 2.0302 | 0.0000 | 200 | 8 | 0 | 0/2 | offline 3B frequency fixed set with 2% boundary replacement, total rate=0.15. |
| 2wikimqa | offline32b_top2 | 0.15 | 103/200 (51.50%) | 103/200 (51.50%) | 0.2656 | 0.0150 | 2.0097 | 0.0000 | 200 | 8 | 0 | 0/2 | offline fixed set from 32B teacher top2-mean aggregation, rate=0.15. |
| 2wikimqa | offline_qk_mean | 0.15 | 0/0 (0.00%) | 0/0 (0.00%) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 8 | 0/0 | PENDING addendum: offline QK mean-score fixed set, rate=0.15. |
| 2wikimqa | offline_qk_mean_boundary2 | 0.15 | 0/0 (0.00%) | 0/0 (0.00%) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 8 | 0/0 | PENDING addendum: offline QK mean-score fixed set with 2% boundary replacement, total rate=0.15. |
| hotpotqa | full_rate1 | 1.00 | 207/260 (79.62%) | 207/260 (79.62%) | 0.3076 | 0.0269 | 1.2716 | 0.0000 | 260 | 11 | 0 | 0/0 | rate=1.0 full recompute baseline; no selector. |
| hotpotqa | online_qk_rate015 | 0.15 | 206/260 (79.23%) | 206/260 (79.23%) | 0.3007 | 0.0308 | 1.2551 | 0.3337 | 260 | 11 | 0 | 0/0 | online FusionRAG-QK selector, rate=0.15. |
| hotpotqa | online_draft_rate015 | 0.15 | 207/260 (79.62%) | 207/260 (79.62%) | 0.3245 | 0.0385 | 1.0741 | 0.1516 | 260 | 11 | 0 | 0/0 | online DraftModel selector with Qwen2.5-3B-Instruct, rate=0.15. |
| hotpotqa | offline3b_mean | 0.15 | 204/260 (78.46%) | 204/260 (78.46%) | 0.3193 | 0.0192 | 0.9275 | 0.0000 | 260 | 11 | 0 | 0/0 | offline fixed set from 3B teacher mean-score aggregation, rate=0.15. |
| hotpotqa | offline3b_freq_boundary2 | 0.15 | 201/260 (77.31%) | 201/260 (77.31%) | 0.3141 | 0.0308 | 0.9278 | 0.0000 | 260 | 11 | 0 | 0/0 | offline 3B frequency fixed set with 2% boundary replacement, total rate=0.15. |
| hotpotqa | offline32b_top2 | 0.15 | 197/260 (75.77%) | 197/260 (75.77%) | 0.3039 | 0.0346 | 0.9274 | 0.0000 | 260 | 11 | 0 | 0/0 | offline fixed set from 32B teacher top2-mean aggregation, rate=0.15. |
| hotpotqa | offline_qk_mean | 0.15 | 0/0 (0.00%) | 0/0 (0.00%) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 11 | 0/0 | PENDING addendum: offline QK mean-score fixed set, rate=0.15. |
| hotpotqa | offline_qk_mean_boundary2 | 0.15 | 0/0 (0.00%) | 0/0 (0.00%) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 11 | 0/0 | PENDING addendum: offline QK mean-score fixed set with 2% boundary replacement, total rate=0.15. |
| triviaqa | full_rate1 | 1.00 | 211/270 (78.15%) | 211/270 (78.15%) | 0.2036 | 0.0370 | 1.2635 | 0.0000 | 270 | 11 | 0 | 0/0 | rate=1.0 full recompute baseline; no selector. |
| triviaqa | online_qk_rate015 | 0.15 | 212/270 (78.52%) | 212/270 (78.52%) | 0.2343 | 0.0407 | 1.2276 | 0.3268 | 270 | 11 | 0 | 0/0 | online FusionRAG-QK selector, rate=0.15. |
| triviaqa | online_draft_rate015 | 0.15 | 214/270 (79.26%) | 214/270 (79.26%) | 0.2137 | 0.0556 | 1.0498 | 0.1520 | 270 | 11 | 0 | 0/0 | online DraftModel selector with Qwen2.5-3B-Instruct, rate=0.15. |
| triviaqa | offline3b_mean | 0.15 | 215/270 (79.63%) | 215/270 (79.63%) | 0.2330 | 0.0630 | 0.9054 | 0.0000 | 270 | 11 | 0 | 0/0 | offline fixed set from 3B teacher mean-score aggregation, rate=0.15. |
| triviaqa | offline3b_freq_boundary2 | 0.15 | 218/270 (80.74%) | 218/270 (80.74%) | 0.2055 | 0.0556 | 0.9047 | 0.0000 | 270 | 11 | 0 | 0/0 | offline 3B frequency fixed set with 2% boundary replacement, total rate=0.15. |
| triviaqa | offline32b_top2 | 0.15 | 156/200 (78.00%) | 156/200 (78.00%) | 0.2186 | 0.0500 | 0.9091 | 0.0000 | 200 | 8 | 0 | 6/0 | offline fixed set from 32B teacher top2-mean aggregation, rate=0.15. |
| triviaqa | offline_qk_mean | 0.15 | 0/0 (0.00%) | 0/0 (0.00%) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 11 | 0/0 | PENDING addendum: offline QK mean-score fixed set, rate=0.15. |
| triviaqa | offline_qk_mean_boundary2 | 0.15 | 0/0 (0.00%) | 0/0 (0.00%) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 11 | 0/0 | PENDING addendum: offline QK mean-score fixed set with 2% boundary replacement, total rate=0.15. |

## 结果文件

- `cross_dataset_summary.csv`
- `cross_dataset_summary.json`

## 2026-07-14 GLM-clean 重新判分

目的：对本目录 `results/` 下已经生成的 prediction CSV 重新用当前 `GLM-5.2` 判分。该步骤不重新跑 generation pipeline，只修正评估口径。

清洗规则：送入 GLM judge 前移除 `<think>...</think>`、残留 think 标签，以及开头的 `Answer:` / `Final Answer:` / `答案:`。不做其他语义改写。

复现命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
python3 MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rejudge_glm_clean_20260714/rejudge_cross_dataset_glm_clean.py \
  > MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rejudge_glm_clean_20260714/rejudge.log 2>&1
```

对应提交：`dea7d39fe6d2e1fd524281d0ac632b3db6dad5a5`；主 README 记录补充提交见后续 git log。

输出文件：

- `rejudge_glm_clean_20260714/rejudged_summary.csv`
- `rejudge_glm_clean_20260714/rejudged_rows.csv`
- `rejudge_glm_clean_20260714/README.md`

覆盖范围：4310 行，missing segment = 0。已重判的方法包括 `full_rate1`、`online_qk_rate015`、`online_draft_rate015`、`offline3b_mean`、`offline3b_freq_boundary2`、`offline32b_top2`。

没有重判 `offline_qk_mean` 和 `offline_qk_mean_boundary2`，原因是它们在当前 README 里仍是 `PENDING addendum`，且 `MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/results/` 下没有对应 CSV 文件。确认命令：

```bash
find MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/results -path "*offline_qk*" -type f | wc -l
# 0
```

GLM-clean 主问题准确率：

| dataset | full_rate1 | online_qk_rate015 | online_draft_rate015 | offline3b_mean | offline3b_freq_boundary2 | offline32b_top2 |
|---|---:|---:|---:|---:|---:|---:|
| 2wikimqa | 117/200 (58.50%) | 112/200 (56.00%) | 106/200 (53.00%) | 104/200 (52.00%) | 105/200 (52.50%) | 106/200 (53.00%) |
| hotpotqa | 236/260 (90.77%) | 226/260 (86.92%) | 228/260 (87.69%) | 229/260 (88.08%) | 225/260 (86.54%) | 227/260 (87.31%) |
| triviaqa | 246/270 (91.11%) | 241/270 (89.26%) | 239/270 (88.52%) | 241/270 (89.26%) | 243/270 (90.00%) | 180/200 (90.00%) |

注意：`triviaqa/offline32b_top2` 只有 200 行旧 CSV，不是完整 270 行，不能和完整 TriviaQA 方法严格横比。

结论：GLM-clean 重判后整体准确率上升，说明旧 judge 低估了部分语义正确答案；但 `rate=0.15` 的 online/offline 方法仍没有整体追平 `full_rate1`，特别是 HotpotQA 和 TriviaQA 上 full recompute 仍是最强或并列最强基线。

## 2026-07-14 补跑 offline32b_top2 / TriviaQA 缺失段

原因：`offline32b_top2/triviaqa` 在旧结果中只有 200/270 行，缺失 `seg_175_200`、`seg_225_250`、`seg_250_270`。旧 `run.log` 显示这些 segment 曾因 device mismatch / rotary_emb 兼容问题失败。

确认命令：

```bash
find MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/results/offline32b_top2/triviaqa -name run.log -print | sort | xargs -r grep -H "FINAL RESULTS\|Traceback\|RuntimeError\|AttributeError"
```

补跑脚本：

```bash
MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rerun_offline32b_top2_trivia_missing.sh
```

启动命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
nohup MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/rerun_offline32b_top2_trivia_missing.sh   > MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/logs/rerun_offline32b_top2_trivia_missing_20260714/driver.outer.log 2>&1 < /dev/null &
```

补跑范围：只跑 `offline32b_top2` + `triviaqa` 的三段缺失 segment：175-200、225-250、250-270。每段使用独立 cache：`/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-crossdataset-offline32b-trivia-missing-cache-20260714/gpu*/seg_*`。

安全策略：不删除旧日志；启动前将旧失败 `run.log` 复制为 `run.log.failed_YYYYMMDD_HHMMSS`，然后写入新的 `run.log`。

