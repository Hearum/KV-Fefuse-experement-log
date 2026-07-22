# Qwen3-235B 参数量泛化实验规划

## 目标

本实验验证 FusionRAG 相关 selection / offline fixed set 方法在更大 Qwen 模型上是否仍然成立。之前主表主要基于 `Qwen3-32B`，本轮把回答模型替换为 `Qwen3-235B-A22B`，其他设置尽量保持一致。

核心问题：

1. `full_rate1`、`online_qk_rate015`、`online_draft_rate015` 在 235B 上的相对关系是否和 32B 一致。
2. 纯 offline fixed set 在更大回答模型下是否仍然有竞争力。
3. offline set 是否只是对 32B 模型偶然有效，换成 235B 后是否退化。
4. 不同数据集上是否保持同一趋势。

## 固定配置

- 仓库：`/raid/home/hming/FusionRAG-pca-analysis`，在 `qjy000` 上访问；`qjy001` 上等价路径为 `/home/hming/FusionRAG-pca-analysis`。
- 回答模型：`/home/hming/models/Qwen3-235B-A22B`。
- model type：`qwen3_moe`。
- prompt：统一 no-thinking prompt，`/no_think` + assistant 侧空 `<think></think>`。
- retrieval：`topk=10`。
- preprocess：`true`。
- preprocess scope：`global`。
- recall method：`bge`。
- BGE：`/mnt/qjhs-sh-lab-01/models/bge-m3`。
- judge：`GLM-5.2` API。
- rate：除 full baseline 外统一 `0.15`。
- 多卡方式：当前为 HF `device_map=auto`，不是 TP=8；速度数据只作为当前实现 wall-clock 参考，不作为高效 serving 下的绝对吞吐结论。

## 数据集顺序

按数据集顺序跑，每个数据集的所有 baseline 跑完后再进入下一个数据集：

1. `musique`
   - 数据：`data/result_reflect.json`
   - 样本：0-200
   - 说明：当前 `qjy001` 已经拉起三组 unified-prompt 235B 实验，先纳入本实验结果。
2. `2wikimqa`
   - 数据：`MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/2wikimqa_reflect.json`
   - 样本：0-200
3. `hotpotqa`
   - 数据：`MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/hotpotqa_reflect.json`
   - 样本：0-260
4. `triviaqa`
   - 数据：`MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/triviaqa_reflect.json`
   - 样本：0-270

## 每个数据集内的方法顺序

先跑核心 online/full baseline，再跑 offline 方法。这样每个数据集可以先得到可解释的上界和在线对照。

1. `full_rate1`
   - `rate=1.0`
   - `reprocess_method=FusionRAG`
   - 含义：完整 online recompute / full attention 对照。

2. `online_qk_rate015`
   - `rate=0.15`
   - `reprocess_method=FusionRAG`
   - 含义：FusionRAG 原始 QK/attention importance selector。

3. `online_draft_rate015`
   - `rate=0.15`
   - `reprocess_method=DraftModel`
   - `draft_model_path=/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`
   - 含义：online draft model selector。注意这里 draft model 仍是 3B，变量只改变回答模型参数量。

4. `offline3b_mean_rate015`
   - `rate=0.15`
   - `offline_fixed_set_method=mean_score`
   - fixed set 来源：对应数据集的 `fixed_sets_*_3b`
   - 含义：3B draft teacher 生成的 offline mean-score fixed set。

5. `offline3b_freq_boundary2_rate015`
   - `rate=0.15`
   - fixed set 来源：3B frequency + 2% boundary replacement。
   - 含义：之前 cross-dataset 中表现较好的纯 offline + boundary 版本。

6. `offline32b_top2_rate015`
   - `rate=0.15`
   - fixed set 来源：对应数据集的 `fixed_sets_*_32b`
   - 含义：32B teacher top2-mean offline fixed set，用来判断强 teacher offline set 换到 235B 回答模型后是否仍泛化。

可选追加：

7. `offline_qk_mean_rate015`
8. `offline_qk_mean_boundary2_rate015`

这两组之前在 cross-dataset 表里是 pending addendum。本轮先放入待完成项，不抢占核心 6 组。

## 机器分配

### qjy001

当前已经运行：

- tmux：`qwen3_235b_three_groups_unified_prompt`
- 目录：`MOTIVATION_EXPERIMENTS/qwen3_235b_three_groups_unified_prompt`
- 任务：`musique/full_rate1 -> online_qk_rate015 -> online_draft_rate015`

处理方式：

- 不停止当前进程。
- 该结果作为本规划中 `musique` 的前三个 baseline。
- 等前三组完成后，再决定是否继续在 qjy001 补 `musique` 的 offline 3 组。

### qjy000

8 张 H20 当前空闲，先负责 cross-dataset 的 235B 实验。由于 235B 使用 `device_map=auto`，单个进程需要 8 卡装载模型，因此 qjy000 上同一时间只跑一个 235B 任务。

队列顺序：

1. `2wikimqa`: 6 个核心方法。
2. `hotpotqa`: 6 个核心方法。
3. `triviaqa`: 6 个核心方法。
4. 若 qjy001 仍未补完 MuSiQue offline，则 qjy000 回头补 `musique` offline 3 组。

## 输出目录

统一输出到：

```text
MOTIVATION_EXPERIMENTS/qwen3_235b_param_scaling_cross_dataset/
```

建议结构：

```text
qwen3_235b_param_scaling_cross_dataset/
  PLAN.md
  RUN_STATUS.md
  launch_qjy000_queue.sh
  summarize_235b_param_scaling.py
  results/
    2wikimqa/
      full_rate1/
      online_qk_rate015/
      online_draft_rate015/
      offline3b_mean_rate015/
      offline3b_freq_boundary2_rate015/
      offline32b_top2_rate015/
    hotpotqa/
    triviaqa/
    musique/
  logs/
  summary_235b_param_scaling.csv
```

## 结果表字段

每一行记录一个 `dataset + method`：

- `dataset`
- `method`
- `model`
- `rate`
- `main_correct`
- `main_total`
- `main_acc`
- `sub_correct`
- `sub_total`
- `sub_acc`
- `avg_f1`
- `avg_em`
- `prompt_eval_mean_s`
- `storage_time_mean_s`
- `selection_time_mean_s`
- `decode_mean_s`（如果日志可解析）
- `found_csv_files`
- `traceback`
- `note`

## 当前注意事项

1. 235B 当前不是 TP=8，而是 `device_map=auto`。速度慢是预期现象。
2. 本轮 prompt 已统一 no-thinking。新实验必须使用当前代码，不能混入旧 235B prompt 结果。
3. 旧目录 `qwen3_235b_three_groups` 里有未统一 prompt 的结果，只保留作调试，不进入主表。
4. 当前 qjy001 上 `qwen3_235b_three_groups_unified_prompt` 的 MuSiQue 前三组可以进入本实验。
5. 所有新结果不能覆盖旧 cross-dataset 32B 结果。
