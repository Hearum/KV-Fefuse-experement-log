# FusionRAG-pca-analysis Agent Guide

本文档给进入本项目的 Codex/Research Agent 使用。开始任何任务前请先阅读本文件，再阅读具体实验目录下的 README/任务书。

## 项目定位

本仓库围绕 FusionRAG、KV Cache Reuse、RAG prefill/recompute、offline fixed token selection、draft model selector、Delta-KV 机制分析等方向做实验探索。

多数任务不是单纯改代码，而是：

- 读懂现有 pipeline；
- 设计小规模可验证实验；
- 跑实验并记录启动命令；
- 把结果、路径、结论写回对应 `MOTIVATION_EXPERIMENTS/<task>/README.md` 或实验日志。

## 常用路径

主仓库路径可能因机器不同而变化：

```bash
/raid/home/hming/FusionRAG-pca-analysis
/home/hming/FusionRAG-pca-analysis
```

如果两个路径都存在，先检查用户指定路径。不要自行移动仓库。

常用 Python：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
```

常用模型：

```bash
/mnt/qjhs-sh-lab-01/models/Qwen3-32B
/home/hming/models/Qwen3-235B-A22B
/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct
/mnt/qjhs-sh-lab-01/models/bge-m3
```

常用数据：

```bash
data/result_reflect.json
MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/2wikimqa_reflect.json
MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/hotpotqa_reflect.json
MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/triviaqa_reflect.json
```

## FusionRAG 入口

常用入口脚本：

```bash
test_fusionrag_reflect_preprocess_exp.py
```

三类 baseline 的含义：

- `full_rate1`：`rate=1.0`，full recompute/full attention 风格上界。
- `online_qk_rate015`：`rate=0.15`，FusionRAG online QK selector。
- `online_draft_rate015`：`rate=0.15`，Qwen2.5-3B-Instruct draft model selector。

关键参数：

```bash
# full_rate1
--rate 1.0
--reprocess_method FusionRAG

# online_qk_rate015
--rate 0.15
--reprocess_method FusionRAG

# online_draft_rate015
--rate 0.15
--reprocess_method DraftModel
--draft_model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct
```

常用公共参数：

```bash
--topk 10
--preprocess true
--recall_method bge
--revert_rope true
--preprocess_scope global
--bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3
```


## Pipeline 和数据集使用

默认 FusionRAG 入口仍是：

```bash
test_fusionrag_reflect_preprocess_exp.py
```

启动实验前需要在 README 或 EXPERIMENT_LOG 中明确记录：

- 使用的模型路径，例如 `Qwen3-32B`、`Qwen2.5-3B-Instruct`、`bge-m3`。
- 使用的数据文件路径，例如 `data/result_reflect.json` 或 cross-dataset reflect 数据。
- 关键 pipeline 参数，例如 `--dataset`、`--model_path`、`--rate`、`--topk`、`--preprocess`、`--preprocess_scope`、`--recall_method`、`--reprocess_method`、`--draft_model_path`、`--revert_rope`。
- 输出目录、cache 目录、实验结果目录三者分别是什么。

如果要复现 setup-standard / cross-dataset 实验，优先先查对应实验目录的 README 和脚本，不要凭记忆拼参数。

重要口径说明：历史 reflect/main-sub wrapper pipeline 与 setup-standard v2 pipeline 不是同一个评测口径，不能把两边的 EM/F1/GLM judge 结果直接当作同源指标比较。引用旧结果时必须标注 pipeline、数据格式、数据路径、runner 脚本和 commit；如果要比较方法优劣，应在同一 pipeline、同一数据文件、同一 cache/preprocess 定义下重跑。

## Preprocess KV Cache 使用规范

Preprocess KV cache 按 `MODEL/DATASET` 维度维护和复用，除非任务本身就是特殊 cache 对比实验，不要为每个新实验重新生成一套 preprocess KV。

必须遵守：

- 同一模型、同一数据集、同一 preprocess 定义下，优先复用已经保存好的 preprocess KV cache。
- 不要让每个 GPU worker 各自维护一份重复 cache；多 worker 应读同一套共享 cache。
- 不要把 cache 路径写进实验结果目录；cache 和 results 必须分开。
- 新建 cache 前先检查已有 cache 的模型、数据集、topk、recall 方法、preprocess_scope、生成脚本和日志。
- 如果因为对比实验必须重新生成 cache，要在实验 README 里说明原因，并写清楚新旧 cache 的差异。
- 清理空间时只清理明确可再生且不含实验结论的 cache；不要删除 README、CSV、JSON summary、judge metadata、脚本、日志或图表。

推荐共享 cache 目录形态：

```bash
<shared-cache-root>/<MODEL_NAME>/<DATASET_NAME>/{kv_cache,preprocess_kv,...}
```

不要使用下面这种模式：

```bash
<experiment>/<worker_gpu_0>/cache
<experiment>/<worker_gpu_1>/cache
```

## MOTIVATION_EXPERIMENTS 维护规范

`MOTIVATION_EXPERIMENTS` 已按实验创建日期整理：

```bash
MOTIVATION_EXPERIMENTS/by_date/YYYY-MM-DD/<experiment>/
```

顶层实验名通常是兼容 symlink，指向 `by_date/YYYY-MM-DD/<experiment>/`。后续新实验也必须继续维护这套 daily 连接结构：真实目录放在当天 `by_date/YYYY-MM-DD/<experiment>/` 下；如需要兼容旧路径或便于发现，在 `MOTIVATION_EXPERIMENTS/<experiment>` 建同名 symlink 指向真实目录。不要再把新的真实实验目录直接堆在 `MOTIVATION_EXPERIMENTS/` 顶层。

每个新实验建立时，先写 plan，再按 plan 执行。推荐最小结构：

```bash
MOTIVATION_EXPERIMENTS/by_date/YYYY-MM-DD/<task>/
  README.md
  PLAN.md
  EXPERIMENT_LOG.md
  scripts/
  results/
  figures/
```

维护要求：

- `PLAN.md` 或 README 开头必须写实验目的、假设、对照组、样本范围、成功/失败判据。
- `EXPERIMENT_LOG.md` 按时间追加当前进度、上下文记忆信息和启动命令、运行机器、GPU、commit、异常和中间观察。
- 所有最终表格、CSV、JSON summary、judge metadata、图表都放在 `results/` 或 `figures/`，不要混在 cache 目录里。
- README 要维护最终结果表、主要结论、复现命令、数据路径、脚本路径和未完成事项。
- 写 README、EXPERIMENT_LOG、阶段性报告和新增说明文档时优先使用中文；代码标识、命令参数、指标名、模型/数据集专有名词可保留英文。

## 结果和 Cache 分离

实验生成结果不要和 cache 放在一起。结果目录和 cache 目录必须分离，避免未来清理 cache 时误删实验结论。

一般应长期保留：

- README、PLAN、EXPERIMENT_LOG。
- 启动脚本、分析脚本、画图脚本。
- 结果 CSV、JSON summary、judge metadata、评测明细、图表。
- 记录实验配置的命令行、环境变量、commit hash、数据路径。

一般可在确认后清理：

- 可从模型和数据重新生成的大体积 KV cache。
- 明确失败且无分析价值的临时 cache。
- 临时 worker 中间目录。

## Commit 规范

以下情况需要及时 commit 保存：

- 主 pipeline 代码有大更新。
- 新增实验接口、环境变量接口、cache 读取逻辑、评测逻辑。
- 做消融实验前后，尤其是会影响 baseline 或评测口径的代码修改。
- 新增或更新关键实验文档、结果表、画图脚本。

每个实验 README/EXPERIMENT_LOG 中都要记录当前实验使用的 commit hash。启动命令建议写成可直接复制运行的形式，并包含关键环境变量。

## 实验记录规范

每个实验目录都应维护中文记录，至少包含：

- README、EXPERIMENT_LOG、阶段性报告和新增说明文档优先使用中文；只有代码标识、命令参数、指标名、论文/模型/数据集专有名词可保留英文。
- 实验目的
- 数据集和样本范围
- 启动命令（包含环境变量和完整参数）
- 当前实验 commit hash
- 代码/脚本路径
- 数据集路径
- cache 路径
- 输出文件路径
- 指标定义
- 结果表格
- 初步结论
- 异常和未解决问题

不要只把结果写在对话里。

推荐结构：

```bash
MOTIVATION_EXPERIMENTS/<task>/
  README.md
  EXPERIMENT_LOG.md
  scripts/
  results/
  figures/
```


## 安全约束

- 不要删除模型权重、实验结果 CSV、README、日志。
- 不要停止用户未明确要求停止的 tmux/session/container。
- 不要复用不确定的 cache 路径；先检查 `run.log` 和实际目录。
- 如果发现结果反常，先排查实现和统计口径，再下结论。

## 面向 KV/Delta-KV 机制任务的提示

若任务涉及 offline KV、preprocess KV、full KV、Delta-KV：

- 先确认 token 对齐。
- 记录 K/V tensor shape、layer/head/token 维度顺序。
- 区分 raw/offline KV、preprocess KV、online recomputed KV、full KV。
- 区分 key 和 value，不要只看 concat。
- 小样本 sanity check 通过后再扩大规模。
