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
