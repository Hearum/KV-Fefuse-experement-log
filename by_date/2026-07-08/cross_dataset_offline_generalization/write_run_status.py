#!/usr/bin/env python3
import subprocess
from datetime import datetime
from pathlib import Path

exp = Path("/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization")
finished = 0
for p in (exp / "results").glob("**/run.log"):
    try:
        if "FINAL RESULTS" in p.read_text(encoding="utf-8", errors="ignore"):
            finished += 1
    except OSError:
        pass
workers = int(subprocess.check_output(
    "ps -ef | grep 'run_cross_dataset_supervisor.sh --worker' | grep -v grep | wc -l",
    shell=True,
    text=True,
).strip())
text = f"""# Cross-Dataset Experiment Run Status

更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 当前状态

- supervisor tmux：`cross_dataset_supervisor`
- accuracy worker：{workers}/8
- 已完整完成 segment：{finished}/180
- 当前阶段：`2wikimqa / online_qk_rate015` 第一批 segment 仍在运行。
- 当前 active cache：`/raid/home/hming/fusionrag-crossdataset-qwen3-cache`。

## 已修正的问题

1. 初版 supervisor 在 score 进程数为 0 时因 `set -euo pipefail` + `grep` 无匹配退出，已修复。
2. 初版 accuracy 共享同一个 dataset cache，多个 segment 并发写 `.pt` 导致 `PytorchStreamReader failed finding central directory`，该轮结果已归档，不进入最终结果。
3. 每 segment 独立 cache 可避免损坏，但重复生成过多 KV，已改成 `worker GPU + dataset` 粒度 cache：同一 worker 串行复用，不并发写。

## 当前观察

`online_qk_rate015` 在 2WikiMQA 上非常慢。原因不是 GPU 空闲，而是 FusionRAG preprocess + global BGE recall 触发大量 cross-document on-demand KV 生成，worker cache 已达到 TB 级。该开销会作为端到端系统成本的一部分记录。

## 查看命令

```bash
tmux attach -t cross_dataset_supervisor
tail -f /raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/logs/supervisor.log
cat /raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/cross_dataset_summary.csv
```
"""
(exp / "RUN_STATUS.md").write_text(text, encoding="utf-8")
print(text)
