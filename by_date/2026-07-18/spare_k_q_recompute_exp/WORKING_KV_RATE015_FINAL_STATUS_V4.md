# Working-KV rate=0.15 稀疏重算阶段报告 v4

## 研究问题

固定 FusionRAG selector 和 rate=0.15，尝试把 native online_qk 对 selected token 的 dense causal attention 重算替换为 block-sparse attention，同时保持 native online_qk 的答案质量，并且减少 online 计算。

full rate=1 只作上界，不是本实验的主要对标。主要对标是同一批样本上的 native online_qk rate=0.15。

## 已验证的正确性基线

Working-KV alpha=1 曾经进入了额外 attention 分支，导致它和 native 不是同一语义。修正后 alpha=1 直接走 native selected-token reprocess 路径，不再传入额外 base/alpha 参数。

50 条 MuSiQue-v2 对照结果：native 与 fixed alpha=1 逐题答案 52/52 一致，GLM judge 均为 20/52。这个结果证明修正后的 Working-KV 接口可以复现 native oracle，但没有节省 attention 计算。

代码提交：1ba0b7b。

## Sparse 结果

| 方法 | 样本 | sparse GLM | 同批 native GLM | 逐题答案差异 | 状态 |
|---|---:|---:|---:|---:|---|
| block top-k=8, block=64 | 50 | 16/50 | 20/50 | 22/50 | 不可用 |
| block top-k=32, block=64 | 10 | 4/10 | 4/10 | 5/10 | 不能证明等价 |
| block top-k=64, block=64 | 5/10 已生成 | 尚无 GLM | 尚无完整对照 | 3/5 exact | 未完成且过慢 |

top-k=64 的启动命令：

    FUSIONRAG_SPARSE_QUERY_CHUNK=32 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py --dataset musique-v2 --method sparse_working_kv_preprocess --rate 0.15 --working-kv-alpha-k 1 --working-kv-alpha-v 1 --sparse-block-size 64 --sparse-block-topk 64 --start 0 --end 10 --gpu 6 --result-root MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_kv_search_10_top64chunk

结果路径：

    MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_kv_search_10_top64chunk/

该进程运行约 21 分钟，仅生成 5/10 条，GPU 利用率持续 100%，未生成 metrics.csv，随后停止。前 5 条 exact match 为 3/5。这个结果不能作为完整 accuracy，但足以说明当前 Python gather prototype 没有实际 online 加速性。

## 计算收益判断

当前 sparse 实现的每层 block routing 和 selected-query gather 使用 Python/PyTorch prototype。top-k 增大到 32 或 64 后，运行时间急剧增加；top-k=64 在 21 分钟只完成 5/10 条。即使 top-k=64 最终 accuracy 接近 native，也不能称为可交付方案，因为没有端到端计算收益证据。

同时，top-k=8 的完整 50 条已经显著低于 native，说明简单的固定 top-k block routing 会损失必要的 causal support。top-k=32 的 10 条虽然 aggregate GLM 恰好与 native 相同，但答案级差异 5/10，不能把 aggregate coincidence 当作等价。

## 当前结论

1. Cache-anchored 的 base plus scaled correction 语义已实现并通过 unit/endpoint guards；native 等价路径已验证。
2. 当前 block-sparse router 尚未找到同时满足质量和收益的配置。
3. 增大 top-k 主要是在用更多 support 换取质量，且当前实现速度更差；这不是“少算重算”的可交付方案。
4. 下一版若继续，应先做 kernel-level batched implementation 和真实 attention support 的离线 upper-bound 分析；不能继续把 Python gather 当作加速结果。

## 复现与版本

- native alpha=1 语义修正：1ba0b7b
- sparse gather 未使用临时张量清理：85ad6f0
- 报告与 shard launcher：4dfb016
- 本报告与 daily-note：33d5449
- 共享 KV cache：/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2

## 三视角审核

### 模型语义审核

PASS。报告使用 native online_qk rate=0.15 作为主要 oracle，明确区分 base cache、candidate KV、working KV，并没有把 sparse candidate 误称为 native 重算结果。

### 系统实现审核

PASS。报告分别记录了 OOM、worker 互相装载、Python gather 过慢和 accuracy 下降；top-k=64 没有 metrics.csv，因此没有伪造 GLM 结果，也没有宣称加速。

### 实验科学审核

PASS。top-k=8 的结论来自 50 条，top-k=32 只作 10 条候选，top-k=64 明确标为未完成。aggregate GLM 与逐题答案差异分开报告，结论范围没有超过证据。

三视角审核均 PASS；若后续补齐 top-k=64，只能作为新版本追加，不能覆盖本版的未完成标记。
