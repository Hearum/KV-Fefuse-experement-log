# Qwen3-235B 参数量泛化实验运行状态

更新时间：2026-07-09 23:30

## 当前机器分配

### qjy000

- tmux：`qwen3_235b_param_qjy000_resume`
- 脚本：`MOTIVATION_EXPERIMENTS/qwen3_235b_param_scaling_cross_dataset/launch_qjy000_resume_queue.sh`
- 当前队列：
  1. `2wikimqa`: 跳过已完成的 `full_rate1`、`online_qk_rate015`，从修复后的 `online_draft_rate015` 重新跑起，然后继续 `offline3b_mean_rate015 -> offline3b_freq_boundary2_rate015 -> offline32b_top2_rate015`
  2. `triviaqa`: 同上 6 组
- 当前状态：已启动，正在 `2wikimqa/online_draft_rate015`。这轮用于替换之前只有 55 条的中断半成品。

### qjy001

- follow-up tmux：`qwen3_235b_param_qjy001_followup`
- follow-up 脚本：`MOTIVATION_EXPERIMENTS/qwen3_235b_param_scaling_cross_dataset/launch_qjy001_followup_queue.sh`
- follow-up 队列：
  1. 补 `musique`: `offline3b_mean_rate015 -> offline3b_freq_boundary2_rate015 -> offline32b_top2_rate015`
  2. 跑 `hotpotqa`: 6 组核心方法。
- 当前状态：已启动，正在 `musique/offline3b_mean_rate015` 加载 Qwen3-235B。
- 修复记录：`offline3b_mean_rate015` 的 fixed-set 目录已从不存在的分段结果目录改为：
  `MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/reflect_draft_smart_global_rate015_full/chunk_fixed_sets_npz`。

## 输出目录

- 主目录：`MOTIVATION_EXPERIMENTS/qwen3_235b_param_scaling_cross_dataset/`
- qjy000 日志：`logs/qjy000_resume_queue.log`
- qjy001 follow-up 日志：`logs/qjy001_followup_queue.log`

## 监控命令

```bash
ssh qjy000 'tmux attach -t qwen3_235b_param_qjy000_resume'
ssh qjy001 'tmux attach -t qwen3_235b_param_qjy001_followup'
```

```bash
ssh qjy000 'tail -f /raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/qwen3_235b_param_scaling_cross_dataset/logs/qjy000_resume_queue.log'
ssh qjy001 'tail -f /home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/qwen3_235b_param_scaling_cross_dataset/logs/qjy001_followup_queue.log'
```
