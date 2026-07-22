# FusionRAG-CD: Causal-Draft Distillation 本机实验

本目录用于把 qjh005 上的 DistillKit smoke 迁回 qjhs 本机继续。

## 方法命名

**FusionRAG-CD (Causal-Draft Distillation)**：蒸馏一个真正的小 causal LM / draft model，不显式训练 selector head；训练完成后读取 student prefill attention 作为 doc token importance。

## 当前状态

- qjh005 已完成 DistillKit tiny local smoke，但真实 Qwen smoke 阻塞于 Hugging Face 网络和模型路径。
- 本机 `/raid/home/hming/FusionRAG-pca-analysis` 已创建本目录。
- 下一步：复用本机已有 Qwen2.5/Qwen3 模型路径，建立本机 smoke 和真实数据 distillation pipeline。

## 模型路径候选

- Teacher/Draft: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`
- Main: `/mnt/qjhs-sh-lab-01/models/Qwen3-32B`

## 计划

1. 先在本机实现最小 causal LM logits distillation smoke，避免依赖外网下载。
2. Student 先用本地小模型或 Qwen2.5 截层方案；teacher 用 Qwen2.5-3B。
3. 训练目标：next-token logits KL/CE，不训练 selector head。
4. smoke 通过后，读取 student attention 生成 token importance，与 full DraftModel selector overlap 对比。


## 2026-07-06 本机 smoke 结果

已在本机完成最小 causal-LM distillation smoke，不依赖 Hugging Face 下载，也不训练 selector head。

命令：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/causal_lm_distill_draftmodel_local/scripts/train_toy_causal_distill.py \
  --out-dir MOTIVATION_EXPERIMENTS/causal_lm_distill_draftmodel_local/outputs/toy_smoke_cpu \
  --steps 10 --batch-size 4 --device cpu
```

输出：

- `outputs/toy_smoke_cpu/toy_student.pt`
- `outputs/toy_smoke_cpu/history.jsonl`
- `outputs/toy_smoke_cpu/attention_importance.json`
- `logs/toy_smoke_cpu.log`

该 smoke 只验证 FusionRAG-CD 的工程链路：causal LM teacher/student KL 蒸馏 + 从 student prefill attention 提取 token importance。下一步需要在空闲 GPU 上把 ToyCausalLM 换成真实 Qwen/DraftModel。

## 2026-07-06 FusionRAG-CD 真实 Qwen 蒸馏启动记录

本轮按“真正 draft model 蒸馏”路线启动，不再训练显式 selector head。

- 方法名：**FusionRAG-CD (Causal-Draft Distillation)**
- Teacher：`/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct` 完整模型
- Student：从同一个 Qwen2.5-3B 初始化，只保留前 4 个 decoder layers
- 训练数据：`MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/data/wikitext103_train.txt`
- token cache：`MOTIVATION_EXPERIMENTS/causal_lm_distill_draftmodel_local/data/wikitext103_qwen_blocks_s256_n20000.pt`
- 目标：next-token logits KL + 0.1 * CE
- 序列长度：256
- 训练规模：20k sequences，最多 2000 optimizer steps
- 并行：GPU0-3，torchrun 4 ranks
- 输出目录：`MOTIVATION_EXPERIMENTS/causal_lm_distill_draftmodel_local/outputs/qwen_cd_wikitext103_4layer_gpu0_3_s256_n20000_step2000`
- 日志：`MOTIVATION_EXPERIMENTS/causal_lm_distill_draftmodel_local/logs/qwen_cd_wikitext103_4layer_gpu0_3_s256_n20000_step2000.log`
- 启动脚本：`MOTIVATION_EXPERIMENTS/causal_lm_distill_draftmodel_local/launch_qwen_cd_gpu0_3.sh`

启动前 smoke：GPU0 单卡 1 step 已通过，确认 teacher/student 加载、4 层截断、KL/CE loss 和 checkpoint 写入正常。

## 2026-07-07 FusionRAG-CD 继续训练到 step 6000

- 训练任务：Qwen2.5-3B teacher -> 4-layer student causal LM distillation。
- 数据：WikiText103 token blocks，seq_len=256，max_sequences=20000。
- 继续训练输出目录：`outputs/qwen_cd_wikitext103_4layer_gpu0_3_s256_n20000_step6000`。
- 恢复点：`training_state_latest.pt`，从 step 3500 继续到 step 6000。
- 最终 step：6000，epoch=8。
- 最终日志指标：loss=5.4949，KL=4.9314，CE=5.6350，lr=1e-5，world_size=4。
- 最终 checkpoint：`training_state_final_step006000.pt`；同时保留 `training_state_step006000.pt` 和 `training_state_latest.pt`。


## 2026-07-07 step6000 验证集 selector overlap 评估

用户指出仅看 train loss 不足，需要验证集，并和完整 3B DraftModel 的 token selection 对齐程度做对比。因此补充复用 HS/native layer4 pipeline 的验证缓存：`teacher_scores` 来自完整 Qwen2.5-3B DraftModel，对 CD 4-layer checkpoint 输出的 doc-token score 计算 KL、top-ratio recall 和 Jaccard。

- 评估脚本：`scripts/eval_cd_layer4_against_full_draft.py`
- CD checkpoint：`outputs/qwen_cd_wikitext103_4layer_gpu0_3_s256_n20000_step6000/training_state_final_step006000.pt`
- 训练状态：step=6000，epoch=8
- Wiki 验证：5000 条
- MuSiQue 验证：全量 teacher cache
- score mode：`attn_prob`
- 输出目录：`eval_cd_vs_full_draft_step6000`

| method | split | KL | R@5% | J@5% | R@10% | J@10% | R@15% | J@15% | R@30% | J@30% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CD 4-layer step6000 | wiki | 0.0083 | 0.4096 | 0.2633 | 0.4812 | 0.3216 | 0.5327 | 0.3679 | 0.5966 | 0.4287 |
| CD 4-layer step6000 | musique | 0.0107 | 0.3813 | 0.2538 | 0.3695 | 0.2367 | 0.3732 | 0.2357 | 0.4400 | 0.2882 |
| native layer4 attn_prob e16 | wiki | 0.0077 | 0.5588 | 0.3966 | 0.5679 | 0.4033 | 0.5908 | 0.4246 | 0.6521 | 0.4868 |
| native layer4 attn_prob e16 | musique | 0.0102 | 0.5044 | 0.3525 | 0.4881 | 0.3319 | 0.5091 | 0.3477 | 0.5708 | 0.4031 |
| HS hidden_head e10 | wiki | 0.0023 | 0.6350 | 0.4750 | 0.6796 | 0.5209 | 0.7099 | 0.5551 | 0.7590 | 0.6144 |
| HS hidden_head e10 | musique | 0.0090 | 0.5518 | 0.3950 | 0.6228 | 0.4653 | 0.6283 | 0.4637 | 0.6973 | 0.5403 |

结论：CD causal LM 蒸馏 checkpoint 虽然训练 loss/KL 有下降，但作为 selector 与完整 3B DraftModel 的 top-token overlap 不如原生 layer4 attn_prob，更明显弱于 HS hidden_head。尤其在 MuSiQue 上，R@15 只有 0.3732，说明只做 next-token causal distillation 没有自然学到足够好的重算 token selection 分布；后续若继续这条线，应优先在真实 selector 目标或 hidden-head/attention score 蒸馏上改，而不是仅延长当前 CD 训练。


### 统一对比表：CD vs 原生 layer4 vs HS hidden_head

说明：teacher 均为完整 Qwen2.5-3B DraftModel 缓存的 doc-token score。`R@x%` 表示候选 selector 选出的 top x% token 对 full DraftModel top x% token 的 recall；`J@x%` 表示二者集合 Jaccard。Wiki 使用 5000 条验证样本，MuSiQue 使用全量 teacher cache。

| method | split | KL | R@5% | J@5% | R@10% | J@10% | R@15% | J@15% | R@30% | J@30% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CD 4-layer step6000 | Wiki val | 0.0083 | 0.4096 | 0.2633 | 0.4812 | 0.3216 | 0.5327 | 0.3679 | 0.5966 | 0.4287 |
| Native layer4 attn_prob | Wiki val | 0.0077 | 0.5588 | 0.3966 | 0.5679 | 0.4033 | 0.5908 | 0.4246 | 0.6521 | 0.4868 |
| HS hidden_head | Wiki val | 0.0023 | 0.6350 | 0.4750 | 0.6796 | 0.5209 | 0.7099 | 0.5551 | 0.7590 | 0.6144 |
| CD 4-layer step6000 | MuSiQue val | 0.0107 | 0.3813 | 0.2538 | 0.3695 | 0.2367 | 0.3732 | 0.2357 | 0.4400 | 0.2882 |
| Native layer4 attn_prob | MuSiQue val | 0.0102 | 0.5044 | 0.3525 | 0.4881 | 0.3319 | 0.5091 | 0.3477 | 0.5708 | 0.4031 |
| HS hidden_head | MuSiQue val | 0.0090 | 0.5518 | 0.3950 | 0.6228 | 0.4653 | 0.6283 | 0.4637 | 0.6973 | 0.5403 |

观察：

1. 原生 layer4 明显强于 CD causal-distill：在 MuSiQue 的 R@15 从 0.3732 提升到 0.5091。
2. HS hidden_head 是三者中最强：Wiki R@15=0.7099，MuSiQue R@15=0.6283。
3. CD 的训练 loss/KL 虽然下降，但 selector overlap 没有转化出来，尤其跨到 MuSiQue 后更弱。


### 补充：原参数未微调 layer4 baseline

上一张统一表中 `Native layer4 attn_prob` 使用的是 teacher-score 训练后的 e16 checkpoint。为了避免和“原始前 4 层”混淆，这里补充未微调 baseline。该 baseline 来自 native layer4 训练日志的 epoch=0：模型只加载 Qwen2.5-3B 原始前 4 层，用最后一层 attention probability 作为 doc-token score，没有经过任何 selector 微调。

| method | split | KL | R@5% | J@5% | R@10% | J@10% | R@15% | J@15% | R@30% | J@30% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CD 4-layer step6000 | Wiki val | 0.0083 | 0.4096 | 0.2633 | 0.4812 | 0.3216 | 0.5327 | 0.3679 | 0.5966 | 0.4287 |
| Native layer4 attn_prob, unfinetuned | Wiki val | 0.0078 | 0.5485 | 0.3868 | 0.5582 | 0.3937 | 0.5801 | 0.4135 | 0.6455 | 0.4794 |
| Native layer4 attn_prob, trained e16 | Wiki val | 0.0077 | 0.5588 | 0.3966 | 0.5679 | 0.4033 | 0.5908 | 0.4246 | 0.6521 | 0.4868 |
| HS hidden_head, trained e10 | Wiki val | 0.0023 | 0.6350 | 0.4750 | 0.6796 | 0.5209 | 0.7099 | 0.5551 | 0.7590 | 0.6144 |
| CD 4-layer step6000 | MuSiQue val | 0.0107 | 0.3813 | 0.2538 | 0.3695 | 0.2367 | 0.3732 | 0.2357 | 0.4400 | 0.2882 |
| Native layer4 attn_prob, unfinetuned | MuSiQue val | 0.0101 | 0.5026 | 0.3510 | 0.4828 | 0.3279 | 0.5081 | 0.3465 | 0.5708 | 0.4031 |
| Native layer4 attn_prob, trained e16 | MuSiQue val | 0.0102 | 0.5044 | 0.3525 | 0.4881 | 0.3319 | 0.5091 | 0.3477 | 0.5708 | 0.4031 |
| HS hidden_head, trained e10 | MuSiQue val | 0.0090 | 0.5518 | 0.3950 | 0.6228 | 0.4653 | 0.6283 | 0.4637 | 0.6973 | 0.5403 |

观察：原生 layer4 attn_prob 微调前后差别很小；真正明显提升的是 HS hidden_head。CD causal LM 蒸馏在当前 step6000 下仍低于未微调原生 layer4 baseline，说明它作为 selector 的 ranking 能力没有自然继承到完整 draft model 的 attention selection 行为。


### CD checkpoint sweep：是否只是训练轮数问题

为排查“CD 训练轮数较少导致效果差”的可能性，对 step2000/2500/3000/3500/4000/4500/5000/6000 分别做同一套验证。Wiki 取 1000 条验证样本，MuSiQue 使用全量 teacher cache。teacher 仍为完整 Qwen2.5-3B DraftModel 的 doc-token score。

| checkpoint | Wiki R@15% | Wiki J@15% | MuSiQue R@15% | MuSiQue J@15% |
|---|---:|---:|---:|---:|
| step2000 | 0.5372 | 0.3717 | 0.3630 | 0.2283 |
| step2500 | 0.5367 | 0.3712 | 0.3664 | 0.2308 |
| step3000 | 0.5352 | 0.3698 | 0.3687 | 0.2324 |
| step3500 | 0.5334 | 0.3682 | 0.3696 | 0.2330 |
| step4000 | 0.5324 | 0.3672 | 0.3708 | 0.2339 |
| step4500 | 0.5314 | 0.3663 | 0.3719 | 0.2346 |
| step5000 | 0.5310 | 0.3659 | 0.3731 | 0.2357 |
| step6000 | 0.5309 | 0.3658 | 0.3732 | 0.2357 |

结论：继续做纯 causal LM distillation 并没有把 selector overlap 推高。Wiki 随训练略微下降，MuSiQue 略升但仍远低于未微调原生 layer4 baseline。因此主要问题不是 checkpoint 选早/选晚，而是训练目标和 selector ranking 不对齐。

