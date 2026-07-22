# CD 4-layer vs Full DraftModel Validation

评估目标：复用 HS/native layer4 训练 pipeline 的验证缓存，用完整 3B DraftModel 缓存分数作为 teacher，比较 CD 4-layer causal distillation checkpoint 的 token ranking/selection 与 full draft 的重合度。

- CD checkpoint: `MOTIVATION_EXPERIMENTS/causal_lm_distill_draftmodel_local/outputs/qwen_cd_wikitext103_4layer_gpu0_3_s256_n20000_step6000/training_state_step004500.pt`
- loaded step/epoch: step=4500, epoch=6
- score mode: `attn_prob`
- wiki val limit: 1000
- musique val limit: all

| split | KL | R@5% | J@5% | R@10% | J@10% | R@15% | J@15% | R@30% | J@30% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wiki | 0.0083 | 0.4077 | 0.2619 | 0.4778 | 0.3185 | 0.5314 | 0.3663 | 0.6003 | 0.4322 |
| musique | 0.0107 | 0.3802 | 0.2530 | 0.3684 | 0.2360 | 0.3719 | 0.2346 | 0.4390 | 0.2874 |
