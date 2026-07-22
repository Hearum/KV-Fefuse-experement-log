# CD 4-layer vs Full DraftModel Validation

评估目标：复用 HS/native layer4 训练 pipeline 的验证缓存，用完整 3B DraftModel 缓存分数作为 teacher，比较 CD 4-layer causal distillation checkpoint 的 token ranking/selection 与 full draft 的重合度。

- CD checkpoint: `MOTIVATION_EXPERIMENTS/causal_lm_distill_draftmodel_local/outputs/qwen_cd_wikitext103_4layer_gpu0_3_s256_n20000_step6000/training_state_step003000.pt`
- loaded step/epoch: step=3000, epoch=4
- score mode: `attn_prob`
- wiki val limit: 1000
- musique val limit: all

| split | KL | R@5% | J@5% | R@10% | J@10% | R@15% | J@15% | R@30% | J@30% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wiki | 0.0083 | 0.4083 | 0.2623 | 0.4812 | 0.3215 | 0.5352 | 0.3698 | 0.6026 | 0.4345 |
| musique | 0.0107 | 0.3760 | 0.2494 | 0.3652 | 0.2337 | 0.3687 | 0.2324 | 0.4349 | 0.2842 |
