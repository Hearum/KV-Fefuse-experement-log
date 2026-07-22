# CD 4-layer vs Full DraftModel Validation

评估目标：复用 HS/native layer4 训练 pipeline 的验证缓存，用完整 3B DraftModel 缓存分数作为 teacher，比较 CD 4-layer causal distillation checkpoint 的 token ranking/selection 与 full draft 的重合度。

- CD checkpoint: `MOTIVATION_EXPERIMENTS/causal_lm_distill_draftmodel_local/outputs/qwen_cd_wikitext103_4layer_gpu0_3_s256_n20000_step2000/training_state_final_step002000.pt`
- loaded step/epoch: step=2000, epoch=3
- score mode: `attn_prob`
- wiki val limit: 1000
- musique val limit: all

| split | KL | R@5% | J@5% | R@10% | J@10% | R@15% | J@15% | R@30% | J@30% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wiki | 0.0083 | 0.4093 | 0.2631 | 0.4836 | 0.3237 | 0.5372 | 0.3717 | 0.6057 | 0.4377 |
| musique | 0.0107 | 0.3691 | 0.2447 | 0.3573 | 0.2281 | 0.3630 | 0.2283 | 0.4308 | 0.2809 |
