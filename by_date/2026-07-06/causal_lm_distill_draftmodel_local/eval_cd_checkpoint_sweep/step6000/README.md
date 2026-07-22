# CD 4-layer vs Full DraftModel Validation

评估目标：复用 HS/native layer4 训练 pipeline 的验证缓存，用完整 3B DraftModel 缓存分数作为 teacher，比较 CD 4-layer causal distillation checkpoint 的 token ranking/selection 与 full draft 的重合度。

- CD checkpoint: `MOTIVATION_EXPERIMENTS/causal_lm_distill_draftmodel_local/outputs/qwen_cd_wikitext103_4layer_gpu0_3_s256_n20000_step6000/training_state_final_step006000.pt`
- loaded step/epoch: step=6000, epoch=8
- score mode: `attn_prob`
- wiki val limit: 1000
- musique val limit: all

| split | KL | R@5% | J@5% | R@10% | J@10% | R@15% | J@15% | R@30% | J@30% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wiki | 0.0083 | 0.4087 | 0.2627 | 0.4772 | 0.3181 | 0.5309 | 0.3658 | 0.5997 | 0.4316 |
| musique | 0.0107 | 0.3813 | 0.2538 | 0.3695 | 0.2367 | 0.3732 | 0.2357 | 0.4400 | 0.2882 |
