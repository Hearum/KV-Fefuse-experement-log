# CD 4-layer vs Full DraftModel Validation

评估目标：复用 HS/native layer4 训练 pipeline 的验证缓存，用完整 3B DraftModel 缓存分数作为 teacher，比较 CD 4-layer causal distillation checkpoint 的 token ranking/selection 与 full draft 的重合度。

- CD checkpoint: `MOTIVATION_EXPERIMENTS/causal_lm_distill_draftmodel_local/outputs/qwen_cd_wikitext103_4layer_gpu0_3_s256_n20000_step6000/training_state_final_step006000.pt`
- loaded step/epoch: step=6000, epoch=8
- score mode: `attn_prob`
- wiki val limit: 8
- musique val limit: 8

| split | KL | R@5% | J@5% | R@10% | J@10% | R@15% | J@15% | R@30% | J@30% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wiki | 0.0082 | 0.3816 | 0.2411 | 0.4572 | 0.3004 | 0.5373 | 0.3703 | 0.6315 | 0.4643 |
| musique | 0.0120 | 0.3958 | 0.2709 | 0.3565 | 0.2213 | 0.3191 | 0.1908 | 0.3570 | 0.2201 |
