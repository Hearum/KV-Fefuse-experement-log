# CD 4-layer vs Full DraftModel Validation

评估目标：复用 HS/native layer4 训练 pipeline 的验证缓存，用完整 3B DraftModel 缓存分数作为 teacher，比较 CD 4-layer causal distillation checkpoint 的 token ranking/selection 与 full draft 的重合度。

- CD checkpoint: `MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/outputs/first4_wikitext_attn_kd_s256_step500/training_state_step007500.pt`
- loaded step/epoch: step=7500, epoch=5
- score mode: `attn_prob`
- wiki val limit: 5000
- musique val limit: all

| split | KL | R@5% | J@5% | R@10% | J@10% | R@15% | J@15% | R@30% | J@30% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wiki | 0.0081 | 0.4008 | 0.2577 | 0.4761 | 0.3178 | 0.5228 | 0.3586 | 0.6033 | 0.4350 |
| musique | 0.0104 | 0.4698 | 0.3212 | 0.4812 | 0.3237 | 0.4952 | 0.3337 | 0.5605 | 0.3935 |
