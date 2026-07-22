# CD 4-layer vs Full DraftModel Validation

评估目标：复用 HS/native layer4 训练 pipeline 的验证缓存，用完整 3B DraftModel 缓存分数作为 teacher，比较 CD 4-layer causal distillation checkpoint 的 token ranking/selection 与 full draft 的重合度。

- CD checkpoint: `MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/outputs/first4_wikitext_attn_kd_s256_step500/training_state_final_step000500.pt`
- loaded step/epoch: step=500, epoch=0
- score mode: `attn_prob`
- wiki val limit: 5000
- musique val limit: all

| split | KL | R@5% | J@5% | R@10% | J@10% | R@15% | J@15% | R@30% | J@30% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wiki | 0.0081 | 0.4533 | 0.2996 | 0.4906 | 0.3296 | 0.5280 | 0.3627 | 0.6077 | 0.4392 |
| musique | 0.0105 | 0.4738 | 0.3285 | 0.4399 | 0.2905 | 0.4454 | 0.2918 | 0.5174 | 0.3533 |
