# CD 4-layer vs Full DraftModel Validation

评估目标：复用 HS/native layer4 训练 pipeline 的验证缓存，用完整 3B DraftModel 缓存分数作为 teacher，比较 CD 4-layer causal distillation checkpoint 的 token ranking/selection 与 full draft 的重合度。

- CD checkpoint: `MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/outputs/first4_wikitext_attn_kd_s256_step500/training_state_step005000.pt`
- loaded step/epoch: step=5000, epoch=3
- score mode: `attn_prob`
- wiki val limit: 5000
- musique val limit: all

| split | KL | R@5% | J@5% | R@10% | J@10% | R@15% | J@15% | R@30% | J@30% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wiki | 0.0081 | 0.3995 | 0.2567 | 0.4752 | 0.3170 | 0.5224 | 0.3582 | 0.6027 | 0.4344 |
| musique | 0.0104 | 0.4689 | 0.3203 | 0.4810 | 0.3234 | 0.4945 | 0.3332 | 0.5607 | 0.3937 |
