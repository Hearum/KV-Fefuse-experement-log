# CD 4-layer vs Full DraftModel Validation

评估目标：复用 HS/native layer4 训练 pipeline 的验证缓存，用完整 3B DraftModel 缓存分数作为 teacher，比较 CD 4-layer causal distillation checkpoint 的 token ranking/selection 与 full draft 的重合度。

- CD checkpoint: `MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/outputs/first4_wikitext_attn_kd_s256_step500/training_state_step010000.pt`
- loaded step/epoch: step=10000, epoch=7
- score mode: `attn_prob`
- wiki val limit: 5000
- musique val limit: all

| split | KL | R@5% | J@5% | R@10% | J@10% | R@15% | J@15% | R@30% | J@30% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wiki | 0.0081 | 0.4003 | 0.2574 | 0.4758 | 0.3175 | 0.5228 | 0.3586 | 0.6031 | 0.4348 |
| musique | 0.0104 | 0.4690 | 0.3206 | 0.4815 | 0.3238 | 0.4949 | 0.3335 | 0.5603 | 0.3933 |
