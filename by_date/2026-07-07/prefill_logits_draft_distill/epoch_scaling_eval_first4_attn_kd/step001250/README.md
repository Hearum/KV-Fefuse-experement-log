# CD 4-layer vs Full DraftModel Validation

评估目标：复用 HS/native layer4 训练 pipeline 的验证缓存，用完整 3B DraftModel 缓存分数作为 teacher，比较 CD 4-layer causal distillation checkpoint 的 token ranking/selection 与 full draft 的重合度。

- CD checkpoint: `MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/outputs/first4_wikitext_attn_kd_s256_step500/training_state_step001250.pt`
- loaded step/epoch: step=1250, epoch=0
- score mode: `attn_prob`
- wiki val limit: 5000
- musique val limit: all

| split | KL | R@5% | J@5% | R@10% | J@10% | R@15% | J@15% | R@30% | J@30% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wiki | 0.0081 | 0.4048 | 0.2613 | 0.4779 | 0.3196 | 0.5243 | 0.3601 | 0.6054 | 0.4372 |
| musique | 0.0104 | 0.4749 | 0.3259 | 0.4821 | 0.3244 | 0.4953 | 0.3339 | 0.5596 | 0.3925 |
