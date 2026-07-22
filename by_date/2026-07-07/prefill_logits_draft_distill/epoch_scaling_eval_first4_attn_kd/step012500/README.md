# CD 4-layer vs Full DraftModel Validation

评估目标：复用 HS/native layer4 训练 pipeline 的验证缓存，用完整 3B DraftModel 缓存分数作为 teacher，比较 CD 4-layer causal distillation checkpoint 的 token ranking/selection 与 full draft 的重合度。

- CD checkpoint: `MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/outputs/first4_wikitext_attn_kd_s256_step500/training_state_final_step012500.pt`
- loaded step/epoch: step=12500, epoch=9
- score mode: `attn_prob`
- wiki val limit: 5000
- musique val limit: all

| split | KL | R@5% | J@5% | R@10% | J@10% | R@15% | J@15% | R@30% | J@30% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wiki | 0.0081 | 0.4003 | 0.2574 | 0.4758 | 0.3175 | 0.5228 | 0.3585 | 0.6032 | 0.4349 |
| musique | 0.0104 | 0.4671 | 0.3187 | 0.4818 | 0.3241 | 0.4949 | 0.3335 | 0.5609 | 0.3938 |
