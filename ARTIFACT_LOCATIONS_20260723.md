# Artifact 路径迁移记录（2026-07-23）

为保持 submodule 可维护，以下大型文件已从本目录移动到：

/raid/home/hming/experiment-artifacts/kv_refuse/2026-07-23/

迁移内容：

- by_date/2026-07-11/kv_lora/results
- by_date/2026-07-12/structured_kv_adapter/results
- by_date/2026-07-06/native_initialized_layer4_distill/checkpoints
- by_date/2026-07-06/causal_lm_distill_draftmodel_local/outputs
- by_date/2026-07-07/prefill_logits_draft_distill/outputs
- by_date/2026-07-07/selector_aware_draft_model/checkpoints
- by_date/2026-07-05/predictor_distill_wikitext/data
- by_date/2026-07-05/predictor_distill_wikitext/checkpoints
- by_date/2026-07-05/predictor_distill_wikitext/teacher_cache_*

迁移采用同一文件系统 mv 完成，没有删除文件。小型 history、config、CSV、README 等元数据保留在 submodule。实验日志必须引用 artifact 的绝对路径。
