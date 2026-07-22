# 实验日志

## 2026-07-16 建立方向

用户问题：能否不用主模型完整重算，而是用已知/离线 attention 或 token score，从 `<i` 的 `V^{l-1}` 加权 readout 出一个 hidden update，再更新 token `i` 的 K/V；并探索 independent vs sequential、direct vs blend 等设计。

当前动作：只建立研究计划，不启动 GPU 实验。qjy000 正在跑 `qwen3_musique_v2_hidden_state_gap` 完整数据集统计，本方向暂不抢占 GPU。

当前 commit：`89bdbb4`

主要结论：

- 这条路线本质是替代主模型 block forward，不是替代 selector。
- 若仍计算完整 QK attention、MLP 或逐 token 全串行，则计算量接近 full recompute，不符合目标。
- 第一阶段应做后层-only、top-m sparse readout、小样本几何 probe，先验证 readout 对 hidden/KV delta 是否有预测性。


补充文档：`FIRST_PROBE_SPEC.md` 已写清楚首个可执行 probe 的输入张量、readout 构造、评价指标和成功判据。


## 2026-07-16 修正为真实 Qwen block 口径

用户提醒：`hidden -> key/value`、`value 加权 -> hidden` 中间有真实模型操作和正则化/归一化，不能用过度简化公式。

已核对 `ktransformers/models/modeling_qwen2.py`：真实层计算是 `input RMSNorm -> q/k/v_proj -> RoPE -> SDPA -> o_proj -> residual -> post-attention RMSNorm -> MLP -> residual`。因此新增 `ACTUAL_MODEL_OPS.md`，后续 surrogate update 以这个计算图为准。

更新后的推荐路线：`score/top-m sparse Value readout -> o_proj -> tiny adapter predicts late-layer hidden correction -> real RMSNorm+Wk/Wv+RoPE projection -> beta blend writeback`。

## 2026-07-16 smoke: late layer value readout probe

commit: 57601b1
machine: qjy003
GPU: 0

启动命令：

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kv_hidden_surrogate_update/scripts/probe_late_layer_value_readout.py \
  --device cuda:0 \
  --max-examples 1 \
  --layers 56-58 \
  --top-ms 16,32 \
  --score-modes uniform,recency,value_norm \
  --max-target-tokens-per-chunk 16 \
  --output-dir MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kv_hidden_surrogate_update/results/smoke_readout_1
```

## 2026-07-16 offline score vs online true score smoke

commit at launch: `0c0b803` plus newly added local script `compare_offline_online_scores.py`，随后会提交文档和脚本。
machine: qjy001
GPU: 0
PID: 1481716

启动命令：

```bash
cd /home/hming/FusionRAG-pca-analysis
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kv_hidden_surrogate_update/scripts/compare_offline_online_scores.py \
  --dataset musique-v2 \
  --max-examples 5 \
  --start-example 0 \
  --device cuda:0 \
  --output-dir MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kv_hidden_surrogate_update/results/offline_online_score_gap_smoke5
```

路径：

- data: `MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/data/musique-v2.jsonl`
- shared preprocess KV: `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2/data/musique-v2-preprocess-10-revert_rope-True/Qwen3-32B`
- offline score cache: `MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/score_cache_full_3b_20260715/musique-v2/score_cache_npz`
- output: `results/offline_online_score_gap_smoke5/`

结果摘要：5 examples 全部长度对齐。offline 原始 score 和 online 原始 score 量级相差很大，因此只用归一化/排序指标判断。rate=0.15 时，`offline3b_mean` Top-k overlap 0.3697，Jaccard 0.2269，NDCG 0.4832；`offline3b_freq_top15` Top-k overlap 0.3707。结论是 offline score 有 coarse prior 价值，但不能直接当作 online true attention 分布。

异常记录：远端 `apply_patch` 对已有文件追加 hunk 不兼容，两次尝试失败且未改坏文件；本段文档使用 Python append 写入。

## 2026-07-16 标记 offline/online score 对比实验设计有误

用户指出刚刚的 `offline score vs online true score` 实验设计有误，需要后续由用户重新指导实验口径。当前处理：

- README 中已把该节标记为“设计有误，暂不作为有效结论使用”。
- `results/offline_online_score_gap_smoke5/` 保留为错误尝试记录，不删除。
- 后续不再引用该节数值作为结论，除非重新设计并复跑。

