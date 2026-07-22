# 首个可执行 Probe 规格

## Probe 名称

`late_layer_value_readout_predicts_hidden_delta`

## 要回答的问题

对于 document token `i`，如果不重算主模型，只用 `<i` 的 cached Value 做一个轻量 readout：

```text
r_i^l = sum_{j in TopM(i)} a_{ij} * V_j^{l-1}
```

这个 `r_i^l` 是否足以预测 full recompute 带来的后层 hidden/KV delta？

如果不能预测 hidden/KV delta，则后续 surrogate updater 很难成功。

## 数据范围

先用小样本，不抢正在跑的 full hidden 任务：

- dataset：`musique-v2`
- examples：5 或 10
- layers：`56-63`
- token subset：offline / draft score top 15% token，另加随机 token 对照
- top-m readout：`m = 8, 16, 32, 64`

## 需要的张量

对每个 example：

1. `h_old_i^{l-1}`：raw/preprocess cache 对应上下文下 token `i` 的 layer input hidden。
2. `h_full_i^{l-1}`：full prompt 下 token `i` 的 layer input hidden。
3. `V_old_j^{l-1}`：cache 中 `<i` token 的 Value。
4. `K_old_i^l, V_old_i^l`：原 cache K/V。
5. `K_full_i^l, V_full_i^l`：full prompt K/V。
6. token score：offline fixed-set score、draft score 或 FusionRAG attention-derived score。

已有脚本可复用：

- `qwen3_musique_v2_hidden_state_gap/scripts/collect_hidden_gap.py`：可抓 layer input hidden。
- `qwen3_musique_v2_full_vs_cache_kv_gap/scripts/collect_musique_v2_full_vs_cache_gap.py`：可抓 full/cache K/V gap。
- `setup_standard_v2_cross_dataset/scripts/build_setup_v2_draft_score_cache.py`：可生成 DraftModel score cache。
- `setup_standard_v2_cross_dataset/scripts/derive_setup_v2_fixed_sets_from_scores.py`：可从 scores 导出 offline fixed sets。

## 构造 readout 分布

第一阶段不要计算完整 QK attention。只测试 cheap score：

### score-only

```text
a_{ij} = softmax(score_j / tau), j in TopM(<i)
```

### score + distance decay

```text
a_{ij} = softmax(score_j / tau - lambda * log(1 + i - j)), j in TopM(<i)
```

### chunk-boundary template

优先选择同 chunk 内靠近 `i` 的高分 token，再补跨 chunk 高分 token。

## 评价指标

### Hidden delta 可预测性

目标：

```text
delta_h_i^l = h_full_i^l - h_old_i^l
```

评价：

- cosine(`r_i^l`, `delta_h_i^l`)
- ridge regression：`[h_old_i, r_i, score_i] -> PCA(delta_h)` 的 R2
- rank-limited reconstruction：预测 PCA coeff 后 relative L2

### K/V delta 可预测性

目标：

```text
delta_K_i^l = K_full_i^l - K_old_i^l
delta_V_i^l = V_full_i^l - V_old_i^l
```

评价：

- relative L2
- cosine
- explained variance
- ridge R2 / PCA coefficient R2

## 成功判据

进入下一阶段 pipeline ablation 的最低要求：

1. 后层 Value delta 的 PCA coefficient R2 明显高于均值/零预测 baseline。
2. `top_m <= 32` 时仍保留大部分预测性。
3. independent 更新和 block-parallel 更新的几何指标差距不大，说明不用完全 token 串行。
4. blend 版本比 direct 更稳定，且存在一个宽容的 beta 区间。

若这些不成立，不应急着接端到端 pipeline。

## 预计输出

- `results/probe_readout_predictability.csv`
- `results/probe_readout_predictability.json`
- `figures/readout_r2_by_layer.png`
- `figures/readout_topm_tradeoff.png`

## 实现注意

1. 不要保存全数据集大 hidden tensor；先小样本按需生成并聚合。
2. 不要写入共享 preprocess KV cache。
3. 不要中断正在跑的 `qwen3_musique_v2_hidden_state_gap` full dataset 任务。
4. 每次启动都记录 commit、命令、数据路径、cache 路径。
