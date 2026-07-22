# Experiment Log

## 2026-07-12：新研究线建立

### 目的

将“用 `Wk'/Wv'` 直接预测 selected document token KV 更新”从概念拆成可证伪实验，独立于旧的 static mean bias、query-prefix 反事实和小样本线性 Adapter 结果。

### 已完成

- 建立 `README.md`、`PLAN.md`、`scripts/`、`results/`、`figures/`。
- 固定 formal target 为 `rate1 full recompute - loaded raw/preprocess KV`。
- 定义 token-local、prefix-conditioned、低维层间递推三类 Adapter。
- 定义严格 main-example-disjoint split、表示上界准入条件和端到端可用标准。

### 本轮命令

```bash
ssh qjy000
cd /raid/home/hming/FusionRAG-pca-analysis
sed -n '1,240p' AGENT.md
sed -n '1,280p' MOTIVATION_EXPERIMENTS/kv_lora/README.md
tail -220 MOTIVATION_EXPERIMENTS/kv_lora/goal.md
```

任务中提到的 `MOTIVATION_EXPERIMENTS/kv_lora/READNE.md` 不存在；实际读取的是该目录 `README.md`。

### 当前结论

尚未启动新实验。现有结果只足以排除“无 context、独立 per-layer/head 的 cached-KV affine mapping”作为可靠方案，不能排除 prefix-conditioned 或带低维层间 state 的 Adapter。下一轮从 Stage 0 target audit 开始。

## 2026-07-12：Stage 0 Target Audit

### 命令

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/audit_formal_targets.py \
  --audit-examples 0 1
```

### 结果

- 2 个 residual 文件、20 个 document span 全部通过 shape、prefix offset、finite value 和 cache 文件对应检查。
- Tensor contract：`[64,1,8,tokens,128]`。
- 数据共有 67 个 subquery 文件，但仅覆盖 36 个 main examples：24 train、7 validation、5 test。
- preprocess 起点首 document mean original gap：K `0.2395`、V `0.3871`；后续 document：K `0.2956`、V `0.6044`。

首文档不接近零并非因果性错误：这里的 base 是 BGE preprocess cache，而 target 是 formal rate1 full recompute。BGE preprocess 本身已经改变 KV，因此不能复用 raw-cache 场景“首文档应为数值零”的断言。后续应额外采集 raw paired target 才能执行该 sanity。

产物：

- `results/stage0_manifest.json`
- `results/stage0_audit.json`

## 2026-07-12：Stage 1 Train-only Feature Basis 上界

### 设计

只使用 main example `<30` 学习每层、每 KV head 的 Delta feature PCA basis；example 30--39 不参与；example `>=40` 严格测试。训练时每 document 均匀采样最多 32 token，测试使用全部 token。test 允许用真实 Delta 做 oracle projection，因此本轮只测试 basis 表示能力，不代表可部署 predictor 效果。

### 命令

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/stage1_feature_basis.py
```

测试覆盖 5 个 main examples、110 个 document instances。

| Kind | Rank | Remaining Delta | Explained Energy | Original Gap | Final KV Error | Delta Cosine |
|---|---:|---:|---:|---:|---:|---:|
| K | 8 | 0.7143 | 48.97% | 0.3664 | 0.2617 | 0.6998 |
| K | 32 | 0.4931 | 75.68% | 0.3664 | 0.1807 | 0.8700 |
| K | 64 | 0.3340 | 88.85% | 0.3664 | 0.1224 | 0.9426 |
| V | 8 | 0.8920 | 20.44% | 0.7804 | 0.6961 | 0.4521 |
| V | 32 | 0.7503 | 43.70% | 0.7804 | 0.5855 | 0.6611 |
| V | 64 | 0.5789 | 66.49% | 0.7804 | 0.4518 | 0.8154 |

### 结论

1. Value 是瓶颈。rank8/16 的 shared feature basis 远不足以替代 full recompute。
2. rank64 对 Value 只回收 `1-0.5789=42.1%` 的 Delta gap，刚超过计划中的 40% 表示准入线；这还是 oracle coefficient 上界。
3. Key 明显更可压缩，rank32 已解释 75.7% 能量，但不能据此忽略 Value 或独立优化 K/V 后直接拼接。
4. rank128 近似零误差只是 128 维 head feature 的完整正交基，不具有低秩意义。
5. 下一轮不训练 rank8/16。只以 rank64 测试 coefficient predictability，并比较 token-local、cached-prefix summary 与 oracle updated-prefix summary；若可预测结果远低于 oracle，转向 chunk-serial/layer-recurrent Adapter。

产物：`results/stage1_feature_basis.json`。

## 2026-07-12：Stage 2 Rank64 Coefficient Predictability Ladder v1

### 设计

固定 Stage 1 的 train-only rank64 basis。测试阶段禁止用真实 Delta 求 coefficient。每层、每 head 使用 Ridge (`alpha=1e-3`) 比较：

- `mean`：固定训练集 Delta mean；
- `position`：token/doc position、prefix token 数、document 长度；
- `token_kv_position`：当前目标 token 的 cached K+V 加位置；
- `token_prefix_kv_position`：再加入当前 document 之前所有 cached K/V 的逐层逐 head mean；
- `oracle_rank64`：使用 test Delta 投影，只作表示上界。

### 命令

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/stage2_predictability_ladder.py
```

| Kind | Method | Remaining Delta | Explained Energy | Final KV Error | Cosine |
|---|---|---:|---:|---:|---:|
| K | mean | 0.9177 | 15.79% | 0.3362 | 0.3973 |
| K | oracle rank64 | 0.3340 | 88.85% | 0.1224 | 0.9426 |
| K | position | 0.9113 | 16.95% | 0.3339 | 0.4119 |
| K | token KV + position | 0.7858 | 38.26% | 0.2879 | 0.6210 |
| K | token + prefix KV + position | 0.8383 | 29.73% | 0.3071 | 0.5740 |
| V | mean | 0.9822 | 3.52% | 0.7665 | 0.1889 |
| V | oracle rank64 | 0.5789 | 66.49% | 0.4518 | 0.8154 |
| V | position | 0.9819 | 3.59% | 0.7663 | 0.1919 |
| V | token KV + position | 0.9212 | 15.14% | 0.7189 | 0.4012 |
| V | token + prefix KV + position | 0.9821 | 3.55% | 0.7664 | 0.3329 |

### 解释

1. 当前 token cached K/V 确实包含 Delta 信息，K/V 均优于 mean，但离 oracle basis 上界很远；简单 `W'(cached KV)` 不足以替代重算。
2. 直接拼接 256 维 per-head prefix K/V mean 后 test 变差。当前不能解释为“prefix 无关”，因为该版本没有用 validation 选择 Ridge alpha，且 24 个 train examples 对 516 维输入容易过拟合。
3. Value 的预测瓶颈远强于 Key：可部署线性模型只解释 15.1% Value Delta 能量，而 basis 上界是 66.5%。
4. 下一轮先将 prefix summary 用 train-only PCA/随机投影压到 8/16/32 维，并在 7 个 validation examples 上选择 Ridge alpha；如果仍不超过 token-local，再测试 updated-prefix oracle summary 与 layer-recurrent state。

产物：`results/stage2_predictability_ladder.json`。

## 2026-07-12：扩展到 73 个独立 Main Examples

### 数据采集

为了检验旧结果是否由 24 个 train examples 导致，在 qjy000 的 8 张 H20 上使用 Qwen3-32B 分片采集 examples 50--99。输出到新目录，不覆盖旧 residual/cache：

```bash
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python \
    MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/collect_formal_residual_range.py \
    --shard $i --num-shards 8 --start-example 50 --end-example 100 \
    --output-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/formal_residual_50_100
done
```

新增 71 个 subquery residual 文件、37 个独立 main examples；8 shards 无 traceback。与旧数据合并后共有 73 个独立 main examples。新 frozen split：`example<70` train（51）、`70<=example<85` validation（11）、`example>=85` test（11）。严格 test 包含 220 个 document instances。

### 扩大后的 Rank64 表示上界

| Kind | Rank | Explained Energy | Remaining Delta | Original Gap | Final KV Error |
|---|---:|---:|---:|---:|---:|
| K | 32 | 75.55% | 0.4945 | 0.3783 | 0.1871 |
| K | 64 | 88.82% | 0.3344 | 0.3783 | 0.1265 |
| V | 32 | 43.86% | 0.7493 | 0.8123 | 0.6086 |
| V | 64 | 66.62% | 0.5777 | 0.8123 | 0.4693 |

与旧 24-train/5-test 的 K/V rank64 `88.85%/66.49%` 几乎相同，证明表示上界结论不是小测试集波动。

### 扩大后的 Predictability

| Kind | Method | Explained Energy | Remaining Delta | Final KV Error |
|---|---|---:|---:|---:|
| K | mean | 15.98% | 0.9166 | 0.3468 |
| K | token KV + position | 40.73% | 0.7698 | 0.2913 |
| K | token + prefix KV + position | 38.30% | 0.7855 | 0.2972 |
| K | oracle rank64 | 88.82% | 0.3344 | 0.1265 |
| V | mean | 3.86% | 0.9805 | 0.7965 |
| V | token KV + position | 18.09% | 0.9050 | 0.7352 |
| V | token + prefix KV + position | 15.78% | 0.9177 | 0.7455 |
| V | oracle rank64 | 66.62% | 0.5777 | 0.4693 |

### 结论

1. 训练 main examples 从24增至51后，token-local K/V分别提高约`2.5/3.0`个百分点，更多独立context确有帮助，但不能解释与oracle之间的巨大差距。
2. 高维 prefix mean 的退化明显缩小，却仍低于 token-local；问题不是简单增加训练轮数可以解决。Ridge为闭式解，本身没有epoch不足问题。
3. 当前证据支持训练真正的非线性 context encoder，但输入必须先结构化降维。下一轮在 validation 上选择 prefix bottleneck维度和正则，然后训练共享两层MLP；不应为64层×8头各训练一个数据不足的独立MLP。

产物：

- `results/formal_residual_50_100/`
- `results/stage1_feature_basis_73examples.json`
- `results/stage2_predictability_ladder_73examples.json`

## 2026-07-12：共享两层 MLP Predictor

### 架构与训练

- 共享网络参数量：199,008。
- 输入：当前 token cached K/V；前序全部 docs K/V mean；最后一个前序 doc K/V mean；token/doc position；layer/head embedding。
- Prefix encoder bottleneck：32；输出：K/V 各 rank64 coefficient。
- train-only coefficient 标准化；AdamW，学习率 `2e-3`，batch 2048，最多40 epochs，validation patience 6。
- 数据：51 train main examples、60,800 sampled rows；11 validation、13,440 rows；11 strict test、220 document instances全token评估。

命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/train_shared_mlp_predictor.py
```

训练曲线：

| Epoch | Train MSE | Validation MSE |
|---:|---:|---:|
| 1 | 0.9936 | **0.8551** |
| 3 | 0.9901 | 0.8562 |
| 5 | 0.9854 | 0.8577 |
| 7 | 0.9796 | 0.8603 |

validation 从第一轮后持续变差，early stopping 选择 epoch 1。继续增加 epoch 只会加重过拟合。

严格 test：

| Kind | Method | Explained Energy | Remaining Delta | Final KV Error |
|---|---|---:|---:|---:|
| K | mean | 15.98% | 0.9166 | 0.3468 |
| K | shared MLP | 16.03% | 0.9164 | 0.3467 |
| K | token-local Ridge | 40.73% | 0.7698 | 0.2913 |

## 2026-07-12：Depth/Residual Recurrent Adapter probe

### 目的

回答最新问题：`l` 层 `V` 来自 `l-1` 层 hidden state，而 `hidden state` 又受到上一层 Value 加权平均影响，那么 `DeltaV^l` 是否应该由 `DeltaV^{l-1}` 或上一层低秩 Value-update state 预测？

本轮只做小型可证伪 probe，不训练大 adaptor。使用已有 Qwen3-8B Wiki `prefix256+target128` layer-update audit 64条样本，固定 48 train / 16 heldout test。每层单独用 PCA(64) 压缩输入，再用 Ridge 预测该层整 token 的 `DeltaK^l` / `DeltaV^l`。

### 初始计划命令

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/analyze_qwen3_8b_delta_recurrence.py \
  --data-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/layer_update_audit_64_stride384 \
  --train-count 48 \
  --feature-rank 64 \
  --alpha 1e-2 \
  --output-json MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/delta_recurrence_probe_48train_16test_rank64.json
```

### 判读

- `prev_dv` 强：只维护上一层 Value delta state 可能足够。
- `prev_h` / `prev_attn` 强但 `prev_dv` 弱：需要 residual/hidden side state 或轻量 hidden adaptor。
- `oracle_current_h` 强但上一层特征弱：`Wv(DeltaH)` 路径成立，但难点是跨层状态转移。
- 所有特征弱：继续做独立 KV L2 predictor 价值低，应优先 functional/logit 监督或 link/adapter tokens。

### 已运行命令

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/analyze_qwen3_8b_delta_recurrence.py \
  --data-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/layer_update_audit_64_stride384 \
  --train-count 48 \
  --feature-rank 64 \
  --alpha 1e-2 \
  --output-json MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/delta_recurrence_probe_48train_16test_rank64.json

CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/analyze_qwen3_8b_delta_recurrence.py \
  --data-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/layer_update_audit_64_stride384 \
  --train-count 48 \
  --feature-rank 256 \
  --alpha 1e-2 \
  --output-json MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/delta_recurrence_probe_48train_16test_rank256.json
```

### 结果

| Target | Feature | Rank64 explained | Rank256 explained |
|---|---|---:|---:|
| K | mean template | 15.19% | 15.19% |
| K | `DeltaV^{l-1}` | 62.02% | 58.87% |
| K | `DeltaK/V^{l-1}` | 70.09% | 68.20% |
| K | `DeltaAttnOut^{l-1}` | 55.77% | 52.90% |
| K | `DeltaH^{l-1}` | 70.07% | 72.15% |
| K | `DeltaH/Attn/V^{l-1}` | 69.80% | 72.68% |
| K | oracle `DeltaH^l` | 73.59% | 77.79% |
| V | mean template | 2.33% | 2.33% |
| V | `DeltaV^{l-1}` | 18.65% | 6.94% |
| V | `DeltaK/V^{l-1}` | 16.35% | 4.56% |
| V | `DeltaAttnOut^{l-1}` | 11.48% | -2.62% |
| V | `DeltaH^{l-1}` | 17.90% | 22.78% |
| V | `DeltaH/Attn/V^{l-1}` | 20.86% | 25.15% |
| V | oracle `DeltaH^l` | 24.00% | 37.22% |

### 结论

1. Key 有明显 depth-recurrent 结构。上一层 `DeltaK/V`、`DeltaH` 或合并状态可以把 `DeltaK^l` 的 heldout explained energy 提高到约 `68%--73%`，远高于 mean template 的 `15.19%`。
2. Value 不支持“只靠上一层 Value delta 递推”。`DeltaV^{l-1}` 对 `DeltaV^l` 只有 `18.65%`，rank256 反而降到 `6.94%`；这更像小样本高维过拟合，而不是稳定可用的低秩递推规律。
3. `DeltaH^{l-1}` / `DeltaH+Attn+V` 对 Value 略强，但也只有 `22.78%/25.15%`。这说明 Value 更新依赖 residual/hidden trajectory，单独的 V tensor 不是充分状态。
4. oracle `DeltaH^l` 对 Value 到 `37.22%`，说明 `Wv(DeltaH)` 方向仍有价值；但下一步必须显式建模或缓存 hidden/normalized-hidden correction，而不是继续用独立 KV coefficient MSE。
5. 实验含义：下一轮优先做 hidden-sidecar adaptor 上界，或在真实 RAG 上走 K-first functional validation。继续扩大“独立 Value PCA coefficient predictor”不是最高价值路线。

## 2026-07-12：Hidden-sidecar static probe

### 目的

验证一个最小假设：如果离线为 document token 缓存低维 hidden sidecar，是否可以不看前缀上下文，直接预测 `DeltaH` 或 `DeltaV`。这对应“preprocess v2 静态 bias”的最乐观版本。

### 脚本

- `scripts/analyze_qwen3_8b_hidden_sidecar_probe.py`

### 命令

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/analyze_qwen3_8b_hidden_sidecar_probe.py \
  --data-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/hidden_vproj_audit_96_stride384 \
  --train-count 64 \
  --ranks 32 64 128 256 \
  --alpha 1e-2 \
  --output-json MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/hidden_sidecar_probe_64train_32test.json
```

### 结果

原始 summary 中 `DeltaH` 的 best layer 为0且 explained=100%，这是因为 layer0 的 hidden input delta 理论上为0。为避免误读，下面报告排除 layer0 后的平均 explained energy。

| Target | Baseline mean | Sidecar rank32 | Sidecar rank64 | Sidecar rank128 | Sidecar rank256 |
|---|---:|---:|---:|---:|---:|
| DeltaH | 12.32% | 35.29% | 34.89% | 33.99% | 31.53% |
| DeltaV | 2.60% | 8.33% | 7.51% | 6.04% | 2.35% |

### 结论

1. 静态 offline `own_h` sidecar 对 `DeltaH` 有弱到中等信号，rank32 约 `35.29%`，但提高rank不增益，说明样本量和上下文缺失很快成为瓶颈。
2. 对 `DeltaV`，静态 sidecar 基本不可用：rank32 只有 `8.33%`，rank256 降到 `2.35%`。
3. 这不支持“每个document离线预存一个固定bias，然后online直接加上去”的 preprocess v2 方案。
4. sidecar 仍可能有价值，但必须作为 online adaptor 的输入，与前序 context/prefix summary 共同预测 `DeltaH` 或通过 functional loss 训练。

### 产物

- `results/qwen3_8b/hidden_sidecar_probe_64train_32test.json`

## 2026-07-12：Context + hidden-sidecar joined probe

### 目的

静态 `own_h` sidecar 对 `DeltaV` 很弱，因此进一步测试它是否需要 online prefix context 才有用。本轮不重新采样，把两个已有数据源按 sample id join：

- `hidden_vproj_audit_96_stride384`：`own_h`、`DeltaH`、`DeltaV`
- `wikitext_delta_train2k_stride384`：`own_kv`、`prefix_features`、`DeltaKV`

join 检查了 `sample`、`token_start`、`prefix_tokens`、`target_tokens` 和 `sampled_positions`，字段一致。

### 脚本

- `scripts/analyze_qwen3_8b_context_sidecar_probe.py`

### 命令

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/analyze_qwen3_8b_context_sidecar_probe.py \
  --hidden-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/hidden_vproj_audit_96_stride384 \
  --delta-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_delta_train2k_stride384 \
  --train-count 64 \
  --ranks 32 64 128 \
  --alpha 1e-2 \
  --output-json MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/context_sidecar_probe_64train_32test.json
```

### 结果

排除 layer0 后的 heldout explained energy：

| Target | Feature | Rank32 | Rank64 | Rank128 |
|---|---|---:|---:|---:|
| DeltaH | mean template | 12.32% | 12.32% | 12.32% |
| DeltaH | own_h | 35.29% | 34.89% | 33.99% |
| DeltaH | prefix | 7.55% | 3.42% | 3.42% |
| DeltaH | own_h + prefix | 30.48% | 31.50% | 30.60% |
| DeltaH | own_kv + prefix | 34.34% | 33.06% | 30.24% |
| DeltaH | own_h + own_kv + prefix | 34.73% | 33.60% | 31.33% |
| DeltaV | mean template | 2.60% | 2.60% | 2.60% |
| DeltaV | own_h | 8.33% | 7.51% | 6.04% |
| DeltaV | prefix | -3.22% | -8.04% | -8.04% |
| DeltaV | own_h + prefix | 4.37% | 3.43% | 1.40% |
| DeltaV | own_kv + prefix | 7.39% | 5.51% | 1.58% |
| DeltaV | own_h + own_kv + prefix | 7.52% | 5.87% | 2.60% |

### 结论

1. 粗 `prefix_features` 不是有效上下文编码。它单独预测 `DeltaV` 为负 explained energy，加入 `own_h` / `own_kv` 后也没有超过静态 `own_h`。
2. 这说明“hidden sidecar + prefix mean/Ridge”不是可用路线；要继续做 Value，需要 token-level context encoder、attention-output/logit supervision，或小型 student Transformer。
3. `own_kv + prefix` 在浅层 best layer 可到约35% `DeltaV` explained，但全层平均只有7%左右，不能作为系统方案。
4. 当前 evidence 更支持 K-first formal validation：先验证修K在真实RAG端到端是否能接近 full rate=1；Value 分支另行设计功能性目标。

### 产物

- `results/qwen3_8b/context_sidecar_probe_64train_32test.json`

## 2026-07-12：K-first formal RAG smoke

### 目的

在真实 FusionRAG formal pipeline 中验证 K-first 假设：如果所有 document token 都重算，但只写回 Key，端到端 answer 是否接近写回完整 K/V。

### 关键实现细节

不能使用 `rate=1`，因为外层在 `rate == 1` 时直接 full prefill，不走 cache/reprocess update mode。这里使用：

```bash
FUSIONRAG_FORCE_ALL_REPROCESS=1
FUSIONRAG_REPROCESS_UPDATE_MODE={kv,k_only,v_only,none}
--rate 0.01
```

`rate` 只负责进入 `load_kv_and_generate` 分支；实际 selected doc tokens 由 `FUSIONRAG_FORCE_ALL_REPROCESS=1` 改成全部 document tokens。

### 待运行命令

公共参数：

```bash
--model_type qwen3
--model_path /mnt/qjhs-sh-lab-01/models/Qwen3-32B
--model_name Qwen3-32B
--data_path ./data/result_reflect.json
--dataset_name musique
--cache_path /raid/home/hming/fusionrag-reflect-qwen3-full-cache
--result_path /raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/formal_kfirst_smoke
--bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3
--topk 10
--recall_method bge
--revert_rope true
--max_samples 42
--start_sample 40
--end_sample 42
--max_cache_len 8192
--max_new_tokens 32
--reprocess_method FusionRAG
--rate 0.01
--preprocess true
```

运行 `kv/k_only/v_only/none` 四个模式。每个模式都写入独立 result_path 子目录，由环境变量记录在日志中。

### 启动异常与修正

第一次启动失败：远端 shell 是 zsh，`common="..."` 标量没有按空格拆分，导致 argparse 把整串参数视为一个 unknown argument。已改为 `bash -lc` 数组。

第二次启动失败：脚本没有 `--max_new_tokens` 参数，已去掉该参数。

第三次启动可运行，但发现四个模式写到同一个 `result_path/config_dir`，可能混写 CSV。已主动停止该轮，改为每个 mode 单独 `result_path`。

### Smoke: example 40--41

命令使用独立目录：

```bash
FUSIONRAG_FORCE_ALL_REPROCESS=1
FUSIONRAG_REPROCESS_UPDATE_MODE={kv,k_only,v_only,none}
--rate 0.01
--max_samples 42 --start_sample 40 --end_sample 42
```

结果：只有 1 个可测试 main example、2 个 subquestions。四种模式全部 `Main 1/1`、`Sub 2/2`，且预测完全相同。该 smoke 只能证明路径可运行，不能判断 K-only。

日志确认 force-all 生效：

```text
force_all_reprocess: selected 1089 doc tokens
force_all_reprocess: selected 1515 doc tokens
```

### Expanded smoke: example 40--49

命令范围：

```bash
--max_samples 50 --start_sample 40 --end_sample 50
```

补跑两个基线：

```bash
preprocess_rate0: --rate 0 --preprocess true
full_rate1:      --rate 1 --preprocess true
```

结果：

| Method | Main Acc | Sub Acc | Avg F1 | Avg EM | Same pred as full rate1 |
|---|---:|---:|---:|---:|---:|
| preprocess_rate0 | 2/5 = 40.00% | 8/11 = 72.73% | 0.4982 | 0.1818 | 3/11 |
| none | 2/5 = 40.00% | 8/11 = 72.73% | 0.3394 | 0.0909 | 5/11 |
| v_only | 2/5 = 40.00% | 7/11 = 63.64% | 0.4833 | 0.1818 | 6/11 |
| kv | 3/5 = 60.00% | 9/11 = 81.82% | 0.4199 | 0.1818 | 11/11 |
| full_rate1 | 3/5 = 60.00% | 9/11 = 81.82% | 0.4199 | 0.1818 | 11/11 |
| k_only | 4/5 = 80.00% | 9/11 = 81.82% | 0.4593 | 0.1818 | 6/11 |

### 结论

1. `kv` 与 `full_rate1` 的 11/11 predictions 完全一致，证明 cache-path force-all KV reprocess 是可信的 full recompute 对照。
2. `k_only` 与 full rate1 的 sub accuracy 相同（9/11），小样本 main accuracy 更高（4/5 vs 3/5）。这不能说明 K-only 优于 full，但说明 K-only 没有明显退化，值得扩大。
3. `v_only` 低于 full/K-only，支持此前 functional ablation：只修 V 而 K/attention weights 仍错，会带来不稳定甚至负收益。
4. `none` 与 preprocess rate0 sub accuracy 相同，说明“重算但不写回”的 ablation 没有凭空提升结果。
5. 下一步应扩大到更多 testable examples；如果 K-only 继续接近 full，Adapter 主线应优先预测 RoPE-aligned Key 或 attention-logit correction。

### 产物

- `results/formal_kfirst_smoke/`
- `results/formal_kfirst_eval_40_50/formal_kfirst_eval_40_50_summary.json`
- `results/formal_kfirst_eval_40_50/formal_kfirst_eval_40_50_row_diffs.json`

## 2026-07-12：K-first formal RAG 扩展到 examples 40--99

### 目的

上一轮 `40--49` 只有 5 个可测试 main examples，K-only 看起来不差但样本太小。本轮扩大到 `example 40--99`，目标是至少覆盖几十个 testable examples，重新判断 K-only 是否真能接近 full recompute。

### 命令

六个方法分 GPU 0--5 并行：

```bash
base=MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/formal_kfirst_eval_40_100
common_base=(
  --model_type qwen3
  --model_path /mnt/qjhs-sh-lab-01/models/Qwen3-32B
  --model_name Qwen3-32B
  --data_path ./data/result_reflect.json
  --dataset_name musique
  --cache_path /raid/home/hming/fusionrag-reflect-qwen3-full-cache
  --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3
  --topk 10
  --recall_method bge
  --revert_rope true
  --max_samples 100
  --start_sample 40
  --end_sample 100
  --max_cache_len 8192
  --reprocess_method FusionRAG
  --preprocess true
)

# preprocess baseline
CUDA_VISIBLE_DEVICES=0 python test_fusionrag_reflect_preprocess_exp.py \
  "${common_base[@]}" \
  --result_path /raid/home/hming/FusionRAG-pca-analysis/$base/preprocess_rate0 \
  --rate 0

# full prefill baseline
CUDA_VISIBLE_DEVICES=1 python test_fusionrag_reflect_preprocess_exp.py \
  "${common_base[@]}" \
  --result_path /raid/home/hming/FusionRAG-pca-analysis/$base/full_rate1 \
  --rate 1

# cache-path force-all reprocess ablations
FUSIONRAG_FORCE_ALL_REPROCESS=1 FUSIONRAG_REPROCESS_UPDATE_MODE=kv     CUDA_VISIBLE_DEVICES=2 python ... --rate 0.01
FUSIONRAG_FORCE_ALL_REPROCESS=1 FUSIONRAG_REPROCESS_UPDATE_MODE=k_only CUDA_VISIBLE_DEVICES=3 python ... --rate 0.01
FUSIONRAG_FORCE_ALL_REPROCESS=1 FUSIONRAG_REPROCESS_UPDATE_MODE=v_only CUDA_VISIBLE_DEVICES=4 python ... --rate 0.01
FUSIONRAG_FORCE_ALL_REPROCESS=1 FUSIONRAG_REPROCESS_UPDATE_MODE=none   CUDA_VISIBLE_DEVICES=5 python ... --rate 0.01
```

实际 pid：

```text
preprocess_rate0 3431198
full_rate1       3431201
kv               3431204
k_only           3431207
v_only           3431210
none             3431213
```

### 结果

覆盖 `42` 个可测试 main examples、`82` 个 subquestions。

| Method | Main Acc | Sub Acc | Avg F1 | Avg EM | Same pred as full rate1 |
|---|---:|---:|---:|---:|---:|
| preprocess_rate0 | 25/42 = 59.52% | 62/82 = 75.61% | 0.5921 | 0.2195 | 28/82 |
| none | 28/42 = 66.67% | 65/82 = 79.27% | 0.4762 | 0.1585 | 48/82 |
| v_only | 21/42 = 50.00% | 56/82 = 68.29% | 0.4775 | 0.1585 | 43/82 |
| k_only | 29/42 = 69.05% | 68/82 = 82.93% | 0.4942 | 0.1707 | 54/82 |
| kv | 37/42 = 88.10% | 77/82 = 93.90% | 0.5283 | 0.1829 | 80/82 |
| full_rate1 | 37/42 = 88.10% | 77/82 = 93.90% | 0.5357 | 0.1829 | 82/82 |

### 结论

1. `kv` 与 `full_rate1` correctness 完全一致（82/82），预测文本 80/82 一致，证明 force-all cache-path reprocess 是可靠对照。
2. `k_only` 比 preprocess rate0 和 none 更好，但离 full/kv 还有明显差距：`82.93%` vs `93.90%` sub accuracy。
3. 这推翻了小样本中“K-only 可能接近 full”的乐观解释。K-only 是有效中间方案，不是完整替代。
4. `v_only` 比所有主要基线都差，说明只修 Value 不可取。
5. 下一步方法应是 **K/logit-first + functional Value correction**：先预测 Key 或 attention logits，再用 attention-output/logit loss 训练轻量 Value/adapter-token 补偿；不要继续单独做 Value L2 coefficient regression。

### 产物

- `results/formal_kfirst_eval_40_100/formal_kfirst_eval_40_100_summary.json`
- `results/formal_kfirst_eval_40_100/formal_kfirst_eval_40_100_row_diffs.json`

## 2026-07-12：Partial Value layer ablation

### 目的

K-only 在 40--99 上优于 preprocess/no-update，但低于 full KV。为了判断 Value correction 是否必须覆盖所有层，本轮测试：

```text
Key: 全层写回重算结果
Value: 只保留部分层的重算结果，其余层恢复 preprocess cache
```

### 代码开关

在 `ktransformers/util/utils.py` 中新增环境变量，仅对 `FUSIONRAG_REPROCESS_UPDATE_MODE=k_only` 生效：

```text
FUSIONRAG_REPROCESS_KEEP_VALUE_LAYERS=last8 / last16 / last32 / all
```

默认不设置时行为不变，仍为原始 K-only。

### 小范围 sanity：example 40--49

命令：

```bash
FUSIONRAG_FORCE_ALL_REPROCESS=1
FUSIONRAG_REPROCESS_UPDATE_MODE=k_only
FUSIONRAG_REPROCESS_KEEP_VALUE_LAYERS={last8,last16,last32,all}
--rate 0.01
--max_samples 50 --start_sample 40 --end_sample 50
```

结果：

| Method | Main Acc | Sub Acc | Avg F1 |
|---|---:|---:|---:|
| k + V last8 | 4/5 = 80.00% | 9/11 = 81.82% | 0.4562 |
| k + V last16 | 4/5 = 80.00% | 9/11 = 81.82% | 0.4731 |
| k + V last32 | 3/5 = 60.00% | 9/11 = 81.82% | 0.4199 |
| k + V all | 3/5 = 60.00% | 9/11 = 81.82% | 0.4199 |

`all` 与 full/kv 小样本结果一致，说明新开关语义正确。

### 扩展：example 40--99

命令：

```bash
FUSIONRAG_FORCE_ALL_REPROCESS=1
FUSIONRAG_REPROCESS_UPDATE_MODE=k_only
FUSIONRAG_REPROCESS_KEEP_VALUE_LAYERS={last8,last16,last32}
--rate 0.01
--max_samples 100 --start_sample 40 --end_sample 100
```

结果：

| Method | Value layers kept | Main Acc | Sub Acc | Avg F1 | Same pred as full |
|---|---:|---:|---:|---:|---:|
| k_only | 0 | 29/42 = 69.05% | 68/82 = 82.93% | 0.4942 | 54/82 |
| k + V last8 | 8 | 29/42 = 69.05% | 67/82 = 81.71% | 0.4930 | 52/82 |
| k + V last16 | 16 | 34/42 = 80.95% | 72/82 = 87.80% | 0.4955 | 53/82 |
| k + V last32 | 32 | 36/42 = 85.71% | 74/82 = 90.24% | 0.5302 | 69/82 |
| kv / full_rate1 | 64 | 37/42 = 88.10% | 77/82 = 93.90% | 0.5357 | 80--82/82 |

### 结论

1. Value 的有效修正集中在中后层，而不是最后8层。
2. `last16` 已经明显回收 K-only 到 full 之间的一部分差距；`last32` 接近 full，但仍缺 3 个 subquestions。
3. 未来 Value adaptor 不应全层同等建模。更合理的是只对后16--32层做 functional Value correction。
4. 结合前面结果，推荐方案变为：全层 Key/logit adapter + 后半层 Value functional adaptor。

### 产物

- `results/formal_partial_v_40_50/`
- `results/formal_partial_v_40_100/formal_partial_v_40_100_summary.json`
| K | oracle rank64 | 88.82% | 0.3344 | 0.1265 |
| V | mean | 3.86% | 0.9805 | 0.7965 |
| V | shared MLP | 3.87% | 0.9805 | 0.7964 |
| V | token-local Ridge | 18.09% | 0.9050 | 0.7352 |
| V | oracle rank64 | 66.62% | 0.5777 | 0.4693 |

### 结论和架构调整

1. 这是实际多轮训练，不再是闭式 Ridge；结果明确排除“只要多训几个 epoch 就会泛化”的解释。
2. 完全共享 MLP 几乎只学到 mean。layer/head embedding 无法替代 per-layer/head 映射，而 Ridge 的独立映射明显更强。
3. 下一版应保留共享的低维 prefix encoder，但使用 per-layer/head low-rank output heads或 grouped heads；这样在参数和层头特异性之间折中。
4. 还需要 `cached-prefix` 与 `oracle updated-prefix` 对照。若后者显著提高，Adapter 必须按 document 串行；若两者都低，必须加入轻量 layer recurrence，而不是继续扩大静态 MLP。

产物：

- `results/shared_mlp_predictor.pt`
- `results/shared_mlp_predictor.json`
- `results/shared_mlp_predictor.log`
- `figures/shared_mlp_training_curve.png`

## 2026-07-12：Prefix Seriality 与分组输出头消融

### Cached Prefix vs Oracle Updated Prefix

用真实preceding-document Delta构造不可部署的updated-prefix KV，保持rank64 basis、Ridge和split相同：

| Prefix input | K explained | V explained |
|---|---:|---:|
| cached-prefix mean | 38.30% | 15.78% |
| oracle updated-prefix mean | 36.85% | 14.36% |

updated-prefix没有提升。当前mean pooling下无证据要求按document串行；该结论只否定“updated mean”特征，不能否定更保留顺序的prefix encoder。

产物：`results/stage2_oracle_updated_prefix.json`。

### 修正 MLP Layer/Head 采样不足

首版每doc只采64个随机layer/head/token组合，平均每个layer-head训练点过少。修正为每doc覆盖全部`64x8=512` layer-head，各随机1个token：训练样本从60,800增至486,400。

平衡共享MLP训练40 epochs，validation从`0.8077`持续降到epoch39的`0.7916`。strict test：

| Model | K explained | V explained |
|---|---:|---:|
| old shared MLP | 16.03% | 3.87% |
| balanced shared MLP | 21.01% | 6.07% |
| per-layer/head Ridge | 40.73% | 18.09% |

采样不足确实影响旧实验，但完全共享输出仍不足。

### Grouped Per-Layer/Head Output Heads

共享token/prefix encoder，输出端为每层每头低秩head：

| Group rank | Parameters | Best epoch | K explained | V explained |
|---:|---:|---:|---:|---:|
| 8 | 约1.3M | 11 | 25.46% | 7.63% |
| 32 | 4,900,512 | 5 | 28.14% | 9.11% |

rank32继续改善，但validation在epoch5后快速过拟合，仍明显弱于闭式Ridge。仅在73个RAG examples上继续增大head rank不合理，需要外部context预训练。

产物：

- `results/shared_mlp_predictor_balanced.{json,pt,log}`
- `results/grouped_head_predictor.{json,pt,log}`
- `results/grouped_head_predictor_rank32.{json,pt,log}`
- `figures/shared_mlp_balanced_training_curve.png`

## 2026-07-12：WikiText预训练调研与Qwen3-8B切换

### 可复用数据

旧draft训练已有严格WikiText-103数据：500k train pairs来自原文前35%，50k validation来自最后10%。旧实验从20k到真实Wiki 500k后，selector recall@15从约61.7%增至76.9%，说明独立语料规模有效；但旧teacher score/logits不是Delta标签，不能直接复用。

Qwen2.5/Qwen3 tokenizer完整vocab不同。按用户要求统一Qwen3：不使用旧IDs，也不采用Qwen2 decode再Qwen3 encode的过渡smoke。最终从539MB原始WikiText按字符slice生成Qwen3-8B token cache：

```text
model=/home/hming/models/Qwen3-8B
slice=[0,0.35]
tokens=42,487,983
dtype=int32
cache size=170MB
```

命令：

```bash
python scripts/prepare_qwen3_wikitext_tokens.py \
  --model-path /home/hming/models/Qwen3-8B \
  --text-file ../predictor_distill_wikitext/data/wikitext103_train.txt \
  --output results/qwen3_8b/wikitext_tokens_train_0_35.pt \
  --start-frac 0 --end-frac 0.35
```

### Qwen3-8B 66-sample Delta Smoke

每条样本为`A=256 tokens, X=128 tokens`，比较`prefill(X)`与`prefill(A+X)`中的X KV；Key反向RoPE shift 256。每条均匀保存8个target tokens、全部36层和8 KV heads，约2.66MB。

66条中50条训练basis、16条严格heldout：

| Kind | Mean original gap | Rank8 energy | Rank32 energy | Rank64 energy |
|---|---:|---:|---:|---:|
| K | 0.2339 | 71.21% | 87.77% | 94.40% |
| V | 0.4203 | 18.59% | 41.59% | 63.71% |

Value gap约为Key的1.8倍且更难压缩，与32B机制结论一致。首次Value PCA出现NaN，经审计tensor均finite、最大绝对值56.25；原因是CPU float32 `eigh`数值不稳，改为float64后结果正常。

产物：

- `PRETRAINING_DATA_PLAN.md`
- `scripts/prepare_qwen3_wikitext_tokens.py`
- `scripts/collect_qwen3_8b_wikitext_delta.py`
- `scripts/analyze_qwen3_8b_wiki_basis.py`
- `results/qwen3_8b/wikitext_tokens_train_0_35.pt`
- `results/qwen3_8b/wikitext_delta_smoke/`
- `results/qwen3_8b/wiki_basis_50train_16test.json`

### 下一 Gate

先在Qwen3-8B扩到2k train + 256独立validation，学习Wiki basis和predictor；随后必须构造8B RAG preprocess-to-full residual做fine-tune/test。只有8B RAG Value explained energy超过无预训练baseline后，才迁移方法到32B。

## 2026-07-12：Qwen3-8B Wiki 3024/256 Predictor

### 目的

验证“低秩basis存在”之后，进一步回答：Delta的低秩系数是否能由轻量Adapter预测。为了避免32B成本，先统一使用Qwen3-8B。

本轮数据仍是Wiki raw/offline到full的synthetic pretraining residual，不是最终RAG preprocess-to-full residual：

```text
A = prefix 256 tokens
X = target 128 tokens
offline/raw = prefill(X)
full = prefill(A + X)中的X部分
DeltaK = RoPE-aligned full K - offline K
DeltaV = full V - offline V
```

每条样本保存8个均匀target tokens、36层、8个KV heads。Key在local/RoPE-aligned坐标比较；Value直接比较。

### 启动命令

生成validation token cache：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/structured_kv_adapter
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/prepare_qwen3_wikitext_tokens.py \
  --model-path /home/hming/models/Qwen3-8B \
  --text-file ../predictor_distill_wikitext/data/wikitext103_train.txt \
  --output results/qwen3_8b/wikitext_tokens_val_90_100.pt \
  --start-frac 0.9 \
  --end-frac 1.0
```

qjy000采集train 0--1999：

```bash
for i in 0 1 2 3 4 5 6 7; do
  start=$((i*250))
  CUDA_VISIBLE_DEVICES=${i} python scripts/collect_qwen3_8b_wikitext_delta.py \
    --token-cache results/qwen3_8b/wikitext_tokens_train_0_35.pt \
    --start ${start} \
    --count 250 \
    --stride 384 \
    --output-dir results/qwen3_8b/wikitext_delta_train2k_stride384
done
```

qjy003采集validation 0--255。qjy003使用共享`/home/hming/FusionRAG-pca-analysis`目录，需要显式设置`PYTHONPATH`，否则会报`ModuleNotFoundError: No module named 'ktransformers'`：

```bash
for i in 0 1 2 3 4 5 6 7; do
  start=$((i*32))
  PYTHONPATH=/home/hming/FusionRAG-pca-analysis CUDA_VISIBLE_DEVICES=${i} \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/collect_qwen3_8b_wikitext_delta.py \
    --token-cache results/qwen3_8b/wikitext_tokens_val_90_100.pt \
    --start ${start} \
    --count 32 \
    --stride 384 \
    --output-dir results/qwen3_8b/wikitext_delta_val256_stride384
done
```

qjy003额外采集train 2000--3023，加速扩大样本：

```bash
for i in 0 1 2 3 4 5 6 7; do
  start=$((2000+i*128))
  PYTHONPATH=/home/hming/FusionRAG-pca-analysis CUDA_VISIBLE_DEVICES=${i} \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/collect_qwen3_8b_wikitext_delta.py \
    --token-cache results/qwen3_8b/wikitext_tokens_train_0_35.pt \
    --start ${start} \
    --count 128 \
    --stride 384 \
    --output-dir results/qwen3_8b/wikitext_delta_train2k_stride384
done
```

为避免训练读到正在写入的`.pt`，创建只包含可`torch.load`样本的hardlink快照：

```text
results/qwen3_8b/wikitext_delta_train_snapshot_verified_2914
results/qwen3_8b/wikitext_delta_train_snapshot_verified_3024
```

训练rank64 predictor：

```bash
PYTHONPATH=/home/hming/FusionRAG-pca-analysis CUDA_VISIBLE_DEVICES=6 \
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/train_qwen3_8b_wiki_predictor.py \
  --train-dir results/qwen3_8b/wikitext_delta_train_snapshot_verified_2914 \
  --val-dir results/qwen3_8b/wikitext_delta_val256_stride384 \
  --output-json results/qwen3_8b/wiki_predictor_rank64_train2928_val256.json \
  --output-pt results/qwen3_8b/wiki_predictor_rank64_train2928_val256.pt \
  --rank 64 \
  --epochs 12 \
  --steps-per-epoch 320 \
  --batch-files 8 \
  --rows-per-file 512 \
  --eval-max-files 128
```

训练rank32 predictor：

```bash
PYTHONPATH=/home/hming/FusionRAG-pca-analysis CUDA_VISIBLE_DEVICES=6 \
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/train_qwen3_8b_wiki_predictor.py \
  --train-dir results/qwen3_8b/wikitext_delta_train_snapshot_verified_3024 \
  --val-dir results/qwen3_8b/wikitext_delta_val256_stride384 \
  --output-json results/qwen3_8b/wiki_predictor_rank32_train3024_val256.json \
  --output-pt results/qwen3_8b/wiki_predictor_rank32_train3024_val256.pt \
  --rank 32 \
  --epochs 12 \
  --steps-per-epoch 320 \
  --batch-files 8 \
  --rows-per-file 512 \
  --eval-max-files 128
```

### 数据规模

| Split | Source | Samples | Token slice |
|---|---|---:|---|
| train | WikiText raw text前35% | 3024 | `wikitext_tokens_train_0_35.pt` |
| validation | WikiText raw text后10% | 256 | `wikitext_tokens_val_90_100.pt` |

完整3024 train快照：

| Kind | Mean original gap |
|---|---:|
| K | 0.2407 |
| V | 0.4421 |

Value gap约为Key的1.84倍。

### 模型定义

`scripts/train_qwen3_8b_wiki_predictor.py`：

- 先按每个layer/head分别学习PCA basis；
- predictor不直接输出完整KV Delta，而是输出K/V PCA coefficient；
- 输入为当前token offline/raw K/V、prefix K/V mean、prefix last mean和简单位置特征；
- 架构为共享own/prefix encoder + per-layer/head rank8 output heads。

### 结果

explained energy定义为`1 - ||Delta - Delta_hat||^2 / ||Delta||^2`。`mean`为每层每头Delta均值模板；`MLP`为可部署轻量predictor；`oracle`为真实Delta投影到同一basis的不可部署上界。

| Train | Val | Rank | Params | K mean | K MLP | K oracle | V mean | V MLP | V oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2928 | 256 | 64 | 760,480 | 15.66% | 55.68% | 95.68% | 1.25% | 7.16% | 66.74% |
| 3024 | 256 | 32 | 594,592 | 15.67% | 55.63% | 89.32% | 1.25% | 7.07% | 43.91% |

rank64的final KV error：

| Kind | Method | Original gap | Final KV error | Delta cosine |
|---|---|---:|---:|---:|
| K | mean | 0.3378 | 0.3102 | 0.3958 |
| K | MLP | 0.3378 | 0.2249 | 0.7462 |
| K | oracle_rank64 | 0.3378 | 0.0702 | 0.9782 |
| V | mean | 0.5590 | 0.5555 | 0.1118 |
| V | MLP | 0.5590 | 0.5387 | 0.2677 |
| V | oracle_rank64 | 0.5590 | 0.3224 | 0.8170 |

### 结论

1. Key已经显示出可预测低秩结构。当前轻量MLP能把K explained energy从mean模板的约15.7%提升到约55.6%，虽然离oracle 95.7%仍有明显距离。
2. Value有低秩可压缩性，但当前输入几乎不能预测其系数。rank64 oracle为66.7%，rank32 oracle为43.9%；但MLP在rank64/rank32都只有约7%。因此问题不是rank64输出维度太高，而是`own KV + prefix mean`不足以决定Value更新。
3. 对KV Adapter设计的直接含义：`Delta ≈ Bc`可作为形式，但K和V不能同等处理。K可以先沿“共享basis + 轻量coefficient predictor”推进；V需要更强的context-conditioned输入，例如prefix token landmark/sketch、attention summary、层内hidden-state或轻量cross-attention，而不是继续扩大同构MLP。
4. 仍不能直接迁移到32B。下一步必须先在Qwen3-8B构造formal RAG preprocess-to-full residual，验证Wiki raw-to-full学到的K可预测性是否迁移到真实FusionRAG pipeline；若V在formal RAG上仍只有个位数explained energy，应重新设计Value Adapter。

### 产物

- `scripts/train_qwen3_8b_wiki_predictor.py`
- `results/qwen3_8b/wikitext_tokens_val_90_100.pt`
- `results/qwen3_8b/wikitext_delta_train2k_stride384/`
- `results/qwen3_8b/wikitext_delta_val256_stride384/`
- `results/qwen3_8b/wikitext_delta_train_snapshot_verified_2914/`
- `results/qwen3_8b/wikitext_delta_train_snapshot_verified_3024/`
- `results/qwen3_8b/wiki_predictor_rank64_train2928_val256.{json,pt}`
- `results/qwen3_8b/wiki_predictor_rank32_train3024_val256.{json,pt}`

### 异常

- qjy003直接用别名SSH会被关闭，改用直连`hming@124.174.50.211`。
- qjy003第一次validation采集未设置`PYTHONPATH`，报`ModuleNotFoundError: No module named 'ktransformers'`；设置`PYTHONPATH=/home/hming/FusionRAG-pca-analysis`后正常。
- 训练前创建hardlink快照，避免读到采集过程中正在写入的`.pt`。

## 2026-07-12：完整Prefix KV Cross-Attention Predictor小样本尝试

### 目的

上一轮predictor只使用`prefix K/V mean`，Value explained energy只有约7%。本轮测试用户提出的问题：

> 能否使用完整prefix K/V预测next/document token的Key/Value变化？

这里不直接flatten完整prefix，而是让target token的offline K/V生成query，对同层同头的完整256-token prefix K/V做一次轻量cross-attention，然后输出PCA coefficient。

### 数据采集

新增脚本：

- `scripts/collect_qwen3_8b_wikitext_delta_fullprefix.py`
- `scripts/train_qwen3_8b_fullprefix_cross_attn.py`

每条样本保存：

```text
own_kv:    [36, 8, 8, 256]
prefix_kv: [36, 8, 256, 256]
delta_kv:  [36, 8, 8, 256]
```

单条文件约`40.1MB`。因此本轮先做小规模：

| Split | Samples | Size |
|---|---:|---:|
| train | 256 | 9.6GB |
| validation | 64 | 2.4GB |

qjy000采集train：

```bash
for i in 0 1 2 3 4 5 6 7; do
  start=$((i*32))
  CUDA_VISIBLE_DEVICES=${i} \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/collect_qwen3_8b_wikitext_delta_fullprefix.py \
    --token-cache results/qwen3_8b/wikitext_tokens_train_0_35.pt \
    --start ${start} \
    --count 32 \
    --stride 384 \
    --output-dir results/qwen3_8b/wikitext_delta_fullprefix_train256_stride384
done
```

qjy003采集validation：

```bash
for i in 0 1 2 3 4 5 6 7; do
  start=$((i*8))
  PYTHONPATH=/home/hming/FusionRAG-pca-analysis CUDA_VISIBLE_DEVICES=${i} \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/collect_qwen3_8b_wikitext_delta_fullprefix.py \
    --token-cache results/qwen3_8b/wikitext_tokens_val_90_100.pt \
    --start ${start} \
    --count 8 \
    --stride 384 \
    --output-dir results/qwen3_8b/wikitext_delta_fullprefix_val64_stride384
done
```

### 训练命令

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/train_qwen3_8b_fullprefix_cross_attn.py \
  --train-dir results/qwen3_8b/wikitext_delta_fullprefix_train256_stride384 \
  --val-dir results/qwen3_8b/wikitext_delta_fullprefix_val64_stride384 \
  --output-json results/qwen3_8b/fullprefix_crossattn_rank64_train256_val64.json \
  --output-pt results/qwen3_8b/fullprefix_crossattn_rank64_train256_val64.pt \
  --rank 64 \
  --epochs 8 \
  --steps-per-epoch 160 \
  --batch-files 4 \
  --rows-per-file 256 \
  --eval-max-files 32
```

模型：

```text
target offline K/V -> own encoder -> query
prefix K/V[256 tokens] -> key/value projection
context = softmax(q @ prefix_key) @ prefix_value
[own_hidden, context, position] -> per-layer/head rank8 output head -> PCA coefficient
```

### 结果

train gap mean：

| Kind | Mean original gap |
|---|---:|
| K | 0.2386 |
| V | 0.4355 |

validation final rows：

| Kind | Method | Explained energy | Remaining Delta | Final KV error | Delta cosine |
|---|---|---:|---:|---:|---:|
| K | mean | 15.56% | 0.9189 | 0.3112 | 0.3945 |
| K | cross_attn | 53.34% | 0.6831 | 0.2313 | 0.7305 |
| K | oracle_rank64 | 95.25% | 0.2180 | 0.0738 | 0.9760 |
| V | mean | 1.16% | 0.9942 | 0.5598 | 0.1079 |
| V | cross_attn | 5.72% | 0.9710 | 0.5467 | 0.2392 |
| V | oracle_rank64 | 63.12% | 0.6073 | 0.3419 | 0.7945 |

### 结论

1. 完整prefix KV cross-attention对Key有效，但没有超过上一轮3024样本mean-prefix MLP的`55.68%`，当前256样本下为`53.34%`。这说明Key的可预测性比较稳健，完整prefix读取不是必要条件，但可能在更大数据/更强模型下继续改善。
2. Value仍然没有解决。即使target token读取完整256-token prefix KV，V explained energy也只有`5.72%`，低于上一轮mean-prefix MLP的`7.16%`，而oracle仍有`63.12%`。因此Value不是缺少低秩basis，而是当前可用输入/模型没有捕捉到决定Value Delta系数的变量。
3. 直接把完整prefix KV交给轻量attention不是充分方案。下一步应把Value单独处理，重点考虑更接近Transformer重算过程的特征：上一层hidden state、attention output/update summary、或显式模拟一小步attention+MLP residual，而不是只从KV cache本身读context。

### 产物

- `results/qwen3_8b/wikitext_delta_fullprefix_train256_stride384/`
- `results/qwen3_8b/wikitext_delta_fullprefix_val64_stride384/`
- `results/qwen3_8b/fullprefix_crossattn_rank64_train256_val64.{json,pt}`
- `results/qwen3_8b/logs/fullprefix_crossattn_rank64_train256_val64.log`

## 2026-07-12：Hidden/v_proj Input Delta Audit

### 目的

上一轮完整prefix KV cross-attention仍无法预测Value Delta。一个可能解释是：Value变化来自hidden state变化，而不是KV cache本身。因此本轮验证：

```text
V_i^l = Wv^l h_i^l
DeltaV_i^l ?= Wv^l Delta h_i^l
```

这里的`h_i^l`不是普通`output_hidden_states`，而是进入每层`v_proj`之前的真实输入，即`input_layernorm`之后、`v_proj`之前的hidden。通过forward hook抓`self_attn.v_proj`的input/output。

### 脚本

- `scripts/collect_qwen3_8b_hidden_vproj_audit.py`
- `scripts/analyze_qwen3_8b_hidden_vproj_audit.py`

每条样本保存：

```text
own_h:   [36, 8 sampled tokens, 4096]
delta_h: [36, 8 sampled tokens, 4096]
delta_v: [36, 8 kv_heads, 8 sampled tokens, 128]
```

单条约`5.31MB`。

### 采集命令

先跑2条smoke：

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/collect_qwen3_8b_hidden_vproj_audit.py \
  --token-cache results/qwen3_8b/wikitext_tokens_train_0_35.pt \
  --start 0 \
  --count 2 \
  --stride 384 \
  --output-dir results/qwen3_8b/hidden_vproj_audit_smoke
```

smoke结果：

```text
sample0: h_gap=0.5831, sampled_v_gap=0.6458, vproj_cache_mismatch=0.0
sample1: h_gap=0.5607, sampled_v_gap=0.5549, vproj_cache_mismatch=0.0
```

随后采96条，64 train + 32 heldout：

```bash
for i in 0 1 2 3 4 5 6 7; do
  start=$((i*12))
  CUDA_VISIBLE_DEVICES=${i} \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/collect_qwen3_8b_hidden_vproj_audit.py \
    --token-cache results/qwen3_8b/wikitext_tokens_train_0_35.pt \
    --start ${start} \
    --count 12 \
    --stride 384 \
    --output-dir results/qwen3_8b/hidden_vproj_audit_96_stride384
done
```

分析命令：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/analyze_qwen3_8b_hidden_vproj_audit.py \
  --data-dir results/qwen3_8b/hidden_vproj_audit_96_stride384 \
  --train-count 64 \
  --output-json results/qwen3_8b/hidden_vproj_audit_64train_32test.json
```

### 一致性检查

96条样本：

```text
vproj_cache_mismatch_mean = 0.0
vproj_cache_mismatch_max  = 0.0
```

说明hook抓到的`v_proj`输出差值和cache中的`DeltaV`完全一致；层位点正确。

平均gap：

| Kind | Mean gap |
|---|---:|
| h input to v_proj | 0.5795 |
| sampled V | 0.5660 |

### PCA / Oracle结果

64条train学习PCA basis，32条heldout测试。

Train rank energy：

| Kind | Rank8 | Rank16 | Rank32 | Rank64 | Rank128 |
|---|---:|---:|---:|---:|---:|
| Delta h | 42.31% | 51.00% | 61.24% | 72.86% | 84.17% |
| Delta V | 36.04% | 51.09% | 69.55% | 88.11% | 100.00% |

Heldout explained energy：

| Kind | Rank8 | Rank16 | Rank32 | Rank64 | Rank128 |
|---|---:|---:|---:|---:|---:|
| Delta h | 37.65% | 42.56% | 47.66% | 52.14% | 56.93% |
| Delta V | 15.90% | 23.91% | 37.35% | 60.42% | 100.00% |

### 结论

1. `DeltaV = Wv(Delta h)`在实现层面成立。本轮hook验证`vproj_cache_mismatch=0`，说明Value cache变化确实可以由进入`v_proj`的hidden input变化线性解释。
2. 但`Delta h`不是更简单的低秩对象。heldout rank64下，`Delta h`只解释`52.14%`，低于`DeltaV`的`60.42%`。rank128下`Delta h`仍只有`56.93%`，说明它在4096维空间里更分散。
3. 因此“先预测Delta hidden，再乘Wv得到DeltaV”不是自动更好的Adapter路线。Wv投影反而把一部分高维hidden变化压缩到了较低维的Value空间。
4. 下一步如果继续研究Value，应更关注Transformer重算中的attention output/update summary，或者直接预测Value低秩系数但提供更接近attention更新的输入；单纯把预测目标从V换成高维hidden不够。

### 产物

- `results/qwen3_8b/hidden_vproj_audit_smoke/`
- `results/qwen3_8b/hidden_vproj_audit_96_stride384/`
- `results/qwen3_8b/hidden_vproj_audit_64train_32test.json`

## 2026-07-12：Layer-wise Sanity 与 Layer-to-Layer Update Audit

### 目的

结合`kv_adaptor/idea2.md`和用户关于“第`l`层Value是否应由第`l-1`层Value/attention update预测”的问题，本轮验证三件事：

1. `layer0`的K/V/h Delta是否接近0，排除token/RoPE/cache对齐问题。
2. prefix影响是否从第一层attention output开始进入residual stream。
3. `DeltaAttnOut^{l-1}`是否能直接解释`DeltaH^l`/`DeltaV^l`，从而支持上一层Value加权平均预测下一层Value Delta。

### 脚本

- `scripts/collect_qwen3_8b_layer_update_audit.py`
- `scripts/analyze_qwen3_8b_layer_update_audit.py`

采集字段：

```text
delta_k:    [36, 8 kv_heads, 8 sampled tokens, 128]
delta_v:    [36, 8 kv_heads, 8 sampled tokens, 128]
delta_h:    [36, 8 sampled tokens, 4096]   # v_proj input
delta_attn: [36, 8 sampled tokens, 4096]   # self_attn output before residual add
```

### Smoke

命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/collect_qwen3_8b_layer_update_audit.py \
  --token-cache results/qwen3_8b/wikitext_tokens_train_0_35.pt \
  --start 0 \
  --count 2 \
  --stride 384 \
  --output-dir results/qwen3_8b/layer_update_audit_smoke
```

结果：

```text
sample0 layer0: K gap=4.46e-4, V gap=3.87e-5, h gap=0.0, attn gap=0.2466
sample1 layer0: K gap=4.90e-4, V gap=9.87e-5, h gap=0.0, attn gap=0.2216
vproj_cache_mismatch = 0.0
```

解释：

- layer0进入attention前的K/V/h几乎无变化，说明token对齐、RoPE-aligned Key和cache位置没有明显错误。
- layer0 attention output已经有明显变化，说明prefix影响从第一层attention output进入residual stream。

### 64-sample Audit

采集命令：

```bash
for i in 0 1 2 3 4 5 6 7; do
  start=$((i*8))
  CUDA_VISIBLE_DEVICES=${i} \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/collect_qwen3_8b_layer_update_audit.py \
    --token-cache results/qwen3_8b/wikitext_tokens_train_0_35.pt \
    --start ${start} \
    --count 8 \
    --stride 384 \
    --output-dir results/qwen3_8b/layer_update_audit_64_stride384
done
```

分析命令：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/analyze_qwen3_8b_layer_update_audit.py \
  --data-dir results/qwen3_8b/layer_update_audit_64_stride384 \
  --output-json results/qwen3_8b/layer_update_audit_64_summary.json
```

### Layer-wise Gap

平均指标：

```text
k_gap_sampled    = 0.3421
v_gap_sampled    = 0.5751
h_gap_sampled    = 0.5814
attn_gap_sampled = 0.7036
vproj_cache_mismatch = 0.0
```

关键layer summary：

| Kind | layer0 | layer1 | layer8 | layer16 | layer24 | layer35 | max layer | max gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K | 0.00046 | 0.1384 | 0.4644 | 0.6289 | 0.6165 | 0.4845 | 20 | 0.6427 |
| V | 0.00010 | 0.3058 | 0.5078 | 0.7694 | 0.5844 | 0.6505 | 18 | 0.8003 |
| h | 0.00000 | 0.3049 | 0.4844 | 0.6621 | 0.6253 | 0.6280 | 17 | 0.6875 |
| attn output | 0.2423 | 0.6261 | 0.7031 | 0.9063 | 0.6985 | 0.7591 | 17 | 0.9405 |

### Cross-layer Signals

```text
cos(DeltaAttnOut^{l-1}, DeltaH^l) mean = 0.2829
cos(DeltaAttnOut^l,     DeltaH^l) mean = 0.0616
cos(DeltaH^l,           DeltaH^{l+1}) mean = 0.8268
correlation(||DeltaH^l||, ||DeltaV^l||) mean = 0.8181
```

代表层：

```text
cos(DeltaAttnOut^0,  DeltaH^1)  = 0.3585
cos(DeltaAttnOut^15, DeltaH^16) = 0.2984
cos(DeltaAttnOut^34, DeltaH^35) = 0.3297
```

### 结论

1. idea2.md要求的最强sanity check通过：layer0 K/V/h Delta接近0。此前Delta统计没有明显token/RoPE/cache错位。
2. prefix影响确实从attention output进入residual stream：layer0 attention output gap约0.242，而layer0 K/V/h约0。
3. 但单独使用上一层attention output解释下一层hidden并不充分：`DeltaAttnOut^{l-1}`到`DeltaH^l`平均cosine只有0.283。
4. `DeltaH`跨层连续性非常强：`DeltaH^l`到`DeltaH^{l+1}`平均cosine为0.827。这说明Value更新更像一个随深度递推的residual trajectory，而不是每层独立由prefix KV或上一层attention output决定。
5. 下一步不应只做`DeltaAttnOut^{l-1} -> DeltaV^l`的单步MLP。更合理的是最小版depth-recurrent predictor，维护一个跨层state code：

```text
code_l = GRU(code_{l-1}, token_feature_l, prefix_summary_l)
coeff_KV_l = decoder_l(code_l)
```

先验证它是否能把Value explained energy从当前`5%-7%`提升到`20%+`。

### 产物

- `results/qwen3_8b/layer_update_audit_smoke/`
- `results/qwen3_8b/layer_update_audit_64_stride384/`
- `results/qwen3_8b/layer_update_audit_64_summary.json`

## 2026-07-12：Depth-Recurrent Predictor 最小版本

### 目的

Layer-update audit显示`DeltaH^l`到`DeltaH^{l+1}`具有很强连续性（平均cosine `0.8268`）。本轮验证最小版depth-recurrent predictor是否能利用这个连续性改善Value预测。

核心变化：

```text
旧模型：每个layer/head独立输出 coefficient
新模型：固定(sample, kv_head, sampled_token)，按36层组成序列
       code_l = GRU(code_{l-1}, own_kv_l, prefix_mean_l, layer/head/pos)
       coeff_l = per-layer/head decoder(code_l)
```

仍使用现有mean-prefix数据，不重新采teacher。目标仍为rank64 K/V PCA coefficient。

### 脚本

- `scripts/train_qwen3_8b_depth_recurrent_predictor.py`

### 命令

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/train_qwen3_8b_depth_recurrent_predictor.py \
  --train-dir results/qwen3_8b/wikitext_delta_train_snapshot_verified_3024 \
  --val-dir results/qwen3_8b/wikitext_delta_val256_stride384 \
  --output-json results/qwen3_8b/depth_recurrent_rank64_train3024_val256.json \
  --output-pt results/qwen3_8b/depth_recurrent_rank64_train3024_val256.pt \
  --rank 64 \
  --epochs 8 \
  --steps-per-epoch 240 \
  --batch-files 8 \
  --seqs-per-file 16 \
  --eval-max-files 128
```

### 结果

```text
train = 3024
validation = 256
best_epoch = 8
parameters = 803,424
train K/V gap = 0.2407 / 0.4421
```

| Kind | Method | Explained energy | Remaining Delta | Final KV error | Delta cosine |
|---|---|---:|---:|---:|---:|
| K | mean | 15.67% | 0.9183 | 0.3102 | 0.3958 |
| K | recurrent | 56.10% | 0.6626 | 0.2238 | 0.7490 |
| K | oracle_rank64 | 95.68% | 0.2078 | 0.0702 | 0.9782 |
| V | mean | 1.25% | 0.9937 | 0.5555 | 0.1118 |
| V | recurrent | 7.44% | 0.9621 | 0.5378 | 0.2728 |
| V | oracle_rank64 | 66.75% | 0.5766 | 0.3224 | 0.8170 |

与之前mean-prefix MLP对比：

| Model | K explained | V explained |
|---|---:|---:|
| mean-prefix MLP rank64 | 55.68% | 7.16% |
| depth-recurrent GRU rank64 | 56.10% | 7.44% |

### 结论

1. 最小depth recurrence只带来非常小的提升：K +0.42点，V +0.28点。
2. 这说明Value预测失败不是单纯因为“每层独立预测”。在仍使用`own KV + prefix mean`输入时，即使加入跨层GRU state，也无法把Value从个位数提升到`20%+`。
3. 当前证据链变得比较清楚：
   - K：共享basis + lightweight predictor可行，稳定在`53%-56%`。
   - V：basis有上界（`~63%-67%`），但KV-only输入、full-prefix cross-attn、hidden target、layer recurrence都没有解决系数可预测性。
4. 下一步优先级应转为functional ablation：先判断系统上是否必须更新V。如果`full K + cached V`已经能恢复大部分attention/output/logit行为，则可以先推进K-only Adapter；如果V必要，再设计更接近full recompute的attention-output/logit监督或link/adapter token路线。

### 产物

- `results/qwen3_8b/depth_recurrent_rank64_train3024_val256.json`
- `results/qwen3_8b/depth_recurrent_rank64_train3024_val256.pt`
- `results/qwen3_8b/logs/depth_recurrent_rank64_train3024_val256.log`

## 2026-07-12：Functional K/V Ablation

### 目的

前面几轮主要看K/V Delta的几何重构误差。`kv_adaptor/idea2.md`指出，系统目标不是最小化KV L2，而是让后续query/decode行为接近full prefill。本轮做一个teacher-forced局部功能指标：

```text
A = prefix 256
X = document/target 128
Q = future query 32

full run = prefill(A + X + Q)
cached X = prefill(X)
```

使用full run中future query的Q，限制它只读document X，比较不同X K/V组合下的attention logits和attention output：

```text
cachedK + cachedV
fullK   + cachedV
cachedK + fullV
fullK   + fullV  # teacher reference
```

注意：这是restricted query->X attention proxy，不是完整端到端RAG accuracy。

### 脚本

- `scripts/analyze_qwen3_8b_functional_kv_ablation.py`

### 命令

先跑2条smoke：

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/analyze_qwen3_8b_functional_kv_ablation.py \
  --token-cache results/qwen3_8b/wikitext_tokens_train_0_35.pt \
  --start 0 \
  --count 2 \
  --stride 416 \
  --output-json results/qwen3_8b/functional_kv_ablation_smoke2.json
```

正式32条：

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/analyze_qwen3_8b_functional_kv_ablation.py \
  --token-cache results/qwen3_8b/wikitext_tokens_train_0_35.pt \
  --start 0 \
  --count 32 \
  --stride 416 \
  --output-json results/qwen3_8b/functional_kv_ablation_32.json
```

### 指标

```text
logit_rel_error = ||Q K_variant^T - Q K_full^T|| / ||Q K_full^T||
output_rel_error = ||P_variant V_variant - P_full V_full|| / ||P_full V_full||
```

其中`P_variant`由对应K的restricted softmax得到；`fullK+cachedV`使用`P_full`，`cachedK+fullV`使用`P_cached`。

### 结果

32 samples summary：

| Variant | Logit rel error | Output rel error |
|---|---:|---:|
| cachedK + cachedV | 0.2744 | 0.8897 |
| fullK + cachedV | 0.0000 | 0.6031 |
| cachedK + fullV | 0.2744 | 1.8550 |
| fullK + fullV | 0.0000 | 0.0000 |

代表层：

| Layer | cachedKV out err | fullK+cachedV out err | cachedK+fullV out err |
|---:|---:|---:|---:|
| 0 | 0.0042 | 0.0001 | 0.0041 |
| 1 | 0.2345 | 0.1756 | 0.1498 |
| 8 | 0.8641 | 0.3221 | 1.7028 |
| 16 | 0.9657 | 0.5684 | 1.4989 |
| 24 | 0.8469 | 0.5588 | 1.8687 |
| 35 | 0.8196 | 0.3466 | 1.4950 |

### 结论

1. K对future query读取document X的attention logits有明显影响：cached K的logit relative error为`0.2744`。
2. 只修K有明显收益：`fullK+cachedV`把attention output relative error从`0.8897`降到`0.6031`。
3. 单独替换V会更糟：`cachedK+fullV`的output error为`1.8550`。这说明V必须配合正确attention weights使用；如果K仍错，full V反而会被错误权重读取，造成更大的局部输出偏差。
4. 这个结果支持K-first/K-only Adapter作为下一步系统路线，但也说明K-only不一定充分：`fullK+cachedV`仍有`0.6031`的局部output error。
5. 下一步应在formal RAG preprocess-to-full上验证`predicted/oracle K + cached V`对真实query/decode的影响，再决定是否继续投入Value Adapter。

### 产物

- `results/qwen3_8b/functional_kv_ablation_smoke2.json`
- `results/qwen3_8b/functional_kv_ablation_32.json`
- `results/qwen3_8b/logs/functional_kv_ablation_32.log`

## 2026-07-12 Scheme B: layer-parallel update probe

### 目的

验证一种非递归 KV update 近似：每一层不依赖上一层 updated hidden，而是用离线缓存的本层 hidden sidecar `own_h[l,t]` 生成 query，独立读取当前 RAG prefix KV，得到 `parallel_context[l,h,t]`，再预测该层 `DeltaK/DeltaV`。

该实验不是端到端 RAG，也没有读取真实 RAG full hidden；`own_h` 来自目标 chunk/doc 单独离线 prefill 的 hook，因此不构成真实 full-context 数据泄漏。

### 脚本

- `scripts/analyze_qwen3_8b_parallel_layer_probe.py`

### Smoke command

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/analyze_qwen3_8b_parallel_layer_probe.py \
  --hidden-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/hidden_vproj_audit_96_stride384 \
  --fullprefix-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_delta_fullprefix_train256_stride384 \
  --max-records 16 --train-count 8 --rank 32 \
  --output-json MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/parallel_layer_probe_smoke16_rank32.json
```

Smoke 结果：K `own_kv` 41.11%，`parallel_context` -53.66%，`own_kv+parallel_context` 41.28%；V `own_kv` -12.36%，`parallel_context` -106.06%，`own_kv+parallel_context` -12.21%。流程可跑通，但 split 太小，仅作 sanity check。

### Formal command

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/analyze_qwen3_8b_parallel_layer_probe.py \
  --hidden-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/hidden_vproj_audit_96_stride384 \
  --fullprefix-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_delta_fullprefix_train256_stride384 \
  --max-records 96 --train-count 64 --rank 64 \
  --output-json MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/parallel_layer_probe_64train_32test_rank64.json
```

### Formal results

96 records, 64 train / 32 test, rank 64, per-layer/per-KV-head PCA+Ridge。

| Target | Feature | Explained Delta Energy | Remaining Delta | Cosine |
|---|---|---:|---:|---:|
| K | mean template | 0.1501 | - | - |
| K | own_kv | 0.5042 | 0.6939 | 0.7051 |
| K | parallel_context | 0.1469 | 0.9192 | 0.4536 |
| K | own_kv + parallel_context | 0.5046 | 0.6936 | 0.7054 |
| V | mean template | 0.0236 | - | - |
| V | own_kv | 0.0376 | 0.9794 | 0.2572 |
| V | parallel_context | -0.1450 | 1.0683 | 0.1241 |
| V | own_kv + parallel_context | 0.0378 | 0.9793 | 0.2572 |

Layer buckets:

| Target/Feature | L00-07 | L08-15 | L16-23 | L24-31 | L32-35 |
|---|---:|---:|---:|---:|---:|
| K own_kv | 0.5233 | 0.6143 | 0.4111 | 0.4577 | 0.5247 |
| K parallel_context | 0.2444 | 0.1560 | 0.0938 | 0.0559 | 0.2216 |
| K own_kv+parallel_context | 0.5233 | 0.6144 | 0.4113 | 0.4580 | 0.5278 |
| V own_kv | 0.1050 | 0.0493 | 0.0114 | -0.0031 | 0.0133 |
| V parallel_context | -0.0067 | -0.1645 | -0.1437 | -0.2301 | -0.2149 |
| V own_kv+parallel_context | 0.1050 | 0.0494 | 0.0116 | -0.0028 | 0.0137 |

### 结论

- `parallel_context` 单独解释力弱，拼接到 `own_kv` 后几乎没有增益：K 从 `0.5042` 到 `0.5046`，V 从 `0.0376` 到 `0.0378`。
- K Delta 仍然有较强的 offline-KV 可预测成分，但这个成分不来自当前实现的 per-layer prefix attention read。
- V Delta 在该 probe 下基本不可预测，说明 Value update 更依赖逐层递归 hidden 轨迹或功能性 supervision。
- 朴素 Scheme B 暂不成立。下一步若继续验证并行/半并行，应把 memory 从 only-prefix 改为 `prefix + target(<i)`，因为真实重算 token_i 会 attend 到前序 prefix 和当前 document 内部 `<i` tokens。

## 2026-07-12 Scheme B revision: prefix + offline previous target memory

### 目的

补测上一轮 only-prefix Scheme B 的缺口：真实 token_i 重算时可见 `prefix + current_document(<i)`，因此本轮采集完整 target/doc 的离线 KV sidecar，并让 sampled token_i 的并行 read memory 变为：

```text
prefix_kv[l] + offline_target_kv[l, :i]
```

注意：`offline_target_kv` 是目标 document 单独离线 prefill 的 KV，不是 full-context KV；输入仍然不使用 full hidden/full target KV/true Delta coefficient。

### 新增脚本

- `scripts/collect_qwen3_8b_wikitext_delta_fullprefix_alltarget.py`
- `scripts/analyze_qwen3_8b_parallel_layer_prevtarget_probe.py`

### Commands

Smoke collect:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/collect_qwen3_8b_wikitext_delta_fullprefix_alltarget.py \
  --token-cache MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_tokens_train_0_35.pt \
  --start 0 --count 4 --stride 384 \
  --output-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_delta_fullprefix_alltarget_smoke4_stride384
```

Smoke probe:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/analyze_qwen3_8b_parallel_layer_prevtarget_probe.py \
  --hidden-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/hidden_vproj_audit_96_stride384 \
  --alltarget-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_delta_fullprefix_alltarget_smoke4_stride384 \
  --max-records 4 --train-count 2 --rank 16 \
  --output-json MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/parallel_layer_prevtarget_probe_smoke4_rank16.json
```

Formal collect:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/collect_qwen3_8b_wikitext_delta_fullprefix_alltarget.py \
  --token-cache MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_tokens_train_0_35.pt \
  --start 0 --count 96 --stride 384 \
  --output-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_delta_fullprefix_alltarget_96_stride384
```

Formal probe:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/analyze_qwen3_8b_parallel_layer_prevtarget_probe.py \
  --hidden-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/hidden_vproj_audit_96_stride384 \
  --alltarget-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_delta_fullprefix_alltarget_96_stride384 \
  --max-records 96 --train-count 64 --rank 64 \
  --output-json MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/parallel_layer_prevtarget_probe_64train_32test_rank64.json
```

### Results

Original full-vs-offline gap over 96 records:

| Target | Mean relative L2 gap | Min | Max |
|---|---:|---:|---:|
| K | 0.2421 | 0.1920 | 0.3300 |
| V | 0.4477 | 0.3440 | 0.6080 |

96 records, 64 train / 32 test, rank 64:

| Target | Feature | Explained Delta Energy | Remaining Delta | Cosine |
|---|---|---:|---:|---:|
| K | own_kv | 0.5042 | 0.6939 | 0.7051 |
| K | parallel_prefix | 0.1469 | 0.9192 | 0.4536 |
| K | parallel_prefix_prevtarget | 0.1715 | 0.9048 | 0.4725 |
| K | own_kv + parallel_prefix | 0.504641 | 0.693554 | 0.705424 |
| K | own_kv + parallel_prefix_prevtarget | 0.504645 | 0.693551 | 0.705427 |
| V | own_kv | 0.0376 | 0.9794 | 0.2572 |
| V | parallel_prefix | -0.1450 | 1.0683 | 0.1241 |
| V | parallel_prefix_prevtarget | -0.1463 | 1.0686 | 0.1267 |
| V | own_kv + parallel_prefix | 0.037793 | 0.979297 | 0.257240 |
| V | own_kv + parallel_prefix_prevtarget | 0.037732 | 0.979328 | 0.257201 |

Layer-bucket explained energy:

| Target/Feature | L00-07 | L08-15 | L16-23 | L24-31 | L32-35 |
|---|---:|---:|---:|---:|---:|
| K parallel_prefix | 0.2444 | 0.1560 | 0.0938 | 0.0559 | 0.2216 |
| K parallel_prefix_prevtarget | 0.2586 | 0.2132 | 0.1091 | 0.0881 | 0.2060 |
| K own_kv+parallel_prefix | 0.5233 | 0.6144 | 0.4113 | 0.4580 | 0.5278 |
| K own_kv+parallel_prefix_prevtarget | 0.5233 | 0.6144 | 0.4112 | 0.4579 | 0.5281 |
| V parallel_prefix | -0.0067 | -0.1645 | -0.1437 | -0.2301 | -0.2149 |
| V parallel_prefix_prevtarget | -0.0070 | -0.1567 | -0.1255 | -0.2423 | -0.2538 |
| V own_kv+parallel_prefix | 0.1050 | 0.0494 | 0.0116 | -0.0028 | 0.0137 |
| V own_kv+parallel_prefix_prevtarget | 0.1050 | 0.0494 | 0.0115 | -0.0029 | 0.0136 |

### Conclusion

- Adding offline previous target KV gives K a small pure-context gain: `0.1469 -> 0.1715` explained energy.
- The gain disappears after adding `own_kv`: `0.504641 -> 0.504645`. So the previous-target sidecar does not add practically useful information beyond the offline target KV itself.
- V remains poorly predictable under this layer-parallel readout.
- This weakens the fully layer-parallel DeltaKV-adapter route. The next promising direction is not more direct per-layer DeltaV L2 regression, but K/logit functional correction plus possibly a mid/late-layer Value functional correction.

## 2026-07-12 Functional K adapter probe: predicted K with shrink gate

### 目的

验证一个更贴近 FusionRAG 目标的 functional 问题：轻量 `own_kv -> DeltaK` predictor 虽然不能完美拟合 DeltaK L2，但是否能减少 future query 读取 document X 时的 attention-logit/output error。

### 新增脚本

- `scripts/analyze_qwen3_8b_predicted_k_functional_probe.py`

脚本先用 train records 拟合 per-layer/per-KV-head PCA+Ridge：

```text
input  = offline own_kv(sampled target tokens), dim 256
output = RoPE-aligned DeltaK, dim 128
rank   = 64
```

Eval 阶段对完整 128 个 target tokens 预测：

```text
predicted_K(lambda) = rotate_to_full_position(offline_K + lambda * predicted_DeltaK)
```

然后复用 teacher-forced full query hidden 计算 query-to-X attention logits/output。full K/V 只作为 metric reference，不作为 predictor input。

### Smoke commands

初版 smoke 曾暴露一个 shape broadcasting bug：`scale=0` 没有等于 cachedK。已修正为保持 `offline_K` shape `[L,1,H,T,F]`，修正后 `predK_s0` 与 cachedK 对齐。

Fixed smoke:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/analyze_qwen3_8b_predicted_k_functional_probe.py \
  --train-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_delta_fullprefix_alltarget_96_stride384 \
  --token-cache MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_tokens_train_0_35.pt \
  --start 96 --count 4 --stride 416 --rank 64 --train-count 64 \
  --scales 0,0.001,0.003,0.01,0.03,0.1,0.3,1.0 \
  --output-json MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/predicted_k_functional_probe_smoke4_scales_rank64_fixed.json
```

Smoke summary：`predK_s0.3` output error `0.6376` vs cached `0.8656`，说明 scale-gated predicted K 有效，继续扩大。

### Formal command

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/analyze_qwen3_8b_predicted_k_functional_probe.py \
  --train-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_delta_fullprefix_alltarget_96_stride384 \
  --token-cache MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_tokens_train_0_35.pt \
  --start 96 --count 64 --stride 416 --rank 64 --train-count 64 \
  --scales 0,0.001,0.003,0.01,0.03,0.1,0.3,0.6,1.0 \
  --output-json MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/predicted_k_functional_probe_64eval_scales_rank64.json
```

### Formal results

64 heldout functional spans, train K predictor mean explained DeltaK energy `0.5896` on train sampled tokens。

| Method | Logit rel error | Output rel error |
|---|---:|---:|
| cachedK_cachedV | 0.2639 | 0.8723 |
| fullK_cachedV | 0.0000 | 0.6067 |
| cachedK_fullV | 0.2639 | 1.7269 |
| fullK_fullV | 0.0000 | 0.0000 |
| predK_s0.001_cachedV | 0.2637 | 0.8718 |
| predK_s0.003_cachedV | 0.2634 | 0.8707 |
| predK_s0.01_cachedV | 0.2625 | 0.8669 |
| predK_s0.03_cachedV | 0.2597 | 0.8556 |
| predK_s0.1_cachedV | 0.2504 | 0.8107 |
| predK_s0.3_cachedV | 0.2287 | 0.6477 |
| predK_s0.6_cachedV | 0.2134 | 0.6304 |
| predK_s1_cachedV | 0.2301 | 0.9246 |

Output-error gap recovery against `cachedK_cachedV -> fullK_cachedV`:

| Method | Recovery |
|---|---:|
| predK_s0.1 | 23.2% |
| predK_s0.3 | 84.6% |
| predK_s0.6 | 91.1% |
| predK_s1.0 | -19.7% |

### 结论

- 这是目前最支持 KV adapter 可行性的结果：简单线性 K predictor + shrink gate 在 functional metric 上回收大部分 K-side benefit。
- 直接 full-scale DeltaK 不可用，会过冲；adapter 设计必须带 gate/scale/norm constraint。
- 下一步应把 gate 从全局 `lambda` 改成可学习的 layer/head/token gate，并在 heldout 上调参或训练，目标函数用 query-attention logits/output，而不是只用 DeltaK L2。
- Value 仍应后置，因为 `cachedK_fullV` 极差，说明 K 未修正时替换 V 会放大错误读取。

## 2026-07-12 Gate calibration: global vs layer-wise K gate

### 目的

上一轮在同一个 64-sample heldout 上扫 `lambda`，结论是 `predK_s0.6` 接近 `fullK_cachedV`。本轮改为 calibration/test split，避免 test-tuned scale。

### 新增脚本

- `scripts/analyze_qwen3_8b_predicted_k_gate_calibration.py`

该脚本复用 `analyze_qwen3_8b_predicted_k_functional_probe.py` 中的 K predictor 训练和 functional eval 逻辑，但增加：

- calibration split 上选择 global scale；
- calibration split 上逐层选择 layer-wise scale；
- 独立 test split 上评估选中 gates。

### Smoke command

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/analyze_qwen3_8b_predicted_k_gate_calibration.py \
  --train-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_delta_fullprefix_alltarget_96_stride384 \
  --token-cache MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_tokens_train_0_35.pt \
  --calib-start 96 --calib-count 4 --test-start 100 --test-count 4 \
  --stride 416 --rank 64 --train-count 64 \
  --scales 0,0.1,0.3,0.6,1.0 \
  --output-json MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/predicted_k_gate_calibration_smoke4_4.json
```

Smoke showed the path works; small-sample layer-wise gate improved over global but was not used for conclusion.

### Formal command

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/analyze_qwen3_8b_predicted_k_gate_calibration.py \
  --train-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_delta_fullprefix_alltarget_96_stride384 \
  --token-cache MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_tokens_train_0_35.pt \
  --calib-start 96 --calib-count 32 --test-start 128 --test-count 64 \
  --stride 416 --rank 64 --train-count 64 \
  --scales 0,0.03,0.1,0.2,0.3,0.45,0.6,0.8,1.0 \
  --output-json MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/predicted_k_gate_calibration_calib32_test64_rank64.json
```

### Formal results

Train K predictor mean explained DeltaK energy: `0.5896`.
Calibration split: 32 spans. Test split: 64 spans.

Calibration:

| Method | Logit rel error | Output rel error |
|---|---:|---:|
| cachedK_cachedV | 0.2517 | 0.8747 |
| fullK_cachedV | 0.0000 | 0.5863 |
| predK_s0.45_cachedV | 0.2057 | 0.5619 |
| cachedK_fullV | 0.2517 | 1.7257 |

Test:

| Method | Logit rel error | Output rel error |
|---|---:|---:|
| cachedK_cachedV | 0.2753 | 0.8807 |
| fullK_cachedV | 0.0000 | 0.5230 |
| predK_global_calib_best_s0.45 | 0.2279 | 0.5738 |
| predK_layerwise_calib_best | 0.2250 | 0.5712 |
| cachedK_fullV | 0.2753 | 1.7387 |

Recovery vs cached-to-fullK gap:

| Gate | Recovery |
|---|---:|
| global scale 0.45 | 85.8% |
| layer-wise scale | 86.5% |

Layer-wise selected scales:

```text
[0.03, 0.6, 0.3, 0.3, 0.3, 0.45, 0.6, 0.8, 0.6, 0.6, 0.6, 0.6,
 0.6, 0.6, 0.6, 0.45, 0.45, 0.45, 0.45, 0.6, 0.45, 0.45, 0.45,
 0.45, 0.45, 0.6, 0.6, 0.45, 0.6, 0.45, 0.45, 0.45, 0.45, 0.45,
 0.45, 0.6]
```

### 结论

- Gate calibration 后，K adapter 结论仍成立，且不是 test-tuned 偶然结果。
- Global gate 已经很强：test output error `0.5738`，回收 `85.8%` K-side gap。
- Layer-wise gate 只略好：`0.5712`，回收 `86.5%`。短期不需要复杂 token-wise gate，先做全局/层级 gate 即可。
- 仍未达到 `fullK_cachedV=0.5230`，说明 predictor 方向或 functional training 还可改进。
- Value 继续后置，因为 `cachedK_fullV=1.7387` 仍然证明 K 错时替换 V 会放大错误。

### Training-set scaling: 64 vs 512 records for K predictor

上一轮已经证明 calibration/test split 下 K adapter 不是 test-tuned 偶然结果。本轮检查一个更基础的问题：当前 `own_kv -> DeltaK` 的低秩线性 predictor 是否只是因为训练样本太少而受限。

实验保持 calibration/test 完全不变，只把 predictor 训练样本从 64 条 sidecar records 扩大到 512 条。注意：predictor 输入只使用离线可获得的 `own_kv`，full-context KV/hidden 只作为 teacher/reference 计算 Delta 和 functional metric，不作为 online 输入，因此不构成真实 RAG 上文泄漏。

命令：

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/analyze_qwen3_8b_predicted_k_gate_calibration.py \
  --train-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_delta_train2k_stride384 \
  --token-cache MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_tokens_train_0_35.pt \
  --calib-start 96 --calib-count 32 --test-start 128 --test-count 64 \
  --stride 416 --rank 64 --train-count 512 \
  --scales 0,0.03,0.1,0.2,0.3,0.45,0.6,0.8,1.0 \
  --output-json MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/predicted_k_gate_calibration_train512_calib32_test64_rank64.json
```

结果对比：

| Train records | Mean explained DeltaK | Test cachedK_cachedV | Test fullK_cachedV | Global gate | Layer-wise gate |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.5896 | 0.8807 | 0.5230 | 0.5738 / 85.8% | 0.5712 / 86.5% |
| 512 | 0.5613 | 0.8807 | 0.5230 | 0.5685 / 87.3% | 0.5653 / 88.2% |

表中 adapter 列格式为 `output rel error / recovery vs cached-to-fullK gap`。

解释：

1. 从 64 到 512，functional output error 有小幅改善：global gate `0.5738 -> 0.5685`，layer-wise gate `0.5712 -> 0.5653`。
2. 改善是真实但不大，说明当前瓶颈不只是样本数。更大的训练集可能继续带来边际收益，但更关键的是 predictor 目标和结构。
3. `train_k_predictor_mean_explained` 下降到 `0.5613` 不代表 functional 变差；它是 DeltaK L2/PCA-Ridge 空间中的均值解释率，而最终目标是未来 query attention 的 output error。这里再次说明应该用 functional metric 作为主指标。
4. 目前最有希望的最小方案仍是：离线 `own_kv` 预测 DeltaK direction，online 只加 calibrated scalar gate。这个方案不需要真实 RAG full hidden，也不需要在线跑完整 Transformer recompute。

下一步：继续跑 train1024/更多 records 看 scaling 是否饱和；若收益继续很小，就转向 functional loss 训练 K adapter，或者加入不泄漏的前序 KV 统计作为输入。

### Training-set scaling: 1024 records result

继续用相同 calibration/test split，把 `own_kv -> DeltaK` predictor 的训练样本扩大到 1024 条。评价口径仍然是 functional future-query attention：用预测后的 K 和 cached V 计算 query 对目标 document span 的 attention logits/output，并与 full-context K/V teacher 对比。

命令：

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts/analyze_qwen3_8b_predicted_k_gate_calibration.py \
  --train-dir MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_delta_train2k_stride384 \
  --token-cache MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/wikitext_tokens_train_0_35.pt \
  --calib-start 96 --calib-count 32 --test-start 128 --test-count 64 \
  --stride 416 --rank 64 --train-count 1024 \
  --scales 0,0.03,0.1,0.2,0.3,0.45,0.6,0.8,1.0 \
  --output-json MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/predicted_k_gate_calibration_train1024_calib32_test64_rank64.json
```

三档训练规模对比：

| Train records | Mean explained DeltaK | Test cachedK_cachedV | Test fullK_cachedV | Global gate | Layer-wise gate |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.5896 | 0.8807 | 0.5230 | 0.5738 / 85.8% | 0.5712 / 86.5% |
| 512 | 0.5613 | 0.8807 | 0.5230 | 0.5685 / 87.3% | 0.5653 / 88.2% |
| 1024 | 0.5544 | 0.8807 | 0.5230 | 0.5694 / 87.0% | 0.5738 / 85.8% |

解释：

1. 512 是当前最好点；1024 没有继续提升，global gate 几乎持平，layer-wise gate 退化。
2. 这说明当前 rank64 PCA+Ridge 的 K adapter 已经不是明显的“小样本不够”问题。继续盲目加样本不是最高优先级。
3. DeltaK L2 空间的 mean explained energy 随 train_count 变大下降，符合“训练集更复杂、固定 rank64 更难覆盖”的现象。但 functional output 并没有同步恶化，说明 L2 DeltaK 解释率不是最终有效性的充分指标。
4. 目前可用结论是：一个完全静态/离线的 own-KV K-bias direction，加上 calibration scale，已经能回收大约 87% 的 K-side functional gap；但剩余 gap 很可能需要更高 rank、functional loss、或 prefix-aware 但不泄漏的输入。

下一步优先级调整：

```text
1. 先做 rank sweep：rank64 vs rank128/256，判断容量瓶颈是否来自低秩 basis。
2. 如果高 rank 仍不提升，转向 functional loss：直接优化 future-query attention logits/output，而不是 DeltaK L2。
3. 再考虑 prefix-aware 输入，但只能使用 online 真实可获得的前序 cached KV 统计，不能使用 full-context hidden。
```

### Prefix length vs KV recompute gap

用户问题：`token_i` 的 KV 重算 gap 是否和上文长度有关？已有 sidecar 固定 `prefix_len=256`，不能直接回答长度趋势，因此补充一个小样本 sweep。

实验设置：固定同一个 target span，只改变 target 前面保留多少 prefix token；比较 target KV 的 offline/cache 结果与 full-context recompute 结果。Key 在计算 gap 前做 RoPE 对齐，Value 直接比较。

```text
Model: Qwen3-8B
Samples: 32
Target length: 128
Prefix lengths: 0, 32, 64, 128, 256, 512, 768
Metric: ||KV_full - KV_offline|| / ||KV_offline||
Result: MOTIVATION_EXPERIMENTS/structured_kv_adapter/results/qwen3_8b/prefix_length_gap_sweep_32samples.json
```

结果：

| Prefix len | K mean | K p50 | K p90 | V mean | V p50 | V p90 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 32 | 0.1635 | 0.1600 | 0.1835 | 0.3454 | 0.3493 | 0.4026 |
| 64 | 0.1877 | 0.1865 | 0.2115 | 0.3780 | 0.3803 | 0.4343 |
| 128 | 0.2138 | 0.2135 | 0.2392 | 0.4134 | 0.4172 | 0.4711 |
| 256 | 0.2423 | 0.2426 | 0.2708 | 0.4382 | 0.4313 | 0.4927 |
| 512 | 0.2705 | 0.2720 | 0.2965 | 0.4628 | 0.4612 | 0.5107 |
| 768 | 0.2873 | 0.2890 | 0.3103 | 0.4761 | 0.4692 | 0.5254 |

解释：

1. KV recompute gap 与上文长度正相关。`prefix_len=0` sanity check 为 0，说明比较口径和 RoPE 对齐没有明显问题。
2. 增长不是线性的，有饱和趋势。K 从 32 到 768 增加 `0.1635 -> 0.2873`；V 从 `0.3454 -> 0.4761`。
3. V 的绝对 gap 一直大于 K，但 K 的相对增长更明显：K 约增长 76%，V 约增长 38%。这说明上文长度带来的新增 context effect 对 K 更敏感，而 V 在很短 prefix 下已经发生较大 residual 更新。
4. 对 adapter 设计的含义：如果要做静态/离线 bias，至少需要考虑 prefix length 或 prefix strength 的 gate；一个固定幅度的 bias 对短上文可能过修，对长上文可能欠修。当前 global scale 可以看作粗糙 gate，后续更合理的是让 gate 依赖在线可获得的 prefix 长度和 prefix KV 统计。

