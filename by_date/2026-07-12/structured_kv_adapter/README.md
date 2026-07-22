# Structured KV Adapter：替代 FusionRAG Document Recompute

本目录独立研究：能否在 FusionRAG online 阶段不重新运行完整 Transformer，而对已缓存 document KV 施加由当前前序 RAG context 决定的轻量结构化更新。

正式目标不是超过 raw/preprocess rate=0，而是逼近同一输入下的 `full rate=1`：

```text
system + doc1 + doc2 + ... + docN + query

B_i = load_kv(source, doc_i)
T_i = rate=1 对 document token 完整重算后的 KV
Delta_i = T_i - B_i
KV_adapter_i = B_i + Delta_hat_i
```

`source` 必须分别讨论 raw 与 BGE preprocess。Key 统一在 local/RoPE-aligned 坐标学习，写回 cache 前再施加 online position RoPE；Value 直接学习。后置 query 不作为 document Delta predictor 输入，因为 causal mask 下它不能影响前面的 document hidden state。

研究计划见 [PLAN.md](PLAN.md)，逐轮命令和结果追加到 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。

## 目录

- `PLAN.md`：研究问题、实验矩阵、准入和停止标准。
- `EXPERIMENT_LOG.md`：每轮设置、命令、结果、异常与结论。
- `scripts/`：数据采集、训练、分析和 pipeline 接口脚本。
- `results/`：manifest、CSV/JSON、checkpoint 与预测结果。
- `figures/`：rate-distortion、分层误差和端到端对比图。

## 当前状态

Stage 0 的 2-example audit 已通过；现有 formal residual 数据包含 67 个 subquery 文件，但只有 36 个独立 main examples（24 train、7 validation、5 test），不能称为完整 50-example 独立测试。

Stage 1 已完成 BGE preprocess 起点的 train-only per-layer/head feature basis 上界。严格 test 上，rank64 对 K/V 分别解释 `88.85%/66.49%` Delta 能量；Value final KV gap 从 `0.7804` 降至 `0.4518`，只回收 `42.1%`，并且仍使用不可部署的 oracle coefficient。rank8/16 不具备继续训练 predictor 的价值；下一步只推进 rank64，并补充 prefix-conditioned predictability ladder。

Stage 2 v1 中，当前 token cached K/V 的线性 rank64 coefficient predictor 只解释 K/V `38.3%/15.1%` Delta 能量，远低于 oracle 的 `88.9%/66.5%`。未经降维和 validation 调参的 256 维 prefix mean 出现过拟合，不能据此否定 context conditioning；下一轮需做低维 prefix summary 和严格 validation 选参。

现已补采 examples 50--99，并形成 73 个独立 main examples：51 train、11 validation、11 test。扩大后 oracle rank64 结论几乎不变（K/V `88.82%/66.62%`）；token-local Ridge 小幅提高到 `40.73%/18.09%`，仍远低于上界。高维 prefix mean 为 `38.30%/15.78%`，仍没有超过 token-local，下一步必须改变 prefix encoder，而不是只增加同类 token 样本。

首个真正多轮训练的共享 MLP 已完成：199,008 参数、60,800 train samples，validation 在 epoch 1 最优并随后持续过拟合。严格 test 的 K/V explained energy 仅 `16.03%/3.87%`，几乎退化为 mean template，明显差于 per-layer/head Ridge。结论不是“epoch 不够”，而是过度共享导致 layer/head-specific mapping 被抹平；下一候选应采用共享 prefix encoder + per-layer/head low-rank output heads，而不是完全共享输出 trunk。

后续发现首版MLP每篇doc只随机采64个layer/head/token组合，layer/head覆盖不公平。改成每篇doc覆盖全部512个layer-head后，共享MLP提高到K/V `21.01%/6.07%`；grouped per-layer/head rank8/rank32输出头进一步提高到`25.46%/7.63%`和`28.14%/9.11%`，但仍低于Ridge `40.73%/18.09%`。因此样本量和层头特异性都重要，当前73个RAG examples不足以训练出强泛化。

现转向Qwen3-8B低成本预训练验证。已从WikiText原始文本统一生成Qwen3-8B token cache：train使用前35%（42,487,983 tokens），validation使用后10%（12,142,879 tokens）。已采集3024条train、256条独立validation的`prefix256+target128` raw-to-full Delta。完整train快照平均原始gap为K/V `0.2407/0.4421`，Value更新幅度约为Key的1.84倍。rank64 oracle在独立validation上解释K/V `95.68%/66.74%`，但当前轻量MLP predictor只解释K/V `55.68%/7.16%`。结论：K具备初步可预测性；V存在可压缩basis，但当前`own KV + prefix mean`输入几乎不能预测V系数，下一步必须改输入特征或训练范式，而不是只扩大同构MLP。预训练设计与scaling gate见`PRETRAINING_DATA_PLAN.md`。

## Qwen3-8B Wiki Predictor 结果

数据定义：

```text
A = prefix 256 tokens
X = target 128 tokens
offline/raw = prefill(X)
full = prefill(A + X)中的X部分
DeltaK = RoPE-aligned full K - offline K
DeltaV = full V - offline V
```

每条样本保存8个均匀target tokens、36层、8个KV heads。train来自WikiText前35%，validation来自最后10%，两者文本区间独立。

| Train | Val | Rank | Predictor params | K mean | K MLP | K oracle | V mean | V MLP | V oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2928 | 256 | 64 | 760,480 | 15.66% | 55.68% | 95.68% | 1.25% | 7.16% | 66.74% |
| 3024 | 256 | 32 | 594,592 | 15.67% | 55.63% | 89.32% | 1.25% | 7.07% | 43.91% |
| 3024 | 256 | 64 depth-recurrent GRU | 803,424 | 15.67% | 56.10% | 95.68% | 1.25% | 7.44% | 66.75% |
| 256 | 64 | 64 full-prefix cross-attn | 858,432 | 15.56% | 53.34% | 95.25% | 1.16% | 5.72% | 63.12% |
| 64/32 hidden audit | 32 | 64 PCA only | oracle only | - | - | - | - | - | V 60.42%, h 52.14% |
| 64 layer-update audit | - | sanity/cosine only | no predictor | layer0 K/V/h≈0 | - | - | attn layer0=0.242 | h-to-next-h cos=0.827 | prev-attn-to-next-h cos=0.283 |
| 32 functional ablation | - | future-query over X | no predictor | cached K logit err=0.274 | - | fullK+cachedV out err=0.603 | cachedKV out err=0.890 | cachedK+fullV out err=1.855 | - |

`mean`表示只用每层每头Delta均值模板；`MLP`表示共享own/prefix encoder + per-layer/head低秩输出头；`oracle`表示真实Delta投影到同一PCA basis，是不可部署上界。

主要观察：

- Key：rank32/rank64 oracle很高，MLP能把Delta explained energy从mean的约15.7%提升到约55.6%，说明Key的低秩系数能被当前轻量特征部分预测。
- Value：rank64 oracle有66.7%，说明Value不是不可压缩；但MLP只有约7%，几乎只比mean模板多6个百分点。rank32降低输出维度后没有改善，说明瓶颈不是rank64输出太难，而是输入特征不足。
- Full-prefix cross-attention小样本尝试显示：让target token直接读取完整256-token prefix KV后，K仍能达到约53.3%，但V只有5.7%。这说明“完整prefix KV”本身不是Value问题的充分解；当前Value系数可能依赖更接近hidden-state/attention-update过程的特征，或者需要不同于K的预测范式。
- Hidden/v_proj audit显示：进入`v_proj`的hidden input变化可以精确解释Value变化，`vproj_cache_mismatch=0`；但`Delta h`本身并不比`DeltaV`更低秩。heldout rank64上，`Delta h` oracle为52.14%，`DeltaV` oracle为60.42%。因此“先预测Delta hidden再过Wv”不是自动更简单的路线。
- Layer-update audit通过了最关键sanity check：layer0的RoPE-aligned K gap约`4.6e-4`，V gap约`1.0e-4`，v_proj input gap为`0`，说明token/RoPE/cache对齐没有明显错误；同时layer0 attention output gap已经为`0.242`，说明prefix影响从第一层attention output开始进入residual stream。跨层上，`DeltaH^l`与`DeltaH^{l+1}`平均cosine为`0.827`，但`DeltaAttnOut^{l-1}`与`DeltaH^l`平均cosine只有`0.283`。因此Value更新更像沿residual trajectory连续传播，而不是仅由上一层attention output单独决定。
- Depth-recurrent GRU小模型用现有3024/256数据验证了“跨层state code”最小版本：K从55.68%小幅到56.10%，V从7.16%小幅到7.44%，没有达到`20%+`的Value gate。说明仅加入layer recurrence、但仍使用`own KV + prefix mean`输入，无法解决Value coefficient预测问题。
- Functional ablation用full future-query的Q读取document X，比较K/V替换的局部功能误差。`cachedK+cachedV`的attention output relative error为`0.890`；`fullK+cachedV`降到`0.603`；`cachedK+fullV`反而升到`1.855`。这说明修K比单独修V更有系统价值，而且V必须和正确attention weights匹配，不能独立替换。
- 对Adapter设计的含义：`Delta ≈ Bc`这个形式对K比较可行，对V需要更强或不同类型的context-conditioned输入。只用prefix K/V mean、last mean、token own KV，甚至小规模完整prefix KV cross-attention，都不足以决定Value更新；hidden/layer/recurrent实验进一步说明，直接换成高维`Delta h`、只用上一层attention output或只加GRU跨层递推都不够。functional ablation支持先推进K-only或K-first Adapter，但`fullK+cachedV`仍有`0.603`局部output error，不能直接认为V可以永久忽略。

## 下一 Gate

Qwen3-8B Wiki raw-to-full已经通过“basis可压缩”gate，但没有通过“Value可预测”gate。下一步不应直接迁移到32B；应先在8B上做两个验证：

1. 构造8B formal RAG preprocess-to-full residual，并优先验证K-only Adapter：`predicted/oracle K + cached V` 是否在真实FusionRAG query/decode上接近 full rate=1。
2. 若K-only在真实RAG中不足，再为V设计attention-output/logit监督或link/adapter token路线；不要继续只优化KV L2或独立V coefficient MSE。

## 2026-07-12 新增路线：Depth/Residual Recurrent Adapter

结合 `kv_adaptor/idea2.md` 和最新讨论，下一步不再把每个 layer/head 的 `DeltaK`、`DeltaV` 视为两个独立回归目标。更合理的目标是：

```text
preceding context
  -> Delta residual / normalized hidden state
  -> frozen Wk/Wv 产生一致的 DeltaK/DeltaV
```

需要先验证一个关键问题：`DeltaV^l` 是否主要可由上一层状态更新解释。因为：

```text
V_i^l = W_V^l h_i^{l-1}
DeltaV_i^l = W_V^l Delta h_i^{l-1}
```

而 `Delta h_i^{l-1}` 又来自上一层 attention output、residual、MLP 和 norm 的共同传播，不等同于 `DeltaV^{l-1}` 本身。因此新增一个小型 sanity/probe：

```text
target:  DeltaK^l / DeltaV^l
features:
  mean baseline
  DeltaV^{l-1}
  DeltaK^{l-1} + DeltaV^{l-1}
  DeltaAttnOut^{l-1}
  DeltaH^{l-1}
  DeltaH^{l-1} + DeltaAttnOut^{l-1} + DeltaV^{l-1}
  oracle DeltaH^l
model:
  per-layer PCA(64) + Ridge, 48 train / 16 heldout samples
```

判读标准：

- 如果 `DeltaV^{l-1}` 明显接近 `DeltaH^{l-1}` / `DeltaAttnOut^{l-1}`，说明只维护低秩 Value-update state 可能足够。
- 如果 `DeltaH` 类输入显著更强，说明 adaptor 需要维护 residual/hidden side state，而不能只从 KV 中递推。
- 如果 `oracle DeltaH^l` 对 V 很高但前一层信号低，说明问题在于跨层状态转移，而不是 `Wv` 投影。
- 如果所有上一层信号都弱，优先转向 functional/logit 监督或 link/adapter tokens，不再继续独立 KV L2 回归。

### Probe 结果

已在 64 条 Qwen3-8B Wiki layer-audit sample 上完成 48 train / 16 heldout probe。

| Target | Feature | Rank64 explained | Rank256 explained |
|---|---|---:|---:|
| K | mean template | 15.19% | 15.19% |
| K | `DeltaV^{l-1}` | 62.02% | 58.87% |
| K | `DeltaK/V^{l-1}` | 70.09% | 68.20% |
| K | `DeltaH^{l-1}` | 70.07% | 72.15% |
| K | `DeltaH/Attn/V^{l-1}` | 69.80% | 72.68% |
| K | oracle `DeltaH^l` | 73.59% | 77.79% |
| V | mean template | 2.33% | 2.33% |
| V | `DeltaV^{l-1}` | 18.65% | 6.94% |
| V | `DeltaK/V^{l-1}` | 16.35% | 4.56% |
| V | `DeltaH^{l-1}` | 17.90% | 22.78% |
| V | `DeltaH/Attn/V^{l-1}` | 20.86% | 25.15% |
| V | oracle `DeltaH^l` | 24.00% | 37.22% |

解释：

- Key 的跨层递推结构明显存在。上一层 `DeltaK/V` 或 `DeltaH` 可解释约 68%--73% 的 heldout `DeltaK^l`，远高于 mean template 的 15%。
- Value 不是简单的上一层 `DeltaV` 递推。`DeltaV^{l-1}` 在 rank64 只有 18.65%，rank256 还因样本不足/过拟合降到 6.94%；这不支持“只维护一个低秩 Value delta state 就能逐层修复 V”。
- `DeltaH^{l-1}` 和合并状态对 Value 略强于 `DeltaV^{l-1}`，但仍只有约 23%--25%。说明 Value 更新更依赖 residual/hidden trajectory，而不是单独的 Value tensor。
- oracle `DeltaH^l` 对 Value 可到 37.22%，但仍不是接近完整恢复；原因是本轮只用 48 train sample 和 PCA/ridge probe，不是直接使用冻结 `Wv`。这说明下一步应显式缓存/预测 normalized hidden correction，再通过主模型 `Wk/Wv` 生成 KV，而不是继续独立预测 Value coefficient。

下一步实验应转为两个分支：

1. **Hidden-sidecar adaptor**：离线为 document token 缓存低维 normalized-hidden sidecar，online 预测 `Delta s_l` 后过冻结 `Wk/Wv`，先验证 oracle sidecar / learned sidecar 的上界。
2. **Functional K-first adaptor**：在真实 RAG query 上优先验证 K-only 或 K-first 更新是否能接近 full attention，因为已有 functional ablation 显示修 K 对 attention logits 和 output 更直接。

### Hidden-sidecar 最小 probe

已复用 `hidden_vproj_audit_96_stride384`，测试最简单的静态 sidecar：

```text
z_l,t = PCA_l(offline own_h_l,t)
z_l,t -> DeltaH_l,t / DeltaV_l,t
```

64 train / 32 heldout，逐层 Ridge。排除 layer0 后结果如下：

| Target | Baseline mean | Sidecar rank32 | Sidecar rank64 | Sidecar rank128 | Sidecar rank256 |
|---|---:|---:|---:|---:|---:|
| DeltaH | 12.32% | 35.29% | 34.89% | 33.99% | 31.53% |
| DeltaV | 2.60% | 8.33% | 7.51% | 6.04% | 2.35% |

解释：

- offline hidden sidecar 对 `DeltaH` 有一定信息，但远不到可以直接恢复 context update 的程度。
- 对真正要修的 `DeltaV`，静态 `own_h` sidecar 基本不可用：rank32 只有 8.33%，rank 更高还过拟合下降。
- 因此“完全静态 preprocess v2 = 给每个 document token 预加一个固定 hidden/V bias”不是当前数据支持的方向。sidecar 如果要有用，必须作为 online adaptor 的输入之一，与前序 context / prefix summary / attention-output 监督结合。
- 下一步应验证 `prefix context + sidecar`，而不是只用 document 自身 sidecar。

### Context + sidecar probe

进一步把 `hidden_vproj_audit_96_stride384` 与 `wikitext_delta_train2k_stride384` 按 sample id 对齐，加入已有 `prefix_features`，测试：

```text
own_h
prefix_features
own_h + prefix_features
own_kv + prefix_features
own_h + own_kv + prefix_features
  -> DeltaH / DeltaV
```

64 train / 32 heldout，逐层 PCA+Ridge，排除 layer0 后主要结果：

| Target | Feature | Rank32 | Rank64 | Rank128 |
|---|---|---:|---:|---:|
| DeltaH | mean template | 12.32% | 12.32% | 12.32% |
| DeltaH | own_h | 35.29% | 34.89% | 33.99% |
| DeltaH | prefix | 7.55% | 3.42% | 3.42% |
| DeltaH | own_h + prefix | 30.48% | 31.50% | 30.60% |
| DeltaH | own_kv + prefix | 34.34% | 33.06% | 30.24% |
| DeltaV | mean template | 2.60% | 2.60% | 2.60% |
| DeltaV | own_h | 8.33% | 7.51% | 6.04% |
| DeltaV | prefix | -3.22% | -8.04% | -8.04% |
| DeltaV | own_h + prefix | 4.37% | 3.43% | 1.40% |
| DeltaV | own_kv + prefix | 7.39% | 5.51% | 1.58% |
| DeltaV | own_h + own_kv + prefix | 7.52% | 5.87% | 2.60% |

解释：

- 当前 `prefix_features` 是非常粗的 per-layer/head prefix KV summary。它单独预测 `DeltaV` 为负 explained energy，加入 `own_h` 后也没有提升。
- 这说明“sidecar + prefix mean”不是可用 adaptor；如果要继续 V，需要更强的 context encoder，例如 causal token-level prefix/target interaction、attention-output/logit supervision，或小型 student Transformer。
- `own_kv + prefix` 对 `DeltaV` 的 best layer 可达约 35%（多在 layer1），但全层平均仍只有 7%左右，说明局部浅层规律不能直接变成全层方案。
- 目前最稳的工程分支仍是 K-first：Key 的递推和 functional ablation 都更强；Value 分支需要重新定义训练目标，而不是继续在粗 prefix summary 上做 Ridge/MLP。

## 下一步：K-first Formal RAG Smoke

目的：在真实 FusionRAG formal pipeline 上验证“只修 Key”是否接近 full recompute，而不是继续只看 Wiki proxy 或 KV L2。

实现方式利用主 pipeline 已有开关：

```text
FUSIONRAG_FORCE_ALL_REPROCESS=1
FUSIONRAG_REPROCESS_UPDATE_MODE=kv / k_only / v_only / none
```

注意不能直接用 `rate=1`，因为 `test_fusionrag_reflect_preprocess_exp.py` 在 `rate == 1` 时会绕过 cache，直接 full prefill。正确 smoke 是设置一个非零 `rate`（例如 `0.01`）进入 `load_kv_and_generate`，再用 `FUSIONRAG_FORCE_ALL_REPROCESS=1` 强制所有 document token 重算。

本轮小样本设置：

```text
model: Qwen3-32B
cache: BGE preprocess cache
dataset: musique result_reflect
examples: main example 40--41
max_new_tokens: 32
modes:
  kv      = 写回重算 K 和 V，近似 full cache recompute
  k_only  = 只写回重算 K，V 保持 preprocess cache
  v_only  = 只写回重算 V，K 保持 preprocess cache
  none    = 重算但不写回 K/V，用于确认 force-all 路径的 no-update baseline
```

判读：

- 如果 `k_only` 的 answer / GLM accuracy 接近 `kv`，说明 K-first adapter 有系统价值。
- 如果 `v_only` 明显差或不稳定，和 Wiki functional ablation 一致：V 必须和正确 K/attention weights 配合。
- 如果 `none` 接近 preprocess rate0，说明 ablation 开关正常。
- 如果 `k_only` 与 `kv` 差距仍大，说明必须继续设计 Value functional adaptor 或 link/adapter tokens。

### K-first Formal RAG Smoke 结果

已完成两个尺度：

1. `example 40--41`：只有 1 个可测试 main example，4种 reprocess update mode 输出完全相同，只能验证路径可运行。
2. `example 40--49`：5 个可测试 main examples、11 个 subquestions，有区分度。

`example 40--49` 结果如下：

| Method | Definition | Main Acc | Sub Acc | Avg F1 | Avg EM | Same pred as full rate1 |
|---|---|---:|---:|---:|---:|---:|
| preprocess rate0 | BGE preprocess cache，不重算 | 2/5 = 40.00% | 8/11 = 72.73% | 0.4982 | 0.1818 | 3/11 |
| none | force-all reprocess，但不写回K/V | 2/5 = 40.00% | 8/11 = 72.73% | 0.3394 | 0.0909 | 5/11 |
| v_only | 只写回V | 2/5 = 40.00% | 7/11 = 63.64% | 0.4833 | 0.1818 | 6/11 |
| kv | 写回K/V，cache路径下的full recompute | 3/5 = 60.00% | 9/11 = 81.82% | 0.4199 | 0.1818 | 11/11 |
| full rate1 | 外层直接full prefill | 3/5 = 60.00% | 9/11 = 81.82% | 0.4199 | 0.1818 | 11/11 |
| k_only | 只写回K，V保持preprocess cache | 4/5 = 80.00% | 9/11 = 81.82% | 0.4593 | 0.1818 | 6/11 |

关键解释：

- `kv` 与 `full rate1` 的 11/11 predictions 完全一致，说明 `FUSIONRAG_FORCE_ALL_REPROCESS=1 + update_mode=kv` 可以作为真实 full recompute 的 cache-path 对照。
- `k_only` 达到和 full rate1 相同的 sub-question accuracy（9/11），且 main accuracy 在这个小样本上更高（4/5 vs 3/5）。不能解释成 K-only 已优于 full，因为样本太小、GLM judge 有噪声；但可以说明 K-only 没有明显崩，并且足以继续扩大验证。
- `v_only` 低于 `kv/full`，且低于 `k_only`，与 Wiki functional ablation 一致：只修V而K仍错，会让正确V被错误attention weights读取，系统收益不稳定。
- `none` 与 preprocess rate0 的 sub accuracy 相同，说明“强制重算但不写回”没有带来虚假收益，ablation 路径基本可信。

下一步：

1. 扩大到至少 50 个 testable main examples，优先比较 `preprocess_rate0 / full_rate1 / kv / k_only / v_only / none`。
2. 如果 `k_only` 在更大样本上接近 `full_rate1`，Adapter 设计应先服务 Key：预测 RoPE-aligned `DeltaK` 或 attention-logit correction。
3. Value 分支不要继续使用独立 L2 coefficient 目标；只有当 K-only 大样本显著落后 full，才投入 `attention-output/logit` 监督或 link/adapter token 方法。

### K-first Formal RAG 扩展结果：example 40--99

已扩展到 `example 40--99`，原 pipeline 自动跳过不可测试样本，最终覆盖：

```text
42 testable main examples
82 subquestions
```

| Method | Definition | Main Acc | Sub Acc | Avg F1 | Avg EM | Same pred as full rate1 |
|---|---|---:|---:|---:|---:|---:|
| preprocess rate0 | BGE preprocess cache，不重算 | 25/42 = 59.52% | 62/82 = 75.61% | 0.5921 | 0.2195 | 28/82 |
| none | force-all reprocess，但不写回K/V | 28/42 = 66.67% | 65/82 = 79.27% | 0.4762 | 0.1585 | 48/82 |
| v_only | 只写回V | 21/42 = 50.00% | 56/82 = 68.29% | 0.4775 | 0.1585 | 43/82 |
| k_only | 只写回K，V保持preprocess cache | 29/42 = 69.05% | 68/82 = 82.93% | 0.4942 | 0.1707 | 54/82 |
| kv | 写回K/V，cache路径下的full recompute | 37/42 = 88.10% | 77/82 = 93.90% | 0.5283 | 0.1829 | 80/82 |
| full rate1 | 外层直接full prefill | 37/42 = 88.10% | 77/82 = 93.90% | 0.5357 | 0.1829 | 82/82 |

解释：

- `kv` 与 `full rate1` 的 correctness 完全一致（82/82），预测文本 80/82 一致，说明 cache-path force-all KV reprocess 仍是可信 full 对照。
- `k_only` 大样本下不再接近 full：sub accuracy `82.93%`，低于 full/kv 的 `93.90%`，但高于 preprocess rate0 的 `75.61%` 和 none 的 `79.27%`。
- 因此 K-only 是有效但不充分的 update。它能回收一部分 full recompute 收益，但不能替代完整 KV 更新。
- `v_only` 明显最差（sub accuracy `68.29%`），进一步证明“只修V而K不修”会破坏读取。
- `none` 高于 preprocess rate0 但低于 K-only/full，说明 reprocess路径和judge存在一些非KV因素或生成随机性/文本差异影响；不过 `kv≈full` 的强一致性说明主要对照仍可信。

更新后的方法判断：

1. **Key-first 是正确优先级，但不是终点。** 应先设计轻量 Key/logit adapter，因为它已经带来端到端增益。
2. **Value 不能完全放弃。** K-only 与 full 还有 `11` 个百分点 sub accuracy gap，需要 Value 或 attention-output 层面的补偿。
3. **下一阶段不应再做独立 Value L2 predictor。** 更合理路线是 `K/logit adapter + functional Value correction`，其中 Value 用 query attention output / logits 监督，而不是直接拟合每层 `DeltaV`。

### Partial Value Layer Ablation

为判断 Value correction 是否必须覆盖所有层，新增一个只对 `k_only` 生效的环境变量：

```text
FUSIONRAG_REPROCESS_KEEP_VALUE_LAYERS=last8 / last16 / last32 / all
```

语义：仍然重算全部 document token，并写回所有 Key；Value 只保留指定层的重算结果，其余层恢复为 preprocess cache 的 Value。默认不设置时就是原始 `k_only`。

`example 40--99` 结果：

| Method | Value updated layers | Main Acc | Sub Acc | Avg F1 | Same pred as full |
|---|---:|---:|---:|---:|---:|
| k_only | 0 | 29/42 = 69.05% | 68/82 = 82.93% | 0.4942 | 54/82 |
| k + V last8 | 8 | 29/42 = 69.05% | 67/82 = 81.71% | 0.4930 | 52/82 |
| k + V last16 | 16 | 34/42 = 80.95% | 72/82 = 87.80% | 0.4955 | 53/82 |
| k + V last32 | 32 | 36/42 = 85.71% | 74/82 = 90.24% | 0.5302 | 69/82 |
| full KV / rate1 | 64 | 37/42 = 88.10% | 77/82 = 93.90% | 0.5357 | 82/82 |

解释：

- 只补最后 8 层 Value 没有收益，甚至略低于 K-only。
- 补最后 16 层 Value 明显提升：sub accuracy 从 `82.93%` 到 `87.80%`。
- 补最后 32 层 Value 接近 full：`90.24%` vs `93.90%`，但仍差 3 个 subquestions。
- 这说明 Value correction 的有效部分集中在中后层，但不是最后几层就够；未来轻量 Value adaptor 可以优先只服务后半层，避免全 64 层同等建模。

更新后的设计方向：

```text
Stage A: Key/logit adapter，全层或多数层
Stage B: Value adaptor，只优先覆盖后 16--32 层
Training target: attention-output / decode-logit functional loss
Avoid: full-layer independent DeltaV L2 regression
```

### Scheme B: layer-parallel update probe

用户提出的并行化假设：正常重算是逐层递归的，`h_i^(l-1)` 更新后再产生第 `l` 层 K/V；如果为了避免完整 Transformer recompute，能否让每一层独立使用离线缓存的 hidden state，直接读当前 RAG 前缀 KV，然后预测本层 K/V gap？

本轮实验只验证这个机制信号是否存在，不训练端到端模型。输入没有使用真实 RAG full hidden，因此不构成数据泄漏：

- `own_h[l,t]`：目标 chunk/doc 单独离线 prefill 时，在每层 `v_proj` 前 hook 到的 hidden sidecar。
- `own_kv[l,h,t]`：目标 chunk/doc 单独离线缓存 K/V。
- `prefix_kv[l,h,p]`：随机 Wiki prefix 的离线 K/V。
- target：同一个目标 token 在 `prefix + target` full prefill 下的 `DeltaK/DeltaV = full - own`。

方案 B 的 parallel feature 构造：对每一层独立，用 `own_h[l,t]` 经过原模型 `q_proj/q_norm/RoPE(full position)` 得到 query，只 attend 到 prefix KV，得到 `parallel_context[l,h,t]`。然后做 per-layer/per-head PCA + Ridge，比较三类输入：

| Feature | 含义 |
|---|---|
| `own_kv` | 只从离线目标 KV 预测 Delta |
| `parallel_context` | 只从本层并行读 prefix 的 context 预测 Delta |
| `own_kv_parallel_context` | 二者拼接 |

设置：Qwen3-8B，Wiki synthetic `96` records，`64` train / `32` test，rank `64`，每层每 KV-head 单独 fit。

结果：

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

分层现象：

- K 的 `own_kv` 在中前层较强，L8--15 平均 explained energy `0.6143`，top layers 为 L8/L10/L7/L35/L11。
- K 的 `parallel_context` 单独较弱，整体只有 `0.1469`，并且拼接后只从 `0.5042` 提升到 `0.5046`，几乎没有增益。
- V 的三种轻量输入都弱：`own_kv` 只有 `0.0376`，`parallel_context` 为负，说明该并行 prefix-read 特征不能解释 Value Delta。

结论：

1. 朴素方案 B，即“每层用离线 hidden 生成 query，只读 prefix KV，然后并行预测本层 Delta”，目前不成立。它没有提供超出 `own_kv` 的有效增益。
2. 这个结果不否定所有并行 adapter，但否定了最简单的 `own_h -> q -> prefix attention context -> DeltaKV` 形式。
3. 失败原因很可能是它缺了重算递归中的两个关键因素：当前 doc 内部 `<i` tokens 的更新轨迹，以及上一层 updated hidden 继续传递到下一层的非线性累积。
4. 下一步如果继续探索非递归/半并行方案，应先把 attention memory 从 only-prefix 改成 `prefix + target(<i)`，再比较；如果仍无增益，则应回到 K/logit-first functional adapter，而不是试图直接并行预测全层 Value Delta。

当前判断：FusionRAG KV adapter 的可行路线仍应优先围绕 Key/logit correction；Value correction 更像中后层的功能性补偿，不能靠当前简单并行 prefix feature 直接完成。

### Scheme B revision: prefix + offline previous target memory

上一轮 Scheme B 只让 sampled token attend 到 prefix KV，缺少真实自回归重算里 token_i 可见的当前 document 内部 `<i` tokens。因此本轮补做一个更严格的版本：

```text
memory(token_i, layer_l) = prefix_kv[l] + offline_target_kv[l, :i]
query(token_i, layer_l)  = q_proj(offline_own_h[l, i]) + q_norm + RoPE(full position)
```

关键约束：`offline_target_kv[l, :i]` 来自 target/doc 单独离线 prefill，不是 full-context previous-token KV，也不是 full hidden；因此仍然没有真实 RAG 数据泄漏。它相当于一种增强 cache sidecar。

新增采集字段：

- `own_target_kv_all`: `[36, 8, 128, 256]`，完整 target/doc 离线 K/V。
- `prefix_kv`: `[36, 8, 256, 256]`，prefix 离线 K/V。
- `delta_kv`: 仍只评估 8 个 sampled target tokens。

正式设置：Qwen3-8B，Wiki synthetic `96` records，`64` train / `32` test，rank `64`，每层每 KV-head 单独 PCA+Ridge。

原始 Delta 幅度：

| Target | Mean relative L2 gap | Min | Max |
|---|---:|---:|---:|
| K | 0.2421 | 0.1920 | 0.3300 |
| V | 0.4477 | 0.3440 | 0.6080 |

结果：

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

解释：

1. 加入 `target(<i)` 后，K 的纯 context feature 有小幅提升：`0.1469 -> 0.1715`。这说明当前 doc 内部前序 token 的离线 KV 确实带来一点额外上下文信号。
2. 但这个信号与 `own_kv` 已有可预测成分高度重合，拼接后几乎没有增益：`0.504641 -> 0.504645`。
3. V 仍然无效，甚至 `parallel_prefix_prevtarget` 仍为负；说明 Value Delta 不是简单由本层 offline hidden query 读取离线 prefix/doc KV 就能恢复。
4. 因此更严格的半并行 Scheme B 仍然不够。真实重算中重要的部分不是“本层一次 attention read 的 context 向量”，而更可能是逐层递归更新后的 hidden trajectory。

当前路线判断：

- 继续追求完全 layer-parallel 的直接 DeltaKV predictor，收益不明显。
- Key 的 `own_kv` 可预测性仍值得利用，但应把目标转成 functional/logit correction，而不是试图用并行 context 直接补 DeltaKV。
- Value correction 应优先按中后层 functional loss 或 attention-output loss 做，而不是按 per-layer L2 Delta 独立拟合。

### Functional K adapter probe: predicted K with shrink gate

前面 per-layer DeltaKV probe 说明：直接解释完整 DeltaKV 很难，尤其 V 很弱。但端到端 RAG 真正关心的是 query 读取 document 时的 attention logits/output 是否接近 full recompute。因此本轮改成 functional proxy：

```text
Train:  offline own_kv(sampled target tokens) -> RoPE-aligned DeltaK
Eval:   predicted_K = rotate_to_full_position(offline_K + lambda * predicted_DeltaK)
Metric: future query tokens attend to document X only, compare logits/output to fullK+fullV teacher
```

训练输入只用 offline target K/V；没有使用 eval 样本的 full Delta coefficient。评价中的 full K/V 和 full hidden 只作为 teacher-forced metric reference。

设置：

- train: `64` Wiki all-target records, 每条 8 sampled tokens, per-layer/per-KV-head PCA+Ridge K predictor, rank `64`。
- eval: `64` heldout functional spans, 每条结构为 `prefix(256) + target/doc(128) + query(32)`。
- variants: cachedK/cachedV, fullK/cachedV, cachedK/fullV, predictedK(lambda)/cachedV。

正式结果：

| Method | K source | V source | Logit rel error | Output rel error |
|---|---|---|---:|---:|
| cachedK_cachedV | offline cache | offline cache | 0.2639 | 0.8723 |
| fullK_cachedV | full recompute K | offline cache | 0.0000 | 0.6067 |
| cachedK_fullV | offline cache | full recompute V | 0.2639 | 1.7269 |
| fullK_fullV | full recompute K | full recompute V | 0.0000 | 0.0000 |
| predK lambda=0.001 | offline + 0.001 predDeltaK | offline cache | 0.2637 | 0.8718 |
| predK lambda=0.003 | offline + 0.003 predDeltaK | offline cache | 0.2634 | 0.8707 |
| predK lambda=0.01 | offline + 0.01 predDeltaK | offline cache | 0.2625 | 0.8669 |
| predK lambda=0.03 | offline + 0.03 predDeltaK | offline cache | 0.2597 | 0.8556 |
| predK lambda=0.1 | offline + 0.1 predDeltaK | offline cache | 0.2504 | 0.8107 |
| predK lambda=0.3 | offline + 0.3 predDeltaK | offline cache | 0.2287 | 0.6477 |
| predK lambda=0.6 | offline + 0.6 predDeltaK | offline cache | 0.2134 | 0.6304 |
| predK lambda=1.0 | offline + 1.0 predDeltaK | offline cache | 0.2301 | 0.9246 |

相对 `cachedK_cachedV -> fullK_cachedV` 的 output-error gap，`lambda=0.6` 回收约 `91.1%`：

```text
cachedK_cachedV output error = 0.8723
fullK_cachedV   output error = 0.6067
predK_s0.6      output error = 0.6304
recovery = (0.8723 - 0.6304) / (0.8723 - 0.6067) = 91.1%
```

解释：

1. K adapter 方向重新变得有希望。虽然直接 DeltaK L2 predictor 只解释约一半 Delta energy，但在 query-attention functional metric 上，一个简单 `own_kv -> DeltaK` ridge predictor 加 shrink gate 已经能显著改善 logits/output。
2. 裸加完整预测 DeltaK 会过冲：`lambda=1.0` 的 output error `0.9246`，比不更新 `0.8723` 还差。因此未来 adapter 不能直接输出无约束 Delta，应带 gate/scale/residual norm control。
3. `cachedK_fullV` 仍然很差：`1.7269`。这再次说明 V 不能在 K 错的时候单独替换；应先修 K/logits，再考虑中后层 V correction。
4. `predK_s0.6` 尚未超过 oracle `fullK_cachedV`，但已经接近：`0.6304` vs `0.6067`。这说明 lightweight K adapter 可能足以回收大部分 K-side functional gain。

当前建议路线：

```text
Stage 1: K/logit adapter with learned or calibrated gate lambda
  input: offline doc KV, layer/head/token metadata, maybe prefix summary
  output: constrained DeltaK or direct logit correction
  loss: functional query-attention logits/output, not only DeltaK L2

Stage 2: Value correction only after K is stabilized
  priority: mid/late layers
  loss: attention-output or decode-logit functional loss
```

### Gate calibration: global vs layer-wise K gate

上一轮 `predK_s0.6` 是直接在 64 条 heldout functional spans 上扫 scale 后报告，可能仍有 test-tuning 风险。因此本轮做严格一点的 calibration/test split：

```text
Train predictor: samples 0--63 all-target sidecar
Calibrate gate: functional spans 96--127
Test gate:      functional spans 128--191
```

流程：

1. 仍使用 per-layer/per-KV-head `own_kv -> DeltaK` PCA+Ridge predictor，rank `64`。
2. 在 calibration split 上，从候选 scale `0, 0.03, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0` 中选择：
   - 一个全局最佳 scale；
   - 每层一个最佳 scale。
3. 在独立 test split 上只评估选出的 gate，不再用 test 选择 scale。

结果：

| Split | Method | Logit rel error | Output rel error |
|---|---|---:|---:|
| calib | cachedK_cachedV | 0.2517 | 0.8747 |
| calib | fullK_cachedV | 0.0000 | 0.5863 |
| calib | best global scale = 0.45 | 0.2057 | 0.5619 |
| test | cachedK_cachedV | 0.2753 | 0.8807 |
| test | fullK_cachedV | 0.0000 | 0.5230 |
| test | predK global gate, scale 0.45 | 0.2279 | 0.5738 |
| test | predK layer-wise gate | 0.2250 | 0.5712 |
| test | cachedK_fullV | 0.2753 | 1.7387 |

Test output-error gap recovery against `cachedK_cachedV -> fullK_cachedV`：

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

解释：

1. 严格 calibration/test 下，K gate 结论仍成立。全局 scale `0.45` 在 test 上把 output error 从 `0.8807` 降到 `0.5738`，接近 fullK_cachedV 的 `0.5230`。
2. Layer-wise gate 只小幅提升：`0.5738 -> 0.5712`，gap recovery `85.8% -> 86.5%`。这说明 scale 的主要收益来自“控制整体 DeltaK 幅度”，不是很复杂的 layer pattern。
3. fullK_cachedV 在 test 上是 `0.5230`，predK layer-wise 是 `0.5712`，仍有剩余 gap，后续可以通过更好的 K predictor 或 functional/logit training 继续缩小。
4. `cachedK_fullV` 仍然很差：`1.7387`。Value 更新必须等 K/logit 稳定后再做。

更新后的建议：

```text
First deployable adapter prototype:
  K path: offline KV -> DeltaK direction predictor -> calibrated global/layer gate -> constrained KV update
  Gate: start with global or layer-wise scalar, not token-wise complex gate
  Training/eval: functional query-attention loss + heldout calibration
  V path: postpone; later add mid/late-layer functional V correction
```

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

