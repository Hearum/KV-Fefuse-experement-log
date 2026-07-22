# Structured KV Adapter 调研计划

## 1. 核心问题

完整重算中，第 `l` 层 token `i` 的更新不是只经过新的 `Wk'/Wv'`：

```text
h_i^(l) = TransformerLayer_l(h_i^(l-1), prefix KV^(l))
K_i^(l) = Wk_l h_i^(l-1)
V_i^(l) = Wv_l h_i^(l-1)
```

因此直接从 cached `K_i,V_i` 预测 `Delta K_i,Delta V_i` 是否可行是待验证假设，不是既定事实。本项目依次回答：

1. cached token state 是否已包含足够信息？
2. 若不足，加入当前 preceding-document KV 的轻量 summary 是否足够？
3. 若仍不足，是否需要一个低维、按 layer/chunk 递推的 side network？
4. Adapter 应一次并行输出所有 token 更新，还是按 chunk 串行、chunk 内并行？

## 2. 不可变实验口径

### 2.1 Target 与 baseline

- Prompt：真实 `system + doc1 + ... + docN + query`。
- Target：同一 prompt、同一 token、同一位置下 `rate=1` full document recompute KV。
- 起点：raw rate=0 与 BGE preprocess rate=0 分开训练、分开报告。
- K：只报告 RoPE-aligned/local Key；V：直接报告。
- 主指标：`original gap = ||T-B||/||B||`、`remaining Delta = ||Delta-Delta_hat||/||Delta||`、`final KV error = ||T-(B+Delta_hat)||/||B||`、cosine、explained variance。
- 最终指标：原 FusionRAG decode + GLM judge 的 main/sub-question accuracy、token F1、latency、显存、额外 cache/参数量。

### 2.2 数据切分

- smoke test：2 examples，只检查 token offset、shape、RoPE 和写回正确性。
- pilot：不少于 50 examples；按 main example 划分 train/validation/test，禁止同一 main example 的 document 跨 split。
- scaling：pilot 有效后才扩大；不得先跑 235B。
- 所有 PCA、标准化、超参和 early stopping 只使用 train/validation；test Delta 不得用于 basis 或 coefficient。

## 3. Adapter 候选范式

### A. 一次性 Token-local Adapter（最低成本）

每层、每 KV head 独立，但 K/V 使用同一低维 latent coefficient：

```text
z_i^l = Encoder_l(Kcache_i^l, Vcache_i^l, token_position, doc_position)
c_i^l = MLP_l(z_i^l)                         # r=8/16/32/64
DeltaK_i^l = U_K^(l,h) c_i^l
DeltaV_i^l = U_V^(l,h) c_i^l
```

这是完全并行、一次性输出所有 document token gap 的下界方案。共享 `c` 用于约束 K/V 一致性；同时保留 K/V 独立 coefficient 作为消融。

### B. Prefix-conditioned 并行 Adapter（主候选）

为每层从当前已加载的前序 document KV 计算固定维度 summary：

```text
p_i^l = PrefixPool({K_j^l,V_j^l | j < doc_i})
c_i^l = MLP_l(z_i^l, p_i^l, position features)
```

比较 mean/max pooling、最后一个 chunk summary、分 head attention pooling。先按 document 生成一个 context code，再一次性预测该 document 全部 token coefficient。该范式按 document 串行，但 document 内 token 并行，不运行原 Transformer attention/MLP。

### C. 低维层间递推 Adapter（表达能力上界）

若 B 明显不足，引入小型 recurrent side state：

```text
s_i^0 = OfflineCode(doc_i)
s_i^l = GRU/2-layer-MLP(s_i^(l-1), p_i^l, layer_embedding)
c_i^l = Head_l(s_i^l)
DeltaK_i^l = U_K^l c_i^l; DeltaV_i^l = U_V^l c_i^l
```

`OfflineCode` 可由缓存 KV 生成，或额外离线存储每 token/layer 32/64 维 hidden code。比较一次整篇输出与逐 document 串行；不做 token 自回归串行，因为其延迟不可接受。

## 4. 分阶段实验

### Stage 0：Target 与信息泄漏审计

1. 复用 formal residual 数据，随机抽查 raw/preprocess 的 `B/T/Delta` shape、offset、token ids。
2. 验证第一个 document 除数值误差外 Delta 接近零，后续 document Delta 随前缀变化。
3. 验证后置 query 变化不改变 rate=1 document target。
4. 检查现有 adapter loader 对 local K、online RoPE、V 写回的顺序。

产物：`results/stage0_audit.json`。任一对齐检查失败则停止训练并修复采集。

### Stage 1：表示能力上界（先判断 rank 是否值得做）

对 K/V 分开及 tied-KV latent，测试 `r={8,16,32,64,128}`：

- per-layer/head oracle SVD；
- train-only global basis，held-out oracle coefficient；
- per-document basis（仅作为存储较大的对照）；
- context-template basis 与 token-wise feature basis。

回答低 rank 缺点是否可接受：rank 越高，参数、cache bandwidth、coefficient 输出和矩阵乘成本线性增加；当 rank 接近 head_dim=128 时，低秩 Adapter 相对直接预测 Delta 的优势基本消失。

准入：在严格 held-out 上，rank32/64 至少恢复 Value Delta 的 60%，并使 final KV gap 相对 preprocess baseline 降低 40%；否则不训练 coefficient predictor，转向局部/chunk/layer recurrence。

### Stage 2：Predictability Ladder

固定 Stage 1 basis，只预测 coefficient，按信息量递增：

1. constant mean；
2. cached X K/V；
3. position、prefix token 数、document rank；
4. cached X + preceding KV summary；
5. oracle preceding updated KV summary（不可部署上界）；
6. 低维层间递推 state。

模型仅用 Linear/Ridge、两层 MLP、轻量 GRU。比较预测 coefficient 与 oracle coefficient 的 R2，同时必须报告最终 K/V gap，避免 coefficient 标度造成误导。

关键诊断：若 4 显著优于 2/3，Adapter 必须 context-conditioned；若 5 显著优于 4，则在线更新应按 document 串行，并将已更新 KV 纳入下一个 document 的 summary；若 4≈5，可一次并行处理全部 docs。

### Stage 3：端到端 KV 写回

只推进 Stage 2 最优且严格 held-out 有效的模型：

- 在原 pipeline 增加独立 `load_kv=raw/preprocess + structured_adapter` 接口；
- rate=0，不调用原 Transformer recompute；
- 检查 K/V 写回、generation logits 和 latency；
- 对照 raw rate0、preprocess rate0、full rate1、现有 selector recompute。

最低可用标准：在至少 50-example 严格 test 上，端到端 accuracy/F1 稳定优于 preprocess rate0，并回收 preprocess 到 full rate1 差距的至少 50%；额外 prefill 延迟显著低于 rate1，且独立复现实验方向一致。

### Stage 4：系统优化

仅在 Stage 3 达标后进行：basis int8/fp8、layer/head pruning、只更新高能后层、selector support、CUDA fused low-rank update、离线 code 存储成本评估。

## 5. 首轮具体执行顺序

1. 建立 frozen manifest：50 examples，main-example-disjoint split。
2. Stage 0 在 2 examples 运行并人工审计一条 document span。
3. 从已有 formal preprocess residual 先做 Stage 1；缺少 raw 配对时再补采集。
4. 只对通过准入的 rank 训练 Stage 2 的 Ridge/MLP。
5. 先在 5 held-out examples 写回生成，确认无异常后跑完整 test 和 GLM judge。

## 6. 必须报告的消融

- raw vs preprocess；K vs V；K-only/V-only/joint K+V。
- tied coefficient vs independent coefficient。
- rank 8/16/32/64/128。
- token-local vs prefix summary vs updated-prefix oracle。
- all layers vs后 16/32 layers；all heads vs energy-selected heads。
- document 内并行 vs document 串行。

## 7. 风险与停止标准

- 若 held-out oracle basis 都不能显著缩小 gap，低秩方向不可行，不用 MLP 掩盖表示瓶颈。
- 若 oracle coefficient 有效而预测失败，问题是输入信息不足；增加 prefix state，不扩大网络盲目拟合。
- 若 KV gap 下降但端到端输出退化，检查 K/V coupling、RoPE、layer accumulation 和误差分布，不能宣称方法有效。
- 任何训练/评估结果必须包含样本数和 split；少量样本只称 smoke test。
- 不删除 cache、结果、README 或日志，不操作 qjy001/235B。
