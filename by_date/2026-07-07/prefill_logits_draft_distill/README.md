# Prefill Logits Draft Distillation 探究路线

## 目标

这条路线研究：能否不直接蒸馏 selector score，而是蒸馏一个真正的小 online draft model，使它在 prefill 阶段的行为接近 teacher model。训练完成后，不直接用 logits 选 token，而是像现有 DraftModel selector 一样，从 student draft model 的 prefill attention 中导出 doc token importance，再用于 FusionRAG 的 online token selection。

这和当前 `selector_aware_draft_model` 主线的区别是：

- selector-score KD：直接拟合 full DraftModel / oracle 给出的 doc-token importance distribution。
- prefill-logits KD：拟合 teacher 在整段 prefill 上的 next-token distribution，让 student 成为一个更像 teacher 的小 LM；selector 能力是从 student attention 中自然读出来的。

## 为什么值得做

如果我们希望得到一个可解释为“小 draft model”的模块，而不是一个专门训练的 selector head，那么 prefill logits KD 更自然：

- 训练目标不绑定固定 top-k / rate；
- student 输出完整语言模型分布，未来可以兼容不同 selection rate；
- 方法更接近常规 LLM 蒸馏 / draft model 训练范式；
- 可能比显式拟合某个 selector 更有泛化性。

## 核心假设

假设：如果 student 在 RAG prompt 的 prefill logits/hidden behavior 上更接近 teacher，那么 student 的 attention-based token importance 也会更接近强 teacher selector，从而提升 FusionRAG 的 online selection 质量。

这个假设必须实验验证，因为 logits 相似不必然意味着 attention 分布相似。

## Teacher 选择

需要区分三种 teacher，不能混用：

| teacher | 含义 | 优点 | 风险/成本 |
|---|---|---|---|
| Qwen2.5-3B full DraftModel | 压缩现有 DraftModel | 成本相对低，直接对齐当前 draft selector 生态 | 只是复刻 draft model，不一定接近 Qwen3 主模型需求 |
| Qwen3-32B main model | 蒸馏主模型 prefill 行为 | 最接近真实 FusionRAG 目标模型 | 成本高，logits 存储/计算压力大 |
| Hybrid teacher | logits 用 Qwen3，attention/selector eval 对齐 full Draft / oracle | 兼顾主模型行为和 selector 评估 | 设计复杂，容易解释不清 |

初步建议：先做 Qwen2.5-3B teacher 的小规模 sanity，验证训练和 eval pipeline；如果 attention selector 指标有效，再切 Qwen3-32B teacher。

## Student 定义

候选 student：

1. `native first4`：Qwen2.5-3B embedding + 前 4 层初始化。
2. `native first8`：前 8 层初始化。
3. `native first12`：前 12 层初始化。
4. 结构裁剪版小 LM：保留完整 LM head / embedding tying，减少层数。

注意：如果 teacher 是 Qwen3-32B，而 student 从 Qwen2.5-3B 初始化，tokenizer/vocab/architecture 是否兼容必须先确认。若 tokenizer 不一致，不能直接做 logits KL，需要统一 tokenizer 或改成 hidden/ranking teacher。

## 数据构造

训练样本应尽量包含 RAG 风格 prompt，而不是只用普通 WikiText continuation。

候选数据：

- WikiText-103：用于通用 prefill KD，成本低，泛化性待验证。
- MuSiQue retrieved passages + question：更接近真实 RAG prompt。
- Calibration RAG prompts：从训练/校准 split 中构造，不泄漏测试答案。
- Mixed corpus：WikiText + RAG prompts 混合，避免 student 只记 RAG 模板。

每条样本结构：

```text
[system/template tokens] + [retrieved documents] + [query/question tokens]
```

Teacher/student 都对同一段 input 做 causal LM forward。训练可只对 query 段、doc 后半段、或全序列位置做 logits KL，需要消融。

## 监督信号

主 loss 应该是 rate-agnostic：

- `KL_logits`: teacher/student next-token logits 分布 KL。
- 可选 `CE`: 对真实 next token 做 CE，防止 student 只拟合 teacher noise。
- 可选 `hidden KD`: 对某些 hidden states 做 MSE/cosine，但不是第一优先级。

不应该使用：

- 固定 top-15 token hard label；
- 固定 residual 5% label；
- 只为某一个 selection rate 优化的 top-k mass loss。

## Logits 存储与计算策略

全 vocab logits 对长 prompt 很大，建议几种实现方式：

1. 在线 KD：teacher 和 student 同步 forward，直接算 KL，不落盘 logits。成本高但实现清晰。
2. Top-k logits cache：只缓存 teacher top-k logits，加一个 tail mass 近似。节省空间，但 KL 近似。
3. Chunked logits cache：按 sequence chunk 落盘，便于复用但 IO 较重。
4. Query-only logits KD：只对 query 段位置算 KL，减少计算量；但可能弱化 doc-token prefill 行为。

Pilot 阶段建议在线 KD 或小规模 top-k logits cache，先验证方向。

## 评估指标

必须同时评估 “LM 蒸馏是否成功” 和 “selector 是否变好”。

### LM 侧

- validation KL logits；
- validation CE / perplexity；
- teacher top-k token recall。

### Selector 侧

用训练后的 student 按 DraftModel selector 方式产生 doc-token score，与强 teacher 比较：

- R@5%、R@10%、R@15%、R@30%；
- Jaccard@5/10/15/30；
- score cosine / Spearman；
- selection latency。

### FusionRAG 真实任务

最终必须跑真实 pipeline：

- `offline10 + student residual5`；
- `student online rate015`；
- 必要时 sweep rate 0.05/0.15/0.30。

对照表必须包含：

| baseline | 说明 |
|---|---|
| online_draft_rate015 | full DraftModel selector |
| offline10_full_draft_residual005 | offline fixed 10% + full Draft residual 5% |
| offline10_middle_residual005 | offline fixed 10% + middle layer residual 5% |
| offline10_layer4_residual005 | offline fixed 10% + layer4 residual 5% |
| HS hidden_head | 显式 selector-head 蒸馏路线 |
| CD causal distill old | 旧 causal LM KD 结果，作为失败/弱 baseline |
| prefill_logits_student | 本路线的新 student |

## 实验阶段

### Stage 0: Pipeline Sanity

目的：确认 logits KD 训练、保存、加载、selector eval 全流程能跑通。

- Teacher：Qwen2.5-3B full DraftModel。
- Student：first4 或 first8。
- 数据：小规模 Wiki/RAG prompt，各 1k-5k。
- 训练：少量 step。
- 只看 training KL 是否下降、selector R@15 是否有非随机信号。

### Stage 1: RAG-domain Small Pilot

目的：验证 RAG prompt 上 logits KD 是否比普通 WikiText 更能提升 MuSiQue selector alignment。

- Teacher：Qwen2.5-3B full DraftModel。
- Student：first4/first8/first12。
- 数据：MuSiQue calibration prompts，不使用测试答案泄漏。
- Eval：MuSiQue validation selector alignment。

### Stage 2: Qwen3 Teacher Pilot

目的：验证蒸馏主模型 prefill behavior 是否比复刻 Qwen2.5 draft 更接近最终任务。

- Teacher：Qwen3-32B main model。
- Student：若 tokenizer/architecture 兼容，使用对应小 Qwen/Qwen3 初始化；否则先不做 logits KL，改成 hidden/ranking teacher。
- Eval：selector alignment + FusionRAG accuracy。

### Stage 3: Full Pipeline Accuracy

如果 Stage 1/2 selector alignment 超过 native middle/first12，再跑完整 FusionRAG：

- 200-example / full dataset；
- rate=0.15 必跑；
- 与 full Draft、middle、HS 同表比较；
- 记录 selection time 和 prompt eval time。

## 风险

1. Logits 相似不保证 attention selector 好。
2. WikiText 训练可能无法迁移到 RAG prompt。
3. Qwen3 teacher 成本高，且 tokenizer/architecture 兼容性需要先确认。
4. 如果 student attention 结构本身不足，即使 logits KD 成功，也可能无法产生高质量 token importance。
5. 需要防止把 selector 问题又变成固定 top-k 分类问题；本路线必须保持 rate-agnostic。

## 当前状态

- 2026-07-07：创建路线文档。尚未启动实验。
- 下一步：先检查 Qwen3/Qwen2.5 tokenizer 兼容性和旧 CD 代码，确定 Stage 0 是复用旧 causal KD 脚本还是重写一个最小在线 logits KD trainer。

## 2026-07-07 数据泄露修正

刚才曾临时从 `data/musique-200.jsonl` 构造过 `musique_rag_prompts.txt`，但尚未启动训练。该做法存在数据泄露风险：如果后续仍在这 200 条 MuSiQue 上报告 selector overlap 或 FusionRAG accuracy，就不能把同一批样本用于 logits KD 训练。

已删除该临时文件，并将这条约束写入路线：

- 禁止使用最终评测集 `data/musique-200.jsonl` / `data/result_reflect.json` 作为训练数据；
- 只允许用于 evaluation 或只读 sanity inspection；
- 训练数据应改为外部预训练语料、明确 disjoint 的 calibration split、或另一个不参与最终评测的 RAG 数据集；
- 如果使用 MuSiQue，必须重新划分 train/calibration/eval，并且最终报告只使用未参与训练的 held-out split。

下一步训练路线改为：先用 WikiText / 外部文本做纯 logits KD smoke；若要 RAG-domain prompt KD，需要先找 disjoint 的 MuSiQue train/dev 或其他 RAG corpus，不能直接用当前 200 条评测样本。

## 2026-07-07 路线收敛：直接小 draft LM 蒸馏

用户纠正：不需要构造 question 或 RAG 风格 prompt。目标就是蒸馏出一个更小的 draft model，本质要求是小模型在普通 prefill 中形成的 attention importance distribution 接近大模型。

因此后续路线调整为：

- 输入：普通连续文本 sequence，不构造 QA/RAG prompt；
- Teacher：完整 Qwen2.5-3B DraftModel；
- Student：同 tokenizer/同架构裁剪成 first4/first8 等小模型；
- 训练目标：标准 causal LM logits KD + 可选 CE，同时加入 rate-agnostic 的 attention-importance KD；
- 不训练 selector head；不使用固定 top-k hard label；
- 训练后用 student 自身 attention 作为 FusionRAG online selector 的 token importance。

旧 CD 只做 logits KL + CE，验证发现 selector overlap 低于未微调 layer4。因此新实验需要补充 attention-importance KD，直接约束“小模型认为哪些历史 token 重要”的分布，而不是只让 next-token logits 变像。

## 2026-07-07 启动：first4 小 draft LM attention-aware distillation

### 设置

- 输入：WikiText-103 连续文本 blocks，不构造 question/RAG prompt。
- Teacher：完整 Qwen2.5-3B-Instruct。
- Student：Qwen2.5-3B-Instruct 初始化后裁剪为前 4 层。
- Loss：`logits_KL + 0.1 * CE + 1.0 * attention_row_KL`。
- attention teacher：完整 teacher 后 18 层 attention 平均。
- attention student：student 最后一层 attention。
- 训练：seq_len=256，max_steps=500，GPU0-3，DDP 4 ranks。
- 输出：`MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/outputs/first4_wikitext_attn_kd_s256_step500`
- 日志：`MOTIVATION_EXPERIMENTS/prefill_logits_draft_distill/logs/first4_wikitext_attn_kd_s256_step500.log`

### 说明

这是“直接蒸馏小 draft LM”的第一版 attention-aware 实验。它不使用 MuSiQue 评测集训练，不使用固定 top-k label，也不训练 selector head。后续评估要看：student 自身 attention 作为 selector 时，是否比旧纯 logits CD 更接近 full DraftModel selector。

### 初始运行状态

训练已启动并稳定输出 loss。早期日志显示：

- step 1: loss=25.1646, logits=24.3316, KL=22.3863, CE=19.4524, attn=0.8330
- step 10: loss=24.1442, logits=23.3944, KL=21.5424, CE=18.5201, attn=0.7498
- step 60: loss=20.2896, logits=19.5839, KL=18.0269, CE=15.5703, attn=0.7056

初步观察：logits/CE/KL 明显下降，attention KL 从 0.83 降到约 0.70 后波动。当前只能说明训练链路正常，不能说明 selector 效果已经改善；必须等 checkpoint 后跑 selector-overlap eval。

### step500 selector-overlap eval

训练完成后，使用 `eval_cd_layer4_against_full_draft.py` 评估 student 自身 attn_prob selector 与完整 Qwen2.5-3B DraftModel teacher score 的重合度。

- checkpoint: `outputs/first4_wikitext_attn_kd_s256_step500/training_state_final_step000500.pt`
- eval output: `eval_first4_attn_kd_step500_vs_full_draft`
- Wiki val: 5000
- MuSiQue val: all 685 teacher-cache pairs

| method | split | KL | R@5 | J@5 | R@10 | J@10 | R@15 | J@15 | R@30 | J@30 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| first4 attn-aware KD step500 | Wiki | 0.0081 | 0.4533 | 0.2996 | 0.4906 | 0.3296 | 0.5280 | 0.3627 | 0.6077 | 0.4392 |
| first4 attn-aware KD step500 | MuSiQue | 0.0105 | 0.4738 | 0.3285 | 0.4399 | 0.2905 | 0.4454 | 0.2918 | 0.5174 | 0.3533 |

对比关键 baseline：

| method | split | R@15 | J@15 |
|---|---|---:|---:|
| Native layer4 attn_prob, unfinetuned | Wiki | 0.5801 | 0.4135 |
| Native layer4 attn_prob, unfinetuned | MuSiQue | 0.5081 | 0.3465 |
| CD logits-only step6000 | Wiki | 0.5327 | 0.3679 |
| CD logits-only step6000 | MuSiQue | 0.3732 | 0.2357 |
| first4 attn-aware KD step500 | Wiki | 0.5280 | 0.3627 |
| first4 attn-aware KD step500 | MuSiQue | 0.4454 | 0.2918 |

结论：attention-aware KD 相比旧 logits-only CD 在 MuSiQue 上有明显改善（R@15 0.3732 -> 0.4454），说明 attention KL 有帮助；但它仍低于未微调原生 layer4 baseline（MuSiQue R@15 0.5081）。当前 500-step 版本不能作为有效改进，需要继续调整训练目标/权重/层数，或者优先测试 first8。

## 2026-07-07 Scaling run: first4 attention-aware KD 10 epoch

用户指出 step500 不足以判断方法，需要先和自身 scaling 比较。已调整：从 step500 checkpoint 继续训练到 10 epoch 总量。

### 训练规模

- 数据：WikiText-103 连续文本 blocks，不含 MuSiQue 评测样本。
- 缓存：`data/wikitext_blocks_s256_n20000.pt`
- 样本量：20,000 sequences。
- sequence length：256。
- 有效 batch：4 GPUs * 1 sample/GPU * grad_accum 4 = 16 sequences/update。
- 1 epoch：20,000 / 16 = 1,250 update。
- 10 epoch：12,500 update。
- 已完成：step500。
- 本次继续：resume from `training_state_latest.pt`，目标 step12500。

### 观察计划

训练时保存每 1 epoch 一个 checkpoint：step1250/2500/.../12500。后续逐个评估 selector overlap，画自身 scaling 曲线，而不是只和其他方法做静态对比。

启动脚本：`launch_first4_attn_kd_wikitext_10epoch_resume.sh`
日志：`logs/first4_wikitext_attn_kd_s256_10epoch_resume.log`

## 2026-07-07 first4 attention-aware KD 10 epoch 结果

完整逐 epoch selector-overlap 结果已写入：`epoch_scaling_eval_first4_attn_kd/summary.csv` 和 `epoch_scaling_eval_first4_attn_kd/README.md`。

核心观察：MuSiQue R@15 从 step500 的 0.4454 提升到 epoch1 的 0.4953，此后 1-10 epoch 基本平台期；Wiki R@15 从 step500 的 0.5280 下降到约 0.5227 并平台期。说明 attention-aware KD 在跨到 MuSiQue 的 selector overlap 上相对自身有明显早期收益，但 20k 数据/first4 容量下 1 epoch 后收益饱和。
