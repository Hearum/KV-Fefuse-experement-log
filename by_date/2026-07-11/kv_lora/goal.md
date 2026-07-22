# KV-LoRA / Delta-KV 机制分析任务书

> 本文档是给后续 Research Agent 使用的任务说明。请先读本文件，再读项目根目录 `AGENT.md`。本任务的目标是做机制分析，不是直接追求一个可发布的系统优化实现。

## 1. 角色

你是一名专注于 Transformer、KV Cache、RAG 推理优化和模型机制分析的科研助手。你的工作方式应该像论文作者：

- 不预设结论，用实验验证假设。
- 当发现新现象时，主动补充实验验证。
- 如果结果与预期相反，要解释原因，而不是忽略。
- 所有实验设置、启动命令、数据路径、图表和结论都必须写回本目录的中文日志/报告，方便其他 agent 接手。

本任务目录：

```bash
/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora
```

建议维护：

```bash
README.md                 # 最终中文研究报告
EXPERIMENT_LOG.md         # 按时间追加每轮实验记录
scripts/                  # 统计/可视化/启动脚本
figures/                  # 图表
results/                  # CSV/JSON/NPY/NPZ 等结果
```

## 2. 研究背景

FusionRAG 的核心问题是：RAG 中文档的 KV cache 可以离线预处理并复用，但真实在线回答时，文档 KV 会受到 query/prompt/full prefill 上下文影响。

对同一个 document token，可以比较两种 KV：

1. **Offline/Raw KV**
   - Document 单独输入模型得到的 KV cache。
   - 近似理解为离线缓存的文档 KV。

2. **Full KV**
   - Query、prompt、召回文档共同 full prefill 后得到的文档 KV。
   - 近似理解为准确但昂贵的在线 full attention KV。

定义：

```text
Delta K = K_full - K_offline
Delta V = V_full - V_offline
```

本阶段核心问题：

```text
Delta KV 到底具有什么结构性质？
```

只有充分理解 Delta KV 的结构，才可能设计低成本 KV update，例如 low-rank adapter、basis + coefficient、token/head/layer selective update、query-conditioned predictor 等。

## 3. FusionRAG 基本启动方法

后续 agent 可能不知道这个项目怎么跑。下面给出最小必要背景。

### 3.1 代码位置

常用工作目录：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
```

在部分机器上共享目录也可能是：

```bash
cd /home/hming/FusionRAG-pca-analysis
```

如果两个路径都存在，优先使用当前机器上已有结果更多的路径。不要移动或删除正在跑的实验目录。

主入口脚本通常是：

```bash
test_fusionrag_reflect_preprocess_exp.py
```

常用 Python：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
```

常用模型路径：

```bash
# 主模型
/mnt/qjhs-sh-lab-01/models/Qwen3-32B
/home/hming/models/Qwen3-235B-A22B

# Draft selector
/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct

# Retriever
/mnt/qjhs-sh-lab-01/models/bge-m3
```

如果只做机制分析，优先使用 Qwen3-32B 或更小样本，不要默认启动 235B。

### 3.2 三个重要 baseline

#### full_rate1

含义：`rate=1.0`，相当于 full recompute/full attention 风格上界。用于得到 full KV 或 full accuracy baseline。

模板：

```bash
PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
REPO=/raid/home/hming/FusionRAG-pca-analysis
MODEL=/mnt/qjhs-sh-lab-01/models/Qwen3-32B
BGE=/mnt/qjhs-sh-lab-01/models/bge-m3

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$REPO/test_fusionrag_reflect_preprocess_exp.py" \
  --model_type qwen3 \
  --model_path "$MODEL" \
  --model_name Qwen3-32B \
  --data_path "$REPO/data/result_reflect.json" \
  --dataset_name musique \
  --cache_path "$REPO/MOTIVATION_EXPERIMENTS/kv_lora/cache/full_rate1" \
  --result_path "$REPO/MOTIVATION_EXPERIMENTS/kv_lora/results/full_rate1" \
  --start_sample 0 \
  --end_sample 5 \
  --rate 1.0 \
  --topk 10 \
  --preprocess true \
  --recall_method bge \
  --reprocess_method FusionRAG \
  --revert_rope true \
  --preprocess_scope global \
  --bge_model_path "$BGE" \
  --device cuda:0
```

#### online_qk_rate015

含义：FusionRAG 默认 online QK selector，真实 query 条件下选择 15% token 重算。

关键参数：

```bash
--rate 0.15
--reprocess_method FusionRAG
```

其余参数和 `full_rate1` 类似。

#### online_draft_rate015

含义：使用 3B draft model 根据真实 query 选择 15% token 重算。

关键参数：

```bash
--rate 0.15
--reprocess_method DraftModel
--draft_model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct
```

其余参数和 `full_rate1` 类似。

### 3.3 当前任务如何拿 KV

本任务不一定需要完整跑 accuracy。更推荐：

1. 先从现有 cache 或已有脚本里找 KV 保存格式。
2. 小样本跑 `start_sample=0, end_sample=1/5` 生成必要 cache。
3. 读取 `key.pt/value.pt` 或中间 `past_key_values`，对齐 document token 的 offline/full KV。
4. 保存轻量的 `npz/csv/json` 统计结果，不要保存重复的大 tensor，除非确实必要。

常见 cache 目录格式类似：

```text
<cache_path>/<model_name>/<dataset_name>/kv_cache/
<cache_path>/<model_name>/<dataset_name>/preprocess_kv_cache_global_topk10_bge/
```

实际路径可能随脚本参数变化，请用 `find` 和 `run.log` 确认，不要靠猜。

## 4. 本任务核心研究问题

### 4.1 Delta-KV 是否具有低秩结构（最重要）

对每一层分别计算：

```text
Delta K = K_full - K_offline
Delta V = V_full - V_offline
```

分析：

- Singular Value Spectrum
- SVD
- Effective Rank
- Stable Rank
- Nuclear Norm
- 90%/95%/99% 累积能量 rank

重点回答：

- Delta KV 是否比原始 KV 更低秩？
- 哪些 layer 最低秩？
- K 与 V 是否性质不同？
- Rank 是否随 layer 增长发生系统变化？

### 4.2 Delta-KV 是否具有稀疏性

分析：

- 元素分布
- L1/L2 norm
- 更新幅度 histogram
- 接近 0 的元素比例
- token/head/dim/block 粒度稀疏性

重点回答：

- 更新是否集中在少数 token？
- 是否只有部分维度变化明显？
- 是否存在 block sparse 或结构化稀疏？

### 4.3 Layer Distribution

统计每层：

```text
||Delta K||
||Delta V||
```

绘制：

- Layer vs Update Magnitude
- 累积贡献曲线

回答：

- 是否主要发生在后几层？
- 是否存在明显 layer boundary？
- 只更新最后若干层能保留多少 Delta-KV 能量？

### 4.4 Head Distribution

统计每个 attention head：

```text
||Delta K||
||Delta V||
```

绘制 Layer × Head heatmap。

回答：

- 是否只有少数 head 被显著修改？
- 是否存在 retrieval head？
- 是否存在跨样本稳定的 head？

### 4.5 Token Distribution

对每个 document token 统计：

- Delta K norm
- Delta V norm
- 相对变化比例
- 与 attention weight / selector score / token position / chunk boundary 的相关性

回答：

- 是否只有少数 token 需要 update？
- retrieval token 是否变化更明显？
- boundary token 是否更特殊？

### 4.6 跨 Query 一致性（重点）

固定同一个 document，更换不同 query，得到：

```text
Delta KV_1, Delta KV_2, ...
```

分析：

- cosine similarity
- PCA / UMAP
- subspace overlap
- intrinsic dimension
- shared basis + query-specific coefficients 是否成立

如果成立，说明 Delta-KV 可能可以用低秩 basis 表示，KV-LoRA 路线才有意义。

### 4.7 Delta-KV 是否可预测

探索能否用轻量模型预测 Delta-KV。

可用输入：

- query embedding
- query hidden state pooling
- document summary embedding
- retrieval score
- token position / layer / head id

可用模型：

- Linear/Ridge Regression
- 小 MLP
- low-rank basis coefficient predictor

不要一开始训练大型 Transformer。评价：

- MSE
- cosine similarity
- explained variance
- top changed token recall

## 5. 推荐实验路线

### Phase 0: 代码和数据定位

必须先回答：

- offline/raw KV 保存在哪里？
- full KV 怎么生成？
- document token 在 full prompt 中如何定位？
- layer/head/token 维度顺序是什么？
- K/V tensor shape 是什么？

把定位过程写入 `EXPERIMENT_LOG.md`。

### Phase 1: 单样本 sanity check

只跑 1 个 example，确认：

- offline KV 和 full KV token 对齐正确；
- Delta-K/V shape 正确；
- norm 非零且数值合理；
- 画出 1 个 layer 的 singular spectrum。

### Phase 2: 多样本统计

扩展到 5/20 个 example，统计 layer/head/token 分布和低秩指标。

### Phase 3: 跨 query 实验

对同一 document 构造或复用多 query，验证 Delta-KV 子空间是否稳定。

### Phase 4: 简单预测/重建实验

只在机制确认后做：

- top-r SVD 重建 Delta-KV；
- shared basis 重建；
- ridge/MLP 预测 basis coefficient。

## 6. 输出要求

最终至少产出：

1. `README.md`：中文研究报告。
2. `EXPERIMENT_LOG.md`：按时间追加的实验日志。
3. `scripts/`：所有统计/可视化/启动脚本。
4. `results/`：CSV/JSON/NPZ 结果。
5. `figures/`：图表。

报告必须包括：

- 实验目的
- 实验设计
- 启动命令
- 数据路径
- 统计口径
- 结果图表
- 机制解释
- 对 KV update / KV-LoRA 方法设计的启发
- 后续问题

## 7. 注意事项

- 不要删除 cache、模型、已有实验结果。
- 不要中断 qjy001 上正在跑的 235B 实验。
- 不要默认跑完整数据集。先 smoke test，再扩大样本。
- 如果磁盘不足，优先使用已有可写大盘；如果没有写权限，先报告，不要随意清理。
- 每次实验结束后必须把结果写入本目录文档，不要只在对话中汇报。
- 如果结果异常，先检查 token 对齐、rope/revert_rope、layer/head 维度、dtype 和 cache 路径。
# KV-LoRA / Delta-KV 下一阶段研究任务

## 项目背景

当前已经完成了 Delta-KV 的第一阶段机制分析。已验证：Full/Offline 存在稳定 Delta；Delta 依赖 document 在 full prompt 的上下文位置而非后置 query；Key 后续仅以 RoPE-aligned 口径分析；Value Delta 更稳定、更大且偏后层；Delta 没有明显元素级稀疏但有 token/layer/head 结构化集中；FusionRAG 仅更新 selector 选中的 token；当前仅证明谱有头部能量，未证明极低秩。

## 当前研究目标

探索 Delta 是否具有可预测的共享结构（shared structure）；若存在共享 Basis，未来可利用轻量 Predictor 替代 Transformer Recompute。

## 第一部分：Shared Subspace Analysis（最高优先级）

固定同一个 document，构造尽可能多不同 query（原始、同义改写、不同表达/长度/推理路径、无关 negative），计算 Delta-K/Delta-V，并分析 PCA、SVD、intrinsic dimension、effective rank、pairwise cosine、CKA、CCA。回答不同 query Delta 是否位于共同低维子空间、是否存在共享 Basis、query 是否仅改变低维系数；估计前几个 PC 的解释方差，以及固定 Basis 仅预测 coefficient 是否可行。

## 第二部分：Rate-Distortion Analysis

逐层以 rank 2/4/8/16/32/64/... 重建 Delta，绘制 relative L2 error、cosine similarity、explained variance。比较 Raw Key、RoPE-aligned Key、Value；回答可压缩性及最容易压缩部分。

## 第三部分：Predictability Analysis

只验证轻量模型：以 query embedding、last hidden state、mean pooling、CLS（如有）为输入，用 Linear/Ridge/两层 MLP 预测 Delta 的低维表示或 PCA coefficient；评价 relative L2、cosine、explained variance、R2，不直接预测完整 KV。

## 第四部分：Delta Basis Analysis

若 shared subspace 成立，研究 Delta≈Bc 的 basis 是否跨 layer/head/document 共享、是否只需几十维。

## 第五部分：Value 优先分析

Value 优先；Key 只分析 RoPE-aligned Key，不再分析 Raw Key。

## 第六部分：主动探索

若有新规律，探索 layer 连续性、频率、head grouping、document position pattern、固定模板、cluster，并解释其对 KV Update 方法设计的意义。

## 输出要求

最终报告必须包含图、表、数据与机制解释，并说明未来 KV Update 方法如何设计；遵循科研探索并在发现更高价值方向时调整计划与原因。


## 13. Per-Document Basis 主实验（下一阶段）

### 13.1 假设

每个 document/chunk 可以与 cached KV 一起离线维护自己的更新 Basis：

```text
Delta_s(target_doc | prefix_context)
  ≈ Mean_s,target + Coef(prefix_context) @ Basis_s,target
```

其中 raw/preprocess、RoPE-aligned K/Value、layer/head 独立。Basis 可以是 document-specific；在线 Adapter 只预测低维 coefficient。后置 query 不参与 document Delta。

### 13.2 Context 构造

固定 target document 的 token 内容和 cache 文件，只改变它之前的 document 集合、顺序与长度。每个 target 至少生成 50 个确定性 context，覆盖：

- 空 prefix、单 document prefix、多 document prefix；
- 相同 prefix 集合的不同顺序；
- 短/中/长 prefix；
- target 位于不同绝对 token offset；
- 相关与较弱相关的前序 chunks。

Prompt 始终为 `system + prefix_docs + target_doc + query`，target 必须位于 query 前。rate=1 重算 prefix 与 target 的全部 token，但指标只截取 target document。每个 context 验证 target offset、token 数和 selected support。

### 13.3 严格数据划分

每个 target 使用 40 train contexts 和 10 held-out contexts。禁止在 held-out Delta 上重新拟合 Basis。至少选择 5 个 cache 完整、长度不同的 target documents；主结论因此至少包含 250 个 context runs，而每个 per-document 结论有 50 个样本。

### 13.4 对照方法

1. No update。
2. Per-document mean Delta template。
3. Per-document shared Basis，rank=1/2/4/8/16/32。
4. Global Basis：由其他 target documents 训练，测试未见 target。
5. Global Basis + document-specific residual Basis。
6. In-context oracle SVD，仅作为不可部署的压缩上界。

Shared Basis 使用训练 contexts 联合拟合；held-out 阶段允许 oracle projection coefficient，先隔离表示能力。只有该上界有效后才训练 coefficient predictor。

### 13.5 两种 Basis 粒度

- Feature basis：把训练 Delta 拼为 `[contexts * target_tokens, head_dim]` 或 `[contexts * target_tokens, kv_heads*head_dim]`；每个 context/token 有独立 coefficient。
- Context-template basis：把每个完整 target Delta flatten 为一个向量，矩阵为 `[contexts, target_tokens*feature]`；每个 context 只产生 r 个 coefficient。

前者对应一次输出每个 token coefficient；后者对应一次输出整篇 document 的 r 个 coefficient。必须同时测试，直接回答 Adapter 应“一次性输出整篇 gap”还是“为每个 token 输出 coefficient”。

### 13.6 判据与后续 Predictor

报告 original gap、Delta recovery、final KV error、explained energy、cosine，并提供 mean/std/median/p10/p90与逐层结果。优先判断：

- per-document Basis 是否明显优于 global Basis；
- context-template 小 rank 是否足够；
- 若只有 token-wise feature Basis 有效，是否能用 prefix summary 并行预测全部 token coefficient；
- cached-prefix 与 oracle updated-prefix predictor 的差距是否要求 chunk 串行。

preprocess full-cache gap 异常必须先审计生成配置；raw/preprocess 不混合训练或统计。


## 14. Fixed-Prefix / Vary-Target Document 实验

目标：固定 `system + A + B + C`，替换不同 target document `X`，在 `system+A+B+C+X+query` 中 rate=1 全量重算，并只分析 X 的 `Delta_X=T_X-B_X`。

该实验回答：

- 不同 X 的 Delta 是否具有共同 layer/head/feature 方向；
- shared Basis 能否泛化到未见 X，而非只在同一 document 的不同 context 间泛化；
- Delta 的幅度、rank、token集中性是否由 X 内容/长度决定；
- cached X 是否足以决定 shared-basis coefficient。

设计：

1. 每个 prefix group 固定同一 example 内的前三个 cache-complete chunks为 A/B/C，其他 cache-complete chunks依次作为 X；X 起始位置固定，因此 Key 不受跨X位置变化混杂。
2. 单组约7个X不足以形成结论；使用至少8个prefix groups，收集至少50个raw/preprocess完整配对X。
3. raw/preprocess、RoPE-aligned K/Value独立。
4. 先报告每个X内部 `[tokens, heads*dim]` oracle rank-distortion。
5. 跨X因长度不同，shared analysis优先使用：
   - per-layer/head feature Basis，将train X的所有token拼接；
   - 每个X的mean-pooled Delta方向、pairwise cosine/CKA；
   - layer/head energy profile cosine与cluster；
   - token position归一化后的energy profile。
6. 按document划分train/test，禁止同一X同时参与Basis学习和测试。比较 no-update、global mean direction、shared feature Basis、per-X oracle Basis。
7. K/V分别报告。Value优先；Key因X offset固定可直接比较当前RoPE坐标，另补local-RoPE对照。

## 15. Strict Prefix-Document Generalization / Rank-Direction Test

上一轮40/10仅保证prefix序列不重复，但train/test复用了相同prefix documents，只能称为seen-document、unseen-combination sanity check。本轮必须使用document-disjoint切分。

对每个固定target `X`：

1. 从同一example剩余9个documents中固定划分5个train-prefix documents与4个test-prefix documents，两组交集必须为空。
2. train只用前者构造40个不同有序prefix；test只用后者构造10个不同有序prefix。manifest中保存pool、split和序列，并在分析前assert交集为空。
3. 使用rate=1全token重算，只截取X的`Delta K/Delta V`；raw/preprocess独立。至少5个X，形成每种cache至少50个严格heldout case。
4. 在40个train prefixes上学习per-X mean与context-template PCA/SVD basis；在10个unseen-prefix-document cases上只允许oracle projection，绘制rank 1/2/4/8/16/32的rate-distortion。
5. 比较no-update、per-X mean、per-X basis oracle、跨X shared basis oracle。回答方向究竟是：(a) 所有prefix变化共享少数方向；(b) 方向只在固定X内部共享；或(c) unseen prefix会产生新方向。
6. 报告每层effective rank、rank-r explained energy、heldout reconstruction error、principal angles/subspace overlap。K/V分开；Key以RoPE-aligned表示为主。
7. 若per-X低rank在strict heldout仍有效，下一步才测试prefix到coefficient的预测；若oracle本身无效，则不训练predictor。

## 16. Strict Prefix-to-Coefficient Predictor

在Section 15确认oracle rank8--16存在表示空间后，禁止测试Delta参与coefficient计算，验证轻量输入能否预测固定X的Value Delta coefficient。

1. 沿用5个X的document-disjoint 40/10 split，共50个heldout cases/cache。
2. 每层仅用40个train Delta学习per-X rank8 basis和Delta mean。
3. 输入消融：`X-only`（固定X只能输出train mean）、prefix位置/总token数/document数、cached-prefix Value整体mean、最后一个prefix document Value mean、prefix Value+position。
4. 高维prefix feature必须只在train split拟合PCA，压到4/8/16维后再做Ridge；alpha用train内部32/8 validation选择。测试特征不得参与归一化/PCA/alpha选择。
5. 比较mean、position、prefix、prefix+position、oracle rank8的absolute final KV gap和remaining Delta；raw/preprocess分开。
6. 若prefix内容在strict split上不能稳定超过position，不进入MLP训练；先扩大不同dataset examples和训练documents。

## 17. Preprocess-v2 Static Value Bias

将per-document static mean Delta从分析对象转化为真正的离线cache格式：`V_v2(X)=V_source(X)+MeanDeltaV_source(X)`，Key保持source cache不变。

1. 新建独立`kv_cache_static_v2`/`preprocess_kv_cache_*_static_v2`目录；禁止覆盖或删除原cache。未修改文件只链接，修改后的Value另存。
2. 先验证磁盘v2加载后的document Value与运行时`V_source+mean`逐元素一致。
3. 在正式FusionRAG文档顺序`system+doc1+...+docn+query`上比较：source rate0、v2 rate0、source rate1 full recompute。
4. 指标：所有doc以及有v2 bias的docs分别报告relative L2/cosine；K/V分开。v2只改Value，因此Key必须与source rate0完全一致。
5. 先用example 0、已有X1--X5做partial-v2 sanity；通过后再为更多documents/examples生成静态bias，至少50 examples评估答案EM/F1/evidence与prefill latency/storage。
6. 必须检查静态bias是否因document在正式pipeline中的顺序/position不同而失效；若有效再研究mean模板低秩压缩和少量校准prefix估计。

## 18. Real Static Bias from Disjoint Calibration Chunks

禁止用测试subquery的full Value构造bias。首轮至少50 examples：

1. Random-cross-example：对每个目标example选3个其他example chunks作为校准prefix；将目标example全部documents做随机和反向两种排列，各full prefill一次；每个X取两次`V_full_calibration-V_raw`平均作为静态bias。
2. 测试时只加载raw K/V并加该bias，rate0生成；测试subquery prefix没有参与bias学习。
3. Top-k对照：使用已有BGE global-topk preprocess cache rate0；只统计cache完整case。
4. 对照full rate1与raw rate0；报告EM/token-F1、与full输出一致率、延迟，并逐position检查first-doc退化。
5. 50 examples通过后扩展全部200；random校准仍有效才研究更少校准排列和bias低秩压缩。

## 19. Calibration Grid: Prefix Documents M x Samples N

静态bias质量同时受校准prefix文档数M和独立Delta样本数N影响，做二维网格：

- `M={0,1,3,5,10}`个跨example prefix documents；
- `N={1,2,4,8,16}`个累计校准样本；每两个样本更换一组跨example prefix，并改变目标documents排列。
- 固定前50 examples，正式subquery context完全heldout，只评Value relative L2以快速覆盖全部组合。
- 同时报告raw和BGE top-k preprocess；按正式doc order/offset分层，检查M/N增大是否稳定降低平均和worst-case gap。
- 若M=0接近M>0，说明主要变量是position/目标文档共现而非随机prefix语义；若增加M显著有效，再优化top-k语义校准。

## 20. Original Pipeline Accuracy Grid

将M/N从KV指标推进到原始FusionRAG+GLM正式评测。代表配置：`M0/N2,N4,N8,N16`形成N曲线；`M1/N16,M3/N16,M5/N16,M10/N16`形成M曲线；额外保留M3/N2连接旧结果。每个bias配置分别跑raw和preprocess底座、rate0、原500-token decode和GLM-5.2 judge。统一对照raw/preprocess rate0与full rate1。

## 21. Strict Offline Random-Prefix Mean + Rank8

纠正known-retrieval-pool泄漏。固定目标文档X，calibration只能从独立examples 50--199抽取前缀，禁止使用X所属main example的其他documents、sub-question或retrieval结果。首轮目标examples 0--4全部unique documents（预计>50 X），每个X使用K=3个外部文档、M=8个独立contexts。

同一次`system+external-prefix+X` full prefill下，分别相对raw/preprocess source计算RoPE-local-aligned Delta K与Delta V。Offline学习per-X Mean和rank8 context basis；online禁止oracle coefficient，使用当前已加载前序KV的轻量统计预测rank8 coefficient。正式比较source rate0、Mean-only、Mean+rank8 predictor、full rate1；raw/preprocess和K/V分别报告。sanity通过后扩大到50 target examples、M=16。
