# KV-LoRA / Delta-KV 机制分析

本目录记录 offline/raw KV、full KV 和 Delta-KV = full - offline 的机制分析。

## 当前口径

- offline/raw KV：对 `system + document` 做 prefill，然后切出 document token span。
- full KV：对 `system + selected documents + query` 做 full prefill，然后按精确 token offset 切出同一 document span。
- KV shape 顺序：`[layers, batch, kv_heads, tokens, head_dim]`。
- 当前阶段只做小样本 sanity check，不启动 235B，不清理任何 cache/模型/历史结果。

## 文件组织

- `scripts/analyze_delta_kv_sanity.py`：小样本 full/offline Delta-KV 采集与统计脚本。
- `results/`：CSV/JSON 统计结果。
- `figures/`：奇异值谱等图。
- `EXPERIMENT_LOG.md`：逐轮实验日志。

## 已知参考

仓库中已有 `MOTIVATION_EXPERIMENTS/preprocess_kv_cache_delta_reflect_full/`，比较的是 raw KV 与 preprocess KV，不是 full-context KV。该参考结果显示 value 的相对变化显著大于 key，后续 full/offline Delta-KV 需要独立验证。

## Phase 1: 单样本 sanity check

命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/kv_lora/scripts/analyze_delta_kv_sanity.py \
  --repo /raid/home/hming/FusionRAG-pca-analysis \
  --model_path /mnt/qjhs-sh-lab-01/models/Qwen3-32B \
  --model_type qwen3 \
  --model_name Qwen3-32B \
  --data_path /raid/home/hming/FusionRAG-pca-analysis/data/result_reflect.json \
  --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3 \
  --device cuda:0 \
  --start_sample 0 \
  --end_sample 1 \
  --max_subquestions 1 \
  --max_docs_per_subquestion 2 \
  --max_cache_len 8192 \
  --max_svd_rank 64
```

输出：

- `results/delta_kv_summary_Qwen3-32B_ex0_1_sub1_docs2.csv`
- `results/delta_kv_layer_metrics_Qwen3-32B_ex0_1_sub1_docs2.csv`
- `results/delta_kv_head_metrics_Qwen3-32B_ex0_1_sub1_docs2.csv`
- `results/delta_kv_token_metrics_Qwen3-32B_ex0_1_sub1_docs2.csv`
- `figures/singular_spectrum_sanity_Qwen3-32B_ex0_1_sub1_docs2.png`

单样本主要观察：

| doc_rank | kind | doc_len | relative L2 | cosine(full, offline) | shape |
|---:|---|---:|---:|---:|---|
| 0 | key | 50 | 0.0087 | 1.0001 | `[64,1,8,50,128]` |
| 0 | value | 50 | 0.0274 | 0.9996 | `[64,1,8,50,128]` |
| 1 | key | 92 | 0.5051 | 0.8729 | `[64,1,8,92,128]` |
| 1 | value | 92 | 0.3896 | 0.9247 | `[64,1,8,92,128]` |

初步判断：

- token 对齐和 shape sanity 通过：offline/full document span 形状一致，Delta 非零。
- 第 1 个 doc 的 Delta 很小，第 2 个 doc 的 Delta 明显大很多，提示 full-offline Delta 强依赖 document 在 full prompt 中的位置和可见的前序 document context；末尾 query 在因果模型中不能反向改变 document KV。后续统计必须保留 `doc_rank/chunk_id/full offset`。
- 单样本中 value 的最大能量集中在最后几层（layer 58-63），但 key 的较大层分布更分散；这需要多样本确认。
- 当前 singular spectrum 不是极端低秩：doc_rank=1 的 top-delta layers 约需要 rank90 27-38、rank95 42-53、rank99 69-76。是否“比原始 KV 更低秩”不能只凭本轮下结论。
- 近零比例不高：以 `abs(delta) < 1% RMS(delta)` 计，doc_rank=1 的 key 约 4.2%，value 约 20.4%；元素级硬稀疏不明显，但 token/head/layer 结构化集中性存在迹象。

## Phase 2: 5-example 小样本统计

命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/kv_lora/scripts/analyze_delta_kv_sanity.py \
  --repo /raid/home/hming/FusionRAG-pca-analysis \
  --model_path /mnt/qjhs-sh-lab-01/models/Qwen3-32B \
  --model_type qwen3 \
  --model_name Qwen3-32B \
  --data_path /raid/home/hming/FusionRAG-pca-analysis/data/result_reflect.json \
  --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3 \
  --device cuda:0 \
  --start_sample 0 \
  --end_sample 5 \
  --max_subquestions 1 \
  --max_docs_per_subquestion 2 \
  --max_cache_len 12000 \
  --max_svd_rank 64
```

输出 tag：`Qwen3-32B_ex0_5_sub1_docs2`。

### 5-example 原始 full-offline 结果

| group | kind | relative L2 mean | relative L2 median | cosine mean | near-zero `<1%RMS` mean |
|---|---|---:|---:|---:|---:|
| all | key | 0.2769 | 0.2574 | 0.9257 | 0.1412 |
| all | value | 0.2077 | 0.1826 | 0.9607 | 0.1989 |
| doc_rank=0 | key | 0.0070 | 0.0086 | 1.0005 | 0.2408 |
| doc_rank=0 | value | 0.0261 | 0.0274 | 0.9995 | 0.2185 |
| doc_rank=1 | key | 0.5468 | 0.5554 | 0.8509 | 0.0415 |
| doc_rank=1 | value | 0.3892 | 0.3796 | 0.9219 | 0.1794 |

关键观察：

- `doc_rank=0` 几乎不变，符合 causal transformer 中前序 token 的 KV 不应受未来 doc/query 影响；少量非零可能来自 SDPA/StaticCache 数值路径差异。
- `doc_rank=1` 的 Delta 稳定显著，说明 full/offline 差异主要来自 document 在 full prompt 中处于不同上下文/位置。
- 原始 key Delta 明显大，但它混入了 RoPE 绝对位置平移，不能直接解释为上下文更新。
- value Delta 不受 RoPE，`doc_rank=1` 均值 `0.3892`，是更直接的上下文更新信号。

Layer/head 分布：

- 原始 value Delta 的能量集中在后层，聚合 diff L2 最大层为 `62, 61, 60, 63, 59, 58, 57, 56`。
- 原始 key Delta 的大层更分散，但该结果受 RoPE 位置平移污染。
- value 的 top head 集中在最后两三层，尤其 layer 62 的多个 head。

Low-rank/sparsity：

- 5-example 平均 rank 指标：key rank90 `40.3`、rank95 `52.9`、rank99 `71.2`；value rank90 `39.8`、rank95 `51.4`、rank99 `68.9`。
- 相对于每个 doc 的 token 数和 1024 hidden KV 维度，这不是极端低秩；但可用 rank-32/64 basis 做重建曲线验证。
- 元素级近零比例仍不高，暂不支持简单 unstructured sparsity；更值得看 layer/head/token 结构化选择。

## Phase 2b: RoPE-aligned key 对照

命令在 Phase 2 基础上增加：

```bash
--include_rope_aligned_key
```

输出 tag：`Qwen3-32B_ex0_5_sub1_docs2_ropekey`。

结果：

| group | kind | relative L2 mean | relative L2 median | cosine mean |
|---|---|---:|---:|---:|
| doc_rank=1 | raw key | 0.5468 | 0.5554 | 0.8509 |
| doc_rank=1 | key_rope_aligned | 0.2049 | 0.1965 | 0.9796 |
| doc_rank=1 | value | 0.3892 | 0.3796 | 0.9219 |

解释：

- 对 offline key 应用与 FusionRAG `revert_rope=True` 加载相同的相对 RoPE shift 后，doc_rank=1 的 key relative L2 从约 `0.55` 降到约 `0.20`。
- 因此原始 K Delta 的主要部分是位置编码差异，不应直接作为 KV-LoRA 要学习的上下文 Delta。
- RoPE 对齐后的 K Delta 仍非零，且聚合最大层转移到中后层 `45-52`；value Delta 仍集中在最后层 `58-63`。
- 后续报告中建议区分三种对象：`raw key delta`、`rope-aligned key delta`、`value delta`。

## 当前结论

1. token/shape 对齐已通过小样本 sanity：KV shape 为 `[64,1,8,T,128]`，对应 Qwen3-32B 的 64 层、8 KV heads、head_dim 128。
2. Delta-KV 强依赖 document 在 full prompt 中的位置；第一个 doc 几乎不变，后续 doc 显著变化。
3. K 的 full-offline 差异必须先做 RoPE 位置对齐，否则会高估 Delta K。
4. V 的上下文 Delta 更稳定、幅度更大，且集中在最后几层和部分 head；KV-LoRA / Delta-KV predictor 应优先从 V 或 rope-aligned K+V 开始。
5. 当前 5-example 证据不支持“极低秩”或“元素级稀疏”的强结论，但支持继续做低秩重建曲线和结构化 layer/head/token 选择。

## 下一步

- 扩到 20 examples 前，先给脚本增加 offset/doc_rank/token-position 的显式列，并把 `doc_rank=0` 近零作为自动 sanity check。
- 做 rank-r 重建曲线：分别评估 raw K、rope-aligned K、V 的 rank 8/16/32/64 explained variance。
- 做跨 query 一致性需要固定同一 document、多 query；当前 `result_reflect.json` 前 5 个样本不满足该设计，需要构造或检索同文档多 query。

## Phase 2：5-example 扩展（仍为小样本）

`results/delta_kv_*_Qwen3-32B_ex0_5_sub1_docs2.*` 覆盖 5 个 example、每个 1 个 sub-question 与 2 篇 document。只看有前缀 context 的第二篇 document：K relative L2 mean/median=`0.5468/0.5554`，V=`0.3892/0.3796`；无前缀 context 的第一篇分别仅 `0.0070/0.0086` 与 `0.0261/0.0274`，应作为 static-cache 的数值基线。

跨 5 个 example，V 的更新能量稳定聚在末层 56--63；K 的高能层则较分散。8 个 KV head 的总能量较均衡，暂未出现单一 head 的显著支配。Token 上，top-10 token 平均覆盖 V `46.7%`、K `12.1%` 的全层 Delta 能量，说明 V 更具 token 级结构化集中性。

低秩结论保持审慎。逐层 `[token, kv_heads*head_dim]` SVD 的 mean rank95 为 K `55.75`、V `53.55`，而 stable rank 约 `5--6`：能量谱有很强头部，但 95% 精确重建仍需要几十个分量。当前证据支持研究 low-rank approximation 的率失真曲线，不支持直接假定一个很小固定 rank 就足够。

## Phase 3：FusionRAG 实际复用/重算 Delta（当前实验口径）

本阶段不再把 `system + docs + query` 的 dense full KV 直接当作唯一分析对象。对同一多 document chunk prompt，分别以 `raw kv_cache` 和 `preprocess_kv_cache` 装载 chunk KV，记录 `load_kv_and_generate` 的两个状态：`before`（cache 复用且 selector 尚未重算）与 `after`（FusionRAG 按 rate 选择 token、与 query 一起 reprocess 后）。核心量为 `Delta_reprocess = KV_after - KV_before`，并按 selected/unselected token、chunk、layer、head 分组。


## Phase 3：FusionRAG 实际重算后的 Delta-KV

这一轮才是与 KV reuse/update 直接相关的比较。对完整多 document chunk prompt，先装载 raw 或 preprocess chunk KV，再运行真实 FusionRAG selector 和 selected-token reprocess。我们记录重算前后的 cache，因此 Delta 是实际系统写回的 `Delta_reprocess`。

在 Qwen3-32B 的 `example=0` 中，sub-question 含 12 document chunk、1313 document token。`rate=0.15` 选择 196 token（14.93%）；未选 1117 token 的 K/V Delta 精确为零，说明系统只直接覆写 selected token。

| KV 起点 | selected Delta-K relative L2 | selected Delta-V relative L2 |
|---|---:|---:|
| raw KV | 0.3251 | 0.5220 |
| preprocess KV | 0.2860 | 0.4689 |

结论：第一，V 比 K 更需要 online update，幅度约高 1.6 倍。第二，preprocess 降低了残余 update（K 约 12%，V 约 10%），但没有消除它，因此它是更好的初始化而不是 online recompute 的替代。第三，Delta 的支撑集严格是 selector 选中的 token；KV-LoRA 的可行路径应是“selector + selected-token Delta predictor/low-rank update”，而不是对全部缓存做统一修正。

本轮已经输出 selected Delta 的逐层 SVD、rank90/95/99、stable/effective rank、元素近零比例、head energy 和 token energy CSV。由于目前只有一个多 chunk example，尚不能声称 Delta 具有跨样本的低秩 shared basis，亦不能训练 predictor；下一步需要扩至 5 examples，并加入 `full_rate1` 作为 Delta target，对比 raw/preprocess/reprocess 到 full 的误差和 selected-energy coverage。

## Phase 4：Shared Subspace 首轮结果（Value 优先）

`goal.md` 要求固定 document、改变 query 检验 shared basis。这里需要一个因果性修正：在现有 `documents -> query` prompt 中，后置 query 不会改变早先 document KV。因此本轮固定一个目标 document，改变其前缀 document context（0/1/2/4/8 chunk），这是实际可使 Value Delta 改变的条件变量。

Value Delta 随前缀从 0 chunk 的数值基线 `0.0280` 增至 1/2/4/8 chunk 的 relative L2 `0.3896/0.8599/0.8833/0.8657`。当 context 已包含至少两个 chunk，Delta 方向具有明显一致性：2--4、2--8、4--8 chunk 的 whole-Delta cosine 分别为 `0.8358/0.7957/0.8136`。这说明存在上下文充分后的稳定更新方向，而不是完全随机的 Delta。

但它还不是一个极低维的全局 shared basis。逐层 pooled token-feature PCA 的 PC1/PC2/PC4 平均仅解释 `14.2%/22.5%/32.7%` 方差，平均 effective rank 为 `59.9`。因此“所有 context、所有 token 共用 2--4 维 Value template”不成立。较合理的 KV update 路线是 per-layer、context-regime-conditioned basis，加上预测其 coefficient；prefix length/document position 应成为轻量 predictor 的输入。该结论仍来自单一目标 document，下一步必须在多个 document 和多个前缀组合上验证。

## 后置 Query 的 Rate=1 反例：当前部署顺序不存在 Query-conditioned Document Delta

为直接检验 goal.md 的多 query 假设，固定全部 12 document chunk（1313 token），保持 `system + docs + query` 顺序，以 rate=1 对所有 document token full prefill；测试原始、改写、不同长度和两个无关 query。四个变体相对原 query 的 document Value KV 都是 `relative L2=0`、`max abs=0`。

这不是“Delta 处在一个很低秩的 query shared subspace”，而是因果掩码导致 query 根本没有进入 document KV 的计算图：所有后置 query 得到同一 Delta。因此在当前 FusionRAG 部署定义中，不能训练 query embedding -> document Delta coefficient predictor。可行的下一步只能是：（1）研究 query 如何预测 selector support；或（2）明确建立 query 位于 document 前的替代 prompt 研究，以验证一个不同于当前部署的 query-conditioned KV update 假设。

## Phase 5：不同 Query 的真实 Rate=1 Reprocess Delta

这轮严格比较同一 document KV 的重算前后。固定 example 0 的 12 个 document chunk（1313 token），raw/preprocess 分开；对 8 个原始、改写、长短、相关和无关 query，均通过真实 `load_kv_and_generate` 路径以 rate=1 重算全部 document token：

```text
Delta_s(q) = KV_after_rate1(s, q) - KV_before(s)
```

所有 run 均确认选中 1313/1313 document token。对 raw/preprocess、RoPE-aligned K/Value、全部 64 层，不同 query 的 Delta 相对原 query 均为 `relative L2=0`、`max absolute difference=0`。重算前 cache 也逐元素相同。

这给 Shared Structure 三个明确答案：

- 不同 query 的 Delta 位于同一“子空间”，但它是退化的完全相同模板：未中心化 query-sample 矩阵 rank=1，中心化后总方差为 0。
- 一个固定 Delta basis/template 可以零误差重建全部 query，但这不是 query-conditioned 低维结构。
- query 不仅没有改变高维方向，连 coefficient 都没有改变。因此当前部署下不存在可学习的 `query -> Delta coefficient` 信号。

机制原因是 document token 位于 query 之前；rate=1 又固定选择全部 document token，query 既不能进入 document KV 的因果计算，也不能通过 selector 改变 support。未来 query predictor 应用于 selector support；Delta vector 本身应由多-document context、document position、raw/preprocess 起点来条件化。

## Phase 6：Query 放在 Document 前时，出现可压缩的 Query-specific Value Delta

作为后置-query不变性的对照，本轮采用 `system + fixed-length query-prefix + all_docs`。固定 example 0 的 12 chunk/1313 document token，构造 8 个原始、改写、相关与无关 query，并将 prefix 统一到 34 token，避免 document position 差异污染 Value 对比。

不同 query 的 Value Delta 不再相同，但存在明显公共结构。raw 起点的 pairwise Delta cosine 平均 `0.853`（范围 `0.800--0.946`），preprocess 为 `0.819`（`0.755--0.933`）。query 间 relative difference 分别平均 `0.533/0.593`，说明 query 确实改变 Delta，而不是只有微小数值扰动。

逐层 centered PCA（排除零方差第 0 层）显示：PC1、PC1--2、PC1--4 平均解释 `54.6%/68.4%/85.9%` query 间方差，effective rank 约 `4.26`。未中心化 PC1 对 raw/preprocess Delta 分别解释 `94.5%/92.7%`，表明可以用“强共享均值模板 + 少量 query-specific basis coefficient”描述本例的 Value Delta。

这是当前最支持 KV-LoRA/Predictor 的证据，但它对应 query-prefix 新计算顺序，不是当前 docs-before-query FusionRAG。并且只有 8 query、单 example，PCA rank 上限为 7；必须通过 held-out query coefficient reconstruction 和跨 example/document basis transfer 后，才能判断轻量 predictor 是否真正成立。

### Query-prefix 下 K/V 必须分开：Key 更稳定，Value 更 Query-sensitive

补充 RoPE-aligned Key 后，K/V 差异清楚。raw/preprocess 起点的 Key Delta pairwise cosine 为 `0.941/0.909`，高于 Value 的 `0.853/0.819`；query 引起的 Delta 相对变化均值为 Key `0.359/0.435`，Value `0.533/0.593`。因此 Value 的 coefficient 随 query 变化更强，不能用 Key 的稳定性代替 Value 结论。

两者的 query residual 维度相近：Key 的 centered PC1/PC4=`56.1%/86.7%`、effective rank=`4.13`；Value 为 `54.6%/85.9%`、`4.26`。未中心化共享模板则 Key 更强（raw/preprocess PC1=`95.9%/94.0%`），Value 为 `94.5%/92.7%`。

方法上应拆成两个 predictor：RoPE-aligned Key 可以用更强的固定模板和较小 coefficient correction；Value 需要更依赖 query 的 coefficient predictor。后续 held-out reconstruction 必须分别报告 K/V，禁止 concat 后只给一个 explained-variance 数字。

## Phase 7：Held-out Query 推翻了“rank-4 已足够”的强结论

16 个等长 query-prefix 按四类划分为 12 train/4 held-out。每层只在 train Value Delta 上学习 basis，并用 oracle coefficient 投影 held-out Delta，因此结果是 basis 可达到的重建上限，不包含 query predictor 的额外误差。

mean template 已很强：raw/preprocess 的 held-out full Delta explained variance 为 `94.1%/92.1%`，relative L2 为 `0.216/0.254`。加入 rank-4 basis 后提高到 `96.4%/95.2%`，relative L2 降至 `0.162/0.189`；rank-8 仅进一步达到 `96.6%/95.5%`。

但只看 query-specific residual，rank-4 held-out explained variance 仅 `44.3%`，rank-8 也只有 `48.4%`，远低于先前8-query in-sample PCA 的 `85.9%`。因此“约4维 query coefficient 足够”的结论不成立；那一数字主要反映训练 query 内拟合。当前可靠结论是：Value Delta 有很强、可泛化的 query-independent mean template，而未见 query 的 residual 不能被一个小型全局线性 basis 充分解释。

方法上应先利用 mean template；residual predictor 需要更多 query、按 query regime 分组或更局部的 per-head/per-token basis。现阶段直接训练 Ridge/MLP coefficient predictor没有充分依据。

## Canonical 主线重置：FusionRAG KV Adapter

后续主线以 `ADAPTER_BENCHMARK.md` 为准。目标是用当前真实 RAG context 条件下的轻量 Adapter 逼近 `Delta = rate1全token重算KV - cache装载后KV`。raw/preprocess、RoPE-aligned K/Value分开，所有结果同时报告 original gap、Delta recovery error、final KV error。

此前 query-prefix Phase 6/7 是反事实机制探索，只作为附录，不再用于判断当前 FusionRAG Adapter 是否可行。下一步是当前部署下的5-example held-out mean/shared-basis benchmark。

## Canonical Stage A：真实 Rate=1 Delta 并非极低秩

Example 0 的真实 rate=1 original gap为：raw K/V=`0.266/0.483`，preprocess K/V=`0.218/0.438`。这是 `||KV_after-KV_before||/||KV_before||`，不是归一化成100%的重建误差。Value的实际gap约为Key的1.8--2倍。

将每层 Delta reshape为 `[1313 tokens, 8 KV heads × 128 dim]` 后，rank-8只解释raw K/V `49.7%/37.9%`、preprocess K/V `47.8%/41.0%`；rank-32约为`68%` K、`59--62%` V；rank-128才达到K `86--87%`、V `81--84%`。因此一个rank 4/8的全层LoRA式Basis不足以替代重算，Value尤其困难。

该结果只否定“全层、全token共享一个极小Basis”的最简单Adapter。下一步应测试per-head和per-chunk低秩性：若局部tensor明显更低秩，合理架构将是chunk串行或causal prefix state、chunk内token并行，并为每个layer/head使用独立小Basis。


## 2026-07-11 Canonical Stage A-50：50 个完整配对样本的 Rate=1 Rank-Distortion

### 实验口径

本轮替代此前单 example 结论，作为当前 canonical baseline。每个有效 example 必须同时存在 raw/preprocess 两种起点，并分别得到 RoPE-aligned K 与 V；cache 缺块的半样本不进入统计。固定真实 `system + docs + query` Pipeline，FusionRAG `rate=1` 对全部 document token 重算：

```text
B = document KV immediately after loading chunk cache
T = document KV after rate=1 reprocess
Delta = T - B
original gap g = ||Delta|| / ||B||
Delta recovery r = ||Delta - Delta_hat|| / ||Delta||
final KV error e = ||T - (B + Delta_hat)|| / ||B||
```

“完全不更新”对应 `r=1 (100%)`，但其实际 final KV error 是 original gap `g`，不是 KV 相差 100%。低秩统计先在每个 example 的每层对 `[doc_tokens, 8 KV heads * 128]` 做 SVD，再对 64 层等权平均，最后跨 50 examples 报告分布，避免长文档获得额外权重。

选中的 50 个完整 example：
`0,1,2,3,4,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,24,25,26,27,32,33,35,36,37,38,39,40,43,44,45,46,50,51,52,53,54,55,56,57,58,59,60,63,64,66`。实际找到 51 个完整样本，按编号排序固定取前 50 个。

### 原始重算 Gap（跨 50 examples）

| cache 起点 | 对象 | mean | std | median | p10--p90 |
|---|---|---:|---:|---:|---:|
| raw | K | 0.2359 | 0.0271 | 0.2359 | 0.2054--0.2696 |
| raw | V | 0.4216 | 0.0578 | 0.4153 | 0.3510--0.4878 |
| preprocess | K | 0.3645 | 0.0549 | 0.3781 | 0.2766--0.4180 |
| preprocess | V | 0.7757 | 0.1366 | 0.8190 | 0.5967--0.9102 |

V 的重算幅度稳定大于 K。raw 下 V/K mean gap 比约 1.79；preprocess 下约 2.13。因此 K/V 必须使用独立 adapter 与误差预算。

这里出现与 example-0 smoke cache 相反的异常：full-cache 上 preprocess gap 比 raw 更大，而不是更小。它可能来自 full preprocess cache 的生成设置、preprocess 语义或 cache 覆盖批次差异。当前只能得出“相对于这个已落盘 preprocess cache，rate=1 target 的 gap 更大”，不能推广为 preprocess 方法本身更差；在训练 adapter 前必须审计 cache 生成命令、模型版本、top-k/recall/scope 与当前加载参数是否完全一致。

### Whole-layer Rank-Distortion（64 层先按 example 平均，再跨 50 examples）

表中每格为 `Delta recovery error / final KV error / explained Delta energy` 的均值。

| 起点/对象 | rank 4 | rank 8 | rank 16 | rank 32 | rank 64 |
|---|---:|---:|---:|---:|---:|
| raw K | .760/.182/41.7% | .698/.166/50.7% | .629/.149/59.8% | .552/.130/68.9% | .470/.110/77.3% |
| raw V | .819/.286/29.4% | .763/.268/38.2% | .695/.246/48.0% | .615/.219/58.6% | .525/.188/69.0% |
| preprocess K | .790/.295/37.3% | .733/.275/45.9% | .668/.250/55.0% | .593/.222/64.4% | .511/.193/73.4% |
| preprocess V | .823/.480/31.2% | .774/.456/38.7% | .714/.424/47.3% | .640/.385/57.2% | .555/.337/67.4% |

### 当前结论

1. **whole-layer rank 4/8 不足以替代重算。** rank 8 仍留下 K 约 70%--73%、V 约 76%--77% 的 Delta L2；“一个很小的全层 LoRA basis 一次性生成全部 token gap”不可行。
2. **谱有压缩性，但不是极低秩。** rank 64 可解释 K 73%--77%、V 67%--69% 的 Delta energy；相应只填掉约 45%--53% 的 Delta L2 gap。是否值得取决于下游输出质量，而不仅是 KV L2。
3. **Value 是主要瓶颈。** 它的 original gap 更大且同 rank 下更难压缩。即使 rank 64，final error 仍为 raw V 0.188、preprocess V 0.337；K 则为 0.110/0.193。
4. **这只否定全层共享小 basis，不否定局部 adapter。** 下一步应在 50 样本上比较 per-head、per-chunk、per-head-per-chunk tensorization，并做跨 example held-out basis，而不是继续增大全层 rank。
5. **架构选择仍未完成。** 若 cached prefix summary 已能预测局部 coefficient，可一次性并行输出；若 oracle updated-prefix 显著优于 cached-prefix，则采用 chunk 串行、chunk 内 token 并行。当前 rank 实验只回答表示上限，尚未回答可预测性。

### 文件

- `scripts/batch_adapter_rank_distortion_50.py`
- `scripts/aggregate_adapter_rank50.py`
- `results/adapter_rank50_parts/`
- `results/qwen3_32b_50ex_rate1_adapter_rank_distortion_global.csv`
- `results/qwen3_32b_50ex_rate1_adapter_rank_distortion_layers.csv`
- `results/qwen3_32b_50ex_rate1_adapter_rank_distortion_summary.json`
- `figures/adapter_rank50_delta_recovery.png`
- `figures/adapter_rank50_final_kv_error.png`
- `figures/adapter_rank50_explained_variance.png`


### 本轮实际命令与异常

```bash
# 0--49 分成 5 个进程分别绑定 qjy000 GPU 1--5；补充 50--59、60--69
CUDA_VISIBLE_DEVICES=<gpu> /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/kv_lora/scripts/batch_adapter_rank_distortion_50.py \
  --start <start> --end <end> --part <part>

MPLBACKEND=Agg /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/kv_lora/scripts/aggregate_adapter_rank50.py
```

首批 0--49 只有 36 个四路完整配对样本；其余样本存在 raw 或 preprocess chunk cache 缺失。补跑 50--69 后共有 51 个完整配对样本。未删除或补造任何 cache，缺块样本严格剔除。所有有效 run 均验证 `selected == document tokens`，即 rate=1 确实重算全部 document token。


## 2026-07-11 Per-Document Basis：5 targets × 50 prefix contexts

### 设计

固定 example 0 中的 5 个 target documents（chunk 1--5），每篇 target 保持 token/cache 不变，从其余 9 个 chunks 构造 50 个不同前序 context，覆盖空、不同长度、不同集合和顺序。Prompt 为 `system + prefix_docs + target_doc + query`；query 后置且固定。每个 context 对 raw/preprocess 分别执行真实 rate=1，确认 target 全 token 被重算，只截取 target Delta。

每篇 document 使用 context 0--39 训练、40--49 held-out。共 250 个 context 定义、500 次 source-conditioned reprocess；最终统计包含 5×10=50 个 held-out target-context。K/V 与 raw/preprocess 分开。

比较两种可部署含义不同的 Basis：

1. **Document-template Basis**：每层把完整 target Delta flatten，每个 context 只使用 r 个系数，一次性产生整篇 document gap。
2. **Token-head feature Basis**：每层每 head 在 128 维 feature 中学习 Basis，每个 target token/head 使用自己的 oracle coefficient。

两者 Basis 都只由40个训练 contexts得到；held-out只投影，不在测试 Delta 上重新拟合。rank 0 是训练 contexts 的 per-document、per-token mean template。

### Original online writeback gap

跨 50 个 held-out target-context，全层平方范数先合并：

| source | K | V |
|---|---:|---:|
| raw | 0.277 | 0.411 |
| preprocess | 0.253 | 0.477 |

这是各自 `B_s -> T_s` 的 source-conditioned online writeback magnitude，不是到共同 dense-full target 的距离。旧 Qwen2.5 实验使用共同 full-context target，确实得到 preprocess 更接近 full：K `0.169→0.122`、V `0.290→0.223`。两者不矛盾；当前结果禁止用于判断 preprocess 与 raw 谁更接近共同 full KV。

### Held-out Document-template Basis

每格为 `Delta recovery error / final KV error`，rank 0 为 mean template。

| source/object | rank0 | rank1 | rank4 | rank8 | rank16 | rank32 |
|---|---:|---:|---:|---:|---:|---:|
| raw K | .549/.150 | .490/.134 | .423/.116 | .384/.105 | .353/.096 | .324/.088 |
| raw V | .535/.218 | .470/.192 | .435/.178 | .408/.166 | .380/.155 | .359/.147 |
| preprocess K | .620/.151 | .553/.135 | .479/.117 | .437/.107 | .400/.097 | .369/.089 |
| preprocess V | .549/.222 | .480/.195 | .446/.181 | .418/.169 | .390/.158 | .369/.149 |

Mean template 已解释 raw K/V `68.8%/69.6%`、preprocess K/V `60.3%/65.9%` Delta energy。加入少量整篇 coefficient 有稳定但有限的 held-out 增益；rank8 仍留下约38%--44%的Delta L2。

### Held-out Token-head Feature Basis

| source/object | rank0 | rank4 | rank8 | rank16 | rank32 | rank64 |
|---|---:|---:|---:|---:|---:|---:|
| raw K | .549/.150 | .456/.125 | .416/.114 | .368/.101 | .303/.083 | .201/.055 |
| raw V | .535/.218 | .477/.194 | .449/.183 | .407/.166 | .347/.141 | .253/.103 |
| preprocess K | .620/.151 | .507/.125 | .461/.114 | .406/.100 | .333/.082 | .221/.053 |
| preprocess V | .549/.222 | .490/.198 | .460/.186 | .418/.169 | .356/.144 | .259/.105 |

低 rank token-head Basis 并未优于 document-template：它只能表达 feature 方向，不能以少量系数捕获整篇 document 的协同更新。到 rank32/64后局部方案更强，但每个 token/head 都需要大量 oracle coefficient，已不像轻量的一次性 LoRA。

### 机制结论与 Adapter 设计

1. **per-document mean template 是目前最稳定的组件。** 不看在线 context，仅缓存一份 document-specific mean Delta，held-out 已填掉约38%--47%的 Delta L2，并解释约60%--70%能量。
2. **存在低维 document-level context variation，但不是极低维。** rank1--8持续改善，然而rank8仍不能高保真替代 Transformer。
3. **残差具有高维局部成分。** token/head rank64才能把final error降至K约0.05、V约0.10，存储与系数预测成本较高。
4. **合理结构是分层 Adapter，而非二选一。** 第一层使用静态 per-document mean/template + 少量document-level context coefficient；第二层只对高残差layer/head/token做局部 correction或保留稀疏recompute。
5. **Value仍是瓶颈。** 相同配置下V final error明显高于K，Value应分配更高rank或更多recompute预算。
6. **Key结论暂受变长prefix的RoPE位置混杂影响。** 当前每个context内B/T位置一致，因此writeback gap有效；但跨context共享Basis还需将Key逆旋转到target-local坐标或使用固定offset cohort后再定论。Value没有该问题。
7. 下一步不直接训练MLP。先补共同dense-full teacher以统一raw/preprocess target，再做cached-prefix/oracle-updated-prefix coefficient predictability，决定并行还是chunk串行。

### 命令与文件

```bash
CUDA_VISIBLE_DEVICES=<0..4> .../python scripts/collect_perdoc_context_deltas.py \
  --example 0 --target <1..5> --contexts 50 --part <t1..t5>
CUDA_VISIBLE_DEVICES=<0..4> .../python scripts/analyze_perdoc_context_basis.py --part <t1..t5>
CUDA_VISIBLE_DEVICES=<0..4> .../python scripts/analyze_perdoc_token_head_basis.py --part <t1..t5>
```

新增：
- `scripts/collect_perdoc_context_deltas.py`
- `scripts/analyze_perdoc_context_basis.py`
- `scripts/analyze_perdoc_token_head_basis.py`
- `scripts/aggregate_perdoc_context_basis.py`
- `results/perdoc_context_deltas/{t1,t2,t3,t4,t5}/`
- `results/perdoc_context_basis_5targets_50contexts_40train10test_{context.csv,summary.json}`
- `results/perdoc_token_head_basis_5targets_50contexts_40train10test.{csv,json}`
- `figures/perdoc_context_basis_5targets_heldout.png`
- `figures/perdoc_basis_granularity_comparison.png`

异常：最初逐层relative error平均时，raw V第0层近零Delta造成极小分母爆炸。正式汇总已改为每个held-out context先跨64层累加 `delta/base/error norm²` 再取比例，旧的逐层相对误差均值不用于结论。


## 2026-07-11 Fixed A+B+C / Vary X：跨 Document Delta 结构

### 实验设计

构造 `system + A + B + C + X + query`，每个 group 固定 A/B/C，只将 X 替换为同一 example 内其他 cache-complete document。X 起始 token offset在组内固定；rate=1重算全部document token，但只截取X的重算前后Delta。8个完整prefix groups×7个X，共56个raw/preprocess完整配对X。example 5/8因preprocess cache缺失被剔除，补跑example 9后达到56。

两级分析：

- per-X internal oracle SVD：每个X、每层独立对 `[X tokens, 8*128]` 分解，测单篇X自身可压缩上限。
- cross-X 7-fold held-out feature Basis：每折拿掉一种target位置，按layer/head用其他X的所有token学习128维feature Basis；未见X只允许oracle token coefficient投影。56个X均恰好作为一次held-out document。

### 幅度与单X内部可压缩性

| source/object | online gap | rank8 energy | rank32 energy | rank64 energy |
|---|---:|---:|---:|---:|
| raw K | .230 | 70.0% | 89.2% | 95.7% |
| raw V | .396 | 56.3% | 82.5% | 91.9% |
| preprocess K | .357 | 64.5% | 85.4% | 93.8% |
| preprocess V | .756 | 54.4% | 79.4% | 90.6% |

每篇X内部K比V更可压缩；rank8对V仍只解释约54%--56%，再次说明Value不是极低秩。

### 不同X的方向与分布特征

同一固定prefix组内：

| source/object | mean-Delta方向 cosine | layer energy profile cosine | layer×head profile cosine | 跨X PC1 / PC4 | effective rank |
|---|---:|---:|---:|---:|---:|
| raw K | .816 | .992 | .988 | 85.0% / 97.4% | 1.91 |
| raw V | .673 | .997 | .992 | 72.6% / 92.3% | 2.69 |
| preprocess K | .603 | .986 | .976 | 78.9% / 96.2% | 2.25 |
| preprocess V | .530 | .996 | .986 | 76.1% / 94.9% | 2.50 |

最明显的新规律是：**不同X的精确feature方向只有中等一致，但Delta能量落在哪些layer/head上高度一致。** layer profile cosine均超过.985，head profile超过.975；Value方向比Key更document-specific，但“更新位置模板”反而非常稳定。

跨X PCA基于每篇X的token-mean Delta，每组仅7篇，因此只能作为组内描述；PC4覆盖92%--97%、effective rank约2--3，说明document-level mean方向存在少量主模态，但不代表token级Delta可由rank4恢复。

### Cross-X held-out Shared Basis

每格为 `Delta recovery error / final KV error / explained energy`：

| source/object | rank4 | rank8 | rank16 | rank32 | rank64 |
|---|---:|---:|---:|---:|---:|
| raw K | .736/.170/45.6% | .652/.150/57.3% | .547/.126/69.9% | .433/.100/81.1% | .294/.068/91.3% |
| raw V | .925/.366/14.5% | .889/.352/20.9% | .834/.330/30.4% | .743/.294/44.7% | .574/.227/67.0% |
| preprocess K | .798/.282/36.2% | .723/.257/47.6% | .623/.223/61.1% | .501/.180/74.9% | .347/.125/87.9% |
| preprocess V | .926/.696/14.2% | .892/.670/20.3% | .838/.629/29.7% | .746/.559/44.3% | .573/.429/67.1% |

对比per-X oracle与cross-X held-out，Value rank64从单X内部约91%--92% energy降到跨X约67%；这证明大量Value低秩方向是document-specific，不能由一个小的全局feature Basis复用。Key transfer明显更好，raw K rank32/64可解释81%/91%。

### 对 Adapter 的含义

1. 固定prefix不能让不同X共享一个小rank Value Basis；rank4/8跨X仅解释约14%/20% Value energy。
2. 可共享的最强结构是layer/head更新预算，而不是完整Delta向量。可训练一个共享router决定哪些layer/head需要更新。
3. Value Basis应优先per-document维护；全局Basis只能作为粗主干，后接document-specific Basis或稀疏recompute。
4. Key更适合共享Basis，Value更适合per-document template。K/V应采用非对称结构。
5. 推荐结构：
   `shared layer/head router + shared Key basis + per-document Value mean/basis + residual selector/recompute`。
6. 当前held-out coefficient仍是oracle projection，尚未证明A+B+C或cached X能预测系数；下一步才做prefix/target feature到coefficient的Ridge预测。

### 文件与命令

- `scripts/collect_fixed_prefix_vary_target.py`
- `scripts/analyze_fixed_prefix_vary_target.py`
- `scripts/analyze_cross_target_shared_basis.py`
- `results/fixed_prefix_vary_target/ex*/`
- `results/fixed_prefix_vary_target_{raw,preprocess}_{k,v}_*.{csv,json}`
- `results/cross_target_shared_basis_{raw,preprocess}_{k,v}_7fold*.{csv,json}`
- `figures/fixed_prefix_vary_target_rank_transfer.png`

```bash
CUDA_VISIBLE_DEVICES=<gpu> .../python scripts/collect_fixed_prefix_vary_target.py --example <id> --part ex<id>
CUDA_VISIBLE_DEVICES=<gpu> .../python scripts/analyze_fixed_prefix_vary_target.py --source <raw|preprocess> --kind <k|v>
CUDA_VISIBLE_DEVICES=<gpu> .../python scripts/analyze_cross_target_shared_basis.py --source <raw|preprocess> --kind <k|v>
```

## 2026-07-11 Per-document Value 系数能否由前序缓存预测（5 documents / 50 heldout contexts）

### 问题与严格基准

固定目标 document `X`，改变它前面的 RAG prefix `A+B+C+...`。目标量始终是 `X` 的 document Value KV：

- 不更新：直接使用 raw/preprocess cached `V_X`；
- full target：在完整 `system+prefix+X` 中对 `X` 全 token 重算得到 `V_X^full`；
- Delta：`V_X^full - V_X^cache`；
- `original_gap = ||V_cache-V_full||/||V_full||`；
- Adapter 后 gap：`final_kv_error = ||V_cache+Delta_hat-V_full||/||V_full||`。

实验使用5个固定X，每个X有50个不同prefix context；前40个训练（其中8个只用于选择Ridge alpha），后10个测试。汇总数字来自每种cache共50个held-out `(X, prefix)` case。每层从训练Delta学习per-document PCA basis，rank取4/8。比较mean template、position-only Ridge、cached-prefix Value mean pooling Ridge，以及使用真实测试Delta投影系数的oracle上界。本轮优先验证Value，尚未训练Key predictor。

### 结果

| cache | 方法 | 原始gap | Adapter后gap | 剩余Delta比例 | 解释Delta能量 |
|---|---|---:|---:|---:|---:|
| raw | 不更新 | 0.411 | 0.411 | 1.000 | 0% |
| raw | mean template | 0.411 | 0.218 | 0.535 | 69.6% |
| raw | position rank8 | 0.411 | 0.203 | 0.497 | 74.2% |
| raw | cached-prefix rank8 | 0.411 | 0.219 | 0.535 | 67.6% |
| raw | oracle rank8 | 0.411 | 0.166 | 0.407 | 82.4% |
| preprocess | 不更新 | 0.477 | 0.477 | 1.000 | 0% |
| preprocess | mean template | 0.477 | 0.222 | 0.549 | 65.9% |
| preprocess | position rank8 | 0.477 | 0.206 | 0.509 | 70.8% |
| preprocess | cached-prefix rank8 | 0.477 | 0.225 | 0.562 | 60.0% |
| preprocess | oracle rank8 | 0.477 | 0.169 | 0.418 | 80.2% |

这里“不更新的剩余Delta=100%”只表示`Delta_hat=0`时Delta误差归一化为1，并不表示KV本身与full相差100%；绝对KV gap是表中的0.411/0.477。

### 结论与判断

1. per-document mean template已经消除约45%--47%的Delta幅度，说明大量更新是固定X自身的稳定更新模板。
2. rank8 oracle还能将最终gap从mean的raw/preprocess 0.218/0.222降到0.166/0.169，证明query-specific/prefix-specific低维系数具有额外价值，但可获得增益约0.052，不能把全部重算误差消掉。
3. 两个位置标量（prefix token数、document数）已稳定优于mean；说明Delta系数首先强依赖位置/长度。
4. 当前1024维cached-prefix Value mean pooling Ridge没有优于position，且方差很大。这不是“prefix不可预测”的证据，而是40个训练context不足以拟合1024维特征，当前形式明显过拟合。
5. 下一步不应直接堆MLP。先把prefix压成低维、因果上有意义的特征：position/RoPE特征、分层Value统计、prefix末端若干token、attention/hidden summary；在>=50训练context/文档和更多X上做nested validation。只有稳定超过position baseline，才值得训练两层MLP。
6. 本轮preprocess gap大于raw只适用于这组source-conditioned online target；不能推翻旧Qwen2.5共同full-target实验中preprocess更近的结论，两者target定义不同。

### 命令与产物

```bash
CUDA_VISIBLE_DEVICES=<0..4> .../.venv/bin/python scripts/predict_perdoc_coefficients.py --part t<1..5> --source <raw|preprocess>
.../.venv/bin/python scripts/aggregate_predict_coefficients.py
```

- `scripts/predict_perdoc_coefficients.py`
- `scripts/aggregate_predict_coefficients.py`
- `results/predict_perdoc_coefficients_t{1..5}_{raw,preprocess}.csv`
- `results/predict_perdoc_coefficients_5targets_50heldout.csv`
- `results/predict_perdoc_coefficients_5targets_50heldout_summary.{csv,json}`
- `figures/predict_perdoc_coefficients_value.png`

## 2026-07-11 Strict document-disjoint prefix / Rank-direction 实验

### 为什么重做

上一轮40/10切分只保证完整prefix序列不重复；train/test仍复用同一批A/B/C documents。本轮对每个固定X把其余9个documents固定拆为5个train-prefix documents和4个test-prefix documents，交集严格为空。train只由前5个构造40个有序prefix，test只由后4个构造10个有序prefix。5个X共50个strict heldout prefixes/cache。所有case均为rate=1全token重算。

本实验仍局限于dataset example 0，因此验证的是同一RAG example内的unseen-prefix-document泛化，不是跨example泛化。

### 完整X Delta的per-document context-template Basis

rank表示：固定X和layer，把每个prefix产生的完整X Delta flatten为一个向量；40个train prefixes组成PCA矩阵。测试时使用真实heldout Delta的oracle投影系数，因此这里只衡量方向/Basis的表示上界，不代表系数已可预测。Key先按`-prefix_tokens`做relative RoPE逆旋转，再比较方向。

| cache/object | no update gap | mean后gap | rank1后gap | rank8后gap | rank16后gap | rank8解释Delta能量 | train effective rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw RoPE-aligned K | .242 | .162 | .146 | .115 | .110 | 77.3% | 10.1 |
| preprocess RoPE-aligned K | .254 | .160 | .145 | .114 | .109 | 73.8% | 10.1 |
| raw V | .373 | .281 | .258 | .229 | .224 | 62.1% | 11.6 |
| preprocess V | .482 | .286 | .263 | .233 | .228 | 64.2% | 11.6 |

关键解释：

1. X的变化不是由一个固定rank-1方向决定。Value rank1只把解释能量从mean的raw/preprocess 43.3%/47.9%提高到51.9%/55.2%。
2. 存在稳定的低维头部，但不是极低秩。rank8达到Value 62%--64%，rank16为64%--66%，之后明显饱和；约34%--36%的Value Delta能量不在train-prefix学到的rank16方向中。
3. Key方向比Value更可压缩；rank8已解释74%--77%，但仍不是单方向现象。
4. train谱effective rank约K 10.1、V 11.6，与rank8--16开始饱和一致。适合“小型basis+residual”，不支持只使用单个LoRA方向。

### 更换X后的共享方向

不同X token数不同。为进行等维跨X比较，本轮把每层X token做mean pooling，再拼接64层/8 heads特征；先减去各X的train mean，再用5个X共200个train context学习shared residual basis。测试仍是50个strict unseen-prefix-document cases。

| object | shared rank1 | shared rank4 | shared rank8 | shared rank16 | shared rank32 |
|---|---:|---:|---:|---:|---:|
| raw RoPE-aligned K | 69.1% | 82.2% | 92.6% | 96.7% | 97.9% |
| preprocess RoPE-aligned K | 59.5% | 76.6% | 90.9% | 95.8% | 97.3% |
| raw V | 46.0% | 55.5% | 65.0% | 74.6% | 79.5% |
| preprocess V | 55.0% | 62.8% | 71.0% | 79.0% | 83.0% |

这些百分比是对完整Delta（包括per-X mean）的解释能量。结论是：更换A+B+C和X后，确实存在跨X共享的粗粒度更新方向，Key尤其强；但Value的逐token细节远弱于token-mean统计。Adapter应分两级：`per-document mean + shared rank8--16 global/context correction + document/token-local residual`。shared basis可负责层/头级整体偏移，不能直接宣称已恢复每个token的KV。

### 产物与命令

```bash
CUDA_VISIBLE_DEVICES=<0..4> .../.venv/bin/python scripts/collect_strict_prefix_deltas.py --example 0 --target <1..5> --part strict_t<1..5>
CUDA_VISIBLE_DEVICES=<gpu> .../.venv/bin/python scripts/analyze_strict_rank_directions.py --source <raw|preprocess> --kind <k|v>
CUDA_VISIBLE_DEVICES=<gpu> .../.venv/bin/python scripts/analyze_strict_cross_x_basis.py --source <raw|preprocess> --kind <k|v>
```

- `scripts/collect_strict_prefix_deltas.py`
- `scripts/analyze_strict_rank_directions.py`
- `scripts/analyze_strict_cross_x_basis.py`
- `results/perdoc_context_deltas/strict_t{1..5}/`
- `results/strict_prefix_rank_{raw,preprocess}_{k,v}.{csv,json}`
- `results/strict_prefix_rank_{raw,preprocess}_{k,v}_overlap.csv`
- `results/strict_cross_x_{raw,preprocess}_{k,v}.csv`

## 2026-07-11 Strict Prefix-to-Coefficient Predictor（非 oracle）

沿用document-disjoint split，并修正生成器默认把empty prefix同时放入train/test的问题：每个X补采一个非空test prefix，最终统一使用40 train + 10非空且只由heldout documents构造的test，raw/preprocess各50个有效case。

固定X，每层仅用train Delta学习rank8 Value basis。测试时禁止读取真实Delta coefficient。输入对照为：X-only（固定X时等价于per-X train mean）、2维position、cached-prefix Value（全prefix token-weighted mean + last-document mean，经train-only PCA压到8维）、prefix Value+position；oracle-r8只作为表示上界。

| cache | 方法 | 原始gap | Adapter后gap | remaining Delta | explained Delta energy |
|---|---|---:|---:|---:|---:|
| raw | X-only / mean | .392 | .271 | .692 | 51.4% |
| raw | position | .392 | .333 | .850 | 11.0% |
| raw | prefix V | .392 | .267 | .680 | 53.2% |
| raw | prefix V + position | .392 | .306 | .781 | 32.9% |
| raw | oracle rank8 | .392 | .238 | .605 | 62.9% |
| preprocess | X-only / mean | .478 | .277 | .679 | 48.5% |
| preprocess | position | .478 | .340 | .867 | -12.4% |
| preprocess | prefix V | .478 | .272 | .670 | 49.9% |
| preprocess | prefix V + position | .478 | .312 | .786 | 20.8% |
| preprocess | oracle rank8 | .478 | .242 | .598 | 59.4% |

结论：

1. 当前prefix Value predictor在strict unseen documents上只将gap改善raw `.0046`、preprocess `.0047`；oracle可改善`.0338/.0344`，即当前输入/模型只兑现约14%的可用低秩增益。
2. 最主要贡献仍是X自己的mean Delta template，而不是当前prefix内容特征。
3. position在seen-document split曾有效，但strict split明显失效且高方差，说明此前结果包含组合分布/文档长度相关性，不能作为可泛化预测依据。
4. prefix+position更差，表明40个train prefixes不足以稳健拟合当前交互；不应继续直接上MLP。
5. 这不等于prefix与Delta无关：oracle仍有约`.034` gap空间。但需要跨多个dataset examples扩大训练，并用更因果的逐层prefix末端hidden/attention summary；必须继续以X-only mean为强baseline。

产物：`scripts/predict_strict_coefficients.py`、`results/strict_predict_strict_t{1..5}_{raw,preprocess}.csv`、`results/strict_predict_5targets_50heldout{,_summary}.csv`。

## 2026-07-11 Preprocess-v2 Static Value Bias：正式RAG顺序sanity

### 实现

对example 0的X1--X5，使用40个strict train-prefix Delta的per-document mean：

`V_static_v2(X) = V_source_cache(X) + mean_train[V_full(X|prefix)-V_source_cache(X)]`。

新建独立cache目录，原cache未覆盖；Key和未修改chunks使用symlink，X1--X5 Value写新tensor。写盘FP16与FP32计算目标的relative error约`.00166`。`rate=0`路径确认会加载磁盘KV并完全跳过selector/recompute。

正式顺序为`system+doc1+...+doc10+query`，query是`What network is National Cycle Route 57 part of?`。以原cache rate1全token重算为target，先报告已有bias的doc1--doc5。

| source | source rate0 Value gap | static-v2 rate0 Value gap | cosine: source -> v2 |
|---|---:|---:|---:|
| raw | .324 | .291 | .935 -> .955 |
| preprocess | .479 | .291 | .861 -> .955 |

逐文档relative L2：

| doc/order | raw source -> v2 | preprocess source -> v2 |
|---|---:|---:|
| X1 / first doc | .000 -> .361 | .404 -> .361 |
| X2 | .387 -> .263 | .384 -> .263 |
| X3 | .380 -> .280 | .395 -> .280 |
| X4 | .447 -> .342 | .354 -> .342 |
| X5 | .404 -> .208 | .857 -> .208 |

### 结论：如何加才有用

1. static-v2在正式文档pipeline中确实有效，尤其preprocess和后序文档；不运行Transformer即可明显缩小Value gap。
2. raw/preprocess v2结果几乎一致不是bug：分别加source-conditioned mean Delta后，都近似得到`mean_train(V_full)`。这说明v2可以成为统一的offline contextualized Value cache，而不必先做现有preprocess再叠加。
3. 不能给每个document永久写一个无条件bias。raw X1在第一个文档位置因没有前序document，offline cache已经等于full（gap 0）；盲加跨位置mean反而制造`.361`误差。
4. 可用的v2应是静态但position-conditioned：离线为每个X按`prefix token/order bucket`保存若干Value版本，在线只根据已知文档offset选择或插值，不读取query、不跑Transformer。最少应有`first/no-prefix`（raw bias=0）和`has-prefix`两档。
5. 当前只是partial-v2（X1--X5）和单example KV sanity，尚未证明答案质量。下一步先做position bucket/插值的KV heldout实验，再扩展到至少50 examples跑EM/F1/evidence和prefill latency；在此之前不改正式cache生成入口。

路径：

- `scripts/build_static_v2_cache.py`
- `scripts/evaluate_static_v2_pipeline.py`
- `results/static_v2_bias_ex0_t{1..5}_{raw,preprocess}_v.pt`
- `results/static_v2_pipeline_ex0.{csv,log}`
- `results/static_v2_pipeline_ex0_meta.json`
- cache：`kv_cache_static_v2_ex0_t1to5/`、`preprocess_kv_cache_global_topk10_bge_static_v2_ex0_t1to5/`

## 2026-07-11 Real Static Bias：50 examples当前结果

### 正式rate=0生成（已完成）

前50 main examples共有67个testable subquestions。所有方法走正式FusionRAG文档pipeline；full rate1为参考，raw/preprocess/static均rate0。生成限制32 tokens，以下F1为当前脚本的归一化token-F1，不是最终官方长decode指标。

| 方法 | cases | 对gold token-F1 | 与full输出token-F1 |
|---|---:|---:|---:|
| full rate1 | 67 | .173 | 1.000 |
| raw KV rate0 | 67 | .104 | .552 |
| random-cross-example static bias (M3,N2) rate0 | 67 | .145 | .627 |
| BGE top-k preprocess KV rate0 | 67 | .146 | .672 |

M3/N2静态bias不看测试subquery prefix，已经恢复raw->full答案F1差距的约59%：`(.145-.104)/(.173-.104)`；表现接近当前top-k preprocess，但仍落后full。需要用官方500-token/evidence judge复核，当前32-token EM均为0，不可作为EM结论。

### 校准组合二维网格：Value relative L2

正式full-context Value为target；raw和BGE preprocess都是rate0缓存。M是额外跨example校准prefix文档数；N是累计Delta样本数。M0仍会把目标example自身documents做不同排列，因此它测试“无需外部prefix，只用同example文档共现/位置校准”。M0/M1/M3/M5均67/67完成，M10当前58/67。

| M | raw | N1 | N2 | N4 | N8 | N16 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | .416 | .374 | .332 | .305 | .290 | **.283** |
| 1 | .416 | .376 | .338 | .311 | .295 | .288 |
| 3 | .416 | .379 | .344 | .318 | .300 | .293 |
| 5 | .416 | .383 | .350 | .320 | .303 | .297 |
| 10* | .414 | .393 | .363 | .329 | .310 | .306 |

`*` M10为58-case中间结果。BGE top-k preprocess rate0在同一Value口径下平均gap为`.754`，明显差于raw；这说明现有top-k preprocess Value不是正式文档顺序的full Value近似，答案收益可能来自Key/生成非线性，不能只看Value L2判断。

当前结论：

1. N是关键变量，1->16在所有M下单调改善；mean bias需要多组校准，N2明显不足。
2. 外部随机文档数M不是越多越好，当前M越大越差；随机跨example语义引入分布偏移。
3. 最优M0/N16说明主要可学习结构来自目标example documents之间的共现、排列和position平均，而不是无关外部文本。
4. 下一步应围绕M0做position-balanced permutations，并比较同example BGE top-k相关docs，而不是继续增加随机外部docs；完成M10后再做正式长decode。

产物：`results/real_static_bias_50/`、`results/bias_calibration_scaling/m{0,1,3,5,10}/`。

### GLM-5.2正式judge补充

对上述67条×4方法已有输出使用项目原`GLM-5.2` endpoint离线judge：

| 方法 | correct/n | GLM accuracy |
|---|---:|---:|
| full rate1 | 17/67 | 25.37% |
| raw KV rate0 | 12/67 | 17.91% |
| BGE preprocess KV rate0 | 15/67 | 22.39% |
| random static M3/N2 rate0 | 17/67 | 25.37% |

static M3/N2在当前67条32-token输出上达到full的judge accuracy，比raw高7.46个百分点、比preprocess高2.98个百分点。由于decode被截到32 tokens且n=67，必须用原pipeline 500-token输出复核，不能视为最终accuracy。

### 原pipeline静态bias加载接口

已在`ktransformers/util/utils.py::load_kv_and_generate`和`test_fusionrag_reflect_preprocess_exp.py`增加：

```bash
--static_value_bias_path <dir>
--static_value_bias_scale 1.0
--static_value_bias_require_all true
```

底座仍由`--preprocess false/true`选择raw/preprocess；随后加载`<example>_<chunk>_value_bias.pt`并加到Value。原pipeline的文档构造、rate0、500-token生成、GLM judge和CSV均不变；结果目录自动附加bias tag，避免覆盖baseline。

### BGE top-k preprocess 学习的静态 Value bias

复用旧`preprocess_kv_cache_global_topk10_bge`作为BGE校准target：

```text
Bias_BGE_raw(X) = V_preprocess_BGE(X) - V_raw(X)
```

正式pipeline加载raw KV后加该bias，得到`raw K + BGE preprocess V`，rate0生成；与raw rate0、完整preprocess rate0（preprocess K+V）和full rate1比较。该实验隔离旧preprocess的Value更新贡献，不使用正式subquery full Value。

前50 examples共写586个可配对bias文件；旧preprocess cache另缺264个raw文件对应项，主要来自未覆盖examples/chunks。正式67个testable subquestions在上一轮均有完整preprocess selected chunks。脚本`build_bge_preprocess_value_bias.py`；正式运行日志`results/original_pipeline_bge_value_bias/logs/rawK_bgeV_rate0.log`。

注意：对preprocess底座而言，同一BGE校准target的source-specific bias是零，因为旧preprocess cache本身就是该target；因此有意义的新分支是`raw K + preprocess V`，完整`preprocess K+V`已是原baseline。

## 2026-07-12 正式结果总览

完整500-token+GLM结果、Raw M/N网格、partial状态和进程清理记录见`CURRENT_RESULTS_SUMMARY.md`。最重要结果：full/raw/preprocess sub accuracy为86.57%/74.63%/80.60%；最佳raw+static仅71.64%；raw K+preprocess V仅47.76%。Value-only L2改善没有提升accuracy，且K/V source不匹配会严重退化，后续应转向匹配的联合K/V静态更新。

## 2026-07-11 M0/N16原Pipeline 500-token + GLM结果

前50 main examples中36个可评main、67个subquestions。五条均使用原`test_fusionrag_reflect_preprocess_exp.py`、正式文档pipeline和GLM-5.2 judge。

| 方法 | Main accuracy | Sub accuracy | F1 | EM |
|---|---:|---:|---:|---:|
| full rate1 | 27/36 = 75.00% | 58/67 = 86.57% | .6080 | .2985 |
| raw KV rate0 | 20/36 = 55.56% | 50/67 = 74.63% | .4439 | .1642 |
| preprocess KV rate0 | 23/36 = 63.89% | 54/67 = 80.60% | .5604 | .2537 |
| raw + M0/N16 Value bias rate0 | 20/36 = 55.56% | 47/67 = 70.15% | .4616 | .2239 |
| preprocess + raw-defined M0/N16 bias rate0 | 17/36 = 47.22% | 43/67 = 64.18% | .4025 | .1343 |

关键纠正：

1. M0/N16将Value L2 gap从raw `.416`降到`.283`，但GLM sub accuracy反而从74.63%降到70.15%。因此L2不是可单独优化的代理目标；attention/生成对某些layer/head/token方向更敏感。
2. raw+static的F1/EM略升但GLM accuracy下降，说明输出表面词重合改善、语义正确case却减少，必须以GLM/evidence为主指标。
3. `preprocess + raw-defined bias`不是公平的source-conditioned方案：当前bias定义为`full_calibration-raw`，直接加到preprocess会重复/错位更新，其显著退化符合预期。后续若测preprocess底座，必须单独学习`full_calibration-preprocess` bias，不能复用raw bias。
4. N16全量平均可能过度修正。M/N正式pipeline网格仍需完成，同时必须加入bias scale `alpha=.25/.5/.75/1.0`，以及layer/head gating；不能依据KV L2选择N。

结果：`results/original_pipeline_m0n16/Qwen3-32B/musique/`及`logs/`。
