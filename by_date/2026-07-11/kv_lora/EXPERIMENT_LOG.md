# KV-LoRA / Delta-KV 实验日志

## 2026-07-11 Phase 0: 代码和数据定位

- 已阅读根目录 `AGENT.md` / `AGENTS.md` 和本目录任务书 `READNE.md`。
- `qjy000` 检查：未发现 tmux 会话输出；8 张 H20 显存基本空闲。未访问或中断 `qjy001` 上的 235B 实验。
- 现有 raw KV 生成逻辑位于 `ktransformers/util/utils.py::prefill_and_save_kv_cache`。对非 system chunk，保存的是 `system + document` forward 后的 document span。
- raw cache 路径在主 runner 中构造为 `<cache_path>/<model_name>/<dataset_name>/kv_cache/{example_id}_{chunk_id}_key.pt/value.pt`。
- `prefill_and_save_kv_cache` 保存的 stacked KV shape 为 `[layers, batch, kv_heads, tokens, head_dim]`；token 维在 stacked tensor 中是第 3 维。
- `test_fusionrag_reflect_preprocess_exp.py` 的 rate=1 分支只做 full recompute generation，没有保存 full document KV。因此本轮新增轻量脚本直接 forward full prompt 并按 token offset 切 document span。
- full prompt 构造沿用 runner 的 `system + docs + question`，Qwen3 query 后缀含 `/no_think` 和空 `<think></think>` 块。

## 2026-07-11 Phase 1 计划

1. 用 Qwen3-32B，只跑 `example_id=0`、第 1 个 sub-question、最多前 2 个 doc。
2. 验证 offline/full document span shape 一致，Delta-K/V 非零且数值合理。
3. 输出 summary/layer/head/token CSV 和一张 singular spectrum 图。
4. 结果正常后才扩到 5 个 example。

## Sanity run: Qwen3-32B_ex0_1_sub1_docs2

- 启动命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/kv_lora/scripts/analyze_delta_kv_sanity.py --repo /raid/home/hming/FusionRAG-pca-analysis --model_path /mnt/qjhs-sh-lab-01/models/Qwen3-32B --model_type qwen3 --model_name Qwen3-32B --data_path /raid/home/hming/FusionRAG-pca-analysis/data/result_reflect.json --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3 --device cuda:0 --start_sample 0 --end_sample 1 --max_subquestions 1 --max_docs_per_subquestion 2 --max_cache_len 8192 --max_svd_rank 64
```

- 命令由 `scripts/analyze_delta_kv_sanity.py` 生成，范围 `example_id=[0, 1)`，每个问题最多 `1` 个 sub-question、每个 sub-question 最多 `2` 个 doc。
- 模型：`/mnt/qjhs-sh-lab-01/models/Qwen3-32B`；数据：`/raid/home/hming/FusionRAG-pca-analysis/data/result_reflect.json`；device：`cuda:0`。
- KV shape 顺序确认：`[layers, batch, kv_heads, tokens, head_dim]`。document token 维度为第 3 维；切片方式为 offline `system_len:system_len+doc_len`，full 按 `system + docs + query` 的累计 offset。
- 输出：`results/delta_kv_summary_Qwen3-32B_ex0_1_sub1_docs2.csv`、`results/delta_kv_layer_metrics_Qwen3-32B_ex0_1_sub1_docs2.csv`、`results/delta_kv_head_metrics_Qwen3-32B_ex0_1_sub1_docs2.csv`、`results/delta_kv_token_metrics_Qwen3-32B_ex0_1_sub1_docs2.csv`、`figures/singular_spectrum_sanity_Qwen3-32B_ex0_1_sub1_docs2.png`。
- 结果摘要：
  - doc_rank=0：key relative L2 `0.0087`，value relative L2 `0.0274`，Delta 很小。
  - doc_rank=1：key relative L2 `0.5051`，value relative L2 `0.3896`，Delta 明显放大。
  - layer energy：value 的最大 Delta 层集中在最后几层 `58-63`；key 的较大层更分散。
  - low-rank：doc_rank=1 的 top Delta layers 需要 rank90 `27-38`、rank95 `42-53`、rank99 `69-76`，不能称为极低秩。
  - sparse：元素级近零比例不高，尤其 doc_rank=1 key 的 `abs(delta)<1%RMS` 只有约 `4.2%`。
- 异常/注意：cosine 有一条 key 为 `1.000120`，略大于 1，可能是 bf16/float32 数值误差；后续统计中 cosine 只作参考。

## Phase 2 run: Qwen3-32B_ex0_5_sub1_docs2

- 启动命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/kv_lora/scripts/analyze_delta_kv_sanity.py --repo /raid/home/hming/FusionRAG-pca-analysis --model_path /mnt/qjhs-sh-lab-01/models/Qwen3-32B --model_type qwen3 --model_name Qwen3-32B --data_path /raid/home/hming/FusionRAG-pca-analysis/data/result_reflect.json --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3 --device cuda:0 --start_sample 0 --end_sample 5 --max_subquestions 1 --max_docs_per_subquestion 2 --max_cache_len 12000 --max_svd_rank 64 2>&1 | tee MOTIVATION_EXPERIMENTS/kv_lora/results/phase2_qwen3_32b_ex0_5_run.log
```

- 输出 tag：`Qwen3-32B_ex0_5_sub1_docs2`。
- 结果摘要：
  - all key relative L2 mean/median：`0.2769 / 0.2574`。
  - all value relative L2 mean/median：`0.2077 / 0.1826`。
  - doc_rank=0 key/value relative L2 mean：`0.0070 / 0.0261`，基本近零。
  - doc_rank=1 key/value relative L2 mean：`0.5468 / 0.3892`，显著增大。
  - value top layers 聚合为 `62,61,60,63,59,58,57,56`，集中在最后层。
  - key top layers 分散，但疑似受 RoPE 位置平移污染。
- 异常/注意：
  - example 4 doc_rank=0 的 Delta 为 0，其他 doc_rank=0 有轻微非零；初步判断是数值路径差异，不影响“doc_rank=0 应近零”的 sanity 结论。
  - 原始 key Delta 不能直接解释为上下文更新，因为 full prompt 中 doc_rank=1 的绝对位置不同。

## Phase 2b run: Qwen3-32B_ex0_5_sub1_docs2_ropekey

- 启动命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/kv_lora/scripts/analyze_delta_kv_sanity.py --repo /raid/home/hming/FusionRAG-pca-analysis --model_path /mnt/qjhs-sh-lab-01/models/Qwen3-32B --model_type qwen3 --model_name Qwen3-32B --data_path /raid/home/hming/FusionRAG-pca-analysis/data/result_reflect.json --bge_model_path /mnt/qjhs-sh-lab-01/models/bge-m3 --device cuda:0 --start_sample 0 --end_sample 5 --max_subquestions 1 --max_docs_per_subquestion 2 --max_cache_len 12000 --max_svd_rank 64 --include_rope_aligned_key 2>&1 | tee MOTIVATION_EXPERIMENTS/kv_lora/results/phase2_qwen3_32b_ex0_5_ropekey_run.log
```

- 输出 tag：`Qwen3-32B_ex0_5_sub1_docs2_ropekey`。
- 口径：`key_rope_aligned` 把 offline key 按 `load_kv_and_generate(..., revert_rope=True)` 中同样的 relative RoPE shift 旋转到 full prompt document offset 后再比较。
- 结果摘要：
  - doc_rank=1 raw key relative L2 mean/median：`0.5468 / 0.5554`。
  - doc_rank=1 key_rope_aligned relative L2 mean/median：`0.2049 / 0.1965`。
  - doc_rank=1 value relative L2 mean/median：`0.3892 / 0.3796`。
  - key_rope_aligned top layers 聚合为 `48,49,45,47,50,52,51,46`。
  - value top layers 仍为 `62,61,60,63,59,58,57,56`。
- 初步结论：
  - 原始 K Delta 的大部分来自 RoPE 位置差异；后续 K 机制分析必须使用 rope-aligned key。
  - V Delta 是更直接的上下文更新信号，在 5-example 小样本上稳定且集中于后层。

## Sanity run: Qwen3-32B_ex0_1_sub1_docs2

- 命令由 `scripts/analyze_delta_kv_sanity.py` 生成，范围 `example_id=[0, 1)`，每个问题最多 `1` 个 sub-question、每个 sub-question 最多 `2` 个 doc。
- 模型：`/mnt/qjhs-sh-lab-01/models/Qwen3-32B`；数据：`/raid/home/hming/FusionRAG-pca-analysis/data/result_reflect.json`；device：`cuda:0`。
- KV shape 顺序确认：`[layers, batch, kv_heads, tokens, head_dim]`。document token 维度为第 3 维；切片方式为 offline `system_len:system_len+doc_len`，full 按 `system + docs + query` 的累计 offset。
- 输出：`results/delta_kv_summary_Qwen3-32B_ex0_1_sub1_docs2.csv`、`results/delta_kv_layer_metrics_Qwen3-32B_ex0_1_sub1_docs2.csv`、`results/delta_kv_head_metrics_Qwen3-32B_ex0_1_sub1_docs2.csv`、`results/delta_kv_token_metrics_Qwen3-32B_ex0_1_sub1_docs2.csv`、`figures/singular_spectrum_sanity_Qwen3-32B_ex0_1_sub1_docs2.png`。

## Sanity run: Qwen3-32B_ex0_5_sub1_docs2

- 命令由 `scripts/analyze_delta_kv_sanity.py` 生成，范围 `example_id=[0, 5)`，每个问题最多 `1` 个 sub-question、每个 sub-question 最多 `2` 个 doc。
- 模型：`/mnt/qjhs-sh-lab-01/models/Qwen3-32B`；数据：`/raid/home/hming/FusionRAG-pca-analysis/data/result_reflect.json`；device：`cuda:0`。
- KV shape 顺序确认：`[layers, batch, kv_heads, tokens, head_dim]`。document token 维度为第 3 维；切片方式为 offline `system_len:system_len+doc_len`，full 按 `system + docs + query` 的累计 offset。
- 输出：`results/delta_kv_summary_Qwen3-32B_ex0_5_sub1_docs2.csv`、`results/delta_kv_layer_metrics_Qwen3-32B_ex0_5_sub1_docs2.csv`、`results/delta_kv_head_metrics_Qwen3-32B_ex0_5_sub1_docs2.csv`、`results/delta_kv_token_metrics_Qwen3-32B_ex0_5_sub1_docs2.csv`、`figures/singular_spectrum_sanity_Qwen3-32B_ex0_5_sub1_docs2.png`。

### 5-example 只读汇总（`doc_rank=1`，即有前序 document context）

- 全局 relative L2：Delta-K mean/median=`0.5468/0.5554`，Delta-V=`0.3892/0.3796`；对照 `doc_rank=0` 为 K=`0.0070/0.0086`、V=`0.0261/0.0274`，后者是 prefix numerical baseline。
- 矩阵化为 `[document_token, kv_heads * head_dim]` 后，按 320 个 layer-record 平均，K 的 rank95=`55.75`、stable rank=`5.46`、effective rank=`76.93`；V 为 `53.55/5.69/74.94`。这表明稳定秩低但 95% 能量 rank 不低，不能把 Delta 直接描述为极低秩。
- layer：V 能量跨样本集中在层 56--63（最高层 62）；K 的高能量层更分散（6、9、15--17、32--33、40 等），未见同等明显的后层集中。
- head：8 个 KV head 的跨样本能量总量相近，没有单一主导 head；当前样本量不足以判定稳定 retrieval head。
- token：每篇 doc 的 top-10 token 平均承载 K `12.1%`、V `46.7%` 的 all-layer Delta energy；V 有明显 token 集中，K 没有强 token 稀疏。

## Sanity run: Qwen3-32B_ex0_5_sub1_docs2_ropekey

- 命令由 `scripts/analyze_delta_kv_sanity.py` 生成，范围 `example_id=[0, 5)`，每个问题最多 `1` 个 sub-question、每个 sub-question 最多 `2` 个 doc。
- 模型：`/mnt/qjhs-sh-lab-01/models/Qwen3-32B`；数据：`/raid/home/hming/FusionRAG-pca-analysis/data/result_reflect.json`；device：`cuda:0`。
- KV shape 顺序确认：`[layers, batch, kv_heads, tokens, head_dim]`。document token 维度为第 3 维；切片方式为 offline `system_len:system_len+doc_len`，full 按 `system + docs + query` 的累计 offset。
- 输出：`results/delta_kv_summary_Qwen3-32B_ex0_5_sub1_docs2_ropekey.csv`、`results/delta_kv_layer_metrics_Qwen3-32B_ex0_5_sub1_docs2_ropekey.csv`、`results/delta_kv_head_metrics_Qwen3-32B_ex0_5_sub1_docs2_ropekey.csv`、`results/delta_kv_token_metrics_Qwen3-32B_ex0_5_sub1_docs2_ropekey.csv`、`figures/singular_spectrum_sanity_Qwen3-32B_ex0_5_sub1_docs2_ropekey.png`。

## 2026-07-11 Phase 3：真实 FusionRAG 多 chunk 复用与重算 Delta

### 目的与口径

- 不再比较简化 dense prompt，而是直接进入 `ktransformers.util.utils.load_kv_and_generate`：装载完整 chunk KV、由 FusionRAG selector 挑选 token、与 query 一起重算并原位覆写 KV。
- 比较 `Delta_reprocess = KV_after_reprocess - KV_before_reprocess`；`before` 是 raw/preprocess chunk KV 已装入 cache 的状态，`after` 是真实 selector 的 selected token 重算完成、decode 前状态。
- 模型/范围：Qwen3-32B，`example_id=0` 的第一个 sub-question；12 document chunk、1313 document token；`rate=0.15`；raw KV 与 preprocess KV 各一次。

### 命令

```bash
CUDA_VISIBLE_DEVICES=4 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/kv_lora/scripts/run_detailed_reprocess_delta.py
```

### 结果与结论

- selector 在 raw/preprocess 两种起点均选择 196/1313 个 document token（14.93%）。未选 1117 token 的 K/V before/after Delta 为精确 0，证明变动来自 selected-token 原位重算，而非统计或 cache 对齐误差。
- raw 起点的 selected-token 平均逐层 relative L2：K=0.3251、V=0.5220；preprocess 起点：K=0.2860、V=0.4689。V 的重算更新持续大于 K（约 1.6 倍）。
- preprocess 相比 raw 已吸收一部分上下文修正：selected K/V 更新幅度分别下降约 12%/10%，但仍有很大的残余 Delta，因此 preprocess KV 不能替代 online selected-token update。
- 这组 Delta 是 KV-LoRA 应直接建模的对象：它严格局限于 selected tokens；“末尾 query 直接改变早先 document KV”的解释不成立，真正的在线变化来自 selected token 在已有多 chunk context 下重新前向计算。
- 已保存逐 layer SVD（rank90/95/99、stable/effective rank）、元素近零比例、layer-head energy、token energy/rank 的详细 CSV。当前只有一个多-chunk example，不能据此宣称跨 query shared basis 或稳定 retrieval head；下一步应扩展到 5 examples，并比较 selected Delta 对 full-offline Delta 的能量/重建覆盖。

### 文件

- `scripts/run_detailed_reprocess_delta.py`
- `results/reprocess_summary_qwen3_32b_ex0_fullchunks_fusionrag_rate0p15_detailed.csv`
- `results/reprocess_layers_qwen3_32b_ex0_fullchunks_fusionrag_rate0p15_detailed.csv`
- `results/reprocess_heads_qwen3_32b_ex0_fullchunks_fusionrag_rate0p15_detailed.csv`
- `results/reprocess_tokens_qwen3_32b_ex0_fullchunks_fusionrag_rate0p15_detailed.csv`

## 2026-07-11 Phase 4：Value Delta 的共享子空间首轮检验

- 任务来源：`goal.md` 的 Shared Subspace Analysis。
- 设计修正：Qwen/FusionRAG 是因果 decoder，后置 query 不会修改 document KV；因此固定目标 document（本例为 sub-question 的最后一个 chunk），通过改变它前面的 document chunks 构造 5 个真正会改变其 KV 的条件：前缀 0/1/2/4/8 chunk，对应 0/50/142/316/1004 prefix token。使用固定后置 query 仅保持 runner prompt 形式。
- 模型：Qwen3-32B；统计对象：Value Delta（full causal-prefix forward - `system+target` offline forward）。
- 输出：`results/value_shared_subspace_context_ex0_targetlast.json` 与 `results/value_shared_subspace_context_ex0_targetlast_layers.csv`。

### 结果

- Value relative L2 随前缀从 0 chunk 的 `0.0280`（数值基线）增加到 1/2/4/8 chunk 的 `0.3896/0.8599/0.8833/0.8657`，说明目标 document 的 Delta 是强烈的前缀-context effect，并在约两个 prefix chunk 后达到饱和量级。
- 大前缀条件之间的 whole-Delta cosine 较高：2 vs 4=`0.8358`、2 vs 8=`0.7957`、4 vs 8=`0.8136`；1 chunk 与大前缀约 `0.239`，0 chunk 与其余接近 0。这表明存在“上下文充分后”的方向稳定区，但不是所有 context 强行共用的单一方向。
- 逐层 pooled token-feature PCA 中，PC1/PC2/PC4 的平均解释方差为 `0.1417/0.2250/0.3274`，平均 effective rank=`59.94`。因此当前证据不支持用仅 2--4 个全局分量重建所有 token/layer Value Delta；shared structure 更可能是分层、分 token 或按 context regime 条件化的 basis。

### 解释

- 未来 KV-LoRA/Predictor 不应直接假设一个跨所有条件、极小 rank 的全局 Value template。
- 更合理的下一步是：只在相近 context regime 内估计 per-layer basis，并预测低维系数；同时把 prefix length/document position 作为 predictor 输入。需要扩至多个固定 document 和更多前缀组合后再训练 Ridge/MLP。

## 2026-07-11 Rate=1 后置 Query 不变性检验（goal.md 的关键设计验证）

- 固定 `example=0` 的全部 12 document chunk（1313 doc token），prompt 顺序严格为 `system + all_docs + query`，对所有 doc token rate=1 full prefill。
- Query 条件：原始、简短改写、无关 weather、无关 France、要求详细推理，共 5 个。
- 输出：`results/query_suffix_rate1_invariance_ex0.json`。
- 结果：每个非原始 query 与原始 query 的 document Value KV 均为 `relative_l2=0.0`、`max_abs=0.0`；cosine 的 `1.000831` 是 float32 cosine 的数值舍入，不能解释为真实差异。
- 结论：在当前 causal `docs -> query` 部署顺序中，后置 query 对 document KV 没有影响。因此 goal.md 中“不同后置 query 下 Delta 的 shared subspace/PCA/coefficient predictor”在此定义下退化：所有 query 的 Delta 完全相同，不存在 query-specific coefficient。
- 方法含义：若目标是 query-conditioned vector update，必须另行定义 `query -> document` 的 prompt，或改为研究当前部署下 query-conditioned selector support；不能将后置-query实验的零 Delta 误报为低秩 predictor 成功。

## 2026-07-11 Phase 5：不同 Query 下 rate=1 Reprocess Delta 的共享结构

### 实验定义

本实验严格采用用户指定的真实重算口径，而不是 dense-full KV 互比。固定 `example=0` 的全部 12 个 document chunk（1313 token）及顺序。对起点 `s ∈ {raw, preprocess}` 和 query `q`：

```text
B_s      = chunk KV 全部装载后、重算前的 document KV
A_s(q)   = rate=1 选择全部 document token 并 reprocess 后的 KV
Delta_s(q) = A_s(q) - B_s
```

测试 8 个 query：原始、简短改写、正式改写、长推理要求、相关但不同问题、weather negative、France negative、math negative。每个 run 均验证 selected document tokens=`1313/1313`。K 是 `revert_rope=true` 后在最终多-document position 上参与重算的对齐口径；Value 单独统计。

### 启动命令

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/kv_lora/scripts/analyze_rate1_query_shared_delta.py
```

### 输出

- `scripts/analyze_rate1_query_shared_delta.py`
- `results/qwen3_32b_ex0_rate1_query_shared_delta_layers.csv`
- `results/qwen3_32b_ex0_rate1_query_shared_delta_manifest.json`
- `results/qwen3_32b_ex0_rate1_query_shared_delta_summary.json`
- `results/qwen3_32b_ex0_rate1_query_shared_delta_run.log`

### 结果

对 raw/preprocess、K/V、全部 64 层，任一 query 变体与原始 query 的 Delta 差异均为：

```text
max relative L2 = 0
max absolute element difference = 0
before-cache max difference = 0
```

cosine 在非零层约为 1；raw-V summary 中出现的最小 cosine=0 来自某些层的 Delta 本身为零，零向量 cosine 没有定义，不能解释为 query 差异。

### 对研究问题的回答

1. 不同 Query 的 Delta 是否位于同一个低维子空间？
   - 是，但属于退化情形：所有 query 的 Delta 逐元素完全相同。未中心化 query×Delta 矩阵的 query 维 rank=1；中心化后矩阵全零、总方差为零。
2. PCA/SVD 是否能找到共享 Basis？
   - 一个固定 Delta template 就可以对同一 KV 起点下的全部 query 做零误差重建。但这不是学到 query-conditioned shared basis，而是因果顺序导致的 query invariance。
3. Query 是否只改变低维系数？
   - 否。系数也没有变化；query 不改变 document reprocess Delta。后置 query 只可能影响 selector support；当 rate=1 固定 support 为全部 document token 时，query 的作用完全消失。

### 方法含义

在当前 `system + docs + query` 部署中，不值得训练 `query embedding -> Delta coefficient` predictor。更直接的模型是每个 context/document-position regime 的固定或 context-conditioned Delta template；query predictor 的任务应是 support selection，而不是 rate=1 Delta vector。该结论目前固定于一个 example，需要在 5 examples 上验证“每个 example 内 query invariance”，但因果机制预期不变。

## 2026-07-11 Phase 6：Query-prefix Value Delta Shared Basis 对照

### 设计

- 固定 example 0 的全部 12 document chunk（1313 doc token）。将 query 放到 document 前：`system + query-prefix + all_docs`，使 document token 能因果访问 query。
- 8 个 query：原始、短改写、正式改写、推理扩展、相关问题、weather/France/math negatives。
- 为隔离 query 内容与 document absolute position，所有 query-prefix 统一 pad 到 34 token；原始长度为 24--34 token。
- Value 优先。`A(q)` 是 query-prefix full prefill 的 document Value；分别定义 `Delta_raw(q)=A(q)-B_raw`、`Delta_preprocess(q)=A(q)-B_preprocess`。raw/preprocess base 来自同一 12 chunk cache。
- 输出：`scripts/analyze_query_prefix_shared_value_delta.py`、`results/qwen3_32b_ex0_query_prefix_fixedlen_value_shared.json`、对应 `_layers.csv` 与运行日志。

### Pairwise 结果

| 起点 | Delta cosine mean | min--max | query 间 relative difference mean | max |
|---|---:|---:|---:|---:|
| raw | 0.8530 | 0.8002--0.9459 | 0.5334 | 0.6274 |
| preprocess | 0.8188 | 0.7551--0.9328 | 0.5934 | 0.6948 |

Query-prefix 后 Delta 明显随 query 改变，但仍具有较强公共方向。preprocess 的 uncentered cosine 稍低，说明相对 preprocess base 的残余 Delta 更 query-sensitive。

### 逐层 query-sample PCA

排除 centered variance 为零的第 0 层后共 63 层。raw/preprocess 的 centered PCA 完全一致（base 在所有 query 间是常数，中心化后抵消）：

| 指标 | 63 层平均 |
|---|---:|
| PC1 explained variance | 0.5459 |
| PC1--2 | 0.6837 |
| PC1--4 | 0.8594 |
| Effective rank | 4.26 |

未中心化 PC1：raw=`0.9453`，preprocess=`0.9267`，说明 Delta 总量有很强共享模板；去掉均值模板后，约 4 个 query-specific 主方向可解释 86% 方差。

### 结论与限制

- 该实验初步支持 `Delta(q) ≈ mean_template + B c(q)`：mean template 解释大部分未中心化能量，少量 centered PCs 描述 query-specific variation。
- raw/preprocess 的 centered basis 相同是代数结果，不意味着二者的总 Delta 完全相同；起点只改变 query-independent offset，query-induced variation 来自同一个 full output `A(q)`。
- 当前只有 8 query、单 example，centered rank 上限为 7；尚未做 held-out query reconstruction，不能宣称 predictor 已验证。
- 下一步应增加 query 数并做 leave-one-query-out：训练 query basis 后预测 held-out coefficient；同时在多个 example 上检验 basis 是否跨 document 共享。

## 2026-07-11 Phase 6b：Query-prefix RoPE-aligned Key 与 Value 分开比较

- 使用与 Phase 6 完全相同的 8 个 query、固定 34-token prefix、12 chunks/1313 doc tokens。
- Key base 对每个 chunk 应用与 FusionRAG `revert_rope` 一致的相对位置旋转；position delta=`34 query tokens + 前序 chunk 累计 tokens`。因此下表是 RoPE-aligned Key，不是 raw-position Key。
- 输出：`scripts/analyze_query_prefix_shared_key_delta.py`、`results/qwen3_32b_ex0_query_prefix_fixedlen_key_shared.json`、对应 `_layers.csv` 和 run log。

| 对象 | 起点 | pairwise cosine mean | min--max | query relative difference mean | max |
|---|---|---:|---:|---:|---:|
| RoPE-aligned K Delta | raw | 0.9415 | 0.9034--0.9938 | 0.3592 | 0.4575 |
| RoPE-aligned K Delta | preprocess | 0.9095 | 0.8542--0.9860 | 0.4352 | 0.5499 |
| V Delta | raw | 0.8530 | 0.8002--0.9459 | 0.5334 | 0.6274 |
| V Delta | preprocess | 0.8188 | 0.7551--0.9328 | 0.5934 | 0.6948 |

逐层 centered PCA：K 的 PC1/PC1--2/PC1--4=`56.1%/69.8%/86.7%`、effective rank=`4.13`；V 为 `54.6%/68.4%/85.9%`、effective rank=`4.26`。未中心化 PC1：raw/preprocess K=`95.9%/94.0%`，V=`94.5%/92.7%`。

结论：K 与 V 都表现为强均值模板加约 4 维 query residual，但 K 的 query 间方向更一致、相对变化更小；V 更 query-sensitive。未来 shared-basis predictor 应分别建模 K/V，不能 concat 后给出单一结论。Value coefficient predictor 是更重要也更困难的部分；RoPE-aligned Key 更接近固定模板，可优先尝试均值模板或更小 coefficient correction。

## 2026-07-11 Phase 7：Query-prefix Value Shared Basis Held-out 验证

### 设计

- 固定 example 0、12 chunks/1313 doc tokens；query 位于 docs 前，并统一到 38 prefix tokens。
- 16 query，覆盖 paraphrase/reasoning/related/negative 四类；每类 3 train + 1 held-out，总计 12 train/4 test。
- 每层只用 12 train Value Delta 拟合 PCA basis；对 held-out query 使用 oracle projection coefficient，测试 rank 0(mean template)、1/2/4/8。oracle coefficient 排除了 predictor 误差，专门检验 basis 的泛化上限。
- raw/preprocess 分开。输出：`scripts/analyze_query_prefix_value_heldout.py`、`results/qwen3_32b_ex0_query_prefix_value_heldout_12train4test.csv`、summary JSON、run log、`figures/query_prefix_value_heldout_rank_curve.png`。

### 结果（64 layer × 4 held-out query 平均）

| 起点 | rank | relative L2 | cosine | full Delta explained | query residual explained |
|---|---:|---:|---:|---:|---:|
| raw | 0 mean | 0.2156 | 0.9541 | 94.06% | 0% |
| raw | 1 | 0.1768 | 0.9629 | 95.80% | 32.29% |
| raw | 2 | 0.1654 | 0.9651 | 96.22% | 42.07% |
| raw | 4 | 0.1620 | 0.9658 | 96.36% | 44.26% |
| raw | 8 | 0.1558 | 0.9669 | 96.57% | 48.42% |
| preprocess | 0 mean | 0.2535 | 0.9593 | 92.07% | 0% |
| preprocess | 1 | 0.2071 | 0.9716 | 94.47% | 32.29% |
| preprocess | 2 | 0.1934 | 0.9746 | 95.04% | 42.07% |
| preprocess | 4 | 0.1894 | 0.9756 | 95.23% | 44.26% |
| preprocess | 8 | 0.1821 | 0.9770 | 95.52% | 48.42% |

### 讨论

- 强共享 mean template 泛化成立：即使不使用 query-specific basis，held-out full Delta 已解释 92--94%。
- query-specific residual 的低维泛化明显弱于 in-sample PCA：训练集内 rank-4 为 85.9%，held-out 仅 44.3%；因此不能声称“4维即可预测 query residual”。
- rank 2 后边际收益迅速下降；rank 2→8 residual explained 仅从 42.1% 增至 48.4%。当前全局 per-layer linear basis 对新 query residual 的覆盖有限。
- raw/preprocess 的 residual explained 完全一致，因为二者仅相差 query-independent base；但 full-Delta mean/template 指标不同。
- 方法建议：固定 mean Delta template 是目前最可靠组件。若要预测 residual，应增加训练 query 或按 query category/局部 regime 建 basis；在此之前不进入 MLP predictor。

## 2026-07-11 Adapter 可行性主线启动

- 停止扩展 query-prefix 支线。
- 第一项实验：真实 `system+docs+query` Pipeline，example 0，raw/preprocess × rate1，RoPE-aligned K/V 分开，逐层 rank 1--128 rate-distortion。
- 输出必须同时包含 original gap、Delta recovery、final KV error，单例通过后再扩5/20 examples。

## 2026-07-11 Canonical Stage A：Example 0 Rate=1 Rank-Distortion

### 设置

- 当前真实 `system+docs+query` Pipeline；12 chunks、1313 doc tokens；raw/preprocess 分开。
- 使用真实 `load_kv_and_generate`，rate=1确认选择1313/1313 tokens；snapshot重算前B与重算后T。
- RoPE-aligned K/V逐层 reshape 为 `[tokens, kv_heads*head_dim]`；`torch.svd_lowrank(q=128,niter=2,seed=0)`，rank=1/2/4/8/16/32/64/128。
- 输出：`scripts/analyze_adapter_rank_distortion.py`、`results/qwen3_32b_ex0_rate1_adapter_rank_distortion.csv`、summary JSON、run log、`figures/adapter_rate1_rank_distortion_ex0.png`。

### 实际 Original Gap（全局）

| 起点 | K `||Delta||/||B||` | V `||Delta||/||B||` |
|---|---:|---:|
| raw | 0.2656 | 0.4833 |
| preprocess | 0.2179 | 0.4378 |

preprocess 将rate1 K/V gap分别降低约18%/9%；V gap仍约为K的1.8--2.0倍。

### 逐层 Rank-Distortion 平均

| 起点/对象 | rank8 explained | rank8 Delta error | rank32 explained | rank64 explained | rank128 explained | rank128 final KV error |
|---|---:|---:|---:|---:|---:|---:|
| raw K | 49.7% | 0.705 | 68.3% | 77.6% | 86.1% | 0.098 |
| raw V | 37.9% | 0.765 | 58.6% | 70.1% | 80.8% | 0.166 |
| preprocess K | 47.8% | 0.721 | 68.5% | 78.5% | 87.2% | 0.077 |
| preprocess V | 41.0% | 0.761 | 62.0% | 73.6% | 84.2% | 0.134 |

### 结论

- 当前整层 token×feature tensorization 不具有极低秩结构。rank 4/8不足以高保真填补gap，尤其Value。
- K比V更可压缩；preprocess对高rank K/V略有改善，但没有把问题变成小rank LoRA。
- rank128仍留下raw/preprocess V约40.4%/38.4%的Delta norm误差（final KV error约0.166/0.134）。
- 这否定了“单个很小全层Basis一次性输出所有token gap”的简单版本，但不否定per-head、per-chunk或局部Basis；下一步优先比较这些tensorization，再决定Adapter并行/分chunk结构。

## 2026-07-11 样本规模修正

用户要求后续验证至少50 examples取平均后再形成结论。此前example 0结果全部降级为sanity。下一轮复用`fusionrag-reflect-qwen3-full-cache`，按GPU分片运行50个有效examples，并汇总均值、标准差和分位数。


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

## 2026-07-11 Prefix -> per-document Value coefficient sanity

- 数据：5个固定target document，每个50个prefix context；0--39训练、40--49测试，每种cache共50个heldout case。
- 定义：比较目标document的cached Value与完整`system+prefix+X`全token重算Value；raw/preprocess分别建立各自online target。
- 表示：逐层per-document PCA basis，rank4/8；预测PCA coefficient，不直接预测完整KV。
- baseline：mean template、prefix长度/document数Ridge、cached-prefix逐层Value mean pooling Ridge、oracle projection。
- raw Value：不更新gap 0.411；mean 0.218；position-r8 0.203；cached-prefix-r8 0.219；oracle-r8 0.166。
- preprocess Value：不更新gap 0.477；mean 0.222；position-r8 0.206；cached-prefix-r8 0.225；oracle-r8 0.169。
- 结论：低维oracle系数有增益，position信息可预测一部分；当前1024维cached-prefix pooling在40个训练context下过拟合，没有超过position/mean。下一轮先做低维prefix feature与扩大训练context，不直接上MLP。
- 注意：`delta_recovery_error=1`仅是`Delta_hat=0`的归一化定义，不是KV gap 100%。

```bash
CUDA_VISIBLE_DEVICES=<0..4> /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/predict_perdoc_coefficients.py --part t<1..5> --source <raw|preprocess>
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python scripts/aggregate_predict_coefficients.py
```

产物：`results/predict_perdoc_coefficients_5targets_50heldout_summary.{csv,json}`、`figures/predict_perdoc_coefficients_value.png`。异常：第一次preprocess启动使用了不存在的conda路径，立即退出且未产生计算；改用项目`.venv`后正常完成。

## 2026-07-11 Strict unseen-prefix-document rank-direction

- 设计：固定5个X；每个X将其余9 docs拆成5 train-pool/4 test-pool，交集为空；40 train prefix + 10 test prefix。raw/preprocess各50个strict heldout case，rate=1。
- Key：对Delta施加`-prefix_tokens` relative RoPE旋转后分析。Value独立分析。
- 完整X context-template结果：raw/preprocess K rank8解释77.3%/73.8%，最终gap .115/.114；raw/preprocess V rank8解释62.1%/64.2%，最终gap .229/.233。rank16 Value仅升至63.7%/65.6%。effective rank K约10.1、V约11.6。
- 判断：不存在一个rank1方向决定X更新；存在rank8--16可压缩头部，但Value仍有约1/3 strict heldout能量无法由该basis覆盖。
- 跨X token-mean pooled shared basis：rank8解释raw/preprocess K 92.6%/90.9%，V 65.0%/71.0%；rank16为K 96.7%/95.8%，V 74.6%/79.0%。共享粗粒度方向成立，逐token Value residual仍document-specific。
- 架构含义：`per-X mean + shared rank8--16 layer/head correction + per-document/token residual selector or recompute`。不能用pooled结果宣称完整KV已恢复。
- 限制：只在example 0内做到prefix-document disjoint，尚未跨dataset example。
- 异常：首次从`kv_lora/`启动，因相对config路径缺失而在数据生成前退出；改从repo root启动后完成。一次scp连接被远端关闭，重试成功，无结果损坏。

命令和产物详见`README.md`同名章节。

## 2026-07-11 Strict non-oracle prefix coefficient predictor

- 修正：原strict test含一个empty prefix，与train empty重复且raw Delta为零；每个X补采一个非空test context，最终每种cache均为50个非空document-disjoint heldout cases。
- Value rank8 coefficient预测；test Delta不参与预测。X-only=per-X mean，prefix feature为cached Value整体mean+last-doc mean，经train-only PCA到8维，Ridge alpha在train内部选择。
- raw final gap：mean .2715，prefix V .2669，position .3334，joint .3064，oracle .2377。
- preprocess final gap：mean .2767，prefix V .2720，position .3395，joint .3117，oracle .2423。
- cached prefix只改善约.0047，约为oracle可用增益的14%；position在strict split退化。当前predictor不可替代recompute，主要有效项仍是per-X mean template。
- 运行中SSH长连接被远端关闭，但远端shell/jobs继续完成；检查确认raw/preprocess各5个CSV、每种方法各50行，无残留进程。

## 2026-07-11 Preprocess-v2 static Value cache sanity

- 构造：X1--X5各取40 train-prefix Delta mean，写`V_v2=V_source+meanDelta`到独立cache；Key/其他文件链接原cache，未覆盖原结果。
- 写盘检查：FP16 relative error约.00166。bias/base norm比raw约.31--.36；preprocess X1--X4约.28--.33，X5异常为.917，因此逐doc报告。
- 正式example0顺序`doc1..doc10`，rate1 full为target；有bias的X1--X5算术平均Value gap：raw `.324 -> .291`，preprocess `.479 -> .291`；cosine raw `.935 -> .955`，preprocess `.861 -> .955`。
- 关键反例：raw first doc原gap为0，静态mean使其变为.361。说明单一position-independent tensor不可直接普适；至少按no-prefix/has-prefix门控，更合理是prefix offset buckets或插值。
- source-specific v2最终几乎相同，因为`source + mean(full-source)=mean(full)`；v2可直接定义为offline contextualized full-Value template。
- 当前仅单example/partial docs的KV sanity，未跑答案指标。下一步先验证position-conditioned static v2，再扩展50 examples。
- 脚本：`build_static_v2_cache.py`、`evaluate_static_v2_pipeline.py`；结果：`static_v2_pipeline_ex0.csv`。

## 2026-07-11 Static-v2 full dataset launch

- 数据：`result_reflect.json`全部200 main examples、全部`should_test` sub-questions，topk10 BGE正式document顺序。
- 完整preprocess cache只覆盖135/200 examples，因此全量统一使用完整raw cache（200/200）。
- v2定义：以`(example, subquery, retrieval order)`为静态cache key，离线full prefill提供contextualized Value；在线保留raw RoPE-shifted Key、注入static full-context Value并rate0生成。
- 对照：full rate1、raw rate0、static-v2 rawK/fullV rate0；`max_new_tokens=32`。8 shards分别运行在qjy000 GPU0--7，JSONL逐条写入并支持resume。
- 重要边界：这是retrieval-set/order-specific static-v2上界，不是一个chunk tensor跨任意RAG集合复用；离线fullV生成成本和存储必须单独报告。
- 首次tmux server异常，未启动任务；改用nohup+PID。首次nohup运行在第一条decode遇到nested list格式后退出且未写结果；修正兼容解码后8 shard正常启动。

```bash
CUDA_VISIBLE_DEVICES=<shard> nohup .../.venv/bin/python scripts/evaluate_static_v2_full_dataset.py --shard <0..7> --num-shards 8 --max-new-tokens 32 > results/static_v2_full_dataset/shard_<n>.log 2>&1 &
```

输出：`results/static_v2_full_dataset/shard_00..07.jsonl`及日志/PID。

## 2026-07-11 Real static bias 50-example launch

- 纠正oracle full-V方法：新bias不读取测试subquery context。
- random-cross-example校准：3个其他example chunks作为prefix；目标example全部docs做随机/反向两种排列，两次full prefill一次性得到所有X的Delta并求平均。
- 正式测试：raw K/V + static random bias、rate0；对照raw rate0/full rate1。top-k分支直接加载现有BGE preprocess cache，缺cache case记为null而非混入。
- 范围：前50 main examples全部testable subquestions，8 shard，32 decode tokens；JSONL resume。
- 脚本：`scripts/evaluate_real_static_bias_50.py`；输出：`results/real_static_bias_50/shard_00..07.jsonl`。
- 启动检查：8 shard均存活且已写首批记录。qjy001未触碰。

## 2026-07-11 Calibration M x N grid launch

- 网格：跨example prefix文档数`M=0,1,3,5,10`；累计bias样本数`N=1,2,4,8,16`。每两个样本换一组prefix并改变目标docs排列。
- 规模：前50 examples全部正式subqueries，指标为full Value target下的relative L2；另含raw/top-k preprocess。
- qjy003 GPU0--7：M0/M1/M5/M10各2 shards；仓库/cache在`/home/hming`，脚本增加跨机器路径检测。
- qjy000：M3共8 shards，利用real-static作业释放的GPU逐步补齐。
- qjy002显存被其他进程占用，未启动；qjy001未触碰。
- qjy003首次因hardcoded `/raid/home`导致import前退出，无模型/结果；修正后8进程存活并加载。脚本：`evaluate_bias_calibration_scaling.py`。
- CPU审计：初版每进程约400 threads、实际1000%--2200% CPU，原因是完整Value GPU->CPU后FP32差分/累加触发PyTorch/OpenMP全核并行；qjy003 load average一度276。将`torch.set_num_threads(4)`、interop=1并设置`OMP_NUM_THREADS=4/MKL_NUM_THREADS=4`后重启resume；当前每进程约110%--125% CPU，机器load持续下降。`nlwp`仍约197主要是CUDA/runtime idle worker，不代表实际占用197 cores。
- 当前结果：50 main/67 subquestions正式32-token生成已完成。gold token-F1 full/raw/random-M3N2/topk为.173/.104/.145/.146；random static恢复raw-full差距约59%，但需官方长decode复核。
- MxN Value gap：M0 N1/2/4/8/16=.374/.332/.305/.290/.283；M1=.376/.338/.311/.295/.288；M3=.379/.344/.318/.300/.293；M5=.383/.350/.320/.303/.297；共同raw=.416。M10 58/67中间结果N16=.306。
- 观察：N单调有效；随机外部prefix文档M增加反而退化。优先M0 position-balanced target-doc permutations。BGE preprocess rate0 Value gap=.754，需与Key和答案指标分开解释。
- GLM-5.2离线judge完成268请求：full/raw/preprocess/static-M3N2 accuracy为17/67(25.37%)、12/67(17.91%)、15/67(22.39%)、17/67(25.37%)。当前输出仅32 tokens，待原pipeline长decode复核。
- 按用户纠正新增原pipeline loader接口：raw/preprocess cache选择后可加`<example>_<chunk>_value_bias.pt`；CLI为`--static_value_bias_path/scale/require_all`，结果目录含bias tag。`py_compile`和`git diff --check`通过。
- 正在8 GPU重新生成/导出前50 examples的M0/N16 bias到`results/static_bias_m0n16_50/`；完成后直接调用原`test_fusionrag_reflect_preprocess_exp.py`跑rate0+GLM，不再使用简化生成脚本作为正式结果。
- M0/N16已导出550个document bias；原pipeline 5-way正式评测已启动：full、raw/preprocess rate0、raw/preprocess+M0N16 rate0。首次zsh字符串参数未拆分，argparse前退出；改为bash数组launcher后5进程正常生成。
- 为回答M/N对最终accuracy的影响，qjy003启动M=0/1/3/5/10校准导出，每次同时保存N=2/4/8/16。watcher将在所需bias各达到550文件后自动启动8个代表配置；每配置依次跑raw和preprocess底座的原pipeline rate0，共16条GLM评测。
- 代表配置：M0N2/M0N4/M0N8（M0N16另有运行）、M1N16、M3N2、M3N16、M5N16、M10N16。脚本：`launch_original_pipeline_bias_grid.sh`。
- Source-specific修正：raw校准原定义为`bias_raw=mean(full-raw)`，不能把它作为preprocess正式方法。利用恒等式`bias_pp=bias_raw+V_raw-V_preprocess`生成preprocess-source bias，无需重跑Transformer。按用户要求保留`preprocess+raw-bias`作为cross-source ablation，不删除/停止该比较；M0N16两条preprocess分支均运行。
- Grid每配置扩展为三条：`raw+raw-bias`、`preprocess+raw-bias`、`preprocess+preprocess-bias`；最后一条才是preprocess KV学习bias的正式定义。转换脚本：`convert_raw_bias_to_preprocess_bias.py`。
- 新增旧BGE top-k preprocess学习bias：逐chunk保存`V_pp_bge-V_raw`，正式测试为`raw K+pp Value` rate0，隔离preprocess Value贡献。前50写586文件，旧pp缺264个raw对应项；67个testable subquestions此前均通过selected-chunk完整性检查。原pipeline+GLM已在GPU6启动。
- 语义修正：M0使用当前main example所有subquestions retrieved-doc并集的排列，属于known-document-pool/transductive上界，不是未见retrieval-set泛化；M>0还追加其他example的第一个unique doc，选择是deterministic offset而非真正随机。后续严格global bias必须逐X只用独立训练examples校准。

## 2026-07-12 Process cleanup and consolidated result

- 按用户要求停止本任务在qjy000/qjy003的pipeline/calibration/watcher/judge进程；验证两机GPU全空闲，qjy001未触碰。未删除任何结果/cache/log。
- 正式500-token+GLM（67 subquestions）：full/raw/preprocess为86.57%/74.63%/80.60%；raw+M0N16为70.15%；preprocess+raw-bias为64.18%；preprocess+source-correct-bias为65.67%；rawK+BGE-preprocessV为47.76%。
- Raw static网格最佳M1N16/M3N16均71.64%，仍低于raw 74.63%。KV L2改善没有转化为accuracy改善。
- 强结论：Value-only替换破坏K/V匹配。完整preprocess K+V 80.60%，rawK+同一preprocessV仅47.76%，后续必须联合分析/更新K和V。
- preprocess grid在停止时仅20--39/67，不能报告accuracy；结果保留。完整汇总见`CURRENT_RESULTS_SUMMARY.md`。
- 实现审计：100 chunks上`pp+bias_pp`与`raw+bias_raw`relative差mean/max仅1.45e-4/1.95e-4，排除source转换/重复加/shape错位。raw bias/base均值.315；pp bias/base均值.729、median .886、max1.049，alpha=1明显可能过更新。
- GLM correctness flips：raw static gain/loss=7/10；pp+raw-bias=1/12；pp+source-bias=3/13。退化是多数净损失。方法问题定位为Value-only K/V失配、scale未校准、M0 transductive校准，不是简单loader bug。

## 2026-07-12 Strict offline Mean+rank8 launch

- 用户方案复述后采用严格定义：每个X独立；prefix只从examples50--199随机取3 docs，不使用目标example其他docs/query/retrieval pool；8 contexts/X。
- 目标为examples0--4全部unique docs，8 shards/qjy000 GPU0--7。一次full capture同时生成K/V；Key逆转prefix-token RoPE shift到local坐标。raw/preprocess base独立计算Delta并保存。
- 输出：`results/strict_offline_random_prefix_m8k3/ex*_chunk*.pt`。下一步从这些数据学习per-X Mean+rank8，并训练非oracle prefix-KV feature coefficient predictor，然后接原pipeline rate0+GLM。
- qjy001未触碰；启动检查8 shard均存活。
- Strict offline采集完成：60 targets×8 contexts，8 shard无异常。6 train/2 heldout rank分析：raw K/V Mean解释70.54%/56.73%，rank5为74.44%/59.37%；preprocess K/V Mean解释92.86%/91.73%，rank5为93.81%/92.23%。rank residual增益很小，当前首先测试联合K+V Mean。8 contexts不足以在heldout下验证rank8，需扩M=16。
- Strict Mean K/V原pipeline sanity完成（examples0--4，7 subs）：full/raw/pp为6/7、5/7、5/7，F1 .7170/.4939/.5414；raw+MeanKV与pp+MeanKV均4/7、F1 .4248。联合K/V仍退化，说明静态random-prefix Mean context mismatch/scale过强，不只是Value-only K/V mismatch。脚本/结果：`build_strict_offline_mean_kv_bias.py`、`original_pipeline_strict_offline_mean_kv/`。
- M0/N16原pipeline完成：full/raw/pp/raw+static/pp+raw-static的sub GLM accuracy为86.57/74.63/80.60/70.15/64.18%；main为75.00/55.56/63.89/55.56/47.22%。raw static虽然Value gap更小且F1/EM略升，但GLM accuracy下降，证明L2代理失效/scale1过修正。
- preprocess static当前复用raw-defined bias，属于source不匹配消融，不能作为正确pp-v2结论；正确做法需学习`full-preprocess` bias。后续网格保留raw正式结果，并增加alpha scale sweep；pp路径需重新导出source-conditioned bias。
- 2026-07-12 用户要求停止实验：已向qjy000/qjy003上本任务启动的kv_lora采集、分析、pipeline和watcher进程发送TERM；复查两机无匹配进程，GPU0--7均空闲。未删除任何cache、模型、结果、CSV、脚本或日志；qjy001未触碰。
