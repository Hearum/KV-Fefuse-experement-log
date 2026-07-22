# KVCOMM-style Anchor Adapter

## 当前结论

本实验研究能否将 FusionRAG document KV 的 Transformer recompute 替换为：

    cached KV + 由当前前序文档匹配得到的 anchor Delta-KV

完整实验设计见 PLAN.md。Phase A 已完成 5 target × 10 held-out contexts 的严格测试。结论是 position-matched anchor 稳定优于 mean/random，但对 Value 的增益太小，没有达到进入大规模端到端实现的预设门槛。

## 已知基线

已有 5 target × 50 context 实验表明：per-document mean template 在 held-out context 上可解释 raw/preprocess Value Delta 能量约 69.6%/65.9%，但旧的静态 K/V mean 在原 pipeline 7 个 subquestion sanity 中从 raw/preprocess 的 5/7 降到 4/7。KV-L2 可恢复不等于生成质量提升，端到端门槛不可省略。

旧 Value-only M0/N16 也将 raw Value gap 从 .416 降至 .283，但 67 个 subquestion 的 GLM accuracy 从 74.63% 降到 70.15%。因此本实验以 GLM accuracy 和 TTFT 作为最终可用性判断，不以 KV L2 单独定结论。

## 数据与路径

- 模型：/mnt/qjhs-sh-lab-01/models/Qwen3-32B
- 严格 Delta 数据：MOTIVATION_EXPERIMENTS/kv_lora/results/perdoc_context_deltas/strict_t{1..5}
- reflect cache：/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/musique
- setup-v2 cache：/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2
- 计划端到端数据：musique-v2，随后 2wikimqa-v2

## Phase A 结果

严格切分包含 5 个固定 target，每个 target 使用 40 个训练 prefix、10 个测试 prefix；训练和测试的 prefix document pool 完全不相交。以下数字先跨 50 个 held-out case 累加平方范数，再计算比例。position top8 的权重只使用 prefix token 数和 document 数；oracle anchor 使用真实测试 Delta 选训练 anchor，只是不可部署上界。

| cache/object | original gap | no-update recovery | mean recovery/final | position top8 recovery/final | oracle recovery/final |
|---|---:|---:|---:|---:|---:|
| raw RoPE-aligned K | .242 | 1.000 | .677/.164 | **.623/.151** | .578/.140 |
| raw V | .372 | 1.000 | .758/.282 | **.735/.274** | .720/.268 |
| preprocess RoPE-aligned K | .252 | 1.000 | .642/.162 | **.592/.149** | .548/.138 |
| preprocess V | .475 | 1.000 | .604/.287 | **.586/.278** | .573/.272 |

主要观察：

1. position top8 在 5/5 target、raw/preprocess、K/V 上都优于 mean，说明 prefix 长度/文档数确实携带可预测信息。
2. 改善幅度有限。相对 mean recovery error，K 平均改善约 8.1%，V 约 3.2%，没有达到计划中至少 10% 的进入门槛。
3. oracle 对 mean 的平均改善也只有 K 15.8%、V 5.9%。Value 是主要性能瓶颈，但即使允许偷看测试 Delta，现有 40-anchor pool 的选择空间也很小。
4. cached-prefix Value matcher 没有优于 position；高维 cached Value mean 不是可靠的 prefix 语义距离。当前最有效 matcher 反而只是两个位置特征。
5. random anchor 始终明显差于 mean/top8，说明任意单个 context offset 不可直接迁移。

因此 Phase A 的结论不是“anchor 完全无效”，而是：**context matching 能稳定改善 K/V L2，但现有 full-rank anchor pool 对关键的 Value 只有很小余量，不足以支持直接进入大规模 dynamic-anchor 端到端实现。**

结果文件：

- results/strict_anchor_transfer/per_case_metrics.csv
- results/strict_anchor_transfer/summary.csv
- results/strict_anchor_transfer/summary.json
- figures/strict_anchor_recovery.png

## 端到端结果

由于 Phase A 未达到预设门槛，本轮没有修改主 pipeline，也没有把 dynamic top8 接入 setup-v2。当前最接近的联合 K/V anchor 端到端对照，是已有 strict offline mean K/V 在原 MuSiQue reflect pipeline 的 7-subquestion sanity：

| 方法 | GLM correct | F1 |
|---|---:|---:|
| full rate1 | 6/7 | .7170 |
| raw rate0 | 5/7 | .4939 |
| preprocess rate0 | 5/7 | .5414 |
| raw + strict mean K/V rate0 | 4/7 | .4248 |
| preprocess + strict mean K/V rate0 | 4/7 | .4248 |

另一条 67-subquestion Value-only 对照中，M0/N16 将 Value L2 gap 从 .416 降到 .283，但 GLM accuracy 从 raw rate0 的 74.63% 降至 70.15%。两组端到端结果共同说明：改善全局 KV L2 并不保证生成质量改善，错误 layer/head/token 的偏置会破坏注意力与 K/V 匹配。

严格地说，dynamic position-top8 的端到端效果尚未测量，不能把 static mean 结果冒充为 dynamic anchor 结果。但 Phase A 显示 dynamic matcher 相对 mean 只改善 Value recovery 约 3%，而 static mean 已出现端到端退化，因此直接投入完整 200-example 实现的收益/风险比很低。下一次若继续，应先做 20-example position top8 + alpha/layer-head gate + rate0.05 residual recompute，不应先跑纯 top8 rate0 的完整数据集。

## 可行性判断

当前版本不构成可用的纯 Adapter 替代方案：

1. **质量证据不足。** Value 的可部署 anchor 增益小，旧静态端到端结果为负。
2. **存储过大。** Qwen3-32B BF16 完整 K+V Delta 每 token 约 256 KiB；100-token document 的一个 full-rank anchor 约 25 MiB，8 anchors 约 200 MiB，接近重复保存多份 KV。
3. **在线 I/O 不轻。** top8 若读取完整 offset，需要读取并混合 8 份与 KV 同尺寸 tensor；虽然不跑 Transformer，但可能转化为严重的存储带宽瓶颈。
4. **低秩不能直接补救。** 既有严格实验中 Value rank8 只解释约 62%--64% Delta energy，且还是 oracle coefficient；压缩后可部署效果只会更低。

仍值得保留的部分是 per-document mean/position template + shared layer/head router + 少量 residual recompute。Anchor 更适合做 calibration/template，不适合按 KVCOMM 原样保存多个 full-rank offset。推荐下一阶段把 position top8 蒸馏为每文档少量 position-conditioned template，并只作用于已验证敏感的 layer/head，再以 rate=0.05 重算兜底。

## 文件

- PLAN.md：完整研究设计与决策门槛。
- EXPERIMENT_LOG.md：命令、commit、运行状态、异常与结果。
- scripts/analyze_strict_anchor_transfer.py：严格 held-out anchor 迁移分析。
- scripts/plot_anchor_transfer.py：结果图。
- results/：CSV/JSON。
- figures/：图表。


## setup-v2 端到端 smoke（commit 96c6f59）

为了不只停留在 KV-L2，本轮把 KVCOMM-style anchor 思路接入了 setup-standard v2 runner。实现方式是：离线对同一个 example 的 document 顺序做 3 个排列，full forward 观测每个 document 在不同 prefix position 下的 Delta-KV，然后对每个 chunk 用 `[1, prefix_token_ratio, doc_rank_ratio]` 做逐元素 ridge，输出原始顺序下的静态 `key_bias/value_bias`。在线阶段不重算，直接在加载 preprocess KV 后加 bias。

新增接口：

- `ktransformers/unified_process_cache.py` 透传 `static_key_bias_path/static_value_bias_path`。
- `setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py` 新增 `position_adapter_raw_rate0` / `position_adapter_preprocess_rate0`，并修正外部 `CUDA_VISIBLE_DEVICES` 不应被 runner 覆盖的问题。
- 构建脚本：`scripts/build_setup_v2_position_biases.py`。

本轮只做两个 MuSiQue-v2 sanity，不扩到 20/200，因为结果已显示当前形式不可用：

| example | 方法 | 结果 | 观察 |
|---:|---|---|---|
| 38 | full_rate1 historical | `TML Entertainment`，正确 | full 上界可答对 |
| 38 | online_draft rate0.15 historical | `TML Entertainment`，正确 | 正常 DraftModel 可恢复 |
| 38 | online_qk rate0.15 historical | `Metalworks Institute`，错 | QK selector 未恢复 |
| 38 | preprocess_rate0 current | `Metalworks Institute`，错 | cache-only baseline 错 |
| 38 | position adapter K+V scale=1 | 重复乱码，错 | 全量 K/V bias 破坏生成 |
| 38 | position adapter V-only scale=0.1/0.25 | `Metalworks Institute`，错 | 小 scale 基本退回 rate0，不能弥补 gap |
| 38 | position adapter K-only scale=0.1 | `Metalworks Institute`，错 | Key 小 scale 也不能恢复 |
| 69 | full_rate1/current preprocess_rate0 | `Sire Records`，错 | 该样本 full 本身错，只能作为稳定性检查 |
| 69 | position adapter K+V scale=1 | 乱码，错 | 再次说明全量 bias 过强 |
| 69 | position adapter V/K-only small scale | `Sire Records`，错 | 小 scale 无有效改变 |

结果文件：

- `results/setup_v2_position_smoke/end2end_smoke_summary.csv`
- `results/setup_v2_position_smoke/end2end_smoke_summary.json`
- 详细日志：`results/setup_v2_position_smoke/logs/`
- adapter 资产：`/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/position_ridge_smoke_preprocess_e38_a3` 和 `...e69_a3`

当前结论：**KVCOMM-style anchor offset 的思想在机制上相关，但不能按“全层全头 full-rank Delta bias 直接加回 KV cache”的形式落地到 FusionRAG。** 主要原因是：

1. full-scale K/V bias 会破坏生成分布，说明预测 Delta 的局部 L2 接近不等于注意力/解码稳定。
2. 小 scale 或只改 K/V 的保守写回不会破坏，但也不能把 rate0 的错误答案拉回 full。
3. setup-v2 MuSiQue 单样本有 20--24 个 document chunk，3-anchor 构建就需要十分钟级，离线多排列 anchor 的成本不可忽略。
4. 真正可继续的方向不是扩大当前 full-rank bias，而是先做 layer/head/token gate 或只对 selector token / late Value 层做校准，并且必须和 rate0.05/0.15 residual recompute 结合验证。

因此，本轮不建议把当前 adapter 扩到完整数据集。若继续，应转向：从 full-correct 且 rate0-wrong 的样本集合中，学习/选择少量稳定 layer/head 的 Value template，再用 online DraftModel 0.05 residual 兜底；否则很容易得到“KV L2 看似改善但答案质量下降”的假阳性。


### Oracle 写回排查

用户指出 position adapter 产生乱码可能是实现问题。为区分实现 bug 与近似方法失败，本轮补做了 exact oracle sanity：对 example 38 的原始 document 顺序直接计算 `full KV - preprocess KV`，保存为静态 K/V bias，再通过同一个 `position_adapter_preprocess_rate0` loader 加回。

结果：oracle K+V scale=1 输出 `TML Entertainment`，与 full_rate1 一致，恢复了 preprocess_rate0 的错误答案 `Metalworks Institute`。

这说明：

1. 静态 bias loader 的基本 shape、chunk 命名、K/V 写回通路是可用的。
2. Key 的本轮 RoPE-local 对齐至少在 oracle setting 下足以恢复 full 输出。
3. 之前 position-ridge K+V scale=1 乱码，不是因为“任何 KV bias 加回都会乱码”，而是因为预测出来的 Delta 方向/幅度不可靠。
4. 当前 negative conclusion 应更精确表述为：**我实现的 position-ridge anchor 近似器不可用；KVCOMM-style exact offset 写回本身在 oracle 下可行。**

与 KVCOMM 原文仍有明显差距：原文是 anchor pool + online matching/offset correction，本轮 position-ridge 只用了 prefix token ratio 和 doc rank 两个位置特征，没有用语义相似 anchor，也没有做 layer/head/token gate 或误差约束。因此不能据此否定 KVCOMM 思路，只能否定当前这个粗糙近似版本。


## KVCOMM-like anchor matching 复核

用户指出前一版 position-ridge 并没有真正参考 KVCOMM。尝试 clone 官方仓库时 GitHub TLS 多次中断，未能在本轮逐行复刻源码；但根据论文/README 的核心思想，本轮实现了更接近 KVCOMM 的最小版本：

- 构建 anchor pool：对同一个 example 的 documents 做 6 个顺序排列，每个 anchor 都保存真实 `full KV - preprocess KV` offset。
- Online matching：对当前 chunk 使用 `prefix length + prefix Value summary` 计算与 anchor 的相似度。
- Offset approximation：选择 top3 anchors，用 softmax 权重加权真实 anchor Delta，输出静态 K/V bias。
- 写回方式仍使用同一个 `static_key_bias_path/static_value_bias_path` loader，因此和 oracle / position-ridge 可直接比较。

脚本：`scripts/build_setup_v2_kvcomm_like_biases.py`。

关键结果仍看 MuSiQue-v2 example 38，因为它满足：full_rate1 正确、preprocess_rate0 错、online_draft rate0.15 正确。

| 方法 | rate | 输出 | 结论 |
|---|---:|---|---|
| full_rate1 historical | 1.0 | `TML Entertainment` | 正确上界 |
| preprocess_rate0 current | 0.0 | `Metalworks Institute` | cache-only 错 |
| online_qk historical | 0.15 | `Metalworks Institute` | QK baseline 错 |
| online_draft historical | 0.15 | `TML Entertainment` | Draft baseline 正确 |
| position-ridge K+V scale=1 | 0.0 | repeated gibberish | 过度简化版本失败 |
| oracle exact K+V scale=1 | 0.0 | `TML Entertainment` | 写回链路正确 |
| KVCOMM-like hybrid a6 top3 K+V scale=1 | 0.0 | `TML Entertainment` | anchor top-k offset 可恢复该样本 |
| KVCOMM-like V-only scale=0.25/0.5 | 0.0 | `Metalworks Institute` | 单独 Value/小 scale 不够 |
| KVCOMM-like K+V scale=0.25 | 0.0 | `Metalworks Institute` | offset 幅度不足 |

结果文件：

- `results/kvcomm_like_e38/summary.csv`
- `results/kvcomm_like_e38/summary.json`
- adapter 资产：`/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/kvcomm_like_hybrid_e38_a6_top3`

修正后的结论：

1. 前一版 position-ridge 乱码不能代表 KVCOMM，它缺少 anchor matching 和真实 offset 加权。
2. KVCOMM-like top-k anchor offset 在 e38 这个 full-correct/rate0-wrong 样本上可以恢复 full 答案，说明这条路线有继续价值。
3. 当前 positive 仍然是 single-case、in-example anchor pool。它还没有证明跨 example / 跨 query / 跨 document 泛化，也没有解决 anchor pool 构建成本。
4. 6 anchors 对 24-doc MuSiQue example 构建约二十多分钟，说明如果每个 example 都靠 full forward 生成 anchor，离线成本很高。后续必须研究 anchor reuse、压缩和只在高价值 token/layer/head 上存 offset。
5. 从效果看，full K+V scale=1 是必要的；V-only 或小 scale 没有恢复答案。这说明该样本需要较完整的 attention-routing + value-content 更新，不能只靠轻微 Value calibration。

下一步最应该做的不是扩大 position-ridge，而是扩大 KVCOMM-like：选 10--20 个 full-correct/rate0-wrong 样本，验证 in-example anchor pool 的恢复率；再做跨 example anchor pool，判断是否能从“需要每个样本 full anchor”变成可复用的 preprocess/adaptor。


## KVCOMM-like targeted gap cases 扩展

为了从 single-case 继续推进，本轮从 MuSiQue-v2 historical outputs 中筛选：`full_rate1` exact-correct 且 `preprocess_rate0` wrong 的样本。200 条中共有 9 个候选，按 context length 选最短 4 个做 KVCOMM-like in-example anchor 测试：38、43、24、54。

设置：

- example 38：沿用 `a6 top3`。
- example 43/24/54：使用 `a4 top2`，即 4 个 document-order anchors，top2 matching。
- matcher：`hybrid = prefix length + prefix Value summary`。
- 写回：full K+V scale=1，rate=0，不做 Transformer recompute。
- 对比：historical full_rate1、preprocess_rate0、online_qk rate0.15、online_draft rate0.15。

结果：

| example | full | preprocess rate0 | online_qk 0.15 | online_draft 0.15 | KVCOMM-like | 结论 |
|---:|---|---|---|---|---|---|
| 38 | TML Entertainment | Metalworks Institute | Metalworks Institute | TML Entertainment | TML Entertainment | 恢复 |
| 43 | Claudia Wells | wrong long answer | think/incomplete | think/incomplete | think/incomplete | 失败 |
| 24 | Francisco Guterres | Jose Ramos-Horta | Jose Ramos-Horta | Francisco Guterres | Francisco Guterres | 恢复 |
| 54 | John D. Loudermilk | Lee Alexander | John D. Loudermilk | John D. Loudermilk | John D. Loudermilk | 恢复 |

Targeted 4-case 统计：

- full_rate1：4/4。
- preprocess_rate0：0/4。
- online_qk rate0.15：1/4。
- online_draft rate0.15：3/4。
- KVCOMM-like in-example anchor：3/4。

结果文件：

- `results/kvcomm_like_candidate_examples.csv`
- `results/kvcomm_like_batch/batch_gap_case_summary.csv`
- `results/kvcomm_like_batch/batch_gap_case_summary.json`
- 详细 endpoint 输出：`results/kvcomm_like_batch/e*_a4_top2_fullkv_scale1/`
- 构建和运行日志：`results/kvcomm_like_batch/logs/`

解释：

1. KVCOMM-like anchor matching 已经不是单个偶然样本：在 4 个 targeted gap cases 中恢复 3 个，和 online_draft rate0.15 持平，明显强于 preprocess_rate0 和 online_qk rate0.15。
2. 这个结果说明 Delta-KV offset 的确包含可复用结构；真实 anchor offset 加权比 position-ridge 强很多。
3. 失败样本 e43 同时也是 online_qk/online_draft 失败样本，输出停留在 reasoning trace，可能不是单纯 KV gap，而是 answer extraction / stop / prompt-following 问题。
4. 仍不能宣称可部署，因为本轮 anchors 是 in-example 通过额外 full forward 构造。若每个 online example 都要先 full forward 生成 anchors，就没有节省计算。真正下一步必须验证跨 example / 跨 query / 跨 document 的 anchor reuse。
5. 成本方面：e43/e24/e54 的 a4 top2 构建均为十几到三十分钟级，doc chunks 19--25 个；e38 a6 top3 也需要二十多分钟级。未来必须减少 anchor 数、只存高价值 layer/head/token、或者从离线 corpus 预构建共享 anchor。

最终阶段判断：

- **不可行的版本**：position-ridge、只用 doc rank/prefix length 的线性外推。
- **已验证可行的机制**：exact oracle KV bias 写回；KVCOMM-like top-k anchor offset 在 targeted gap cases 上有 3/4 恢复。
- **尚未解决的关键问题**：anchor 如何离线复用、如何跨样本泛化、如何降低存储/构建成本、如何选择 layer/head/token gate。

因此，下一阶段推荐路线是：固定 9 个 MuSiQue-v2 gap candidates 全部跑完 in-example KVCOMM-like，随后做 leave-one-example-out / cross-example anchor pool。如果 cross-example anchor 仍有效，才值得进入完整数据集和速度评估；如果只能 in-example 有效，则它更像 oracle calibration，不能作为 FusionRAG preprocess v2。


## Offline Prefix Bank 可部署性检查

用户指出 in-example anchor 预先知道当前会召回哪些 doc chunk，因此不能作为真正 online 可部署方案。为此本轮补做更严格的 offline-prefix-bank sanity：

- target 仍是 MuSiQue-v2 example 38 的每个 doc chunk。
- anchors 不再来自 example 38 自己的 doc 顺序排列。
- anchors 的 prefix 来自其它 examples（1--10，排除 target）的 docs，每个 anchor prefix 取 3 个 docs。
- 对每个 target chunk，构造 `system + offline_prefix_docs + target_chunk`，观测 `full KV - preprocess KV`。
- online matching 仍用当前真实 prefix summary 去选 offline anchors，写回 full K+V scale=1。

结果：

| setting | output | correct |
|---|---|---:|
| full_rate1 | TML Entertainment | 1 |
| preprocess_rate0 | Metalworks Institute | 0 |
| in-example KVCOMM-like | TML Entertainment | 1 |
| offline-prefix-bank a3/top2/pdocs3 | Equity Music Group | 0 |

解释：

1. 当前可部署版 offline prefix bank 没能恢复 e38，说明 in-example 3/4 positive 很大程度依赖“同一组目标 docs 的不同排列 anchors”。
2. 只用其它 examples 的随机/固定 prefix docs 构造 anchor，还不能提供足够准确的 offset。
3. 这并不否定 KVCOMM-like 思路，但说明 anchor bank 的来源是关键变量：必须让 offline anchors 覆盖目标 chunk 真实会遇到的 prefix distribution。
4. 未来可部署路线应改为 per-chunk anchor bank，prefix 不是随机 examples，而应来自该 chunk 的历史召回邻居、BGE top-k 相关 chunk、同 document 邻近 chunk、或训练集 retrieval trace。还需要 confidence gate；匹配低时 fallback 到 DraftModel recompute。

最终可行性判断更新：

- in-example anchor：机制正信号强，targeted 4 cases 恢复 3/4，但不可部署。
- offline random-prefix anchor：e38 失败，当前不可用。
- 下一步只有在“离线 anchor bank 的 prefix 分布接近真实 RAG prefix”时才值得继续扩大。

## BGE-neighbor Prefix Bank 可部署性检查

为了进一步回答“如果 online 前不知道当前 example 会召回哪些 doc chunk，该怎么做”，本轮把 offline prefix bank 的来源从随机其它 examples 改成全局 BGE 相似 chunk：

- 对每个目标 chunk，使用 `prepare_data` 已生成的全局 FAISS / BGE `context_rank`。
- 每个目标 chunk 排除自己后，取 BGE top-neighbor chunks，按每 3 个 neighbor 组成一个 anchor prefix。
- 仍然构造 `system + bge_neighbor_prefix + target_chunk`，观测 `full KV - preprocess KV`。
- online 写回仍是 full K+V scale=1，rate=0，不做 Transformer recompute。

结果：

| setting | output | correct |
|---|---|---:|
| full_rate1 | TML Entertainment | 1 |
| preprocess_rate0 | Metalworks Institute | 0 |
| in-example KVCOMM-like | TML Entertainment | 1 |
| other-example offline-prefix-bank | Equity Music Group | 0 |
| BGE-neighbor prefix-bank a3/top2/pdocs3 | Metalworks Institute | 0 |

解释：

1. BGE-neighbor 版本比 random other-example 更接近可部署，因为 anchor prefix 来自离线可获得的相似 chunk，而不是当前 example 的真实 doc 排列。
2. 但它仍没有恢复 e38，输出退回到 preprocess_rate0 的 `Metalworks Institute`。
3. 这说明当前 KVCOMM-like 正信号主要来自 in-example anchors 覆盖了真实 context-conditioned Delta；只用语义相似 chunk 的离线 prefix 还不能学到足够准确的 KV offset。
4. 下一步如果继续这条线，应优先做 retrieval-trace bank：用训练集真实 RAG 检索轨迹中“某 chunk 经常出现在哪些 prefix 后面”的上下文分布来构造 anchor，而不是仅用 BGE 相似度。
5. 当前可部署结论仍为 negative：naive offline prefix bank 和 BGE-neighbor prefix bank 都不能直接替代 recompute。

## BGE-neighbor Prefix Bank 9-case 扩展

用户指出 e38 单样本不能判断方法不可行，因此本轮把 BGE-neighbor prefix bank 扩展到全部 9 个 MuSiQue-v2 targeted gap candidates。这 9 个样本满足：`full_rate1` 正确，而 `preprocess_rate0` 至少在严格答案上失败。

设置保持一致：

- source：preprocess KV。
- prefix source：每个 target chunk 的全局 BGE top-neighbor chunks，排除 target 自己。
- num anchors：3。
- top-k anchors：2。
- prefix docs per anchor：3。
- matcher：hybrid = prefix length similarity + prefix Value summary cosine。
- 写回：full K+V bias scale=1，rate=0，不做 online Transformer recompute。

结果表：

| example | gold | BGE-neighbor output | exact | contains |
|---:|---|---|---:|---:|
| 38 | TML Entertainment | Metalworks Institute | 0 | 0 |
| 43 | Claudia Wells | Rebecca Sweitzer | 0 | 0 |
| 24 | Francisco Guterres | José Ramos-Horta | 0 | 0 |
| 54 | John D. Loudermilk | John D. Loudermilk | 1 | 1 |
| 14 | Jennifer Connelly | LaTanya Richardson | 0 | 0 |
| 68 | Claudia Wells | Claudia Wells | 1 | 1 |
| 66 | ancient Egyptian religion | Egyptian | 0 | 0 |
| 13 | Tom Hood | Frances Freeling Broderip's sibling is Tom Hood. | 0 | 1 |
| 63 | Matthew Lawrence | Matthew Lawrence | 1 | 1 |

9-case 统计，使用 contains 作为宽松实体命中口径：

| method | contains correct |
|---|---:|
| preprocess_rate0 | 1/9 |
| online_qk rate0.15 | 4/9 |
| online_draft rate0.15 | 7/9 |
| BGE-neighbor prefix bank | 4/9 |

严格 exact 口径下，BGE-neighbor prefix bank 为 3/9。

解释更新：

1. e38 单样本失败不能说明方法不可行；9-case 后可以更准确地说：BGE-neighbor prefix bank 有弱正信号，明显好于 preprocess_rate0，但不足以替代 online_draft。
2. BGE-neighbor 的 4/9 contains 与 online_qk rate0.15 接近，但低于 online_draft 的 7/9，也低于 in-example KVCOMM-like 在前 4 个 gap cases 上的 3/4。
3. 成功样本 e54/e68/e63 都是 BGE-neighbor 输出直接命中实体；e13 输出句子包含正确实体但不是严格 exact。
4. 失败样本 e38/e24/e14 仍基本回到错误实体；e43 从原来的 reasoning/incomplete 变成 `Rebecca Sweitzer`，说明 bias 确实改变了生成轨迹，但不是正确方向。
5. 当前最合理结论不是“BGE-neighbor 不可行”，而是“naive BGE-neighbor anchor bank 不够强”。它能恢复一部分 gap，但没有覆盖 full recompute 所需的真实 prefix-conditioned Delta。
6. 下一步需要比较 anchor 来源：BGE neighbor、同 example doc permutation、retrieval-trace prefix、same-document neighboring chunks。若 retrieval-trace bank 能接近 in-example anchor，才有机会形成可部署 KV adaptor。

结果文件：

- `results/offline_prefix_bank/bge_neighbor_9case_summary.csv`
- `results/offline_prefix_bank/bge_neighbor_9case_summary.json`
- 每个样本 endpoint 输出：`results/offline_prefix_bank/e*_bge_neighbor_a3_top2_pdocs3_fullkv_scale1/`
- 构建和运行日志：`results/kvcomm_like_batch/logs/*bge_neighbor*`
## 2026-07-16：Mixed Anchor Pool 扩池计划

目标：验证更多离线上下文变化能否提高 Delta 的覆盖能力。对每个目标文档 `d_i` 构建 12 组 anchor：3 组来自全局 BGE 近邻，9 组来自数据集其他 example 的随机文档；每组包含 3 个前缀文档。online 仅使用已缓存 preprocess Value 的均值摘要进行匹配，从 12 组中选择 top-4，加权合成 `Delta K/Delta V`，不运行主模型重算。

分阶段判据：

1. 先跑现有 9 个 `full_rate1` 正确、`preprocess_rate0` 错误的 targeted gap cases。
2. 与旧 BGE-only 3-anchor（3/9 exact、4/9 contains）、online QK（4/9 contains）和 online Draft（7/9 contains）比较。
3. 若 mixed pool 明显超过旧 BGE-only，至少达到 5/9 contains，才扩大到 MuSiQue-v2 全 200 条；否则先分析 anchor 命中和权重，不直接消耗完整数据集构建成本。

固定配置：`prefix_source=mixed, bge_anchors=3, random_anchors=9, prefix_docs=3, topk_anchors=4, matcher=hybrid, temperature=0.07, random_seed=20260716`。

### 扩池结果

| 方法 | targeted exact | targeted contains |
|---|---:|---:|
| preprocess rate=0 | - | 1/9 |
| BGE-only，3 anchors，top-2 | 3/9 | 4/9 |
| mixed，3 BGE + 9 random，top-4 | 4/9 | 4/9 |
| online QK，rate=0.15 | - | 4/9 |
| online Draft，rate=0.15 | - | 7/9 |

Mixed pool 恢复了 e38，并保留 e54/e68/e63；但 e13 从 BGE-only 的 contains-correct 回退为错误，因此 contains 总数没有增加。195 个有 metadata 的 chunk 中，random anchor 占 top-4 槽位的 48.46%，平均获得 40.50% 融合权重，说明随机组并非没有被选择，而是其 Delta 插值不稳定。

结论：增加随机前缀覆盖有弱正信号，但当前 matcher 无法可靠判断哪些随机 Delta 可以迁移。结果未达到预注册的 5/9 contains 门槛，因此没有启动 MuSiQue-v2 全 200 条。下一步应先学习或校准 Delta compatibility/confidence，而不是继续机械增加随机池容量。

结果文件：`results/mixed_anchor_pool/mixed_anchor_9case_summary.csv`、`results/mixed_anchor_pool/mixed_anchor_9case_summary.json`。

## 2026-07-16：Anchor Pool Oracle Compatibility 上界计划

在训练 compatibility gate 前，先验证 mixed pool 是否包含足够接近真实 Delta 的候选。分析专用 oracle 对每个目标 chunk 计算真实 `Delta = full(current prefix + d_i) - preprocess(d_i)`，按 Value Delta relative L2 从 12 个 anchor 中选择 top-1，并将该 Delta 写回 endpoint。

该实验不代表可部署方法，因为选择阶段使用真实 Delta；它只回答候选池覆盖上界。若 oracle endpoint 仍不能明显超过 mixed matcher 的 4/9，则无需训练 gate，应继续改变 anchor 构造；若达到至少 7/9，说明主要瓶颈是 compatibility ranking，可以继续训练只读取 cached prefix statistics 的轻量 gate。

### Oracle top-1 结果

Oracle Value-L2 top-1 得到 4/9 exact、4/9 contains，没有超过 mixed matcher。正确样本为 e54、e14、e68、e13；e38 和 e63 在 oracle top-1 下回退。

在保存了完整 candidate metadata 的 192/200 个 chunk 上：最近候选的 Value relative L2 均值为 0.712、中位数为 0.786；普通 matcher top-1 候选均值为 0.809；matcher score 与负 oracle error 的平均 Spearman 相关仅 0.347。Oracle 最近候选中 BGE 131 次、random 61 次。

含义：matcher 确实不够准确，但更主要的问题是单个候选 Delta 覆盖不足。即使使用真实 Delta 排名，选到的最好候选通常仍保留约 71% 的相对 L2 误差。下一步先验证 12 个 Delta 的线性 span 上界，而不是直接训练 gate。

结果：`results/oracle_compatibility/oracle_value_top1_9case_summary.{csv,json}`。

## 2026-07-17：12-anchor Oracle Linear Span

不再从 12 个 Delta 中只选一个，而是 offline 解 joint K/V ridge coefficient：`Delta_hat = sum_j c_j Delta_j`。这是分析上界，系数直接使用真实 Delta 求得，不能作为 online 方法。

四例 stage-1 endpoint 为 2/4：e38、e54 正确，e43、e24 错误，没有达到扩展剩余 5 例的 3/4 门槛。90 个 chunk 的平均 Value/Key relative L2 为 0.599/0.581，中位数为 0.652/0.607；平均 coefficient L2 为 0.523，最大单系数绝对值为 0.696，未出现系数爆炸。

Linear span 比最近单 anchor 明显降低 L2，但仍保留约 60% Delta residual，且没有转化为 endpoint 提升。下一步做同 chunk 的 32-anchor scaling probe，区分“basis 数量不足”和“随机前缀方向本身不覆盖真实上下文 Delta”。

结果：`results/oracle_span/oracle_kv_lstsq_stage1_summary.{csv,json}`。

## 2026-07-17：Anchor Count Scaling 12 -> 32

在 e38/e43/e24/e54 各取 5 个固定 chunk，用同一 seed 将 mixed pool 从 3 BGE + 9 random 扩到 3 BGE + 29 random。20 个同源 chunk 上，Value relative L2 从 0.5860 降至 0.5763，仅改善 1.66%；Key 从 0.5693 降至 0.5478，仅改善 3.78%。32-anchor coefficient L2 均值 0.5115，最大绝对系数 0.5928。

结论：随机 anchor 数从 9 增至 29 后，span 覆盖基本饱和，未达到约 0.35 的预设 residual 门槛，因此不跑 32-anchor endpoint。下一步不再机械增加随机池，而是使用训练/RAG 数据里真实出现的 preceding-document sequence 构造 retrieval-trace anchors。

结果：`results/oracle_span_a32_probe/oracle_span_a12_vs_a32_probe.{csv,json}`。

## 2026-07-17：Retrieval-trace Anchor（last 3 docs）

对每个目标 chunk，从其他 example 中找到 BGE 相似 document occurrence，并把该 occurrence 之前紧邻的 3 个真实 RAG 文档作为 anchor prefix。相同 20 chunks 上，trace A12 的 Value/Key relative L2 为 0.5784/0.5570，相比 mixed A12 的 0.5860/0.5693 仅改善 1.30%/2.16%，与 mixed A32 接近。

分样本看，trace 对 e43/e24 明显更好，但对 e38/e54更差。这说明真实共现前缀有价值，但统一截断为 3 docs 破坏了长前缀位置/上下文条件。下一轮保留 occurrence 的全部 preceding documents，验证完整 retrieval trace。

结果：`results/oracle_span_trace_probe/oracle_span_anchor_source_probe.{csv,json}`。

## 2026-07-17：Retrieval-trace Full-prefix Probe 与 e24 Endpoint

将 retrieval-trace anchor 从“只保留 occurrence 前 3 个文档”改为保留该 occurrence 的全部 preceding documents；仍严格排除目标 example。相同 20 个 probe chunks 上：

| anchor pool | Value relative L2 | Key relative L2 |
|---|---:|---:|
| mixed A12 | 0.5860 | 0.5693 |
| mixed A32 | 0.5763 | 0.5478 |
| retrieval trace A12，last 3 docs | 0.5784 | 0.5570 |
| retrieval trace A12，full prefix | **0.5510** | **0.5038** |

Full-prefix trace 相比 mixed A12 将 Value/Key residual 分别降低 5.98%/11.52%，说明真实 prefix 长度和位置是 Delta span 的重要条件。分样本结果仍不稳定：e24 为 0.243/0.218，但 e38/e54 仍为 0.671/0.612 和 0.744/0.686。

由于 e24 的 probe residual 首次低于约 0.35，补齐其全部 22 个 document chunks 并做严格 endpoint。端到端仍输出 `Jose Ramos-Horta`，gold 为 `Francisco Guterres`，EM/ROUGE 均为 0。完整 22 个 chunks 上，Value relative L2 mean/median 为 0.266/0.224，Key 为 0.217/0.165；即使整体 residual 已显著下降，最终答案仍未恢复。

e24 的 Commission 线索位于 chunk 4/5，答案实体文档位于 chunk 10，Ambelau 线索位于 chunk 20/21。对应 chunk 4、10、20 的 Value/Key residual 分别为 0.256/0.236、0.243/0.179、0.438/0.389。chunk 10 同时包含 Francisco Guterres 与 Jose Ramos-Horta，说明该样本对局部表示/路由误差高度敏感。当前结论是：全局 K/V relative L2 不是充分的 endpoint 指标，后续需分析关键 chunk 的按层和按 token 误差，而不是直接扩大完整数据集。

对 chunks 2/7/12/17/20/21 的按层诊断显示，Value 目标能量平均 96.17% 位于 48-63 层，Value 误差能量 97.17% 也位于该段；Key 对应为 49.57% 和 57.81%。但高误差 chunk 的四个层段会同步升高，例如 chunk 21 的 Value 分段 relative L2 为 0.803/0.858/0.771/0.837，而低误差 chunk 17 为 0.090/0.058/0.099/0.141。因此失败不是一个固定层段的孤立异常，更像是 chunk 级 anchor span 覆盖不足，以及“所有层和 K/V 共用一组 12 维系数”约束过强。下一步先测每层独立 12 维 coefficient 的 oracle 上界；只有它显著降低 residual 并恢复 endpoint，才值得研究 coefficient predictor。

### Layer-wise coefficient 上界

在 chunks 2/4/7/10/12/17/20/21 上，将 shared 12 维 coefficient 扩成每层独立的 `64 x 12` coefficient。Value residual 均值仅从 0.3539 降到 0.3510，平均相对改善 0.75%；Key 从 0.2966 降到 0.2908，改善 1.87%。关键 chunk 10 的 Value/Key 仅从 0.2431/0.1792 变成 0.2417/0.1759，chunk 20 从 0.4380/0.3888 变成 0.4349/0.3805。

结论：增加 64 倍 coefficient 自由度没有解决问题，shared coefficient 不是主要瓶颈，当前 12 个 retrieval-trace Delta basis 本身没有覆盖目标 Delta。layer-wise 版本不补完整 22 chunks，也不做 endpoint。下一步保持 shared coefficient，做 retrieval-trace A12 -> A32 paired scaling；若仍接近饱和，则停止这类 anchor span 路线，不训练 coefficient predictor。

结果：`results/oracle_span_trace_layerwise_probe/layerwise_vs_shared_summary.csv`；实现提交 `067b102`。

### Retrieval-trace A12 -> A32 scaling

保持 shared joint K/V coefficient，在同 8 个 chunks 上将 full-prefix retrieval-trace anchors 从 12 扩到 32。Value residual mean 从 0.3539 降到 0.3315，按 chunk 平均相对改善 6.32%；Key 从 0.2966 降到 0.2715，改善 8.36%。32 个 prefix sequence 在每个 chunk 中均唯一，平均来自 10.6 个 source examples，因此改善有限不能归因于简单重复池。

收益高度依赖位置：chunk 2 的 V/K 改善 20.97%/22.47%，chunk 4 改善 14.93%/18.04%，但答案 chunk 10 仅改善 1.73%/3.11%，chunk 7/12/17 也接近饱和；Ambelau chunk 20 为 7.29%/15.36%。A32 coefficient L2 均值 0.594，最大单系数绝对值 0.790，没有系数爆炸。

这说明更多真实 trace directions 对部分 chunk 有用，但不能稳定扩大所有 document Delta 的 span。补齐 e24 全部 22 chunks 后，Value mean/median 从 A12 的 0.2664/0.2236 降到 A32 的 0.2547/0.2122；Key 从 0.2170/0.1648 降到 0.2023/0.1618。按 chunk 平均相对改善只有 3.76%/5.66%。严格 endpoint 仍输出 `Jose Ramos-Horta`，gold `Francisco Guterres`，EM/ROUGE 为 0。

因此 A32 未通过 endpoint gate：不扩 MuSiQue-v2 200 条，不训练 coefficient predictor，也不继续增加同类 anchor 数。这组 oracle 已经使用真实目标 Delta 求 coefficient，online predictor 只可能更差。下一阶段若继续替代 recompute，adapter 必须读取当前真实 preceding KV/hidden summary 并生成 anchor span 之外的新 Delta 方向，而不能只在离线静态 Delta bank 内选取或线性组合。

结果：`results/oracle_span_trace_a32_probe/a12_vs_a32_summary.csv`、`results/oracle_span_trace_a32_e24/a12_vs_a32_full22_summary.csv`；launcher 提交 `552b784`，完整 e24/endpoint launcher 提交 `e25a917`。

结果目录：`results/oracle_span_trace_fullprefix_probe/`、`results/oracle_span_trace_fullprefix_e24/`。构建/endpoint 提交为 `a120daf`，按层诊断提交为 `069e6bd`。
