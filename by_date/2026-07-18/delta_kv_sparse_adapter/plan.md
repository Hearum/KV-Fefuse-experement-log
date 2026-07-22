# Delta-KV Sparse Adapter 实验计划

## 目标
验证 sparse attention 只作为近似重算路径，生成 `K_sparse/V_sparse`，然后对原缓存 KV 做比例插值，而不是删除 attention token。

## 定义
- `K_cache/V_cache`: raw 或 preprocess offline cache。
- `K_sparse/V_sparse`: sparse recompute 路径得到的目标 document token KV。
- `Delta_K = K_sparse - K_cache`，`Delta_V = V_sparse - V_cache`。
- `K_adapter = K_cache + alpha_K * Delta_K`。
- `V_adapter = V_cache + alpha_V * Delta_V`。

## 端点验收
1. alpha=0 必须等价于原 rate=0 pipeline。
2. alpha=1 必须等价于写入 K_sparse/V_sparse 的 pipeline。
3. 先做 2 个样本，保存逐层/逐 token KV 最大误差，再扩展到 20 个样本。
4. 只有端点通过，才运行 200 条 MuSiQue-v2。

## 对比
raw 和 preprocess 分开；`alpha_K/alpha_V` 先同时取 `0, 0.25, 0.5, 0.75, 1.0`，后续再做 K/V 分开扫描。指标为 EM、F1、去除 think 后 GLM judge，以及 KV 插值误差。

## 重要区别
上一轮 `sparse_block_*` 是直接限制最终 attention 的可见 token，没有生成 Delta-KV，因此结果不能用于本实验。

## 运行记录
- 创建日期：2026-07-18
- 工作树：`hming/add-pca-analysis`
- shared cache：`/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2`

## Sanity 结果

- commit：`0bacfef`（alpha 独立 result root），adapter 实现 commit：`58324ac`。
- MuSiQue-v2 example 1--2，raw/preprocess，alpha=0/1 四个端点均完成，无 shape/OOM 错误。
- 四个 CSV 已按 alpha 分开保存于 `results/sanity2/`；每组 2 条 Question。
- 小样本四组去除 think 后的答案没有观察到 alpha=0 与 alpha=1 的答案差异；这只说明该 2 样本 sanity 未暴露差异，不代表 KV 数值相同。
- 初次 sanity 中 raw alpha=1 漏传参数、raw alpha=0 产生混合 CSV，均已移到 `results/sanity2/invalid_partial/`，不纳入结果。


## Sweep20 结果（2026-07-19）

### 启动与复现
- 代码 commits：58324ac（插值 adapter）、0bacfef（alpha 独立结果目录）、9b1ebe1（sanity 记录）。
- 命令模板：python MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py --method sparse_delta_adapter_preprocess|sparse_delta_adapter_raw --dataset musique-v2 --start 0 --end 20 --rate 0.15 --adapter-alpha {0,0.25,0.5,0.75,1.0} --block-size 64 --topk-blocks 8 --head-chunk-size 2 --query-chunk-size 512
- raw/preprocess 共 10 组，每组 20 条；结果目录：results/sweep20/。
- 使用共享 cache：/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2。

### 完整性
| load KV | alpha | 有效样本 |
|---|---:|---:|
| preprocess | 0, 0.25, 0.5, 0.75, 1.0 | 20/20 每组 |
| raw | 0, 0.25, 0.5, 0.75, 1.0 | 20/20 每组 |

### 当前自动指标（仅作方向性检查）
| load KV | alpha=0 | alpha=0.25 | alpha=0.5 | alpha=0.75 | alpha=1 |
|---|---:|---:|---:|---:|---:|
| preprocess：EM | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| raw：EM | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| preprocess：粗略去 think F1 | 约 1.03% | 约 1.03% | 约 1.03% | 约 1.03% | 约 1.03% |
| raw：粗略去 think F1 | 约 2.75% | 约 2.75% | 约 2.75% | 约 2.75% | 约 2.75% |

这里的 EM/F1 不能作为最终性能：生成 CSV 中包含 think 内容，且部分输出没有规范闭合标签。因此本表只用于确认各组已跑完和观察 alpha 是否造成明显输出变化；正式结论必须使用统一的 think 清理和 GLM judge。

### 阶段性观察
1. 20 样本上五个 alpha 的粗略答案指标完全没有拉开，说明目前不能据此证明插值比例有效或无效。
2. raw 的粗略结果高于 preprocess，但样本很少且指标解析不可靠，不能解释为 raw 更好。
3. alpha=0/1 端点已经完成执行层面的 sanity，没有 shape/OOM 错误；但仍需用 KV 数值对比确认 alpha=0 是否逐元素等价 cache、alpha=1 是否等价 sparse writeback。
4. 下一步应先对这 200 条输出做统一 GLM judge，再进行 200 条完整数据集；若 alpha 仍无差异，再检查 adapter 是否只作用于目标 token，以及 cache index 对齐是否正确。

## Post-blend full result (2026-07-19)

运行方式：使用已有 FUSIONRAG_REPROCESS_KV_BLEND_BETA post-recompute blend，beta=1-alpha；同时设置 FUSIONRAG_STRICT_REPROCESS_ABLATION=1 和 FUSIONRAG_CLEAN_STRICT_ABLATION=1。MuSiQue-v2 共 200 样本，分成 4 个 50 样本 segment，使用共享 preprocess cache。

GLM judge command：
SETUP_V2_REJUDGE_METHODS=postblend_alpha0,postblend_alpha1,postblend_alpha2,postblend_alpha3,postblend_alpha4 SETUP_V2_REJUDGE_DATASETS=musique-v2 SETUP_V2_REJUDGE_RATES=0.15 SETUP_V2_REJUDGE_OUT_DIR=.../results/glm_postblend3 GLM_REJUDGE_WORKERS=30 python rejudge_setup_v2_glm_clean.py

正式 GLM 结果：
| alpha | beta | rows | GLM correct | GLM Acc |
| 0.00 | 1.00 | 200 | 67 | 33.5% |
| 0.25 | 0.75 | 200 | 68 | 34.0% |
| 0.50 | 0.50 | 200 | 74 | 37.0% |
| 0.75 | 0.25 | 200 | 72 | 36.0% |
| 1.00 | 0.00 | 200 | 72 | 36.0% |

统一 think 清理后的自动 EM/F1（与历史 setup-v2 EM/F1 清洗实现可能不同，仅作辅助）：
| alpha | EM | F1 |
| 0.00 | 20.50% | 30.52% |
| 0.25 | 22.50% | 30.85% |
| 0.50 | 23.50% | 32.19% |
| 0.75 | 22.50% | 31.27% |
| 1.00 | 21.50% | 30.86% |

覆盖检查：每个 alpha 的 4 个新 segment 合计 200 条、问题唯一。旧 seg_0_200 文件未计入。GLM 输出：results/glm_postblend3/rejudged_summary.csv。

结论：post-blend 严格路径的 alpha=0 已通过 2 样本端点验证，与 rate=0 输出一致；完整 200 样本上 alpha=0.5 最好（GLM 37.0%），高于 alpha=0 的 33.5%，但仍低于历史 full attention 41.5% 和 native online DraftModel 41.0% 的 MuSiQue-v2 参考结果。因此该方案能恢复部分 KV gap，但当前不能替代完整重算；alpha=0.5 是后续优化和 profiling 的候选点。

## K/V split sanity (2026-07-19)

固定 beta=0.5、strict post-blend，分别只混合 Key 和只混合 Value，MuSiQue-v2 前 2 样本。
- Key-only：Maria Bello -> Maggie Gyllenhaal；Manhattan Project -> Manhattan Project。
- Value-only：Maria Bello -> Maggie Gyllenhaal；Manhattan Project -> James Chadwick was a participant of the Cavendish Laboratory at the University of Cambridge.
- 两组均输出正常文本，无 think 截断或 OOM。
- 结果目录：results/kv_split_sanity/key 和 results/kv_split_sanity/value。
- 该 sanity 只用于确认 K/V 分开接口可运行，不作为完整性能结论。


## K/V split full experiment (2026-07-19)

### 目的
在严格的 post-recompute KV blend 下，固定 beta=0.5（重算结果权重 alpha=0.5），分别只更新 Key 或只更新 Value，比较哪一部分解释联合 K/V 更新的收益。两组都是对原始缓存做重算混合，不是直接覆盖缓存。

### 启动与数据
Key-only 在 qjy000，Value-only 在 qjy001；每组 4 个 50 样本分段，共 200 个 MuSiQue-v2 样本。共享 setup-v2 preprocess cache。环境变量为 FUSIONRAG_REPROCESS_KV_BLEND_BETA=0.5、FUSIONRAG_REPROCESS_KV_BLEND_MODE=key 或 value、FUSIONRAG_STRICT_REPROCESS_ABLATION=1、FUSIONRAG_CLEAN_STRICT_ABLATION=1。启动脚本为 MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py，结果目录为 results/kv_split_full/key 和 results/kv_split_full/value。每组有效 200/200，问题唯一。

### GLM judge
使用 MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/rejudge_setup_v2_glm_clean.py，对两个结果目录统一评测。命令使用 SETUP_V2_REJUDGE_METHODS=postblend_key05,postblend_value05、SETUP_V2_REJUDGE_DATASETS=musique-v2、SETUP_V2_REJUDGE_RATES=0.15、GLM_REJUDGE_WORKERS=30。输出为 results/glm_kv_split/rejudged_rows.csv 和 results/glm_kv_split/rejudged_summary.csv。

### 完整结果
| 方法 | 更新部分 | 样本 | GLM 正确 | GLM Acc |
|---|---|---:|---:|---:|
| post-blend alpha=0.5 | Key-only | 200 | 75 | 37.5% |
| post-blend alpha=0.5 | Value-only | 200 | 73 | 36.5% |
| post-blend alpha=0.5 | Key+Value | 200 | 74 | 37.0% |

### 解释
在本实验误差范围内，Key-only 与 Value-only 都能恢复约同等量级的性能；联合更新没有超过单独更新。因此当前数据不支持 Value 单独主导收益，也不支持 K 和 V 必须联合才有效。更准确的结论是：在 alpha=0.5、rate=0.15 下，K/V 各自都能改变模型行为，但收益很小且没有出现可加性。与联合 alpha=0.5 的 37.0% 基本一致，说明之前 alpha sweep 的峰值不是由某一个分支单独造成的。

对照：strict post-blend 联合 alpha=0 的 rate=0 为 33.5%，联合 alpha=0.5 为 37.0%，历史 full attention rate=1 为 41.5%。K/V split 仍明显低于 full recompute，不能作为完整重算的替代方案。

实验提交：结果和文档已写入工作树，提交哈希见 git log。


## K/V paired analysis (2026-07-19)

对 glm_kv_split/rejudged_rows.csv 中的 200 个相同问题按 Question 对齐 Key-only 和 Value-only，并与 glm_postblend3 中的联合 alpha=0.5 对齐。

| Key-only | Value-only | 联合 K+V alpha=0.5 | 样本数 |
|---|---|---|---:|
| 错 | 错 | 错 | 118 |
| 对 | 对 | 对 | 66 |
| 对 | 错 | 对 | 5 |
| 错 | 对 | 对 | 3 |
| 对 | 错 | 错 | 4 |
| 错 | 对 | 错 | 4 |

Key-only 独有正确 9 个，Value-only 独有正确 7 个；联合方法的 74 个正确样本为 66 个共同正确加上 Key-only 独有中的 5 个和 Value-only 独有中的 3 个。没有出现 Key-only 与 Value-only 都错误、而联合 K+V 正确的样本。

### 新结论
当前 alpha=0.5、rate=0.15 的完整数据上，没有观察到 K/V 的正协同增益。联合更新的收益主要来自不同分支各自覆盖的一部分样本，而不是两个更新共同修复了单独分支无法修复的问题。下一步若继续优化，应优先研究 token/layer/head 级别的选择或独立 alpha，而不是简单增大联合 adapter 的容量；同时需要检查这是否只在 MuSiQue-v2 的固定 alpha 上成立。


## Layer-selective smoke (2026-07-19)

### 目的
在 beta=0.5、严格 post-blend、rate=0.15 下，检查联合 K/V 更新是否可以只保留后部 Transformer 层。该实验只做 20 个 MuSiQue-v2 样本，作为是否扩大完整数据集的 sanity check。

### 启动方式
使用 setup_standard_v2 的 run_setup_v2_task.py、method=online_qk、dataset=musique-v2、start=0、end=20、rate=0.15、共享 setup-v2 preprocess cache。三组环境分别为：

- all_layers：FUSIONRAG_REPROCESS_KV_BLEND_BETA=0.5，严格模式，写回全部层。
- last16：额外设置 FUSIONRAG_REPROCESS_KEEP_KEY_LAYERS=first16 和 FUSIONRAG_REPROCESS_KEEP_VALUE_LAYERS=first16，表示只保留后 16 层的重算写回。
- last8：额外设置 FUSIONRAG_REPROCESS_KEEP_KEY_LAYERS=first24 和 FUSIONRAG_REPROCESS_KEEP_VALUE_LAYERS=first24，表示只保留后 8 层的重算写回。

结果目录：results/layer_selective_smoke/{all_layers,last16,last8}。GLM judge 使用同一份 rejudge_setup_v2_glm_clean.py，结果为 results/glm_layer_smoke/rejudged_summary.csv。

### 结果
| 方法 | 样本 | GLM 正确 | GLM Acc |
|---|---:|---:|---:|
| 全层 K+V | 20 | 6 | 30.0% |
| 仅最后 16 层 K+V | 20 | 6 | 30.0% |
| 仅最后 8 层 K+V | 20 | 6 | 30.0% |

### 解释
20 个样本上三组完全相同，说明该 smoke 没有发现后层截断造成退化，也没有证明后层足以替代全层。由于样本量太小，不能据此报告完整性能；下一步只有在扩大到至少 50 或 200 样本后，才能判断后层是否真能保留联合 adapter 的收益。当前结果支持继续做后层 full-data 验证。


## 重复实验说明（2026-07-19）

本目录后续追加的 layer-selective smoke 不是新的研究主线。回查后确认，部分 K/V 写回和 layer selective 方向已有历史实验：

- by_date/2026-06-29/kv_update_rate_sweep_nojudge：Qwen2.5-7B MuSiQue，完整 rate sweep，已有 kv、v_only、k_only 的端到端结果。
- by_date/2026-07-12/structured_kv_adapter：已有 layer update audit、parallel layer probe、K-first/V-only 相关分析。
- by_date/2026-06-26/recompute_v_only_ablation：已有 V-only 消融。

因此 results/layer_selective_smoke 只作为 Qwen3-32B MuSiQue-v2、post-blend alpha=0.5 的重复 sanity，不扩大到完整数据集，也不作为新的主结论。后续主线回到尚未充分回答的问题：在严格的统一 pipeline 下，K/V split 的样本级差异、layer/head/token 级选择，以及可预测的结构化 adapter。
