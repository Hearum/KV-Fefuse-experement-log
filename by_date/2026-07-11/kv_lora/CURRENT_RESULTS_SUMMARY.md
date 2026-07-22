# KV-LoRA / Static Bias 当前结果汇总（2026-07-12）

## 运行状态

已停止本任务在qjy000和qjy003启动的pipeline、calibration、watcher与judge进程。两台机器GPU均为空闲；qjy001未触碰。结果文件、cache、日志均保留。

## 原FusionRAG Pipeline正式结果

设置：Qwen3-32B，前50 main examples（36个testable main / 67个subquestions），正式RAG文档流程，500-token decode，GLM-5.2 judge。除full外均为rate=0。

| 方法 | Main GLM accuracy | Sub GLM accuracy | F1 |
|---|---:|---:|---:|
| full rate1 | 27/36 = 75.00% | 58/67 = **86.57%** | .6080 |
| raw K + raw V | 20/36 = 55.56% | 50/67 = **74.63%** | .4439 |
| preprocess K + preprocess V | 23/36 = 63.89% | 54/67 = **80.60%** | .5604 |
| raw K/V + M0N16 raw-source V bias | 20/36 = 55.56% | 47/67 = 70.15% | .4616 |
| preprocess K/V + M0N16 raw-source V bias | 17/36 = 47.22% | 43/67 = 64.18% | .4025 |
| preprocess K/V + M0N16 preprocess-source V bias | 19/36 = 52.78% | 44/67 = 65.67% | .4368 |
| raw K + BGE-preprocess V | 9/36 = 25.00% | 32/67 = 47.76% | .3719 |

`preprocess-source bias = mean(full-preprocess)`通过`bias_raw + V_raw - V_preprocess`严格换算。它比把raw-source bias直接加到preprocess略好，但仍显著差于不加bias的preprocess baseline。

## Raw底座M/N正式网格（完整67/67）

这些配置使用`raw K/V + raw-source static V bias`，rate0，原pipeline+GLM。

| 配置 | Main GLM accuracy | Sub GLM accuracy | F1 |
|---|---:|---:|---:|
| M0N2 | 50.00% | 65.67% | .4388 |
| M0N4 | 55.56% | 67.16% | .4659 |
| M0N8 | 52.78% | 68.66% | .4482 |
| M0N16 | 55.56% | 70.15% | .4616 |
| M1N16 | 52.78% | **71.64%** | **.4761** |
| M3N2 | 50.00% | 68.66% | .4531 |
| M3N16 | 50.00% | **71.64%** | .4702 |
| M5N16 | 50.00% | 70.15% | .4638 |
| M10N16 | 44.44% | 65.67% | .4524 |

趋势：N增大通常改善，但不是严格单调；M过大退化。最佳static配置M1N16/M3N16仍低于raw baseline 74.63%，更低于preprocess 80.60%。

## KV指标与生成指标的矛盾

Value relative L2网格中，raw gap约.416，M0N16降至.283；但GLM accuracy从raw 74.63%降至70.15%。因此“更接近full Value”不保证答案更好。可能原因包括Key未同步更新、K/V不匹配、均值bias破坏少数关键token/head，以及L2受大量非关键维度主导。

BGE消融更强地证明K/V耦合：完整preprocess K+V达到80.60%，但raw K+preprocess V仅47.76%。旧preprocess的收益不能归因于Value单独替换，必须保留匹配的Key或联合结构。

## 未完成/部分结果

qjy003被停止时，preprocess+raw-bias网格仅完成20--39/67条，不报告accuracy；preprocess+preprocess-source bias网格尚未进入完整评测。对应CSV/日志保留在`results/original_pipeline_bias_grid/`，可断点续跑。

早期32-token离线judge中M3N2曾得到17/67=25.37%，与full相同；该结果受短decode影响，已被上述500-token正式结果取代，不应作为主结论。

## 当前研究判断

1. 静态Value mean template在表示层面有效，但作为独立Value Adapter尚不能提升正式accuracy。
2. 下一步不应继续只优化Value L2或扩大M/N；优先分析匹配的RoPE-aligned Key bias，以及联合K/V更新。
3. 需要按layer/head/token重要性评估bias，而非全tensor均匀相加；可测试selector支撑集上的静态K/V模板。
4. M0使用当前main example全部retrieved-doc并集，属于known-document-pool/transductive上界，不是全局未见retrieval-set泛化。

## Static bias实现审计

针对“是否操作错误”做了文件级和结果级检查：

- loader在raw/preprocess Value加载后只加一次bias；rate0不再修改document KV；未发现重复写入、token shape错位或preprocess后覆盖。
- 对100个chunks验证`V_pp+bias_pp`与`V_raw+bias_raw`，relative L2差mean/max为`1.45e-4/1.95e-4`，仅FP16换算误差；source-specific公式和加载实现正确。
- raw bias/base norm比mean=.315；preprocess bias/base mean=.729、median=.886、max=1.049。preprocess-source更新幅度过大，alpha=1未经heldout调参，属于方法设计缺陷。
- GLM翻转：raw->raw+M0N16有7个wrong->correct、10个correct->wrong；preprocess->preprocess+raw-bias为1 gain/12 loss；preprocess->preprocess+source-bias为3 gain/13 loss。退化来自系统性over-update，不是少数judge噪声。

因此没有证据表明代码把bias加错；但实验手法有三项明确问题：Value-only导致K/V失配、固定scale=1过强、M0校准使用测试main-example文档池。后续必须先做heldout alpha/层头门控和联合K/V，再讨论static adapter有效性。

## 关键路径

- 正式baseline/M0N16：`results/original_pipeline_m0n16/`
- Raw M/N网格：`results/original_pipeline_bias_grid/`
- BGE Value消融：`results/original_pipeline_bge_value_bias/`
- GLM短decode sanity：`results/real_static_bias_50/glm_judge/`
- Bias文件：`results/static_bias_m0n16_50*`、`results/static_bias_grid_50/`

## Strict Offline Random-Prefix最新结果

目标examples0--4共60个unique documents；每个X用8个独立calibration contexts，每个prefix含3个来自examples50--199的随机docs。训练前6个context、heldout后2个；K为local-RoPE-aligned。由于只有6个训练样本，去均值后最大rank5，本轮不能严格报告rank8。

| source/object | Mean explained energy | Oracle rank4 | Oracle rank5 | Mean remaining Delta |
|---|---:|---:|---:|---:|
| raw K | 70.54% | 74.15% | 74.44% | .543 |
| raw V | 56.73% | 59.11% | 59.37% | .658 |
| preprocess K | **92.86%** | 93.74% | 93.81% | .267 |
| preprocess V | **91.73%** | 92.19% | 92.23% | .288 |

rank增益很小：raw K/V从Mean到rank5仅+3.90/+2.64个百分点；preprocess仅+0.95/+0.50。说明独立随机prefix下Delta主要是source/document-specific固定Mean，context coefficient信号弱。下一步优先把source-specific Mean K+V联合接入原pipeline；同时把contexts扩到16以验证rank8，但只有rank8相对Mean产生稳定增益时才训练coefficient predictor。

### Strict Offline Mean K/V 原pipeline结果

使用前6个external-prefix contexts的Mean K和Mean V，raw/preprocess source独立；Key在local-RoPE坐标加bias后再做online offset。目标examples0--4，共5 main/7 subquestions，500-token+GLM。

| 方法 | Sub GLM | F1 | Main all-correct |
|---|---:|---:|---:|
| full rate1 | 6/7 = 85.71% | .7170 | 4/5 |
| raw rate0 | 5/7 = 71.43% | .4939 | 3/5 |
| preprocess rate0 | 5/7 = 71.43% | .5414 | 3/5 |
| raw + strict Mean K/V | 4/7 = 57.14% | .4248 | 2/5 |
| preprocess + strict Mean K/V | 4/7 = 57.14% | .4248 | 2/5 |

两种source更新后输出/指标相同，符合它们都近似指向同一calibration mean full-KV。联合K/V仍退化，排除“只因Value-only K/V不匹配”这一单一解释；更根本问题是随机外部prefix下的静态Mean与正式RAG context不匹配，以及scale=1过更新。样本仅7个subquestions，只作sanity否定信号，需alpha sweep后才决定是否扩大。
