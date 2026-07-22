# Offline Fixed Token Selector 实验计划

## 1. 需求理解

我们现在要验证一个新的系统假设：

```text
FusionRAG 在线阶段需要根据 query 选择一部分 doc tokens 做重算。
但前面的实验发现，不同 query 下被选中的 token 集合存在明显稳定部分。
因此，可以尝试在 offline 阶段提前为每个 example / retrieved document sequence 固定选出 0.15 比例 token。
在线阶段直接使用这个固定集合，或者只在固定集合之外补少量 query-specific tokens。
```

这个实验的关键约束是：

```text
不能提前使用该 example 的真实测试问题来定义 offline token set。
否则 offline set 会泄漏答案，导致实验结果不可用。
```

所以本实验不是简单复用前面 full-attention 真实问题得到的 anchor set，而是要设计不会泄漏真实 query 的 offline token 定义方法，并与在线 selector 做公平对比。

## 2. 核心问题

我们要回答三个问题：

```text
Q1. 不看当前真实 query，仅凭 offline 固定 token set，能不能接近 online selector 的效果？

Q2. offline 固定 token set 是否能覆盖 online selector 反复选中的稳定部分？

Q3. 如果 offline 固定 set 不够，offline fixed anchors + online query residual 是否能在更低 selector 开销下保持质量？
```

如果成立，潜在系统创新点是：

```text
Document tokens contain query-stable update anchors.
These anchors can be selected offline without observing the online query.
Online selection only needs to recover query-sensitive residual tokens.
```

## 3. 数据泄漏规则

### 禁止使用

定义某个 example 的 offline token set 时，不能使用：

```text
1. 该 example 的 main question。
2. 该 example 的 sub questions。
3. 该 example 的 answer。
4. 用该 example 真实问题跑出来的 full-attention distribution。
5. 用该 example 真实问题跑出来的 FusionRAG-QK / draft selected tokens。
```

这些都属于测试 query 信息，会造成泄漏。

### 允许使用

可以使用：

```text
1. 该 example 的 retrieved documents / chunks 本身。
2. 文档 token 的位置、边界、长度、标点、标题等结构信息。
3. 其他 example 的问题作为 calibration queries。
4. 与当前 example 无关的通用 synthetic calibration prompts。
5. 不含当前真实 query 的 draft/QK/full-attention 分布。
```

注意：

```text
如果使用其他 example 的问题作为 calibration queries，
这些问题只能用于定义当前 example 的 offline set；
不能使用当前 example 自己的问题。
```

## 4. Offline Token Set 定义方法

所有方法默认选择总 doc tokens 的 0.15 比例。

### 4.1 Random Per-Document

每个 retrieved document 内按长度比例随机选 15% tokens。

用途：

```text
空白对照。
判断 offline fixed set 的收益是否超过随机选择。
```

### 4.2 Position / Boundary Heuristic

只用文档结构定义 token：

```text
每个 document 内优先选择：
  - document 开头若干 tokens；
  - document 结尾若干 tokens；
  - 标点/换行附近 tokens；
  - Document: 标记附近 tokens；
  - 每个 chunk 的局部尾部 tokens。
```

用途：

```text
检验稳定 hotspot 是否主要来自结构/位置 bias。
```

### 4.3 Calibration-QK Stable Set

使用其他 example 的 queries 作为 calibration queries。

对当前 example 的 fixed doc sequence：

```text
1. 取 m 个 unrelated calibration queries。
2. 对每个 calibration query，运行 FusionRAG-QK selector。
3. 每个 calibration query 得到 top 15% token set。
4. 统计每个 token 被选中的频率。
5. 选择频率最高的前 15% tokens 作为 offline fixed set。
```

这个方法不使用当前 example 的真实问题，但使用了 FusionRAG-QK 的 selector 机制。

用途：

```text
验证 FusionRAG-QK 是否确实存在可 offline 收敛的 fixed set。
```

### 4.4 Calibration-Draft Stable Set

和 Calibration-QK 类似，但 selector 换成 draft model：

```text
1. 用其他 example 的 queries 作为 calibration queries。
2. draft model 对当前 example 的 doc sequence 选择 importance tokens。
3. 对 token selected frequency 排序。
4. 固定选择 top 15% tokens。
```

用途：

```text
验证 draft selector 的稳定集合能否 offline 化。
```

### 4.5 Calibration-Full-Attention Stable Set

使用其他 example 的 queries 作为 calibration queries，对当前 fixed doc sequence 跑 full attention：

```text
1. calibration query 不是当前 example 的真实 query。
2. 统计 query-to-doc attention score。
3. 对每个 calibration query 取 top 15% tokens。
4. 按 selected frequency 或 mean attention score 排序。
5. 固定选择 top 15% tokens。
```

注意：

```text
这不是可部署方法，因为 offline full attention 计算重。
但它可以作为 offline fixed set 的上界分析。
```

### 4.6 Per-Document Normalized Variants

上面 4.3-4.5 都需要做两个版本：

```text
global_top:
  在整个拼接 doc token 序列里选 top 15%。

per_doc_top:
  每个 document 内各自选 top 15%，再合并。
```

原因：

```text
之前实验发现 global_top 有明显 sequence-tail / recency bias。
per_doc_top 可以控制每个 document 的 token budget，避免最后一个文档独占 fixed set。
```

## 5. Online 对照组

需要和以下在线 selector 对比：

```text
1. online_fusionrag_qk
   当前 FusionRAG 原始在线 QK selector。

2. online_draft
   draft model 在线 selector。

3. random_per_doc
   在线随机选择同样比例 token。

4. no_update / rate=0
   不重算 token，对照下界。

5. full_update / rate=1
   全部 token 重算，对照上界。
```

## 6. Hybrid 方法

如果 offline fixed set 单独效果不够，可以继续测试：

### 6.1 Offline Fixed + QK Residual

总 rate 固定为 0.15，例如：

```text
offline fixed anchors: 0.10
online QK residual:    0.05
```

或者：

```text
offline fixed anchors: 0.075
online QK residual:    0.075
```

Residual 选择方式：

```text
在线 QK selector 排序后，跳过已经在 offline fixed set 中的 tokens，
从剩余 tokens 里补足预算。
```

### 6.2 Offline Fixed + Draft Residual

同上，但 residual 由 draft selector 产生。

用途：

```text
看能否用较小的 online selector 预算补足 query-specific 部分。
```

## 7. 评测指标

### 7.1 质量指标

第一阶段先不开 LLM judge：

```text
F1
EM
answer length
invalid / empty answer ratio
```

如果某些方法接近 online baseline，再补：

```text
GLM judge accuracy
```

### 7.2 系统指标

```text
selector latency
online update/recompute latency
end-to-end TTFT
total wall time
```

对于 offline fixed set：

```text
online selector latency 应接近 0；
但需要记录 offline set lookup / mask construction 的开销。
```

### 7.3 机制指标

```text
Jaccard(offline fixed set, online_fusionrag_qk selected set)
Jaccard(offline fixed set, online_draft selected set)
coverage of online selected set by offline fixed set
coverage of full-attention hotspot by offline fixed set
selected token position distribution
per-document selected token distribution
```

其中 coverage 定义为：

```text
coverage(A by B) = |A ∩ B| / |A|
```

比如：

```text
coverage(online_qk by offline_fixed)
  = online QK 选中的 tokens 中，有多少比例已经包含在 offline fixed set 里。
```

## 8. 第一阶段实验矩阵

先只跑 rate=0.15，避免实验过大。

| group | selector | offline definition | online query used for selection | leakage risk |
|---|---|---|---|---|
| lower bound | no_update | none | no | none |
| baseline | online_fusionrag_qk | none | yes | allowed, online method |
| baseline | online_draft | none | yes | allowed, online method |
| control | random_per_doc | doc length only | no | none |
| offline | qk_calib_global | unrelated calibration queries | no current query | no |
| offline | qk_calib_per_doc | unrelated calibration queries | no current query | no |
| offline | draft_calib_global | unrelated calibration queries | no current query | no |
| offline | draft_calib_per_doc | unrelated calibration queries | no current query | no |
| upper analysis | fullattn_calib_per_doc | unrelated calibration queries | no current query | no for current example |

建议 calibration query 数：

```text
m = 32 unrelated queries
```

第一阶段样本数：

```text
先用已有 109 或 200 个样本中的一小组 sanity，例如 20-30 个；
确认 pipeline 正确后再扩展到完整样本。
```

## 9. 预期结果解读

### 情况 A：offline fixed 接近 online selector

如果：

```text
offline qk/draft calibration set 的 F1/EM 接近 online_fusionrag_qk；
selector latency 明显降低；
coverage online selected set 较高；
```

说明：

```text
FusionRAG 的 online selection 中存在较大的 query-stable 成分，
可以 offline 化。
```

### 情况 B：offline fixed 明显低于 online selector，但 hybrid 接近

如果：

```text
offline fixed 单独不够；
offline fixed + small online residual 接近 online selector；
```

说明：

```text
token selection 可以分解为 stable anchors + query-sensitive residual。
这是更合理的系统设计方向。
```

### 情况 C：offline fixed 和 hybrid 都不好

说明：

```text
前面的 stable set 更多是 attention/selector 现象，
不一定能直接转化为 answer quality。
```

此时应回到机制解释，而不是继续强行作为系统优化。

## 10. 需要实现的代码改动

### 10.1 先构建 Rate-Independent Score/Rank Cache

这一步是后续实验的基础。不要直接只保存某个 rate 下的 selected tokens，因为后面会不断调整：

```text
rate = 0.05 / 0.10 / 0.15 / 0.30 / 0.50 / ...
selection rule = global_top / per_doc_top / frequency / mean_score / hybrid residual
selector = QK / draft / full-attention / heuristic
```

如果只保存 selected set，每换一个 rate 或规则就要重新跑 selector，代价高，也容易引入不一致。因此第一步应该保存 rate-independent 的 token score/ranking cache。

#### 10.1.1 Cache 保存粒度

对每个 example，固定它的 retrieved documents / chunks。对每个 calibration query，保存：

```text
example_id
calibration_query_id
calibration_query_source_example
calibration_query_source_sub_id
calibration_query_text
selector_type
doc_token_count
doc_boundaries
token_scores
token_rank
```

其中：

```text
token_scores:
  长度为 doc_token_count 的 float array。
  表示每个 doc token 在该 calibration query 下的 importance score。

token_rank:
  长度为 doc_token_count 的 int array。
  rank=1 表示该 query 下分数最高的 token。
```

对于不同 selector，score 含义不同：

```text
QK selector:
  保存 FusionRAG-QK importance score。

Draft selector:
  保存 draft model 给出的 token importance score / attention-derived score。

Full-attention oracle:
  保存 query-to-doc attention mass。
```

如果某个 selector 当前实现只输出 selected indices，不输出完整 score，需要优先修改它，让它输出完整 token_scores。实在拿不到完整 score，至少保存：

```text
selected_indices_at_dense_rate
```

但这只是 fallback，不作为主设计。

#### 10.1.2 Cache 文件格式

建议保存为 `.npz` + `.jsonl`：

```text
offline_score_cache/
  qk_calib_scores.npz
  qk_calib_meta.jsonl
  draft_calib_scores.npz
  draft_calib_meta.jsonl
```

`.npz` 存大数组：

```text
scores: float16/float32, shape [num_cases, doc_token_count_padded]
ranks: int32, shape [num_cases, doc_token_count_padded]
valid_mask: bool, shape [num_cases, doc_token_count_padded]
doc_boundaries: object 或单独 jsonl 保存
```

`.jsonl` 存 metadata：

```json
{
  "case_id": 0,
  "example_id": 3,
  "calibration_query_id": 17,
  "calibration_query_source_example": 42,
  "calibration_query_source_sub_id": -1,
  "selector_type": "qk",
  "doc_token_count": 1243,
  "doc_boundaries": [[0, 120], [120, 255], ...],
  "uses_current_example_question": false
}
```

必须显式记录：

```text
uses_current_example_question = false
```

这是防止数据泄漏的审计字段。

#### 10.1.3 从 Cache 派生 Offline Set

有了 score/rank cache 后，后续所有 offline fixed set 都由 cache 离线派生：

```text
global_frequency@rate:
  对每个 calibration query，根据 token_scores 全局取 top rate；
  统计 token 被选中的频率；
  最终选频率最高的 top rate。

per_doc_frequency@rate:
  对每个 calibration query，每个 document 内单独取 top rate；
  统计每个 token 的 selected frequency；
  每个 document 内再选 frequency 最高的 top rate。

mean_score@rate:
  对每个 token 在 calibration queries 上取 mean score；
  按 mean score 选 top rate。

rank_borda@rate:
  对每个 token 汇总 rank，例如取 mean rank 或 reciprocal rank；
  按汇总 rank 选 top rate。
```

这样后续要看：

```text
0.15 / 0.30 / 0.50
global vs per-doc
frequency vs mean score
QK vs draft
fixed-only vs fixed+residual
```

都不需要重新跑 selector。

#### 10.1.4 当前主方法的派生规则

主方法仍然定义为：

```text
QK-Calib-PerDoc-Frequency@0.15
```

具体从 cache 派生：

```text
1. 对每个 example，读取它的 QK calibration score cache。
2. 对每个 calibration query，在每个 document 内取 QK score top 15%。
3. 统计每个 token 被 calibration queries 选中的次数 freq(token)。
4. 每个 document 内按 freq(token) 降序选择 top 15%。
5. 如果 freq 相同，用 mean_score(token) 降序作为 tie-break。
6. 输出该 example 的 offline_fixed_set。
```

注意：

```text
这个 fixed set 的定义只依赖 calibration queries，不依赖当前 example 的真实测试 query。
```

### 10.2 Offline set 生成脚本

新增脚本：

```text
tools_build_offline_fixed_token_sets.py
```

功能：

```text
输入:
  - dataset
  - model/tokenizer paths
  - selector type: qk / draft / fullattn / random / heuristic
  - calibration query source
  - rate
  - global_top or per_doc_top

输出:
  - offline_fixed_sets.jsonl 或 npz
  - 每个 example 的 selected token indices
  - calibration query ids
  - selected frequency / score
  - doc boundary metadata
```

这个脚本不跑模型，只读取 10.1 的 score/rank cache，并按指定规则派生 fixed set。

### 10.3 Pipeline 支持 fixed selector

在现有 real pipeline 中增加 selector：

```text
selector=fixed_offline
```

加载 offline fixed set：

```text
--offline_set_path ...
```

在线阶段：

```text
不再调用 QK/draft selector；
直接使用预先保存的 token indices 做 update/recompute。
```

### 10.4 Hybrid selector

增加：

```text
selector=fixed_plus_qk_residual
selector=fixed_plus_draft_residual
```

参数：

```text
--fixed_rate
--residual_rate
--total_rate
```

## 11. 实验记录要求

所有结果写入：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/
```

计划文件：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/README_plan.md
```

每次实验都记录：

```text
1. 启动命令
2. git branch / commit
3. 数据集范围
4. rate
5. calibration query 来源
6. 是否使用当前真实问题
7. 输出文件
8. 主要结果
9. 当前结论
```

并同步追加到总日志：

```text
MOTIVATION_EXPERIMENTS/motivation_problem_scan.md
```

## 12. 当前建议的执行顺序

```text
Step 1:
  先实现 rate-independent score/rank cache。
  保存每个 calibration query 对每个 doc token 的完整 score/rank。

Step 2:
  从 cache 离线派生不同 rate / 不同规则的 offline fixed set。
  只比较 offline set 和 online selector set 的 Jaccard / coverage。

Step 3:
  确认没有泄漏且 overlap 有信号后，跑小规模 answer pipeline。

Step 4:
  如果 fixed offline 有希望，再跑 full sample。

Step 5:
  再做 hybrid selector。
```

这样可以避免一开始就改大 pipeline，先把核心假设验证清楚。

## 13. Phase 2：真实生成流程验证

### 13.1 实验目标

把 phase1b 得到的 chunk-local fixed set 接入真实 `load_kv_and_generate()` 路径，验证它是否能替代在线 FusionRAG-QK selector。

对照组：

```text
online_fusionrag_qk：
  在线使用当前 query 计算 importance，并选 rate=0.15 token update。

offline_chunk_qk：
  离线为每个 chunk 保存 selected_local_indices；
  在线只做 local-to-global index mapping，不计算 selector。

random_chunk：
  每个 chunk 随机固定选 rate=0.15 token，作为空白对照。
```

### 13.2 启动方式

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
CUDA_VISIBLE_DEVICES=0 TMPDIR=/raid/home/hming/tmp \
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
tools_offline_fixed_selector_phase2_generation.py \
  --model-path /mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct \
  --num-examples 20 \
  --rate 0.15 \
  --max-new-tokens 50
```

### 13.3 输出

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/
phase2_generation_rate015/answer_detail.csv

MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/
phase2_generation_rate015/summary.csv

MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/
phase2_generation_rate015/summary.json

MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/
phase2_generation_rate015/README.md
```

### 13.4 结果

| selector | n | ROUGE-1 | EM | TTFT(s) | KV load(s) | selection(s) | update+query(s) | selected doc tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| online_fusionrag_qk | 20 | 0.2434 | 0.2000 | 0.4116 | 0.0599 | 0.1139 | 0.2378 | 1103.2 |
| offline_chunk_qk | 20 | 0.2492 | 0.2000 | 0.2706 | 0.0556 | 0.0000 | 0.2150 | 1098.6 |
| random_chunk | 20 | 0.2259 | 0.2000 | 0.2710 | 0.0559 | 0.0000 | 0.2151 | 1098.6 |

### 13.5 当前判断

```text
1. offline_chunk_qk 可以直接去掉在线 selector：
   selection 0.1139s -> 0.0000s。

2. rate=0.15 时，TTFT 从 online_fusionrag_qk 的 0.4116s
   降到 offline_chunk_qk 的 0.2706s，下降约 34.3%。

3. 20 个样本上 offline_chunk_qk 的 ROUGE-1/EM 没有低于 online QK，
   但样本量太小，不能作为最终 accuracy 结论。

4. random_chunk 的速度接近 offline_chunk_qk，
   但 ROUGE-1 更低，说明收益不是单纯来自“少做 selection”，
   fixed set 仍需要 calibration/QK 信号。

5. 下一步需要做 rate sweep：
   rate=0.05/0.10/0.15/0.30/0.50，
   对比 online_fusionrag_qk、offline_chunk_qk、random_chunk。
```

## 14. Phase 2b：selector 来源与 set 构造方法横向对比

### 14.1 为什么先不做 rate sweep

直接扫 rate 会混淆两个问题：

```text
1. offline fixed set 的来源是否重要？
   QK / draft / position / random

2. fixed set 的构造方式是否重要？
   frequency stable set / mean-score stable set
```

因此先固定 `rate=0.15`，横向比较不同方法。

### 14.2 Fixed set 构造

脚本：

```text
tools_offline_fixed_selector_phase1d_set_methods.py
```

输出：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/
phase1d_selector_set_methods_rate015/
```

方法：

```text
qk_frequency_per_chunk
qk_mean_score_per_chunk
draft_frequency_per_chunk
draft_mean_score_per_chunk
position_tail_per_chunk
position_boundary_per_chunk
random_per_chunk
```

held-out online QK selected set 作为参照时：

| method | Jaccard | coverage |
|---|---:|---:|
| qk_frequency_per_chunk | 0.6960 | 0.8152 |
| qk_mean_score_per_chunk | 0.6910 | 0.8120 |
| draft_frequency_per_chunk | 0.4367 | 0.6032 |
| draft_mean_score_per_chunk | 0.4306 | 0.5973 |
| position_boundary_per_chunk | 0.2224 | 0.3578 |
| random_per_chunk | 0.0794 | 0.1463 |
| position_tail_per_chunk | 0.0703 | 0.1283 |

### 14.3 真实生成结果

输出：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/
phase2_selector_methods_rate015/
```

| selector | ROUGE-1 | EM | TTFT(s) | selection(s) |
|---|---:|---:|---:|---:|
| online_fusionrag_qk | 0.2434 | 0.2000 | 0.4113 | 0.1136 |
| offline_qk_freq | 0.2492 | 0.2000 | 0.2720 | 0.0000 |
| offline_qk_mean | 0.2493 | 0.2000 | 0.2717 | 0.0000 |
| offline_draft_freq | 0.2492 | 0.2000 | 0.2716 | 0.0000 |
| offline_draft_mean | 0.2459 | 0.2000 | 0.2722 | 0.0000 |
| position_boundary | 0.1870 | 0.1500 | 0.2715 | 0.0000 |
| position_tail | 0.2419 | 0.2000 | 0.2727 | 0.0000 |
| random_chunk | 0.2259 | 0.2000 | 0.2723 | 0.0000 |

### 14.4 当前判断

```text
1. QK-frequency 与 QK-mean 都是强候选，二者差距很小。

2. draft-based fixed set 与 online QK 的集合重叠更低，
   但 20 样本生成指标没有明显掉分，需要扩大样本数确认。

3. position_boundary 明显不可靠。

4. position_tail 生成指标尚可，但 set overlap 极低，
   说明 tail bias 不能作为主要解释。

5. random_chunk 低于 QK/draft 方法，说明不能随机固定 token。

下一步优先扩大样本数，而不是立刻扫 rate：
  online_fusionrag_qk
  offline_qk_freq
  offline_qk_mean
  offline_draft_freq
  random_chunk
```
