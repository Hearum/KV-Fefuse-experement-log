
## 2026-07-09 235B online_qk 正确性排查记录

### 背景
用户指出 Qwen3-235B 下 `online_qk_rate015` 效果明显偏低，怀疑实现不等价。暂停后续 grid/cross-dataset 实验，先检查实现正确性。

### 已做检查 1：force-all cache/recompute 等价性
- 命令核心环境：`FUSIONRAG_FORCE_ALL_REPROCESS=1`
- 模型：`Qwen3-235B-A22B`
- 数据：MuSiQue 前 3 个 main question
- 路径：`MOTIVATION_EXPERIMENTS/qwen3_235b_equivalence_debug/force_all_cache_path_reuse_online_qk_cache/full_0_3/`
- cache：复用 `/home/hming/fusionrag-qwen3-235b-three-groups-unified-prompt-cache/online_qk_rate015`
- 含义：仍走 FusionRAG cache/recompute 路径，但强制选择全部 doc token 重算，用来检验 cache/recompute 是否能复现 full attention 路径。

结果：force-all 与历史 `full_rate1` 前 3 条 correctness 完全一致，答案也高度一致：
- idx0：full 正确，force-all 正确，online_qk 正确。
- idx1：full 错误，force-all 错误，online_qk 错误。
- idx2：full 错误，force-all 错误，online_qk 正确。

初步结论：235B 的普通 full path 和强制全量重算的 cache/recompute path 没有表现出明显不等价；`online_qk_rate015` 低分暂时不能归因于全局 cache/mask/recompute 主路径完全坏掉。

### 已做检查 2：模型实现差异
- 32B 路径：`model_type=qwen3` 动态导入 `/mnt/qjhs-sh-lab-01/wjh/FusionRAG/ktransformers/models/modeling_qwen3.py`。
- 235B 路径：`model_type=qwen3_moe` 使用本仓库新增 `ktransformers/models/modeling_qwen3_moe.py`。
- 235B 配置：`attn_impl=eager`, `num_hidden_layers=94`, `num_attention_heads=64`, `num_key_value_heads=4`。
- 32B 配置：`attn_impl=eager`, `num_hidden_layers=64`, `num_attention_heads=64`, `num_key_value_heads=8`。

### 当前判断
235B 的低分更可能来自 QK selector / importance 分数，而不是 full cache/recompute 主路径。下一步应验证：
1. `FUSION_SKIP_CHUNK=1` 默认会跳过 system 和第一个 doc chunk；需要测试 `FUSION_SKIP_CHUNK=0` 是否显著改变 online_qk。
2. 保存 235B online_qk 的 selected token trace，检查选中的 token 是否集中在合理文档/位置。
3. 对比 235B online_qk 选择集合与 full-attention oracle / draft selector 的重合度，判断是 selector 定义问题还是 MoE adapter 的 importance 计算问题。

### 已做检查 3：`FUSION_SKIP_CHUNK` 与 selected token trace

目的：验证 235B `online_qk_rate015` 低分是否由默认 `FUSION_SKIP_CHUNK=1` 导致。该默认值会在 QK importance 计算中跳过 `context_id <= 1`，即跳过 system 和第一个 doc chunk。

实验设置：
- skip=0：`FUSION_SKIP_CHUNK=0`，路径 `MOTIVATION_EXPERIMENTS/qwen3_235b_selector_debug/online_qk_skip0/full_0_20/`。
- skip=1：默认设置，路径 `MOTIVATION_EXPERIMENTS/qwen3_235b_selector_debug/online_qk_skip1_trace/full_0_20/`。
- 两组均设置 `FUSIONRAG_SAVE_SELECTED_DIR` 保存 selected token trace。
- 为节省 235B 资源，只保留前几条 trace 后停止。

第一条样本结果：
- 默认 skip=1：输出 `57 is part of the United Kingdom's National Cycle Network.`，判定正确。
- skip=0：输出 `2017, the 16,575 mile network was used for over 786 million trips.`，判定错误。

第一条样本的 chunk-level selected token 分布：

| setting | passage_idx=1/chunk1 | passage_idx=2/chunk2 | passage_idx=5/chunk5 | passage_idx=8/chunk8 | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| skip=1 | 0/50 = 0.0% | 23/92 = 25.0% | 19/52 = 36.5% | 8/301 = 2.7% | 默认跳过第一个 doc chunk，但答案正确 |
| skip=0 | 20/50 = 40.0% | 21/92 = 22.8% | 18/52 = 34.6% | 8/301 = 2.7% | 纳入第一个 doc chunk 后，答案反而错误 |

额外发现：
- trace 中 `chunk_ids` 为 `[0, 1, 2, ..., 10, 1, 2]`，不是解析错误。
- 原因是 preprocess 阶段为每个 doc 融合 top-k 相似文档 KV，而回答阶段又拼入当前 sub-question 的 retrieved docs，因此实际上下文是“preprocess 后的 doc KV + 当前 query/retrieved docs”的混合结构。

阶段结论：
- `FUSION_SKIP_CHUNK=1` 不是 235B 低分的直接 bug。简单改成 skip=0 会改变 selected token 分布，并且在第一条样本上使结果变差。
- 当前更可疑的是 QK importance 的打分定义与“preprocess doc KV + 当前 retrieved docs”这种上下文构造之间存在偏差；也可能是 235B MoE 的 attention/score 分布和 32B dense 模型不同，导致同一套 QK selector 不再稳定。
- 后续不应继续基于 235B `online_qk_rate015` 直接做大规模 grid；应先对比 `online_qk`、`online_draft`、full-attention oracle 的 selected set overlap，并分析 full 正确但 online_qk 错误的样本中，gold/relevant doc token 是否被 selector 覆盖。

### 已做检查 4：32B 与 235B 是否真的是同一套 pipeline

结论：32B 与 235B 走的是同一个 `test_fusionrag_reflect_preprocess_exp.py` pipeline，包括 preprocess doc KV、online FusionRAG-QK selector、selected token recompute、query decode。差异在模型 wrapper：
- 32B：`model_type=qwen3`，导入 `/mnt/qjhs-sh-lab-01/wjh/FusionRAG/ktransformers/models/modeling_qwen3.py`。
- 235B：`model_type=qwen3_moe`，导入本仓库 `ktransformers/models/modeling_qwen3_moe.py`。

代码检查：
- 脚本会强制 `config._attn_implementation = "sdpa"`。
- 32B 实际走老 `Qwen2SdpaAttention` 中的 FusionRAG importance 逻辑。
- 235B 走 MoE attention 后置的 FusionRAG importance 逻辑。
- 两边 QK importance 公式等价：最后一层 `query_states @ context_key`，softmax 后按 head/query 聚合写入 `importance_cache`。

prompt 检查：
- `model_family_map` 原先缺少 `qwen3_moe/qwen3moe`，会把 235B 映射为 `Qwen2.5`。
- 但当前 `Qwen2.5` 与 `Qwen3` 的 `2wikimqa` system prompt 完全相同，因此这不是本次异常根因。
- 已修复代码：将 `qwen3_moe/qwen3moe` 映射到 `Qwen3`，避免后续混淆。

selected trace 对比：同一 example/sub-question，32B 与 235B 的上下文结构完全一致：
- `chunk_ids = [0, 1, 2, ..., 10, 1, 2]`
- 说明“preprocess 后 doc KV + 当前 retrieved docs”的混合上下文不是 235B 特有，32B 也是这样。

32B vs 235B selected token overlap：

| sample | context structure same | selected doc tokens | intersection | Jaccard | overlap ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| ex0_sub0 | yes | 196 vs 196 | 104 | 0.3611 | 53.06% |
| ex1_sub0 | yes | 194 vs 194 | 107 | 0.3808 | 55.15% |

观察：
- 两边 pipeline 与上下文结构一致，但实际 selected token 集合重合度只有约 53%-55%。
- ex1_sub0 中，gold answer `Corfe Mullen, Dorset, England` 位于 chunk1。235B 不是没有选到关键 token；它在重复出现的 chunk1 中甚至选到了更多 `Corfe/Mullen/England` 相关 token，但 235B full/force-all 本身也答错该样本。因此该样本不能证明 online_qk selector 错，而是 235B 模型本身回答行为不同。

更新后的判断：
- 用户指出“32B 也是同一个 pipeline”是对的。此前不能把混合上下文或 skip chunk 解释成 235B 独有实现问题。
- 当前更合理的解释是：同一个 QK selector 在 32B dense 和 235B MoE 上得到的 token 排序差异较大；其中一部分差异是模型本身 full attention 行为不同，另一部分才可能是 online_qk 相对 full 的退化。
- 下一步应只分析 `235B full 正确但 235B online_qk 错误` 的样本，排除模型本身答错的样本，再看 selector 是否漏掉必要 token。

### 已做检查 5：旧 `online_qk_rate015` CSV 是否还能代表当前实现

目的：用户指出 32B 也使用同一套 pipeline，因此继续怀疑 235B 低分是否来自当前实现本身。为避免把旧实验结果和当前代码混在一起，本轮先抽取旧表中 `full_rate1` 正确但 `online_qk_rate015` 错误的样本，在当前代码下重新跑一条最小 trace。

旧表对齐结果：
- `full_rate1`：217/250 sub-question 正确。
- 旧 `online_qk_rate015`：177/250 sub-question 正确。
- 两者同时正确：164。
- `full_rate1` 正确但旧 `online_qk_rate015` 错误：53。
- 旧 `online_qk_rate015` 正确但 `full_rate1` 错误：13。

复现实验设置：
- 样本：CSV row 13，对应 `example_id=10, sub_q_idx=1`。
- 问题：`Which organization did George Packer volunteer for?`
- 标准答案：`Peace Corps`
- 旧 `online_qk_rate015` 输出：`700 Club`，判错。
- 当前重新运行路径：`MOTIVATION_EXPERIMENTS/qwen3_235b_selector_debug/full_correct_qk_wrong_sample10/full_10_11/`
- selected trace：`MOTIVATION_EXPERIMENTS/qwen3_235b_selector_debug/full_correct_qk_wrong_sample10/selected_traces/example010_sub01_FusionRAG_rate0p15.json`
- 仍使用相同 235B 模型、相同 cache 路径、相同 `rate=0.15`、相同 `FusionRAG` QK selector；只按当前代码重新跑该 main sample。

当前复现结果：
- `sub_q_idx=0`：`Who is the author of The Unwinding?`，输出 `乔治·帕克 (George Packer)`，judge 判对。
- `sub_q_idx=1`：`Which organization did George Packer volunteer for?`，输出 `George Packer volunteered for the Peace Corps.`，judge 判对。

selected token 观察：

| context part | chunk id | doc len | selected | selected ratio | 备注 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2 | 203 | 4 | 1.97% | 包含 `Peace Corps` 的文档第一次出现 |
| 12 | 2 | 203 | 18 | 8.87% | 包含 `Peace Corps` 的文档第二次出现 |

关键词覆盖：
- 第一次出现的 `Peace` / `Corps` token 未被选中。
- 第二次出现中 `Corps` token 被选中，`Peace` token 未被选中。
- 但当前输出仍正确，说明该样本不能继续作为“当前 QK selector 必然漏掉答案 token”的证据。

阶段结论：
- 旧 `online_qk_rate015` 表中至少有一个 `full 正确 / online_qk 错误` 样本，在当前代码下已经不可复现。
- 因此旧 235B `online_qk_rate015=177/250` 的结果应标记为 stale，不能继续作为当前实现正确性判断依据。
- 接下来如果要比较 235B 大模型效果，应重新跑一轮统一 prompt、统一代码版本下的 `full_rate1 / online_qk_rate015 / online_draft_rate015`，再做 selector 分析。

### 自动记录：235B current rerun `online_qk_rate015`

- 完成时间：2026-07-09 20:22:46
- CSV：`MOTIVATION_EXPERIMENTS/qwen3_235b_current_rerun_20260709/online_qk_rate015/full_0_200/Qwen3-235B-A22B/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv`
- run.log：`MOTIVATION_EXPERIMENTS/qwen3_235b_current_rerun_20260709/online_qk_rate015/full_0_200/run.log`
- 原始 CSV 行数：250
- 唯一 `(Main Question, Sub Question)`：248
- 重复行：2
- Sub accuracy：179/248 (72.18%)
- Main accuracy：74/135 (54.81%)
- Avg F1：0.4495
- Avg EM：0.1331

```json
{
  "csv": "MOTIVATION_EXPERIMENTS/qwen3_235b_current_rerun_20260709/online_qk_rate015/full_0_200/Qwen3-235B-A22B/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv",
  "rows": 250,
  "unique_rows": 248,
  "duplicate_rows": 2,
  "sub_correct": 179,
  "sub_total": 248,
  "sub_acc": 0.7217741935483871,
  "main_correct": 74,
  "main_total": 135,
  "main_acc": 0.5481481481481482,
  "avg_f1": 0.4494872366833654,
  "avg_em": 0.13306451612903225
}
```

### 自动记录：235B current rerun `online_draft_rate015`

- 完成时间：2026-07-09 22:33:53
- CSV：`MOTIVATION_EXPERIMENTS/qwen3_235b_current_rerun_20260709/online_draft_rate015/full_0_200/Qwen3-235B-A22B/musique/DraftModel_global_topk10_bge/rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv`
- run.log：`MOTIVATION_EXPERIMENTS/qwen3_235b_current_rerun_20260709/online_draft_rate015/full_0_200/run.log`
- 原始 CSV 行数：250
- 唯一 `(Main Question, Sub Question)`：248
- 重复行：2
- Sub accuracy：204/248 (82.26%)
- Main accuracy：93/135 (68.89%)
- Avg F1：0.4959
- Avg EM：0.1532

```json
{
  "csv": "MOTIVATION_EXPERIMENTS/qwen3_235b_current_rerun_20260709/online_draft_rate015/full_0_200/Qwen3-235B-A22B/musique/DraftModel_global_topk10_bge/rate_0.15_draft_Qwen2.5-3B-Instruct_revert_rope.csv",
  "rows": 250,
  "unique_rows": 248,
  "duplicate_rows": 2,
  "sub_correct": 204,
  "sub_total": 248,
  "sub_acc": 0.8225806451612904,
  "main_correct": 93,
  "main_total": 135,
  "main_acc": 0.6888888888888889,
  "avg_f1": 0.495881671170459,
  "avg_em": 0.1532258064516129
}
```
