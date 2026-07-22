# Working-KV 稀疏重算实验日志

## 目标与约束

本轮以 `goal.md` 为唯一规范，验证 selected document token 在每一层产生 candidate KV 后，先与原 cache 融合为 working KV，再执行本层 attention：

```text
working_K = (1 - alpha) * base_K + alpha * candidate_K
working_V = (1 - alpha) * base_V + alpha * candidate_V
```

同层 selected token 采用 layer-parallel 更新。若 `i < j` 且 token j 的 attention support 包含 token i，j 必须读取 i 在本层的 working KV。query prefill 不参与融合，继续对完整 mixed cache 使用 dense attention。

## 分支和提交

| 阶段 | 分支 / commit | 说明 |
|---|---|---|
| 起点 | `exp/sparse-kv-working-blend-20260719` / `d2dcbd3` | 从 PCA 分析分支切出的隔离实验分支 |
| 错误原型 | `e519e24` | 首次把融合放到 attention 前，但使用全局阶段判断，可能污染 query |
| 阶段隔离 | `44890a9` | 改为 document reprocess 显式参数，query 不传 |
| 测试与 runner | `8366cb6` | 增加 alpha 端点测试和明确方法名 |
| 真实路径修正 | `0fe51cc` | 发现正式 pipeline 使用 `models/modeling_qwen3.py`，迁移实现并撤销错误副本修改 |
| 10 样本 launcher | `ade0cd6` | 固定 GPU 的 preprocess alpha 矩阵 |
| host 路径修正 | `656827d` | qjy000 使用 `/raid/home`，qjy001 使用 `/home` |
| 50 样本 launcher | `5287bb8` | raw/preprocess、Dense/Sparse 的 50 样本矩阵 |

## 已发现并订正的问题

1. 首版用 `q_len > 1` 和全局环境变量判断融合阶段，会把多 token query prefill 误判成 document recompute。已改成只由 document reprocess 调用传入 `reprocess_kv_alpha`。
2. 仓库同时存在 `models/modeling_qwen3.py` 与 `ktransformers/models/modeling_qwen3.py`。setup-v2 正式 pipeline 实际导入前者；首次小模型测试导入后者，造成“测试通过但 32B 不生效”。已撤销错误副本改动，测试和正式 pipeline 统一导入 `models.modeling_qwen3`。
3. 不能用 alpha=1 运行产生的深层 candidate 验证 alpha=0.25 的静态线性插值。第 1 层以后 candidate 依赖前层 alpha-conditioned hidden state；正确关系是每层使用该次运行实际产生的 candidate 与 base 融合。
4. 第一次批量启动脚本在 zsh 中拆分参数失败，任务立即退出，没有模型进程和有效结果。后续 launcher 改为 Python 固定任务表。
5. qjy001 不存在 `/raid/home/hming`，launcher 初版未启动两组任务。已增加 host→repo root 显式映射。

## Correctness 测试

脚本：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/test_working_kv_semantics.py
```

当前结果：

```text
PASS working-KV endpoints, immutable base snapshots, pre-attention scatter,
selected-predecessor read, dependency stats, and query isolation
```

已验证：

- Dense/Sparse alpha=0 的 selected-position KV 等于 base KV；
- alpha=1 等于未融合 candidate；
- layer 0 的 alpha=0.25 满足显式插值；
- 原 base cache 对象未被实验副本修改；
- query prefill 未重新融合 document KV；
- sparse working KV 无 NaN/Inf。
- attention 算子实际读到 scatter 后的 working K/V；
- 构造的 `i < j` 跨 block 用例中，j 实际读取 i 的 working V；
- router 输出 selected-predecessor dependency coverage、真实 block 数和真实 KV token 数；
- working-KV 仅允许 SDPA 且 `output_attentions=False`，超 32768 selected tokens 明确拒绝，避免静默破坏 layer-parallel 语义；
- working-KV 主路径每层 cache 索引迁移到对应 cache device；自动测试覆盖正常/异常 `past_tokens` 恢复。真实多设备 cache 尚未测试，不声称多设备已验证。

## 2026-07-19 首轮三智能体审核与订正

三名独立 Codex reviewer 分别从 Transformer 因果语义、系统/cache 正确性、科研方法学审查。共同指出旧提交 `5287bb8` 的 50 样本矩阵只能作为探索结果，不能作为最终结论，原因包括：生产路径没有显式 immutable selected-position base snapshot、测试没有观察真实 attention 输入、没有 dependency coverage、没有 delta-direction/oracle-alpha、alpha=0 未同提交重跑、alpha 选择与最终报告没有验证/测试划分。

本轮已完成的实现订正：

1. document reprocess 前逐层保存 selected-position base K/V；每层只从该只读快照计算 working K/V，再写入 working cache；
2. working K/V 在本层 attention 前写回，测试直接捕获 attention 收到的 K/V；
3. 修复 eager 路径误用未定义 `kwargs`；working-KV runner 明确只支持 SDPA；
4. 对超过 32768 selected tokens 的 working-KV 请求 fail-fast，暂不宣称支持该范围；
5. 新增 selected predecessor dependency coverage、effective blocks 和 effective KV tokens；
6. 修复多设备 cache position 与异常路径 cache 长度恢复。

仍需完成后才能形成最终方法结论：

- 修订提交上的 32B alpha 端点回归；
- 同 commit、同 cache、同样本的完整 50 样本验证矩阵，包括实际 alpha=0 与 raw Sparse alpha=0.25；
- 冻结 alpha 后在独立样本段测试，并给 paired bootstrap CI；
- dense/sparse delta direction、relative L2 与 per-layer oracle alpha；
- 真实 dependency coverage 与 attention support 统计；
- 质量、系统代价和当前 PyTorch 原型不等于 fused-kernel 加速的边界说明。

## 1 样本 Qwen3-32B 真实 pipeline sanity

设置：

- dataset：MuSiQue-v2；
- cache：`/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2`；
- load KV：preprocess；
- rate：0.15；
- block size：64；
- block top-K：8；
- commit：`0fe51cc5933b4c871403e77fe9c63c9c183105b2`。

端点结果：

| 方法 | alpha | 第 1 个样本答案 | 对照 |
|---|---:|---|---|
| Dense working KV | 0 | Mary Bono | 与历史 preprocess_rate0 完全一致 |
| Dense working KV | 0.5 | Maggie Gyllenhaal | 中间更新 |
| Dense working KV | 1 | Salma Hayek | 与历史 online_qk@0.15 完全一致 |
| Sparse working KV | 0.5 | Maggie Gyllenhaal | 中间更新 |
| Sparse working KV | 1 | Salma Hayek | sparse replace |

这组只验证端点和文本正常，不用于准确率结论。

## 10 样本 preprocess alpha 矩阵

启动脚本：

```bash
python3 MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/launch_working_kv_preprocess_10.py
```

commit：`656827d4db558e433b41a5cfb472f984c7b59bbb`

| Attention support | alpha | N | EM | F1 |
|---|---:|---:|---:|---:|
| Dense/full causal prefix | 0 | 10 | 30.00% | 35.58% |
| Dense/full causal prefix | 0.25 | 10 | 30.00% | 35.58% |
| Dense/full causal prefix | 0.5 | 10 | 30.00% | 35.58% |
| Dense/full causal prefix | 0.75 | 10 | 30.00% | 36.94% |
| Dense/full causal prefix | 1 | 10 | 20.00% | 30.28% |
| MoBA block sparse | 0 | 10 | 30.00% | 35.58% |
| MoBA block sparse | 0.25 | 10 | 30.00% | 35.58% |
| MoBA block sparse | 0.5 | 10 | 30.00% | 38.65% |
| MoBA block sparse | 0.75 | 10 | 30.00% | 39.17% |
| MoBA block sparse | 1 | 10 | 30.00% | 35.58% |

初步解释：

- alpha=0 的 Dense/Sparse 都回到 preprocess baseline，说明 write-back 端点正确；
- Dense alpha=1 在该 10 样本子集明显低于 cache-anchored alpha，支持“直接 replacement 会破坏已有 cache 信息”；
- sparse alpha=0.5/0.75 的 F1 暂时较高，但 N=10 不能用于选最佳 alpha；
- 当前 PyTorch block router/gather 原型的 wall time 比 full-support selected-query attention 更慢，不能据此声称端到端加速。需要将算法 KV support 降低和未 fused 原型开销分开报告。

## 旧提交上正在运行的 50 样本探索矩阵

启动脚本：

```bash
python3 MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/launch_working_kv_50.py
```

commit：`5287bb8c874b425a7417f2b73aa12fa7e14d8fb8`

结果目录：

```text
MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results_working_kv_50/
```

覆盖：

- preprocess：Dense alpha=0.25/0.5/0.75/1，Sparse alpha=0.25/0.5/0.75/1；
- raw：Dense alpha=0.25/0.5/0.75/1，Sparse alpha=0.5/0.75/1；
- alpha=0 的质量使用已经通过端点测试的对应 rate=0 baseline，避免重复计算。

该矩阵启动后首轮审核发现上述证据缺口，因此不再作为最终冻结结果。任务不中断，完成后只用于观察趋势；最终表必须在订正后的冻结 commit 上按验证/测试划分重跑。

## 修复版真实 32B 端点回归

- runtime commit：`ea73a0e2a35dcbebdfcfe914fc663959eddf9717`；
- dataset/cache：MuSiQue-v2，共享 Qwen3-32B setup-v2 preprocess cache；
- 样本：`seg_0_1`；rate=0.15；block size=64；Top-K=8。

| 方法 | alpha | 输出 |
|---|---:|---|
| Dense + preprocess | 0 | `Mary Bono` |
| Dense + preprocess | 1 | `Salma Hayek` |
| Sparse + preprocess | 0 | `Mary Bono` |
| Sparse + preprocess | 1 | `Answer: Salma Hayek` |

alpha=0 与 rate=0 输出对齐，alpha=1 与 selected-token replacement 输出对齐。该回归只证明实现端点，不证明任务质量。

## 修复版 50 样本验证矩阵

启动脚本：

```bash
python3 MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/launch_working_kv_fix2_50.py
```

启动器 commit：`0437d520c0c64cc3ba000030b0fca8418e2492bb`。运行时核心语义 commit 为 `ea73a0e`；后续 commit 只增加 no-op trace callback、分析脚本和 stats-only 字段，不改变未启用 stats/trace 时的数值路径。结果目录：

```text
results_working_kv_fix2_validation_50/
```

完整覆盖 raw/preprocess × Dense/Sparse × alpha={0,0.25,0.5,0.75,1}。每个方法独占结果目录，读取同一 MODEL/DATASET cache；EM/F1/GLM 在 runner 内自动执行。旧队列被重新分配时只终止等待中的 bash 父进程，正在运行的 Python 子进程未中断。

已完成的首批 Dense 暂定结果：

| Load KV | alpha | N | EM | F1 | GLM |
|---|---:|---:|---:|---:|---:|
| preprocess | 0 | 50 | 22% | 32.92% | 32% |
| preprocess | 0.75 | 50 | 24% | 33.55% | 34% |
| raw | 0 | 50 | 14% | 23.43% | 26% |
| raw | 0.75 | 50 | 28% | 38.51% | 40% |

这是 alpha-selection 段的中间结果，不是最终 test 结论。preprocess 的变化很小；raw 的提升较大但 N=50 CI 宽，必须等完整 alpha 扫描后冻结配置，并在独立样本段复验。

## Delta direction 与 oracle alpha

脚本：

```bash
python scripts/run_working_kv_delta_trace.py ...
python scripts/analyze_working_kv_delta_trace.py ...
```

- commit：`8c5a299`；
- 样本：MuSiQue-v2 前 5 个 example；
- 每个 example、每层固定采样前 32 个 selected tokens 的全部 KV heads/dim；
- Dense 与 Sparse 使用相同 cache、selector、样本和 selected-token 顺序；
- 结果是 sampled-token 可行性分析，不能外推为全部 token 的无偏估计。

| Load KV | Delta | Cosine | Sparse/Dense norm | Oracle alpha | Oracle relative L2 |
|---|---|---:|---:|---:|---:|
| raw | K | 0.498 | 1.564 | 0.318 | 0.867 |
| raw | V | 0.468 | 1.481 | 0.316 | 0.884 |
| preprocess | K | 0.545 | 1.293 | 0.421 | 0.839 |
| preprocess | V | 0.562 | 1.186 | 0.474 | 0.827 |

解释：preprocess 下 Sparse delta 与 Dense delta 更接近，幅度失配也更小，但方向余弦仍只有约 0.54–0.56。即使使用后验 oracle 标量，仍留下 82.7%–83.9% relative L2。因此只能得出：在这 5 个 examples、每层前 32 selected tokens、trajectory alpha=1 的偏置 trace 上，global scalar 不能重建 Dense delta。不能据此直接推广到全部 token/数据，也不能把偏离因果归因给 router。

## Sparse router 真实支持

配置：preprocess、alpha=0.5、block size=64、Top-K=8、MuSiQue-v2 第 1 个 example。stats 路径执行 full QK 仅用于分析，不能计入速度。

| 指标 | 数值 |
|---|---:|
| layer-head records | 4096 |
| selected predecessor dependency coverage | 7.79% |
| attention mass recall | 28.38% |
| effective KV/query | 480.6 |
| dense causal KV/query | 9901.1 |
| KV support fraction | 4.85% |
| preserve-all dependency blocks/query | 125.5 |
| preserve-all dependency KV/query | 8003.6 |
| preserve-all support fraction | 80.83% |

这说明当前 Top-K=8 MoBA prototype 虽把 exact-attention support 降到 4.85%，但漏掉 92.21% 的 selected-predecessor pairs 和 71.62% 的 dense attention mass。该 support 错误发生在 alpha blending 之前，不能靠标量 alpha 修复。当前 Python router 还比 dense selected-query kernel 更慢，因此只能声称理论 support 下降，不能声称端到端加速。

若强制保留所有 earlier-selected blocks，coverage 可按定义达到 100%，但在该 example 的静态预算估算中支持膨胀到 125.5 blocks/query、8003.6 KV/query，即 dense causal support 的 80.83%。这只说明该样本下不具吸引力，不等价于实际 preserve-all 质量/延迟实验或总体否决。

## 2026-07-19 修复版 validation 完成

完整矩阵为 raw/preprocess × Dense/Sparse × `alpha={0,0.25,0.5,0.75,1}`，每组 N=50。启动命令：

```bash
python3 MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/launch_working_kv_fix2_50.py
```

启动器 commit `0437d52`，核心 Working-KV 语义 commit `ea73a0e`。汇总文件：`working_kv_fix2_validation_summary.csv`、`working_kv_fix2_validation_summary_paired.csv`。

实际 test 前对 raw 候选按主指标 GLM、tie-break F1、再选较小 alpha，Dense raw 和 Sparse raw 均冻结 `alpha=0.75`；full-rate1 只作 control，不参与选择。这里存在一处必须保留的 protocol deviation：早期文字把规则写成覆盖全部 raw/preprocess 条件，但执行时用未量化的“稳定增益”排除了 preprocess 非零 alpha，仅保留 `alpha=0` 参考。若机械执行原文字规则，Dense-preprocess 应选 `alpha=1.0`、Sparse-preprocess 应选 `alpha=0.5`，但二者没有 frozen-test 数据。因此最终确认性结论只覆盖实际测试的 raw 候选，不能外推到 preprocess 非零 alpha。validation full-rate1 为 EM 28%、F1 39.25%、GLM 42%。

## 2026-07-19 Frozen test（源 rows 51--200，N=150）

候选与 rate=0 对照使用以下命令模式；不同方法和 Sparse 分片只改变 `--method/--working-kv-alpha/--start/--end/--gpu`，各自写独立目录：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py \
  --dataset musique-v2 --method dense_working_kv_raw --rate 0.15 \
  --working-kv-alpha 0.75 --start 50 --end 200 --gpu 0 \
  --result-root MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results_working_kv_fix2_test_150
```

full-rate1 原单卡任务已生成源 rows 51--145 左右，但 judge 尚未开始。为完成严格对照而不覆盖其 CSV，另在 qjy000/qjy001 空闲 GPU 上按互斥区间运行 10 个分片：`50:70,70:90,90:110,110:130,130:145,145:156,156:167,167:178,178:189,189:200`。结果根为 `results_working_kv_fix2_test_150_full_shards/`。每个分片沿用 runner 内集成的 EM/F1/GLM judge。10 个分片完整产出后，停止了仍在重复计算的原单卡 full 进程 PID 2627660；未删除其已生成 CSV。

分片启动示例：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py \
  --dataset musique-v2 --method full_rate1 --start 145 --end 156 --gpu 0 \
  --result-root MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results_working_kv_fix2_test_150_full_shards
```

汇总脚本在 commit `28adeb9` 增加多结果根合并、唯一问题计数、相同重复去重和冲突重复 fail-fast：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/summarize_working_kv_test.py \
  --root MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results_working_kv_fix2_test_150 \
  --root MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results_working_kv_fix2_test_150_full_shards \
  --summary MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/working_kv_fix2_test_summary.csv \
  --paired MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/working_kv_fix2_test_summary_paired.csv \
  --integrity MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/working_kv_fix2_test_integrity.json
```

| 方法 | EM | F1 | GLM |
|---|---:|---:|---:|
| preprocess alpha=0 | 20.00% | 34.38% | 33.33% |
| raw alpha=0 | 15.33% | 29.22% | 30.67% |
| Dense raw alpha=0.75 | 20.00% | 34.44% | 33.33% |
| Sparse raw alpha=0.75 | 14.67% | 28.50% | 27.33% |
| full rate=1 | 24.67% | 40.29% | 41.33% |

Paired 结论：

- Dense raw 0.75 vs raw 0：EM `+4.67pp`，F1 `+5.23pp`，GLM `+2.67pp`；三项 95% paired bootstrap CI 均覆盖 0，GLM up/down `13/9`，McNemar `p=0.5235`。
- Sparse raw 0.75 vs raw 0：EM `-0.67pp`，F1 `-0.72pp`，GLM `-3.33pp`；GLM up/down `6/11`，McNemar `p=0.3323`。
- Dense raw 0.75 vs full：F1 `-5.84pp`，GLM `-8pp`，GLM CI `[-14,-2]pp`，McNemar `p=0.0169`。
- Sparse raw 0.75 vs full：F1 `-11.79pp`，GLM `-14pp`，GLM CI `[-20.67,-7.33]pp`，McNemar `p=0.000104`。

最终判断：validation 上 raw blend 的趋势未在 test 复现。Dense raw blend 相对 raw0 只有正向点估计且 CI 覆盖 0，同时没有节省 attention；实际冻结的 Top-K=8 Sparse-raw blend 不优于 raw rate=0，不能作为可用方案。Sparse/Dense preprocess 非零 alpha 因上述 protocol deviation 未进入 frozen test，保持未回答状态。

## 2026-07-19 首轮最终报告三审订正

三名 reviewer 均确认核心 Working-KV 语义和逐样本 test 数值，但首轮均为 NOT PASS。订正如下：

1. 报告与 `goal.md` 补充 Qwen3 cached K 的真实坐标：`W_K` 后还有 head-wise `k_norm` 与绝对位置 RoPE；V 不经过 RoPE。
2. 删除“Dense blend 已有价值”“validation 被高估”等过强措辞。独立 test 只能支持正向点估计且 CI 覆盖 0；validation/test 差异也可能来自抽样波动或连续切分偏移。
3. 将 delta oracle 结论严格限制在 5 examples、每层前 32 selected tokens、alpha=1 trajectory；不再断言 per-layer scalar gate 一定不足。
4. 后续 Top-K/router 开发不得继续使用 rows 51--200 作为独立 test；需新增 dev 和未触碰 holdout。
5. 新增 `frozen_evidence/validation` 和 `frozen_evidence/test`，保存完整逐样本行、原始 CSV SHA256、统一 question-set hash 和 post-hoc provenance 限制。历史 worker 未在启动前写入 git HEAD，因此 alpha 冻结时序和 full 同源性有日志支持，但不是密码学预注册。
6. 汇总器现在比较整行内容；只有所有字段相同的重复才去重，预测文本或 judge rationale 不同会 fail-fast。
7. 历史和当前 runner 都允许缺失 KV/FAISS artifact 持锁回填，所以撤回“worker 保证只读”的声明。曾尝试的 `--require-existing-cache` 没覆盖目录创建和 FAISS 写入，第二轮系统审核判定不完整，已删除。后续应先单 worker warmup、核对 cache manifest，再启动并行阶段。
8. runner 改为 `--gpu` 无条件覆盖继承的 `CUDA_VISIBLE_DEVICES` 并打印最终绑定，避免静默跑错物理卡。
9. manifest 加入 `custom_cache.py`、`unified_process_cache.py`、runner、汇总/冻结脚本 hash；cache 增加排序后均匀抽样内容 hash，并明确不是完整 900GB cache hash。语义 guards 与 runner GPU guard 使用不同 commit 字段。

新增 guard 测试：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/test_setup_v2_runner_guards.py
```

输出：`PASS setup-v2 deterministic GPU binding`。该测试不声称 cache 严格只读。

## 2026-07-19 第四轮审阅后补充计划

第四轮 reviewer 指出 frozen test 未覆盖按原统一规则应选出的 Dense-preprocess `alpha=1.0` 和
Sparse-preprocess `alpha=0.5`，且 `goal.md` 要求的 K/V 分离 alpha 尚未执行。补充实验保持 source rows
1--50 为 validation、51--200 为 frozen test，不在 test 上重新选择参数：

1. frozen test 补跑 Dense-preprocess `(alpha_k,alpha_v)=(1,1)` 和 Sparse-preprocess `(0.5,0.5)`；
2. validation 只做一轴 K/V ablation，不做二维网格。围绕四个已选共享 alpha，分别跑 K-only `(a,0)`
   和 V-only `(0,a)`；与已有 `(0,0)`、`(a,a)` 比较 K/V 贡献；
3. 将 test 不确定性解释限定为该 150 行测试集合上的 question-level 条件推断，并补充基于近重复问题
   cluster 的敏感性分析，不把逐行 bootstrap/McNemar 当成对任意新问题总体的无条件推断；
4. 固化 Delta `.pt`、router `.jsonl` primary evidence，补齐可执行生成命令，并记录旧 trace 缺少显式
   example/selected-position 标识的限制；`task.md` 作为用户最初 sparse-MoBA 描述一并纳入版本控制，
   `goal.md` 仍是用户明确指定的最终规范。

实现保持单一 Working-KV pipeline：新增可选 `--working-kv-alpha-k/v`，未提供时回退到原
`--working-kv-alpha`。启动器：

启动器使用控制端的 `qjy000/qjy003` SSH alias，必须从控制端执行：

```bash
ssh qjy000 'cat /raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/launch_confirmatory_and_kv_ablation.py' | python3 -
```

第一次误从 qjy000 计算节点内部直接运行，因节点不能解析控制端 SSH alias 而在启动首个 worker 前失败；
第二次从控制端执行时发现本地不应创建远端日志目录，也在启动 worker 前失败。启动器随后改为由每条远端命令
执行 `mkdir -p`，以下正式任务使用修复后的 commit。

第三次从控制端执行成功启动 qjy000 的 8 个 frozen-test worker，但在首个 qjy003 task 前发现 qjy003
repo 根目录是 `/home/hming/FusionRAG-pca-analysis` 而非 qjy000 的 `/raid/home/...`，因此 qjy003 的 8 个
validation worker 尚未启动。启动器改为显式 host-to-root 映射并支持 `--host` 过滤；修复提交后只执行
`--host qjy003`，不会重复启动 qjy000 分片：

```bash
ssh qjy000 'cat /raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/launch_confirmatory_and_kv_ablation.py' | python3 - --host qjy003
```

固定分配为 qjy000 GPUs 0--7 跑两个 preprocess frozen-test 候选的 4+4 个互斥分片，qjy003 GPUs
0--7 各跑一个 N=50 K-only/V-only validation 条件。所有 worker 复用同一 Qwen3-32B/MuSiQue-v2
shared cache，结果与 cache 分离。

## 2026-07-19 补充实验结果

正式补充任务均完成，最终实现/启动器 commit：`ddeabdb`；K/V alpha 实现 commit：`a7fa0c6`；primary
Delta/router evidence commit：`e0aa789`。

Frozen test（rows 51--200，N=150）：

| 方法 | alpha_k | alpha_v | EM | F1 | GLM |
|---|---:|---:|---:|---:|---:|
| dense preprocess | 1.0 | 1.0 | 21.33% | 35.71% | 36.00% |
| sparse preprocess | 0.5 | 0.5 | 18.67% | 33.37% | 32.67% |

Dense preprocess alpha=1 相对既有 preprocess alpha=0 的 paired delta 为 EM/F1/GLM `+1.33/+1.33/+2.67pp`，
逐行 bootstrap CI 均覆盖 0；Sparse preprocess 没有同源 frozen sparse-alpha0 control，因此只记录绝对结果。

Validation K/V ablation（rows 1--50，N=50）新增条件：

| 条件 | K-only GLM | V-only GLM | 共享 alpha GLM |
|---|---:|---:|---:|
| dense raw, a=0.75 | 28% | 34% | 40% |
| sparse raw, a=0.75 | 22% | 30% | 34% |
| dense preprocess, a=1.0 | 30% | 32% | 38% |
| sparse preprocess, a=0.5 | 24% | 40% | 34% |

完整 EM/F1/GLM 和 paired CI 在 `working_kv_alpha_kv_validation_summary.csv`、
`working_kv_alpha_kv_validation_paired.csv`。总体方向是 V-only 通常比 K-only 更好，Sparse-preprocess
上最明显，但 N=50 只能作为机制线索，不能作为最终泛化结论。

新增可复现命令：

```bash
# 完整固定 GPU 启动器，从控制端执行
ssh qjy000 'cat /raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/launch_confirmatory_and_kv_ablation.py' | python3 -

# K/V alpha 汇总
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/summarize_working_kv_alpha_kv.py \
  --root MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results_working_kv_alpha_kv_validation \
  --expected-n 50 \
  --output MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/working_kv_alpha_kv_validation_summary.csv

/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/summarize_working_kv_alpha_kv.py \
  --root MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results_working_kv_confirmatory_preprocess_test \
  --expected-n 150 \
  --output MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/working_kv_confirmatory_preprocess_test_summary.csv

# Dense preprocess alpha=1 vs frozen preprocess alpha=0 的 paired 分析
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/compare_working_kv_alpha_kv.py \
  --new-root MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results_working_kv_confirmatory_preprocess_test \
  --baseline-root MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results_working_kv_fix2_test_150 \
  --output MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/working_kv_confirmatory_preprocess_test_paired_vs_alpha0.csv \
  --new-method dense_working_kv_preprocess alpha_k_1p0__alpha_v_1p0 dense_pp_alpha1 \
  --baseline-method dense_working_kv_preprocess alpha_0p0 dense_pp_alpha0
```

结果路径、cache 路径和每个 GPU/segment 映射见 launcher 与前文记录。

## 2026-07-19 全量 MuSiQue-v2 baseline 对比计划

目标：在完整 MuSiQue-v2 200 examples 上，用 runner 默认集成的 GLM judge，同时记录 EM/F1，比较当前
Working-KV 方案与已有 baseline。已有 GLM baseline 直接复用历史统一评测结果，不重新生成：full rate=1
`83/200=41.5%`，online draft rate=0.15 `82/200=41.0%`，online qk `75/200=37.5%`，uniform alpha=0.1
`83/200=41.5%`，offline mean/top2 `81/200=40.5%`，preprocess rate=0 `65/200=32.5%`，raw rate=0
`59/200=29.5%`。

本轮新增 8 个 Working-KV 条件，每个切为 `0:100`、`100:200` 两个分片，共 16 workers，固定分配 qjy000
GPU 0--7 与 qjy003 GPU 0--7：Dense/Sparse raw/preprocess shared alpha，以及 validation 中最有解释力的
raw K-only/V-only 和 sparse-preprocess K-only/V-only。所有条件使用 rate=0.15、同一 shared cache，GLM
由 runner 自动调用，不再手动后评测。

启动命令：

```bash
ssh qjy000 'cat /raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/launch_working_kv_full_musique.py' | python3 -
```

启动后发现初版映射把同一条件的两个分片同时放到同一 GPU，已立即停止本轮刚启动的进程；没有影响已有
baseline/cache。修复后映射为每个方法 index 使用同编号 GPU，`0:100` 在 qjy000、`100:200` 在 qjy003，
因此 8 个方法正好使用两台机器各 8 张卡。

批量 SSH 提交随后出现连接断开，目录只生成空目录；单卡 smoke 已成功完成并自动 GLM 判定。最终启动改为
每台机器一个控制会话、本地启动 8 个 `start_new_session` worker：

```bash
ssh qjy000 'cd /raid/home/hming/FusionRAG-pca-analysis && python3 MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/start_working_kv_full_host.py --start 0'
ssh qjy003 'cd /home/hming/FusionRAG-pca-analysis && python3 MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts/start_working_kv_full_host.py --start 100'
```
