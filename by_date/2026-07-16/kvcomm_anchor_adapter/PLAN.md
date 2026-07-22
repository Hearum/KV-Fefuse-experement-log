# KVCOMM-style Anchor Adapter 探究计划

## 1. 研究目标

目标不是直接训练完整 KV predictor，而是检验以下可部署近似：

```text
Delta_KV(X | current_prefix)
  ~= sum_j w_j(current_prefix, anchor_prefix_j) * Delta_KV_anchor_j(X)

KV_hat = KV_cache + Delta_KV_hat
```

其中 `X` 是目标 document/chunk，anchor 只来自训练 prefix。在线阶段不能读取当前样本的真实 Delta，也不能用测试 Delta 选择权重。

最终要回答：

1. 相似 prefix 是否对应相似 Delta-KV？
2. nearest/top-k anchor 是否稳定优于 document mean template？
3. anchor offset 是否还能压成 rank 4/8/16/32 的低秩表示？
4. 该近似写回 FusionRAG 后，能否在不运行 document Transformer recompute 的情况下接近 rate=1 生成质量？
5. 计算、存储和端到端收益是否足以构成可用方案？

## 2. 为什么先用 MuSiQue

第一阶段使用历史 reflect MuSiQue 严格 context 数据，不重复生成 Qwen3-32B KV：

- 已有 5 个固定 target document；
- 每个 target 有 40 个 train prefix 和 10 个 held-out prefix；
- train/test 使用的 prefix documents 完全不相交；
- 每个 case 都由真实 FusionRAG `rate=1` 重算获得；
- raw/preprocess、K/V 均已保存。

它适合验证表示与检索假设，但不能作为最终 benchmark。端到端阶段使用 setup-standard `musique-v2`，因为它是标准单问题 RAG 格式、200 examples、已有统一 cache 与 full/raw/preprocess/online baseline。通过后再扩展到 `2wikimqa-v2`，检验更多支持文档和不同上下文长度下的泛化。

## 3. 状态与指标定义

对 cache source `s in {raw, preprocess}`：

```text
B_s     = 当前 multi-document prompt 装载 cache 后、重算前的目标文档 KV
T_s     = 相同 prompt 下 rate=1 全 token 重算后的目标文档 KV
Delta_s = T_s - B_s
```

报告：

```text
original_gap = ||Delta|| / ||B||
recovery     = ||Delta - Delta_hat|| / ||Delta||
final_error  = ||Delta - Delta_hat|| / ||B||
energy       = 1 - ||Delta - Delta_hat||^2 / ||Delta||^2
cosine       = cos(Delta, Delta_hat)
```

`No update` 的 recovery 是 1，仅表示没有恢复 Delta；KV 实际差距由 original_gap 表示。

Key 先按 prefix token offset 逆 RoPE 到 target-local 坐标。Value 直接比较。K/V、raw/preprocess 禁止拼接成一个指标。

## 4. Phase A：严格 held-out Anchor 可迁移性

### 数据切分

- 5 个固定 target X。
- 每个 X：40 train contexts / 10 test contexts。
- train/test prefix document pool 严格不相交。
- 共 50 个 held-out target-context，raw/preprocess 各一套。

### 可部署特征

1. `position`：prefix token 数、prefix document 数。
2. `cached_prefix_v`：从已缓存 prefix Value 在若干层做 token-weighted mean，不运行 Transformer。
3. `hybrid`：position 距离与 cached-prefix 距离的归一化组合。

### 对照方法

- `no_update`
- `mean_anchor`：40 个训练 Delta 的均值。
- `random_anchor`
- `position_nn`、`position_topk`
- `cached_v_nn`、`cached_v_topk`
- `hybrid_topk`
- `oracle_anchor`：用真实 test Delta 选训练 anchor，只作 anchor pool 表示上界，不可部署。

top-k 扫描 `k=1/2/4/8`。权重只由 prefix feature 距离产生。

### Phase A 成功判据

进入后续阶段至少满足：

- 可部署 top-k 在 K 或 V 上的 held-out recovery 相对 mean template 改善至少 10%；
- 改善在 5 个 target 中至少 4 个方向一致；
- nearest/top-k 明显优于 random anchor；
- oracle anchor 明显优于 mean，证明 anchor pool 本身有选择价值。

若 oracle 都不优于 mean，停止 KVCOMM 路线；若 oracle 有效但 feature top-k 无效，问题在 matcher，而不是 offset 表示。

## 5. Phase B：Anchor Rate-Distortion

仅在 Phase A 通过后执行。对每个 document、layer、head 的训练 anchor Delta 学 basis：

```text
Delta_anchor ~= Mean_X + B_X,r c_anchor
```

比较 full-rank anchor 与 rank `4/8/16/32/64`：

- per-document context-template basis；
- per-document per-head feature basis；
- shared Key basis + document-specific Value basis。

同时报告恢复率、最终 KV error、每文档额外存储量和在线乘加量。低秩方案必须相对 full-rank anchor 保留至少 90% 的端到端收益，否则不称为轻量 Adapter。

## 6. Phase C：端到端 FusionRAG

### 第一轮：MuSiQue-v2 20-example sanity

方法矩阵：

| 方法 | rate | 是否重算 document Transformer | 作用 |
|---|---:|---|---|
| full rate1 | 1.0 | 是 | 质量上界 |
| raw rate0 | 0 | 否 | 空白对照 |
| preprocess rate0 | 0 | 否 | 现有离线方法 |
| online DraftModel | 0.15 | 选中 token 重算 | 当前高效 baseline |
| mean anchor adapter | 0 | 否 | 静态 offset 对照 |
| top-k anchor adapter | 0 | 否 | KVCOMM-style 主方法 |
| low-rank top-k anchor | 0 | 否 | 最终轻量候选 |
| top-k anchor + sparse residual recompute | 0.05 | 少量 | 混合候选 |

所有方法同一数据文件、同一 Qwen3-32B、同一 shared preprocess cache；生成同时计算 EM/F1/GLM judge。先跑 20 examples，确认输出与 KV 写回正常，再扩到 200。

### 第二轮：MuSiQue-v2 完整 200 examples

主要指标：GLM accuracy；辅助指标：EM、F1、TTFT、prefill/recompute wall time、额外 anchor I/O、峰值显存。成功标准：

- 相对 preprocess rate0 恢复至少 50% 的 `full rate1 - preprocess rate0` GLM accuracy 差距；
- TTFT 明显低于 full rate1，并优于或接近 online DraftModel rate0.15；
- 结果在至少 3 个随机 anchor 构建 seed 下稳定。

### 第三轮：2WikiMQA-v2 外部验证

不调 matcher 超参，直接迁移 MuSiQue-v2 选定配置。若性能下降，按 prefix 文档数、总 token 长度和 answer-support 文档位置分桶，判断 anchor 是否只适合某种 context regime。

## 7. 实现约束

- 不为本实验创建 per-worker preprocess cache；复用 `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2/<dataset>`。
- anchor 资产和结果 CSV 分开；可再生的大 tensor 不放进 Git。
- matcher 不得访问 online full Delta、full hidden 或更新后 prefix KV。
- 当前 prompt 是 `system + docs + query`，query 不用于 document Delta matcher；query 仍可用于 residual selector。
- 任何正式运行记录 commit、命令、机器/GPU、数据路径、cache 路径和输出路径。

## 8. 决策树

```text
feature top-k > mean > random
  -> 做低秩压缩 -> MuSiQue-v2 端到端

oracle > mean，但 feature top-k <= mean
  -> 改 matcher/低维 prefix feature，不训练完整 KV predictor

oracle <= mean
  -> 放弃 anchor selection，保留 mean/template + sparse recompute 路线

KV L2 改善但端到端下降
  -> 不继续优化 L2；转向 layer/head gating 与 logits-aware 校准
```
