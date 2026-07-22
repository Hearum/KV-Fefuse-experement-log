# Shallow Draft Selector Probe

## 目的

探索类似 MTP / shallow draft 的方向：不使用 lexical/BM25 等静态启发式，而是只用 DraftModel 的浅层 attention/hidden computation 近似完整 DraftModel selector。

当前代码扩展：

- `ktransformers/util/utils.py` 中 `_fusionrag_compute_draft_doc_attention_scores` 新增 `score_layers` 和 `stop_after_score_layers`。
- 默认行为不变：不指定时仍使用原来的后半层 RRF score。
- 当设置 `--draft_layer_selection fixed --draft_fixed_layer N` 时，DraftModel selector 只使用第 N 层 attention score，并默认在第 N 层后停止 draft forward。
- 也可用环境变量 `FUSIONRAG_DRAFT_SCORE_LAYERS` 指定层，例如 `0`, `1`, `2`, `4`, `first4`, `last`。

## Smoke 设置

- model: Qwen3-32B
- draft model: Qwen2.5-3B-Instruct
- dataset: MuSiQue reflect pipeline
- example: `start_sample=0,end_sample=1`
- rate: 0.15
- compared methods: default full DraftModel selector, fixed layer 0/1/2/4

## Smoke 结果

```csv
method,selected_count,intersection_with_default,recall_default,precision_vs_default,jaccard_vs_default,selection_time,score_forward_time,correct,pred
default,227,227,1.0,1.0,1.0,0.22971500499988906,0.22859614799381234,True,<think>  </think>  National Cycle Network
layer0,227,118,0.5198237885462555,0.5198237885462555,0.35119047619047616,0.11815586299053393,0.1170461930159945,True,<think>  </think>  Answer: The National Cycle Network
layer1,227,133,0.5859030837004405,0.5859030837004405,0.4143302180685358,0.1148062810243573,0.11367827499634586,True,<think>  </think>  Answer: The National Cycle Network
layer2,227,139,0.6123348017621145,0.6123348017621145,0.44126984126984126,0.12121406799997203,0.1200317130133044,True,<think>  </think>  Answer: The National Cycle Network
layer4,227,176,0.775330396475771,0.775330396475771,0.6330935251798561,0.12711441001738422,0.1260049499978777,True,<think>  </think>  National Cycle Network

```

## 初步观察

- fixed layer 4 对 default DraftModel selector 的 recall 达到 77.5%，Jaccard 63.3%，比 layer 0/1/2 明显更接近。
- score forward time 从 default 0.229s 降到 layer4 0.126s；layer0-2 也在 0.11-0.12s 左右。
- 单样本 smoke 不能说明 accuracy，但说明浅层 draft selector 代码路径可用，并且存在“更浅层仍能近似完整 selector”的信号。

## 下一步

1. 跑 20 examples selector-only alignment，不做 GLM judge accuracy。
2. 指标：selected-token recall/Jaccard、residual recall、chunk recall、selection time。
3. 若 layer4/first4 在 20 examples 上保持较高 recall，再跑完整 pipeline accuracy：offline fixed 0.15 + shallow draft residual 0.05。

## 20-example Selector Alignment

设置：`start_sample=0,end_sample=20`，共 32 个 sub-question traces。default 表示原始 DraftModel selector；layerN 表示只用 draft model 第 N 层 attention score，并在该层后停止前向。

| method | n | token recall vs default | token Jaccard | chunk recall | freq-base residual recall | score forward time(s) |
|---|---:|---:|---:|---:|---:|---:|
| default | 32 | 100.00% | 100.00% | 100.00% | 100.00% | 0.1452 |
| layer0 | 32 | 37.14% | 23.13% | 64.34% | 53.30% | 0.0092 |
| layer1 | 32 | 44.18% | 28.79% | 79.84% | 56.11% | 0.0127 |
| layer2 | 32 | 46.08% | 30.35% | 95.26% | 54.75% | 0.0162 |
| layer4 | 32 | 62.15% | 45.63% | 99.52% | 57.53% | 0.0236 |

### 20-example Answer Summary (rough, not final accuracy)

| method | Main Acc | Sub Acc |
|---|---:|---:|
| default | 15/18 (83.33%) | 29/32 (90.62%) |
| layer0 | 11/18 (61.11%) | 23/32 (71.88%) |
| layer1 | 11/18 (61.11%) | 23/32 (71.88%) |
| layer2 | 11/18 (61.11%) | 24/32 (75.00%) |
| layer4 | 14/18 (77.78%) | 26/32 (81.25%) |

### 20-example Takeaway

- layer4 is the best shallow candidate among tested layers: it keeps substantially more of the default DraftModel selected tokens than layers 0-2 while still reducing score-forward time by a large margin.
- layer0-2 are very fast but lose too much selector overlap, so they are risky as direct residual selectors.
- Next step: run full pipeline for `offline fixed 0.15 + shallow layer4 residual 0.05`, and compare against full Draft residual 0.05 in the unified main table.
