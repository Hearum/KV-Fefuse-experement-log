# Full Accuracy Rerun Baselines

完整 `data/result_reflect.json`，同一套 reflect pipeline，同一套 GLM-5.2 judge。

| label | rate | Main Acc | Sub Acc | F1 | EM | prompt/full-prefill mean(s) | storage mean(s) | selection mean(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rate1_full_attention | 1.00 | 104/135 (77.04%) | 218/250 (87.20%) | 0.5666 | 0.2280 | 0.1167 | 0.0000 | 0.0000 |
| online_qk_rate030 | 0.30 | 94/135 (69.63%) | 199/250 (79.60%) | 0.5083 | 0.1920 | 0.2584 | 0.0253 | 0.1043 |
| online_qk_rate015 | 0.15 | 84/135 (62.22%) | 189/250 (75.60%) | 0.4878 | 0.2120 | 0.2311 | 0.0249 | 0.1032 |
| rate0_no_doc_recompute | 0.00 | 75/135 (55.56%) | 175/250 (70.00%) | 0.4510 | 0.1600 | 0.1011 | 0.0250 | 0.0000 |

说明：`rate=1` 是 full attention/full recompute baseline，不走 cache reuse selection；`prompt/full-prefill mean(s)` 对 rate=1 是完整上下文 prefill，对 FusionRAG 路径是脚本打印的 prompt eval duration。

## Offline Fixed Set 现有结果

下面是之前 `rate=0.15` 的 offline fixed set 方法对比，输出目录：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/phase2_selector_methods_rate015/
```

注意：这组目前是 `20 examples` 的真实生成小样本结果，不是完整 `data/result_reflect.json` accuracy。它用于判断 offline set 方法是否有希望；最终结论仍需按完整数据集 + GLM judge 重跑。

| selector | n | ROUGE-1 | EM | TTFT(s) | storage(s) | selection(s) | update+query(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| online_fusionrag_qk | 20 | 0.2434 | 0.2000 | 0.4113 | 0.0604 | 0.1136 | 0.2374 |
| offline_qk_freq | 20 | 0.2492 | 0.2000 | 0.2720 | 0.0568 | 0.0000 | 0.2153 |
| offline_qk_mean | 20 | 0.2493 | 0.2000 | 0.2717 | 0.0570 | 0.0000 | 0.2148 |
| offline_draft_freq | 20 | 0.2492 | 0.2000 | 0.2716 | 0.0567 | 0.0000 | 0.2149 |
| offline_draft_mean | 20 | 0.2459 | 0.2000 | 0.2722 | 0.0573 | 0.0000 | 0.2150 |
| position_boundary | 20 | 0.1870 | 0.1500 | 0.2715 | 0.0564 | 0.0000 | 0.2151 |
| position_tail | 20 | 0.2419 | 0.2000 | 0.2727 | 0.0564 | 0.0000 | 0.2163 |
| random_chunk | 20 | 0.2259 | 0.2000 | 0.2723 | 0.0567 | 0.0000 | 0.2156 |

当前解读：

```text
1. offline_qk_freq / offline_qk_mean / offline_draft_freq 在小样本上没有明显低于 online QK。
2. offline 方法把 selection 时间从约 0.1136s 降到 0。
3. random_chunk 质量低于 QK/draft offline set，说明不能只是随机固定 token。
4. position_boundary 明显差，position_tail 虽然小样本指标尚可，但之前 set overlap 很低，不能作为主解释。
5. 下一步要把 offline_qk_freq、offline_qk_mean、offline_draft_freq 按完整数据集 + GLM judge 重跑。
```
