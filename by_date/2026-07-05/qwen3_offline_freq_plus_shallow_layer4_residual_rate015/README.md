# Qwen3 Offline Frequency + Shallow Layer4 Draft Residual

## 设置

- model: Qwen3-32B
- dataset: MuSiQue reflect pipeline, 200 examples
- base offline set: `draft_smart_frequency_global`, rate=0.15
- online residual: DraftModel, residual rate=0.05
- shallow setting: `FUSIONRAG_DRAFT_SCORE_LAYERS=4`, so residual DraftModel selector uses only layer 4 score and stops after layer 4.
- judge: GLM-5.2

## 结果

| method | Main Acc | Sub Acc | F1 | EM | residual tokens | residual select time | prompt eval time |
|---|---:|---:|---:|---:|---:|---:|---:|
| freq_shallow_layer4_residual005 | 96/135 (71.11%) | 204/250 (81.60%) | 0.2034 | 0.0120 | 70.8 | 0.0238 | 0.9929 |

## 对比基线

- full Draft residual counterpart `freq_draft005`: Main 103/135 (76.30%), Sub 217/250 (86.80%), residual select time about 0.1410s.
- pure offline `draft_smart_frequency_global`: Main 94/135 (69.63%), Sub 203/250 (81.20%).
- online Draft rate=0.15: Main 99/135 (73.33%), Sub 209/248 (84.27%).
- full recompute rate=1: Main 105/135 (77.78%), Sub 215/250 (86.00%).

## 初步结论

Shallow layer4 residual substantially reduces online residual selection time, but the quality gain over pure offline is small and far below full Draft residual. Current single-layer shallow selector is therefore not sufficient; the next variants should test first-k-layer aggregation, slightly deeper layers, or a trained lightweight predictor.
