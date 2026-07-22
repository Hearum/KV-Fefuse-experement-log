# Low-Cost Draft Selector Probe

## Stage 1: 20-example Selector Alignment

Default is the original full DraftModel selector using the latter half of DraftModel layers. Other methods use partial or sparse DraftModel layers through `FUSIONRAG_DRAFT_SCORE_LAYERS`.

| method | n | token recall vs default | token Jaccard | chunk recall | residual recall | score forward time(s) |
|---|---:|---:|---:|---:|---:|---:|
| default | 32 | 100.00% | 100.00% | 100.00% | 100.00% | 0.1452 |
| first4 | 32 | 47.89% | 31.94% | 98.10% | 54.52% | 0.0210 |
| first8 | 32 | 63.99% | 47.72% | 98.86% | 60.36% | 0.0366 |
| first12 | 32 | 66.56% | 50.48% | 100.00% | 60.08% | 0.0538 |
| layer4_8_12 | 32 | 65.10% | 48.71% | 100.00% | 56.40% | 0.0536 |
| layer8_12_16 | 32 | 66.59% | 50.34% | 100.00% | 57.18% | 0.0689 |
| middle | 32 | 72.21% | 57.04% | 100.00% | 65.62% | 0.0764 |
| last | 32 | 56.61% | 39.80% | 94.63% | 58.09% | 0.1401 |

### 20-example Answer Sanity

| method | Main Acc | Sub Acc |
|---|---:|---:|
| default | 15/18 (83.33%) | 29/32 (90.62%) |
| first4 | 10/18 (55.56%) | 21/32 (65.62%) |
| first8 | 14/18 (77.78%) | 26/32 (81.25%) |
| first12 | 16/18 (88.89%) | 29/32 (90.62%) |
| layer4_8_12 | 14/18 (77.78%) | 27/32 (84.38%) |
| layer8_12_16 | 15/18 (83.33%) | 29/32 (90.62%) |
| middle | 14/18 (77.78%) | 28/32 (87.50%) |
| last | 12/18 (66.67%) | 25/32 (78.12%) |

## Stage 1 Takeaway

This table is a selector-quality screen. Full accuracy should only be run for candidates that preserve high overlap with the default DraftModel selector while reducing score-forward time.

## Stage 2: Full Pipeline Accuracy

Configuration: offline fixed set `draft_smart_frequency_global` at 10% plus partial Draft residual at 5%.

| method | Main Acc | Sub Acc | residual select(s) | prompt eval(s) |
|---|---:|---:|---:|---:|
| layer4 residual5 | 92/135 (68.15%) | 198/250 (79.20%) | 0.0235 | 0.8817 |
| first12 residual5 | 92/135 (68.15%) | 197/250 (78.80%) | 0.0524 | 0.9139 |
| middle residual5 | 95/135 (70.37%) | 202/250 (80.80%) | 0.0743 | 0.9328 |
| full Draft residual5 | 97/135 (71.85%) | 208/250 (83.20%) | 0.1406 | 1.0007 |

Takeaway: `middle` is the best partial Draft tradeoff in this run, but it still leaves a clear quality gap to full Draft residual. `first12` is not better than the cheaper layer4 setting.
