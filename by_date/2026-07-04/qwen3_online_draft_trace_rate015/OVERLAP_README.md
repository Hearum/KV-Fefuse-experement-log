# Online Draft vs Offline Hybrid Token Overlap

| subset | n subq | Jaccard | online recall by offline | offline precision vs online | online tokens | offline tokens |
|---|---:|---:|---:|---:|---:|---:|
| all | 250 | 0.3004 | 0.4248 | 0.5008 | 244.2 | 208.7 |
| draft_correct_offline_wrong | 38 | 0.2902 | 0.4100 | 0.4917 | 220.3 | 184.8 |
| offline_correct_draft_wrong | 30 | 0.3046 | 0.4282 | 0.5051 | 234.8 | 200.2 |
| draft_correct_qk_wrong | 47 | 0.2936 | 0.4158 | 0.4939 | 236.7 | 201.3 |
| qk_correct_draft_wrong | 15 | 0.3204 | 0.4538 | 0.5169 | 298.9 | 263.6 |

Interpretation: `online recall by offline` is the fraction of online draft-selected tokens also covered by offline hybrid. Low values mean offline misses many online draft choices.
