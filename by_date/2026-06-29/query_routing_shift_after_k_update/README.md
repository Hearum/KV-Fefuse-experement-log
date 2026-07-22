# Query Routing Shift after K/V Update

## 目的

验证 K update 是否改变 query 对文档 chunk 的 attention routing，从而解释 `k_only` 常强于 `v_only` 的现象。

## 设置

- 模型: Qwen2.5-7B-Instruct
- KV 初始状态: preprocess top-10 KV cache
- 代表性 case: 4 个，来自 `kv` 正确而 `v_only` 容易错误的样本。
- mode: `preprocess`, `k_only`, `v_only`, `kv`
- 指标: query attention mass 在 selected tokens、包含 gold answer 的 chunk、包含 wrong prediction 的 chunk 上的占比。

## 汇总表

| case | mode | selected mass | gt chunk mass | wrong chunk mass | gt last-layer | wrong last-layer | top chunks |
|---|---|---:|---:|---:|---:|---:|---|
| art_brut_lead_singer | preprocess | 0.1758 | 0.0612 | 0.0335 | 0.1312 | 0.0751 | `[{"doc_chunk": 5, "mass": 0.03364921361207962}, {"doc_chunk": 10, "mass": 0.03352007269859314}, {"doc_chunk": 6, "mass": 0.03248543664813042}, {"doc_chunk": 2, "mass": 0.03058335743844509}, {"doc_chunk": 12, "mass": 0.03058335743844509}]` |
| art_brut_lead_singer | k_only | 0.2264 | 0.1051 | 0.0345 | 0.1186 | 0.1240 | `[{"doc_chunk": 12, "mass": 0.08548138290643692}, {"doc_chunk": 11, "mass": 0.03968803212046623}, {"doc_chunk": 10, "mass": 0.034470222890377045}, {"doc_chunk": 5, "mass": 0.029279440641403198}, {"doc_chunk": 6, "mass": 0.025882072746753693}]` |
| art_brut_lead_singer | v_only | 0.1625 | 0.0377 | 0.0367 | 0.0568 | 0.0888 | `[{"doc_chunk": 6, "mass": 0.03684736043214798}, {"doc_chunk": 10, "mass": 0.036724671721458435}, {"doc_chunk": 5, "mass": 0.03096088394522667}, {"doc_chunk": 3, "mass": 0.027178989723324776}, {"doc_chunk": 9, "mass": 0.02662021480500698}]` |
| art_brut_lead_singer | kv | 0.2299 | 0.1184 | 0.0310 | 0.1603 | 0.0785 | `[{"doc_chunk": 12, "mass": 0.09352483600378036}, {"doc_chunk": 11, "mass": 0.03913489356637001}, {"doc_chunk": 10, "mass": 0.03104846179485321}, {"doc_chunk": 2, "mass": 0.024858007207512856}, {"doc_chunk": 5, "mass": 0.023674756288528442}]` |
| national_dream_author | preprocess | 0.1558 | 0.0295 | 0.1017 | 0.0421 | 0.2438 | `[{"doc_chunk": 8, "mass": 0.039134226739406586}, {"doc_chunk": 5, "mass": 0.036344967782497406}, {"doc_chunk": 7, "mass": 0.034510593861341476}, {"doc_chunk": 10, "mass": 0.03179275617003441}, {"doc_chunk": 3, "mass": 0.03159257024526596}]` |
| national_dream_author | k_only | 0.1672 | 0.0368 | 0.0892 | 0.0632 | 0.2460 | `[{"doc_chunk": 10, "mass": 0.06839559227228165}, {"doc_chunk": 1, "mass": 0.03680397570133209}, {"doc_chunk": 8, "mass": 0.034013934433460236}, {"doc_chunk": 7, "mass": 0.03385739400982857}, {"doc_chunk": 2, "mass": 0.029510335996747017}]` |
| national_dream_author | v_only | 0.1543 | 0.0275 | 0.1023 | 0.0455 | 0.2143 | `[{"doc_chunk": 5, "mass": 0.03809010609984398}, {"doc_chunk": 8, "mass": 0.0379372239112854}, {"doc_chunk": 7, "mass": 0.034084826707839966}, {"doc_chunk": 3, "mass": 0.030734142288565636}, {"doc_chunk": 10, "mass": 0.030553951859474182}]` |
| national_dream_author | kv | 0.1628 | 0.0351 | 0.0862 | 0.0619 | 0.2380 | `[{"doc_chunk": 10, "mass": 0.06576452404260635}, {"doc_chunk": 1, "mass": 0.03514264151453972}, {"doc_chunk": 7, "mass": 0.03191584348678589}, {"doc_chunk": 8, "mass": 0.031602926552295685}, {"doc_chunk": 2, "mass": 0.030067235231399536}]` |
| bartrams_bridge_water | preprocess | 0.1466 | 0.0815 | 0.0212 | 0.1520 | 0.0313 | `[{"doc_chunk": 1, "mass": 0.04359697550535202}, {"doc_chunk": 10, "mass": 0.0379292331635952}, {"doc_chunk": 7, "mass": 0.03258746489882469}, {"doc_chunk": 8, "mass": 0.031038830056786537}, {"doc_chunk": 9, "mass": 0.02987193875014782}]` |
| bartrams_bridge_water | k_only | 0.1184 | 0.1141 | 0.0188 | 0.2260 | 0.0224 | `[{"doc_chunk": 10, "mass": 0.05988989397883415}, {"doc_chunk": 1, "mass": 0.05423467233777046}, {"doc_chunk": 4, "mass": 0.028571821749210358}, {"doc_chunk": 7, "mass": 0.027856459841132164}, {"doc_chunk": 5, "mass": 0.026798170059919357}]` |
| bartrams_bridge_water | v_only | 0.1453 | 0.0812 | 0.0210 | 0.1529 | 0.0288 | `[{"doc_chunk": 1, "mass": 0.04253893718123436}, {"doc_chunk": 10, "mass": 0.038647815585136414}, {"doc_chunk": 7, "mass": 0.03261761739850044}, {"doc_chunk": 8, "mass": 0.03089636191725731}, {"doc_chunk": 5, "mass": 0.030059123411774635}]` |
| bartrams_bridge_water | kv | 0.1215 | 0.1168 | 0.0193 | 0.2321 | 0.0171 | `[{"doc_chunk": 10, "mass": 0.06422960013151169}, {"doc_chunk": 1, "mass": 0.052606258541345596}, {"doc_chunk": 4, "mass": 0.028705330565571785}, {"doc_chunk": 7, "mass": 0.026423171162605286}, {"doc_chunk": 5, "mass": 0.02527538500726223}]` |
| first_african_american_candidate | preprocess | 0.1608 | 0.0521 | 0.1360 | 0.1048 | 0.2449 | `[{"doc_chunk": 2, "mass": 0.04821294546127319}, {"doc_chunk": 7, "mass": 0.04111265391111374}, {"doc_chunk": 3, "mass": 0.03798164054751396}, {"doc_chunk": 5, "mass": 0.03110845759510994}, {"doc_chunk": 1, "mass": 0.030284367501735687}]` |
| first_african_american_candidate | k_only | 0.1888 | 0.0564 | 0.1363 | 0.1418 | 0.2581 | `[{"doc_chunk": 10, "mass": 0.058812301605939865}, {"doc_chunk": 2, "mass": 0.048242006450891495}, {"doc_chunk": 7, "mass": 0.04025474935770035}, {"doc_chunk": 3, "mass": 0.034084927290678024}, {"doc_chunk": 1, "mass": 0.02932368963956833}]` |
| first_african_american_candidate | v_only | 0.1583 | 0.0518 | 0.1356 | 0.0914 | 0.2493 | `[{"doc_chunk": 2, "mass": 0.046897124499082565}, {"doc_chunk": 7, "mass": 0.041590895503759384}, {"doc_chunk": 3, "mass": 0.03761938586831093}, {"doc_chunk": 1, "mass": 0.03046782687306404}, {"doc_chunk": 4, "mass": 0.030236339196562767}]` |
| first_african_american_candidate | kv | 0.1888 | 0.0572 | 0.1379 | 0.1294 | 0.2729 | `[{"doc_chunk": 10, "mass": 0.058161649852991104}, {"doc_chunk": 2, "mass": 0.04943956807255745}, {"doc_chunk": 7, "mass": 0.04092513769865036}, {"doc_chunk": 3, "mass": 0.03555431216955185}, {"doc_chunk": 1, "mass": 0.030879270285367966}]` |

## 输出

- `routing_shift_summary.csv`
- `routing_shift_detail.json`
