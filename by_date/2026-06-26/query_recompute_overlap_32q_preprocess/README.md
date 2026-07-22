
## 运行 20260626_212157

- 时间: 2026-06-26T21:21:57
- 目的: 验证固定同一文档时，使用 preprocess KV 的主模型 QK selector 在不同 query 下是否稳定。
- 命令参数: `example_start=0`, `num_examples=20`, `max_passages_list=[10]`, `rate_list=[0.1, 0.2, 0.3, 0.5]`, `native_count=11`, `control_count=21`, `block_size=16`
- preprocess KV: `/raid/home/hming/fusionrag-pca-top1-top10-cache-20/data/musique-pca-subset-preprocess-10-revert_rope-True/Qwen2.5-7B-Instruct`
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - score 来源: preprocess top-10 KV cache 中的 context K，与 query hidden state 计算逐层 QK attention，softmax 后对 layer/head/query token 求和。
  - `score_cosine`: query importance score 向量余弦相似度。
  - `score_l2_rel`: score 相对 L2 差异，越小说明分数越接近。
  - `chunk_cosine`: 选中 token 落到各 chunk 的分布相似度。
- 完成 PreprocessKV example 0, passages=10, rate=0.1: token_jaccard_mean=0.6644, block_jaccard_mean=0.7820, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 0, passages=10, rate=0.2: token_jaccard_mean=0.6975, block_jaccard_mean=0.8858, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 0, passages=10, rate=0.3: token_jaccard_mean=0.7416, block_jaccard_mean=0.9449, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 0, passages=10, rate=0.5: token_jaccard_mean=0.8219, block_jaccard_mean=0.9836, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 1, passages=10, rate=0.1: token_jaccard_mean=0.6392, block_jaccard_mean=0.7820, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 1, passages=10, rate=0.2: token_jaccard_mean=0.6794, block_jaccard_mean=0.8900, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 1, passages=10, rate=0.3: token_jaccard_mean=0.7322, block_jaccard_mean=0.9503, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 1, passages=10, rate=0.5: token_jaccard_mean=0.8171, block_jaccard_mean=0.9932, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 2, passages=10, rate=0.1: token_jaccard_mean=0.7593, block_jaccard_mean=0.8433, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 2, passages=10, rate=0.2: token_jaccard_mean=0.7699, block_jaccard_mean=0.8945, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 2, passages=10, rate=0.3: token_jaccard_mean=0.8011, block_jaccard_mean=0.9487, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 2, passages=10, rate=0.5: token_jaccard_mean=0.8479, block_jaccard_mean=0.9910, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 3, passages=10, rate=0.1: token_jaccard_mean=0.6279, block_jaccard_mean=0.7900, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 3, passages=10, rate=0.2: token_jaccard_mean=0.6669, block_jaccard_mean=0.8906, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 3, passages=10, rate=0.3: token_jaccard_mean=0.7160, block_jaccard_mean=0.9414, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 3, passages=10, rate=0.5: token_jaccard_mean=0.7976, block_jaccard_mean=0.9825, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 4, passages=10, rate=0.1: token_jaccard_mean=0.6804, block_jaccard_mean=0.7863, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 4, passages=10, rate=0.2: token_jaccard_mean=0.7256, block_jaccard_mean=0.9086, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 4, passages=10, rate=0.3: token_jaccard_mean=0.7646, block_jaccard_mean=0.9584, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 4, passages=10, rate=0.5: token_jaccard_mean=0.8413, block_jaccard_mean=0.9932, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 5, passages=10, rate=0.1: token_jaccard_mean=0.6521, block_jaccard_mean=0.7610, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 5, passages=10, rate=0.2: token_jaccard_mean=0.6810, block_jaccard_mean=0.8749, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 5, passages=10, rate=0.3: token_jaccard_mean=0.7220, block_jaccard_mean=0.9415, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 5, passages=10, rate=0.5: token_jaccard_mean=0.7973, block_jaccard_mean=0.9883, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 6, passages=10, rate=0.1: token_jaccard_mean=0.7155, block_jaccard_mean=0.8143, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 6, passages=10, rate=0.2: token_jaccard_mean=0.7595, block_jaccard_mean=0.8692, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 6, passages=10, rate=0.3: token_jaccard_mean=0.7955, block_jaccard_mean=0.9198, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 6, passages=10, rate=0.5: token_jaccard_mean=0.8405, block_jaccard_mean=0.9797, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 7, passages=10, rate=0.1: token_jaccard_mean=0.6413, block_jaccard_mean=0.8045, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 7, passages=10, rate=0.2: token_jaccard_mean=0.6820, block_jaccard_mean=0.9047, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 7, passages=10, rate=0.3: token_jaccard_mean=0.7219, block_jaccard_mean=0.9508, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 7, passages=10, rate=0.5: token_jaccard_mean=0.8010, block_jaccard_mean=0.9844, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 8, passages=10, rate=0.1: token_jaccard_mean=0.6710, block_jaccard_mean=0.7969, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 8, passages=10, rate=0.2: token_jaccard_mean=0.6920, block_jaccard_mean=0.8703, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 8, passages=10, rate=0.3: token_jaccard_mean=0.7301, block_jaccard_mean=0.9276, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 8, passages=10, rate=0.5: token_jaccard_mean=0.8152, block_jaccard_mean=0.9867, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 9, passages=10, rate=0.1: token_jaccard_mean=0.6663, block_jaccard_mean=0.8113, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 9, passages=10, rate=0.2: token_jaccard_mean=0.6850, block_jaccard_mean=0.8914, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 9, passages=10, rate=0.3: token_jaccard_mean=0.7379, block_jaccard_mean=0.9372, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 9, passages=10, rate=0.5: token_jaccard_mean=0.8159, block_jaccard_mean=0.9838, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 10, passages=10, rate=0.1: token_jaccard_mean=0.6981, block_jaccard_mean=0.8135, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 10, passages=10, rate=0.2: token_jaccard_mean=0.7163, block_jaccard_mean=0.8897, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 10, passages=10, rate=0.3: token_jaccard_mean=0.7481, block_jaccard_mean=0.9320, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 10, passages=10, rate=0.5: token_jaccard_mean=0.8210, block_jaccard_mean=0.9824, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 11, passages=10, rate=0.1: token_jaccard_mean=0.6459, block_jaccard_mean=0.7770, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 11, passages=10, rate=0.2: token_jaccard_mean=0.6813, block_jaccard_mean=0.9026, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 11, passages=10, rate=0.3: token_jaccard_mean=0.7238, block_jaccard_mean=0.9470, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 11, passages=10, rate=0.5: token_jaccard_mean=0.8026, block_jaccard_mean=0.9877, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 12, passages=10, rate=0.1: token_jaccard_mean=0.6611, block_jaccard_mean=0.7866, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 12, passages=10, rate=0.2: token_jaccard_mean=0.6975, block_jaccard_mean=0.8840, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 12, passages=10, rate=0.3: token_jaccard_mean=0.7508, block_jaccard_mean=0.9540, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 12, passages=10, rate=0.5: token_jaccard_mean=0.8327, block_jaccard_mean=0.9937, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 13, passages=10, rate=0.1: token_jaccard_mean=0.6482, block_jaccard_mean=0.7750, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 13, passages=10, rate=0.2: token_jaccard_mean=0.6967, block_jaccard_mean=0.9193, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 13, passages=10, rate=0.3: token_jaccard_mean=0.7323, block_jaccard_mean=0.9568, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 13, passages=10, rate=0.5: token_jaccard_mean=0.8089, block_jaccard_mean=0.9895, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 14, passages=10, rate=0.1: token_jaccard_mean=0.6533, block_jaccard_mean=0.7824, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 14, passages=10, rate=0.2: token_jaccard_mean=0.6784, block_jaccard_mean=0.8876, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 14, passages=10, rate=0.3: token_jaccard_mean=0.7118, block_jaccard_mean=0.9450, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 14, passages=10, rate=0.5: token_jaccard_mean=0.7841, block_jaccard_mean=0.9934, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 15, passages=10, rate=0.1: token_jaccard_mean=0.6755, block_jaccard_mean=0.8329, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 15, passages=10, rate=0.2: token_jaccard_mean=0.7369, block_jaccard_mean=0.9156, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 15, passages=10, rate=0.3: token_jaccard_mean=0.7772, block_jaccard_mean=0.9466, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 15, passages=10, rate=0.5: token_jaccard_mean=0.8445, block_jaccard_mean=0.9798, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 16, passages=10, rate=0.1: token_jaccard_mean=0.6608, block_jaccard_mean=0.7853, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 16, passages=10, rate=0.2: token_jaccard_mean=0.7236, block_jaccard_mean=0.8732, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 16, passages=10, rate=0.3: token_jaccard_mean=0.7602, block_jaccard_mean=0.9117, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 16, passages=10, rate=0.5: token_jaccard_mean=0.8327, block_jaccard_mean=0.9756, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 17, passages=10, rate=0.1: token_jaccard_mean=0.6961, block_jaccard_mean=0.8104, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 17, passages=10, rate=0.2: token_jaccard_mean=0.7089, block_jaccard_mean=0.8964, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 17, passages=10, rate=0.3: token_jaccard_mean=0.7523, block_jaccard_mean=0.9443, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 17, passages=10, rate=0.5: token_jaccard_mean=0.8267, block_jaccard_mean=0.9868, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 18, passages=10, rate=0.1: token_jaccard_mean=0.6874, block_jaccard_mean=0.8178, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 18, passages=10, rate=0.2: token_jaccard_mean=0.7166, block_jaccard_mean=0.9223, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 18, passages=10, rate=0.3: token_jaccard_mean=0.7526, block_jaccard_mean=0.9566, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 18, passages=10, rate=0.5: token_jaccard_mean=0.8239, block_jaccard_mean=0.9857, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 19, passages=10, rate=0.1: token_jaccard_mean=0.6889, block_jaccard_mean=0.8428, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 19, passages=10, rate=0.2: token_jaccard_mean=0.7203, block_jaccard_mean=0.8958, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 19, passages=10, rate=0.3: token_jaccard_mean=0.7574, block_jaccard_mean=0.9445, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 19, passages=10, rate=0.5: token_jaccard_mean=0.8277, block_jaccard_mean=0.9919, score_cosine_mean=1.0000, score_bad_value_total=0

### PreprocessKV 汇总: passages=10, rate=0.1

- 输出: `preprocesskv_batch_20260626_212157_20examples_10passages_rate0.1.json`
- 耗时: 434.3s
- token_jaccard mean over examples: 0.6716 (min=0.6279, max=0.7593)
- block_jaccard mean over examples: 0.7998 (min=0.7610, max=0.8433)
- score_cosine mean over examples: 1.0000 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1369 (min=0.1139, max=0.1873)

### PreprocessKV 汇总: passages=10, rate=0.2

- 输出: `preprocesskv_batch_20260626_212157_20examples_10passages_rate0.2.json`
- 耗时: 434.8s
- token_jaccard mean over examples: 0.7058 (min=0.6669, max=0.7699)
- block_jaccard mean over examples: 0.8933 (min=0.8692, max=0.9223)
- score_cosine mean over examples: 1.0000 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1369 (min=0.1139, max=0.1873)

### PreprocessKV 汇总: passages=10, rate=0.3

- 输出: `preprocesskv_batch_20260626_212157_20examples_10passages_rate0.3.json`
- 耗时: 435.4s
- token_jaccard mean over examples: 0.7465 (min=0.7118, max=0.8011)
- block_jaccard mean over examples: 0.9430 (min=0.9117, max=0.9584)
- score_cosine mean over examples: 1.0000 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1369 (min=0.1139, max=0.1873)

### PreprocessKV 汇总: passages=10, rate=0.5

- 输出: `preprocesskv_batch_20260626_212157_20examples_10passages_rate0.5.json`
- 耗时: 436.4s
- token_jaccard mean over examples: 0.8200 (min=0.7841, max=0.8479)
- block_jaccard mean over examples: 0.9866 (min=0.9756, max=0.9937)
- score_cosine mean over examples: 1.0000 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1369 (min=0.1139, max=0.1873)

### PreprocessKV 运行 20260626_212157 结束

- 全量输出: `preprocesskv_batch_20260626_212157_all_cases.json`
