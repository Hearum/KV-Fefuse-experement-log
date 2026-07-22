
## DraftModel selector 运行 20260626_184238

- 时间: 2026-06-26T18:42:38
- 目的: 验证固定同一文档时，不同 query 触发的 DraftModel 重算 token 是否稳定。
- 启动参考: `/mnt/qjhs-sh-lab-01/wjh/FusionRAG/run_task.py DraftModel ... --draft_model_path /mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`
- 命令参数: `example_start=0`, `num_examples=20`, `max_passages_list=[3, 5, 10]`, `rate_list=[0.1, 0.2]`, `native_count=8`, `control_count=8`, `block_size=16`
- DraftModel: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct`, layer_selection=`rrf`, rrf_k=18
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - score 来源: draft model query→doc attention，经 `rrf` 聚合；rate 阶段复用同一 score，离线调用 `smart_query_selection`。

## 运行 20260626_184239

- 时间: 2026-06-26T18:42:39
- 目的: 验证固定同一文档时，使用 preprocess KV 的主模型 QK selector 在不同 query 下是否稳定。
- 命令参数: `example_start=0`, `num_examples=20`, `max_passages_list=[3, 5, 10]`, `rate_list=[0.1, 0.2]`, `native_count=8`, `control_count=8`, `block_size=16`
- preprocess KV: `/raid/home/hming/fusionrag-pca-top1-top10-cache-20/data/musique-pca-subset-preprocess-10-revert_rope-True/Qwen2.5-7B-Instruct`
- 指标:
  - `token_jaccard`: 不同 query 选中 token 集合 Jaccard。
  - `block_jaccard`: 选中 16-token block 集合 Jaccard。
  - score 来源: preprocess top-10 KV cache 中的 context K，与 query hidden state 计算逐层 QK attention，softmax 后对 layer/head/query token 求和。
  - `score_cosine`: query importance score 向量余弦相似度。
  - `score_l2_rel`: score 相对 L2 差异，越小说明分数越接近。
  - `chunk_cosine`: 选中 token 落到各 chunk 的分布相似度。
- 完成 DraftModel example 0, passages=3, rate=0.1: token_jaccard_mean=0.7196, block_jaccard_mean=0.7539, score_cosine_mean=0.9689, score_bad_value_total=0
- 完成 DraftModel example 0, passages=3, rate=0.2: token_jaccard_mean=0.6847, block_jaccard_mean=0.7787, score_cosine_mean=0.9689, score_bad_value_total=0
- 完成 PreprocessKV example 0, passages=3, rate=0.1: token_jaccard_mean=0.7009, block_jaccard_mean=0.8010, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 0, passages=3, rate=0.2: token_jaccard_mean=0.7240, block_jaccard_mean=0.9163, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 1, passages=3, rate=0.1: token_jaccard_mean=0.5821, block_jaccard_mean=0.6899, score_cosine_mean=0.9358, score_bad_value_total=0
- 完成 PreprocessKV example 1, passages=3, rate=0.1: token_jaccard_mean=0.6684, block_jaccard_mean=0.8380, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 1, passages=3, rate=0.2: token_jaccard_mean=0.6305, block_jaccard_mean=0.7099, score_cosine_mean=0.9358, score_bad_value_total=0
- 完成 PreprocessKV example 1, passages=3, rate=0.2: token_jaccard_mean=0.6916, block_jaccard_mean=0.8947, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 2, passages=3, rate=0.1: token_jaccard_mean=0.7894, block_jaccard_mean=0.8860, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 2, passages=3, rate=0.2: token_jaccard_mean=0.7999, block_jaccard_mean=0.9484, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 2, passages=3, rate=0.1: token_jaccard_mean=0.7655, block_jaccard_mean=0.8372, score_cosine_mean=0.9908, score_bad_value_total=0
- 完成 DraftModel example 2, passages=3, rate=0.2: token_jaccard_mean=0.7695, block_jaccard_mean=0.8625, score_cosine_mean=0.9908, score_bad_value_total=0
- 完成 PreprocessKV example 3, passages=3, rate=0.1: token_jaccard_mean=0.7453, block_jaccard_mean=0.8906, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 3, passages=3, rate=0.2: token_jaccard_mean=0.7997, block_jaccard_mean=0.9314, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 3, passages=3, rate=0.1: token_jaccard_mean=0.7576, block_jaccard_mean=0.7747, score_cosine_mean=0.9708, score_bad_value_total=0
- 完成 DraftModel example 3, passages=3, rate=0.2: token_jaccard_mean=0.7511, block_jaccard_mean=0.8242, score_cosine_mean=0.9708, score_bad_value_total=0
- 完成 PreprocessKV example 4, passages=3, rate=0.1: token_jaccard_mean=0.7080, block_jaccard_mean=0.8318, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 4, passages=3, rate=0.2: token_jaccard_mean=0.7569, block_jaccard_mean=0.9076, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 4, passages=3, rate=0.1: token_jaccard_mean=0.6908, block_jaccard_mean=0.7797, score_cosine_mean=0.9760, score_bad_value_total=0
- 完成 DraftModel example 4, passages=3, rate=0.2: token_jaccard_mean=0.7078, block_jaccard_mean=0.8307, score_cosine_mean=0.9760, score_bad_value_total=0
- 完成 PreprocessKV example 5, passages=3, rate=0.1: token_jaccard_mean=0.6981, block_jaccard_mean=0.8305, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 5, passages=3, rate=0.2: token_jaccard_mean=0.7125, block_jaccard_mean=0.9294, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 5, passages=3, rate=0.1: token_jaccard_mean=0.6581, block_jaccard_mean=0.7097, score_cosine_mean=0.9801, score_bad_value_total=0
- 完成 DraftModel example 5, passages=3, rate=0.2: token_jaccard_mean=0.7107, block_jaccard_mean=0.7940, score_cosine_mean=0.9801, score_bad_value_total=0
- 完成 PreprocessKV example 6, passages=3, rate=0.1: token_jaccard_mean=0.7502, block_jaccard_mean=0.8498, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 6, passages=3, rate=0.2: token_jaccard_mean=0.7662, block_jaccard_mean=0.9160, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 6, passages=3, rate=0.1: token_jaccard_mean=0.6810, block_jaccard_mean=0.7430, score_cosine_mean=0.9726, score_bad_value_total=0
- 完成 DraftModel example 6, passages=3, rate=0.2: token_jaccard_mean=0.7058, block_jaccard_mean=0.7811, score_cosine_mean=0.9726, score_bad_value_total=0
- 完成 PreprocessKV example 7, passages=3, rate=0.1: token_jaccard_mean=0.6991, block_jaccard_mean=0.8181, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 7, passages=3, rate=0.2: token_jaccard_mean=0.7308, block_jaccard_mean=0.9092, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 8, passages=3, rate=0.1: token_jaccard_mean=0.7161, block_jaccard_mean=0.8669, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 8, passages=3, rate=0.2: token_jaccard_mean=0.7428, block_jaccard_mean=0.9147, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 7, passages=3, rate=0.1: token_jaccard_mean=0.6493, block_jaccard_mean=0.7168, score_cosine_mean=0.9705, score_bad_value_total=0
- 完成 DraftModel example 7, passages=3, rate=0.2: token_jaccard_mean=0.6562, block_jaccard_mean=0.7733, score_cosine_mean=0.9705, score_bad_value_total=0
- 完成 PreprocessKV example 9, passages=3, rate=0.1: token_jaccard_mean=0.6700, block_jaccard_mean=0.8339, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 9, passages=3, rate=0.2: token_jaccard_mean=0.7239, block_jaccard_mean=0.9242, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 8, passages=3, rate=0.1: token_jaccard_mean=0.7592, block_jaccard_mean=0.8161, score_cosine_mean=0.9776, score_bad_value_total=0
- 完成 DraftModel example 8, passages=3, rate=0.2: token_jaccard_mean=0.7376, block_jaccard_mean=0.8335, score_cosine_mean=0.9776, score_bad_value_total=0
- 完成 PreprocessKV example 10, passages=3, rate=0.1: token_jaccard_mean=0.7054, block_jaccard_mean=0.8085, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 10, passages=3, rate=0.2: token_jaccard_mean=0.7516, block_jaccard_mean=0.9303, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 9, passages=3, rate=0.1: token_jaccard_mean=0.5639, block_jaccard_mean=0.6425, score_cosine_mean=0.9476, score_bad_value_total=0
- 完成 DraftModel example 9, passages=3, rate=0.2: token_jaccard_mean=0.6626, block_jaccard_mean=0.7865, score_cosine_mean=0.9476, score_bad_value_total=0
- 完成 PreprocessKV example 11, passages=3, rate=0.1: token_jaccard_mean=0.7092, block_jaccard_mean=0.7876, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 11, passages=3, rate=0.2: token_jaccard_mean=0.7052, block_jaccard_mean=0.8979, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 12, passages=3, rate=0.1: token_jaccard_mean=0.7501, block_jaccard_mean=0.8801, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 12, passages=3, rate=0.2: token_jaccard_mean=0.7512, block_jaccard_mean=0.9447, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 10, passages=3, rate=0.1: token_jaccard_mean=0.6848, block_jaccard_mean=0.7733, score_cosine_mean=0.9525, score_bad_value_total=0
- 完成 DraftModel example 10, passages=3, rate=0.2: token_jaccard_mean=0.7232, block_jaccard_mean=0.8384, score_cosine_mean=0.9525, score_bad_value_total=0
- 完成 PreprocessKV example 13, passages=3, rate=0.1: token_jaccard_mean=0.7227, block_jaccard_mean=0.8152, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 13, passages=3, rate=0.2: token_jaccard_mean=0.7556, block_jaccard_mean=0.8937, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 11, passages=3, rate=0.1: token_jaccard_mean=0.6432, block_jaccard_mean=0.6827, score_cosine_mean=0.9621, score_bad_value_total=0
- 完成 DraftModel example 11, passages=3, rate=0.2: token_jaccard_mean=0.7596, block_jaccard_mean=0.8680, score_cosine_mean=0.9621, score_bad_value_total=0
- 完成 PreprocessKV example 14, passages=3, rate=0.1: token_jaccard_mean=0.6848, block_jaccard_mean=0.7847, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 14, passages=3, rate=0.2: token_jaccard_mean=0.7144, block_jaccard_mean=0.8999, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 12, passages=3, rate=0.1: token_jaccard_mean=0.7665, block_jaccard_mean=0.8185, score_cosine_mean=0.9806, score_bad_value_total=0
- 完成 DraftModel example 12, passages=3, rate=0.2: token_jaccard_mean=0.7445, block_jaccard_mean=0.8474, score_cosine_mean=0.9806, score_bad_value_total=0
- 完成 PreprocessKV example 15, passages=3, rate=0.1: token_jaccard_mean=0.7527, block_jaccard_mean=0.8796, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 15, passages=3, rate=0.2: token_jaccard_mean=0.7946, block_jaccard_mean=0.9335, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 DraftModel example 13, passages=3, rate=0.1: token_jaccard_mean=0.7396, block_jaccard_mean=0.7781, score_cosine_mean=0.9761, score_bad_value_total=0
- 完成 DraftModel example 13, passages=3, rate=0.2: token_jaccard_mean=0.7061, block_jaccard_mean=0.8187, score_cosine_mean=0.9761, score_bad_value_total=0
- 完成 PreprocessKV example 16, passages=3, rate=0.1: token_jaccard_mean=0.6841, block_jaccard_mean=0.8209, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 16, passages=3, rate=0.2: token_jaccard_mean=0.7371, block_jaccard_mean=0.9113, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 DraftModel example 14, passages=3, rate=0.1: token_jaccard_mean=0.6169, block_jaccard_mean=0.7120, score_cosine_mean=0.9584, score_bad_value_total=0
- 完成 DraftModel example 14, passages=3, rate=0.2: token_jaccard_mean=0.6666, block_jaccard_mean=0.8486, score_cosine_mean=0.9584, score_bad_value_total=0
- 完成 PreprocessKV example 17, passages=3, rate=0.1: token_jaccard_mean=0.7072, block_jaccard_mean=0.8458, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 17, passages=3, rate=0.2: token_jaccard_mean=0.7367, block_jaccard_mean=0.9168, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 18, passages=3, rate=0.1: token_jaccard_mean=0.7124, block_jaccard_mean=0.8867, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 18, passages=3, rate=0.2: token_jaccard_mean=0.7343, block_jaccard_mean=0.9384, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 15, passages=3, rate=0.1: token_jaccard_mean=0.6633, block_jaccard_mean=0.7471, score_cosine_mean=0.9674, score_bad_value_total=0
- 完成 DraftModel example 15, passages=3, rate=0.2: token_jaccard_mean=0.6915, block_jaccard_mean=0.8192, score_cosine_mean=0.9674, score_bad_value_total=0
- 完成 PreprocessKV example 19, passages=3, rate=0.1: token_jaccard_mean=0.7316, block_jaccard_mean=0.8389, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 19, passages=3, rate=0.2: token_jaccard_mean=0.7588, block_jaccard_mean=0.8959, score_cosine_mean=1.0000, score_bad_value_total=0

### PreprocessKV 汇总: passages=3, rate=0.1

- 输出: `preprocesskv_batch_20260626_184239_20examples_3passages_rate0.1.json`
- 耗时: 65.8s
- token_jaccard mean over examples: 0.7153 (min=0.6684, max=0.7894)
- block_jaccard mean over examples: 0.8397 (min=0.7847, max=0.8906)
- score_cosine mean over examples: 1.0000 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1303 (min=0.0732, max=0.2100)

### PreprocessKV 汇总: passages=3, rate=0.2

- 输出: `preprocesskv_batch_20260626_184239_20examples_3passages_rate0.2.json`
- 耗时: 65.9s
- token_jaccard mean over examples: 0.7444 (min=0.6916, max=0.7999)
- block_jaccard mean over examples: 0.9177 (min=0.8937, max=0.9484)
- score_cosine mean over examples: 1.0000 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1303 (min=0.0732, max=0.2100)
- 完成 DraftModel example 16, passages=3, rate=0.1: token_jaccard_mean=0.6816, block_jaccard_mean=0.7648, score_cosine_mean=0.9580, score_bad_value_total=0
- 完成 DraftModel example 16, passages=3, rate=0.2: token_jaccard_mean=0.7193, block_jaccard_mean=0.8242, score_cosine_mean=0.9580, score_bad_value_total=0
- 完成 PreprocessKV example 0, passages=5, rate=0.1: token_jaccard_mean=0.6874, block_jaccard_mean=0.8057, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 0, passages=5, rate=0.2: token_jaccard_mean=0.7247, block_jaccard_mean=0.9000, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 17, passages=3, rate=0.1: token_jaccard_mean=0.7201, block_jaccard_mean=0.7411, score_cosine_mean=0.9684, score_bad_value_total=0
- 完成 DraftModel example 17, passages=3, rate=0.2: token_jaccard_mean=0.7508, block_jaccard_mean=0.8536, score_cosine_mean=0.9684, score_bad_value_total=0
- 完成 DraftModel example 18, passages=3, rate=0.1: token_jaccard_mean=0.5395, block_jaccard_mean=0.5945, score_cosine_mean=0.9588, score_bad_value_total=0
- 完成 DraftModel example 18, passages=3, rate=0.2: token_jaccard_mean=0.6579, block_jaccard_mean=0.7630, score_cosine_mean=0.9588, score_bad_value_total=0
- 完成 PreprocessKV example 1, passages=5, rate=0.1: token_jaccard_mean=0.6423, block_jaccard_mean=0.8202, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 1, passages=5, rate=0.2: token_jaccard_mean=0.6864, block_jaccard_mean=0.8827, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 19, passages=3, rate=0.1: token_jaccard_mean=0.7121, block_jaccard_mean=0.8138, score_cosine_mean=0.9605, score_bad_value_total=0
- 完成 DraftModel example 19, passages=3, rate=0.2: token_jaccard_mean=0.7160, block_jaccard_mean=0.8316, score_cosine_mean=0.9605, score_bad_value_total=0

### DraftModel 汇总: passages=3, rate=0.1, mode=rrf

- 输出: `draft_batch_20260626_184238_20examples_3passages_rate0.1_rrf.json`
- 耗时: 78.4s
- token_jaccard mean over examples: 0.6797 (min=0.5395, max=0.7665)
- block_jaccard mean over examples: 0.7445 (min=0.5945, max=0.8372)
- score_cosine mean over examples: 0.9667 (min=0.9358, max=0.9908)
- score_l2_rel mean over examples: 0.2351 (min=0.1296, max=0.3265)

### DraftModel 汇总: passages=3, rate=0.2, mode=rrf

- 输出: `draft_batch_20260626_184238_20examples_3passages_rate0.2_rrf.json`
- 耗时: 78.5s
- token_jaccard mean over examples: 0.7076 (min=0.6305, max=0.7695)
- block_jaccard mean over examples: 0.8144 (min=0.7099, max=0.8680)
- score_cosine mean over examples: 0.9667 (min=0.9358, max=0.9908)
- score_l2_rel mean over examples: 0.2351 (min=0.1296, max=0.3265)
- 完成 PreprocessKV example 2, passages=5, rate=0.1: token_jaccard_mean=0.7625, block_jaccard_mean=0.8970, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 2, passages=5, rate=0.2: token_jaccard_mean=0.7762, block_jaccard_mean=0.9388, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 3, passages=5, rate=0.1: token_jaccard_mean=0.7157, block_jaccard_mean=0.8745, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 3, passages=5, rate=0.2: token_jaccard_mean=0.7420, block_jaccard_mean=0.9013, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 0, passages=5, rate=0.1: token_jaccard_mean=0.6631, block_jaccard_mean=0.7359, score_cosine_mean=0.9618, score_bad_value_total=0
- 完成 DraftModel example 0, passages=5, rate=0.2: token_jaccard_mean=0.6645, block_jaccard_mean=0.7799, score_cosine_mean=0.9618, score_bad_value_total=0
- 完成 PreprocessKV example 4, passages=5, rate=0.1: token_jaccard_mean=0.6934, block_jaccard_mean=0.8119, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 4, passages=5, rate=0.2: token_jaccard_mean=0.7471, block_jaccard_mean=0.9151, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 1, passages=5, rate=0.1: token_jaccard_mean=0.6005, block_jaccard_mean=0.6287, score_cosine_mean=0.9466, score_bad_value_total=0
- 完成 DraftModel example 1, passages=5, rate=0.2: token_jaccard_mean=0.6495, block_jaccard_mean=0.7704, score_cosine_mean=0.9466, score_bad_value_total=0
- 完成 PreprocessKV example 5, passages=5, rate=0.1: token_jaccard_mean=0.6524, block_jaccard_mean=0.7979, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 5, passages=5, rate=0.2: token_jaccard_mean=0.6840, block_jaccard_mean=0.9015, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 6, passages=5, rate=0.1: token_jaccard_mean=0.7410, block_jaccard_mean=0.8193, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 6, passages=5, rate=0.2: token_jaccard_mean=0.7507, block_jaccard_mean=0.8906, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 2, passages=5, rate=0.1: token_jaccard_mean=0.7479, block_jaccard_mean=0.8466, score_cosine_mean=0.9846, score_bad_value_total=0
- 完成 DraftModel example 2, passages=5, rate=0.2: token_jaccard_mean=0.7687, block_jaccard_mean=0.8734, score_cosine_mean=0.9846, score_bad_value_total=0
- 完成 PreprocessKV example 7, passages=5, rate=0.1: token_jaccard_mean=0.6825, block_jaccard_mean=0.8136, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 7, passages=5, rate=0.2: token_jaccard_mean=0.7102, block_jaccard_mean=0.9031, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 3, passages=5, rate=0.1: token_jaccard_mean=0.7098, block_jaccard_mean=0.7277, score_cosine_mean=0.9381, score_bad_value_total=0
- 完成 DraftModel example 3, passages=5, rate=0.2: token_jaccard_mean=0.7296, block_jaccard_mean=0.8051, score_cosine_mean=0.9381, score_bad_value_total=0
- 完成 PreprocessKV example 8, passages=5, rate=0.1: token_jaccard_mean=0.7307, block_jaccard_mean=0.8709, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 8, passages=5, rate=0.2: token_jaccard_mean=0.7537, block_jaccard_mean=0.8825, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 4, passages=5, rate=0.1: token_jaccard_mean=0.7341, block_jaccard_mean=0.7889, score_cosine_mean=0.9664, score_bad_value_total=0
- 完成 DraftModel example 4, passages=5, rate=0.2: token_jaccard_mean=0.7204, block_jaccard_mean=0.8404, score_cosine_mean=0.9664, score_bad_value_total=0
- 完成 PreprocessKV example 9, passages=5, rate=0.1: token_jaccard_mean=0.7036, block_jaccard_mean=0.8380, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 9, passages=5, rate=0.2: token_jaccard_mean=0.7092, block_jaccard_mean=0.9085, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 10, passages=5, rate=0.1: token_jaccard_mean=0.7095, block_jaccard_mean=0.8325, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 10, passages=5, rate=0.2: token_jaccard_mean=0.7240, block_jaccard_mean=0.9026, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 5, passages=5, rate=0.1: token_jaccard_mean=0.6348, block_jaccard_mean=0.6668, score_cosine_mean=0.9748, score_bad_value_total=0
- 完成 DraftModel example 5, passages=5, rate=0.2: token_jaccard_mean=0.6817, block_jaccard_mean=0.8155, score_cosine_mean=0.9748, score_bad_value_total=0
- 完成 PreprocessKV example 11, passages=5, rate=0.1: token_jaccard_mean=0.6919, block_jaccard_mean=0.8094, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 11, passages=5, rate=0.2: token_jaccard_mean=0.7178, block_jaccard_mean=0.8979, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 6, passages=5, rate=0.1: token_jaccard_mean=0.7144, block_jaccard_mean=0.7488, score_cosine_mean=0.9702, score_bad_value_total=0
- 完成 DraftModel example 6, passages=5, rate=0.2: token_jaccard_mean=0.7136, block_jaccard_mean=0.8099, score_cosine_mean=0.9702, score_bad_value_total=0
- 完成 PreprocessKV example 12, passages=5, rate=0.1: token_jaccard_mean=0.7114, block_jaccard_mean=0.7970, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 12, passages=5, rate=0.2: token_jaccard_mean=0.7425, block_jaccard_mean=0.9171, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 13, passages=5, rate=0.1: token_jaccard_mean=0.7349, block_jaccard_mean=0.8607, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 13, passages=5, rate=0.2: token_jaccard_mean=0.7589, block_jaccard_mean=0.9166, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 7, passages=5, rate=0.1: token_jaccard_mean=0.6978, block_jaccard_mean=0.7536, score_cosine_mean=0.9702, score_bad_value_total=0
- 完成 DraftModel example 7, passages=5, rate=0.2: token_jaccard_mean=0.6778, block_jaccard_mean=0.7961, score_cosine_mean=0.9702, score_bad_value_total=0
- 完成 PreprocessKV example 14, passages=5, rate=0.1: token_jaccard_mean=0.6894, block_jaccard_mean=0.8070, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 14, passages=5, rate=0.2: token_jaccard_mean=0.7402, block_jaccard_mean=0.8972, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 8, passages=5, rate=0.1: token_jaccard_mean=0.7032, block_jaccard_mean=0.7989, score_cosine_mean=0.9736, score_bad_value_total=0
- 完成 DraftModel example 8, passages=5, rate=0.2: token_jaccard_mean=0.7574, block_jaccard_mean=0.8313, score_cosine_mean=0.9736, score_bad_value_total=0
- 完成 PreprocessKV example 15, passages=5, rate=0.1: token_jaccard_mean=0.7454, block_jaccard_mean=0.8618, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 15, passages=5, rate=0.2: token_jaccard_mean=0.7764, block_jaccard_mean=0.9284, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 16, passages=5, rate=0.1: token_jaccard_mean=0.6972, block_jaccard_mean=0.8275, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 16, passages=5, rate=0.2: token_jaccard_mean=0.7272, block_jaccard_mean=0.8908, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 DraftModel example 9, passages=5, rate=0.1: token_jaccard_mean=0.6280, block_jaccard_mean=0.7652, score_cosine_mean=0.9326, score_bad_value_total=0
- 完成 DraftModel example 9, passages=5, rate=0.2: token_jaccard_mean=0.6949, block_jaccard_mean=0.8292, score_cosine_mean=0.9326, score_bad_value_total=0
- 完成 PreprocessKV example 17, passages=5, rate=0.1: token_jaccard_mean=0.6789, block_jaccard_mean=0.8266, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 17, passages=5, rate=0.2: token_jaccard_mean=0.6986, block_jaccard_mean=0.9001, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 10, passages=5, rate=0.1: token_jaccard_mean=0.6722, block_jaccard_mean=0.7720, score_cosine_mean=0.9385, score_bad_value_total=0
- 完成 DraftModel example 10, passages=5, rate=0.2: token_jaccard_mean=0.7011, block_jaccard_mean=0.8206, score_cosine_mean=0.9385, score_bad_value_total=0
- 完成 PreprocessKV example 18, passages=5, rate=0.1: token_jaccard_mean=0.6726, block_jaccard_mean=0.8096, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 18, passages=5, rate=0.2: token_jaccard_mean=0.7177, block_jaccard_mean=0.9153, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 19, passages=5, rate=0.1: token_jaccard_mean=0.7056, block_jaccard_mean=0.8264, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 19, passages=5, rate=0.2: token_jaccard_mean=0.7433, block_jaccard_mean=0.9035, score_cosine_mean=1.0000, score_bad_value_total=0

### PreprocessKV 汇总: passages=5, rate=0.1

- 输出: `preprocesskv_batch_20260626_184239_20examples_5passages_rate0.1.json`
- 耗时: 83.7s
- token_jaccard mean over examples: 0.7024 (min=0.6423, max=0.7625)
- block_jaccard mean over examples: 0.8304 (min=0.7970, max=0.8970)
- score_cosine mean over examples: 1.0000 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1288 (min=0.0727, max=0.2057)

### PreprocessKV 汇总: passages=5, rate=0.2

- 输出: `preprocesskv_batch_20260626_184239_20examples_5passages_rate0.2.json`
- 耗时: 83.8s
- token_jaccard mean over examples: 0.7315 (min=0.6840, max=0.7764)
- block_jaccard mean over examples: 0.9047 (min=0.8825, max=0.9388)
- score_cosine mean over examples: 1.0000 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1288 (min=0.0727, max=0.2057)
- 完成 DraftModel example 11, passages=5, rate=0.1: token_jaccard_mean=0.6967, block_jaccard_mean=0.7673, score_cosine_mean=0.9587, score_bad_value_total=0
- 完成 DraftModel example 11, passages=5, rate=0.2: token_jaccard_mean=0.7809, block_jaccard_mean=0.8505, score_cosine_mean=0.9587, score_bad_value_total=0
- 完成 DraftModel example 12, passages=5, rate=0.1: token_jaccard_mean=0.6932, block_jaccard_mean=0.7321, score_cosine_mean=0.9733, score_bad_value_total=0
- 完成 DraftModel example 12, passages=5, rate=0.2: token_jaccard_mean=0.7456, block_jaccard_mean=0.8431, score_cosine_mean=0.9733, score_bad_value_total=0
- 完成 PreprocessKV example 0, passages=10, rate=0.1: token_jaccard_mean=0.6854, block_jaccard_mean=0.7966, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 0, passages=10, rate=0.2: token_jaccard_mean=0.7217, block_jaccard_mean=0.8958, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 13, passages=5, rate=0.1: token_jaccard_mean=0.7140, block_jaccard_mean=0.7805, score_cosine_mean=0.9746, score_bad_value_total=0
- 完成 DraftModel example 13, passages=5, rate=0.2: token_jaccard_mean=0.7116, block_jaccard_mean=0.8162, score_cosine_mean=0.9746, score_bad_value_total=0
- 完成 DraftModel example 14, passages=5, rate=0.1: token_jaccard_mean=0.6630, block_jaccard_mean=0.7475, score_cosine_mean=0.9629, score_bad_value_total=0
- 完成 DraftModel example 14, passages=5, rate=0.2: token_jaccard_mean=0.7196, block_jaccard_mean=0.8549, score_cosine_mean=0.9629, score_bad_value_total=0
- 完成 DraftModel example 15, passages=5, rate=0.1: token_jaccard_mean=0.6585, block_jaccard_mean=0.7178, score_cosine_mean=0.9557, score_bad_value_total=0
- 完成 DraftModel example 15, passages=5, rate=0.2: token_jaccard_mean=0.7360, block_jaccard_mean=0.8439, score_cosine_mean=0.9557, score_bad_value_total=0
- 完成 PreprocessKV example 1, passages=10, rate=0.1: token_jaccard_mean=0.6544, block_jaccard_mean=0.7907, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 1, passages=10, rate=0.2: token_jaccard_mean=0.6933, block_jaccard_mean=0.8964, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 16, passages=5, rate=0.1: token_jaccard_mean=0.6776, block_jaccard_mean=0.7537, score_cosine_mean=0.9562, score_bad_value_total=0
- 完成 DraftModel example 16, passages=5, rate=0.2: token_jaccard_mean=0.7043, block_jaccard_mean=0.7991, score_cosine_mean=0.9562, score_bad_value_total=0
- 完成 PreprocessKV example 2, passages=10, rate=0.1: token_jaccard_mean=0.7771, block_jaccard_mean=0.8621, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 2, passages=10, rate=0.2: token_jaccard_mean=0.7882, block_jaccard_mean=0.9045, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 17, passages=5, rate=0.1: token_jaccard_mean=0.7043, block_jaccard_mean=0.7806, score_cosine_mean=0.9728, score_bad_value_total=0
- 完成 DraftModel example 17, passages=5, rate=0.2: token_jaccard_mean=0.7214, block_jaccard_mean=0.8553, score_cosine_mean=0.9728, score_bad_value_total=0
- 完成 DraftModel example 18, passages=5, rate=0.1: token_jaccard_mean=0.6382, block_jaccard_mean=0.7094, score_cosine_mean=0.9466, score_bad_value_total=0
- 完成 DraftModel example 18, passages=5, rate=0.2: token_jaccard_mean=0.7166, block_jaccard_mean=0.8627, score_cosine_mean=0.9466, score_bad_value_total=0
- 完成 DraftModel example 19, passages=5, rate=0.1: token_jaccard_mean=0.6780, block_jaccard_mean=0.7867, score_cosine_mean=0.9621, score_bad_value_total=0
- 完成 DraftModel example 19, passages=5, rate=0.2: token_jaccard_mean=0.7098, block_jaccard_mean=0.8286, score_cosine_mean=0.9621, score_bad_value_total=0

### DraftModel 汇总: passages=5, rate=0.1, mode=rrf

- 输出: `draft_batch_20260626_184238_20examples_5passages_rate0.1_rrf.json`
- 耗时: 115.9s
- token_jaccard mean over examples: 0.6815 (min=0.6005, max=0.7479)
- block_jaccard mean over examples: 0.7504 (min=0.6287, max=0.8466)
- score_cosine mean over examples: 0.9610 (min=0.9326, max=0.9846)
- score_l2_rel mean over examples: 0.2548 (min=0.1604, max=0.3292)

### DraftModel 汇总: passages=5, rate=0.2, mode=rrf

- 输出: `draft_batch_20260626_184238_20examples_5passages_rate0.2_rrf.json`
- 耗时: 116.1s
- token_jaccard mean over examples: 0.7152 (min=0.6495, max=0.7809)
- block_jaccard mean over examples: 0.8263 (min=0.7704, max=0.8734)
- score_cosine mean over examples: 0.9610 (min=0.9326, max=0.9846)
- score_l2_rel mean over examples: 0.2548 (min=0.1604, max=0.3292)
- 完成 PreprocessKV example 3, passages=10, rate=0.1: token_jaccard_mean=0.6407, block_jaccard_mean=0.8001, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 3, passages=10, rate=0.2: token_jaccard_mean=0.6861, block_jaccard_mean=0.8982, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 4, passages=10, rate=0.1: token_jaccard_mean=0.7069, block_jaccard_mean=0.8052, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 4, passages=10, rate=0.2: token_jaccard_mean=0.7486, block_jaccard_mean=0.9167, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 0, passages=10, rate=0.1: token_jaccard_mean=0.7084, block_jaccard_mean=0.7820, score_cosine_mean=0.9584, score_bad_value_total=0
- 完成 DraftModel example 0, passages=10, rate=0.2: token_jaccard_mean=0.7203, block_jaccard_mean=0.8260, score_cosine_mean=0.9584, score_bad_value_total=0
- 完成 PreprocessKV example 5, passages=10, rate=0.1: token_jaccard_mean=0.6743, block_jaccard_mean=0.7861, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 5, passages=10, rate=0.2: token_jaccard_mean=0.7009, block_jaccard_mean=0.8806, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 1, passages=10, rate=0.1: token_jaccard_mean=0.6029, block_jaccard_mean=0.6530, score_cosine_mean=0.9594, score_bad_value_total=0
- 完成 DraftModel example 1, passages=10, rate=0.2: token_jaccard_mean=0.6400, block_jaccard_mean=0.7819, score_cosine_mean=0.9594, score_bad_value_total=0
- 完成 PreprocessKV example 6, passages=10, rate=0.1: token_jaccard_mean=0.7415, block_jaccard_mean=0.8265, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 6, passages=10, rate=0.2: token_jaccard_mean=0.7776, block_jaccard_mean=0.8746, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 2, passages=10, rate=0.1: token_jaccard_mean=0.7273, block_jaccard_mean=0.8269, score_cosine_mean=0.9647, score_bad_value_total=0
- 完成 DraftModel example 2, passages=10, rate=0.2: token_jaccard_mean=0.7661, block_jaccard_mean=0.8437, score_cosine_mean=0.9647, score_bad_value_total=0
- 完成 PreprocessKV example 7, passages=10, rate=0.1: token_jaccard_mean=0.6623, block_jaccard_mean=0.8144, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 7, passages=10, rate=0.2: token_jaccard_mean=0.6973, block_jaccard_mean=0.9072, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 3, passages=10, rate=0.1: token_jaccard_mean=0.6112, block_jaccard_mean=0.6699, score_cosine_mean=0.9351, score_bad_value_total=0
- 完成 DraftModel example 3, passages=10, rate=0.2: token_jaccard_mean=0.6774, block_jaccard_mean=0.8345, score_cosine_mean=0.9351, score_bad_value_total=0
- 完成 PreprocessKV example 8, passages=10, rate=0.1: token_jaccard_mean=0.6803, block_jaccard_mean=0.8054, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 8, passages=10, rate=0.2: token_jaccard_mean=0.7063, block_jaccard_mean=0.8783, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 4, passages=10, rate=0.1: token_jaccard_mean=0.6977, block_jaccard_mean=0.7899, score_cosine_mean=0.9566, score_bad_value_total=0
- 完成 DraftModel example 4, passages=10, rate=0.2: token_jaccard_mean=0.7393, block_jaccard_mean=0.8267, score_cosine_mean=0.9566, score_bad_value_total=0
- 完成 PreprocessKV example 9, passages=10, rate=0.1: token_jaccard_mean=0.6863, block_jaccard_mean=0.8163, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 9, passages=10, rate=0.2: token_jaccard_mean=0.7012, block_jaccard_mean=0.8989, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 10, passages=10, rate=0.1: token_jaccard_mean=0.7221, block_jaccard_mean=0.8282, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 10, passages=10, rate=0.2: token_jaccard_mean=0.7396, block_jaccard_mean=0.8948, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 5, passages=10, rate=0.1: token_jaccard_mean=0.6513, block_jaccard_mean=0.7066, score_cosine_mean=0.9638, score_bad_value_total=0
- 完成 DraftModel example 5, passages=10, rate=0.2: token_jaccard_mean=0.6780, block_jaccard_mean=0.7995, score_cosine_mean=0.9638, score_bad_value_total=0
- 完成 PreprocessKV example 11, passages=10, rate=0.1: token_jaccard_mean=0.6664, block_jaccard_mean=0.7899, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 11, passages=10, rate=0.2: token_jaccard_mean=0.7035, block_jaccard_mean=0.9118, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 6, passages=10, rate=0.1: token_jaccard_mean=0.7054, block_jaccard_mean=0.7559, score_cosine_mean=0.9581, score_bad_value_total=0
- 完成 DraftModel example 6, passages=10, rate=0.2: token_jaccard_mean=0.7219, block_jaccard_mean=0.8270, score_cosine_mean=0.9581, score_bad_value_total=0
- 完成 PreprocessKV example 12, passages=10, rate=0.1: token_jaccard_mean=0.6910, block_jaccard_mean=0.8089, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 12, passages=10, rate=0.2: token_jaccard_mean=0.7226, block_jaccard_mean=0.8971, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 7, passages=10, rate=0.1: token_jaccard_mean=0.6651, block_jaccard_mean=0.7516, score_cosine_mean=0.9547, score_bad_value_total=0
- 完成 DraftModel example 7, passages=10, rate=0.2: token_jaccard_mean=0.6988, block_jaccard_mean=0.8354, score_cosine_mean=0.9547, score_bad_value_total=0
- 完成 PreprocessKV example 13, passages=10, rate=0.1: token_jaccard_mean=0.6733, block_jaccard_mean=0.7901, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 13, passages=10, rate=0.2: token_jaccard_mean=0.7159, block_jaccard_mean=0.9212, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 14, passages=10, rate=0.1: token_jaccard_mean=0.6895, block_jaccard_mean=0.8056, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 14, passages=10, rate=0.2: token_jaccard_mean=0.7022, block_jaccard_mean=0.8921, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 8, passages=10, rate=0.1: token_jaccard_mean=0.6528, block_jaccard_mean=0.7398, score_cosine_mean=0.9466, score_bad_value_total=0
- 完成 DraftModel example 8, passages=10, rate=0.2: token_jaccard_mean=0.6896, block_jaccard_mean=0.7946, score_cosine_mean=0.9466, score_bad_value_total=0
- 完成 PreprocessKV example 15, passages=10, rate=0.1: token_jaccard_mean=0.6886, block_jaccard_mean=0.8429, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 15, passages=10, rate=0.2: token_jaccard_mean=0.7452, block_jaccard_mean=0.9170, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 DraftModel example 9, passages=10, rate=0.1: token_jaccard_mean=0.6581, block_jaccard_mean=0.7632, score_cosine_mean=0.9432, score_bad_value_total=0
- 完成 DraftModel example 9, passages=10, rate=0.2: token_jaccard_mean=0.7007, block_jaccard_mean=0.8303, score_cosine_mean=0.9432, score_bad_value_total=0
- 完成 PreprocessKV example 16, passages=10, rate=0.1: token_jaccard_mean=0.6642, block_jaccard_mean=0.7931, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 PreprocessKV example 16, passages=10, rate=0.2: token_jaccard_mean=0.7315, block_jaccard_mean=0.8726, score_cosine_mean=0.9999, score_bad_value_total=0
- 完成 DraftModel example 10, passages=10, rate=0.1: token_jaccard_mean=0.7011, block_jaccard_mean=0.7543, score_cosine_mean=0.9297, score_bad_value_total=0
- 完成 DraftModel example 10, passages=10, rate=0.2: token_jaccard_mean=0.7171, block_jaccard_mean=0.8264, score_cosine_mean=0.9297, score_bad_value_total=0
- 完成 PreprocessKV example 17, passages=10, rate=0.1: token_jaccard_mean=0.7124, block_jaccard_mean=0.8156, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 17, passages=10, rate=0.2: token_jaccard_mean=0.7205, block_jaccard_mean=0.8975, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 DraftModel example 11, passages=10, rate=0.1: token_jaccard_mean=0.7082, block_jaccard_mean=0.7774, score_cosine_mean=0.9345, score_bad_value_total=0
- 完成 DraftModel example 11, passages=10, rate=0.2: token_jaccard_mean=0.7202, block_jaccard_mean=0.8212, score_cosine_mean=0.9345, score_bad_value_total=0
- 完成 PreprocessKV example 18, passages=10, rate=0.1: token_jaccard_mean=0.6881, block_jaccard_mean=0.8157, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 18, passages=10, rate=0.2: token_jaccard_mean=0.7224, block_jaccard_mean=0.9237, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 19, passages=10, rate=0.1: token_jaccard_mean=0.7067, block_jaccard_mean=0.8467, score_cosine_mean=1.0000, score_bad_value_total=0
- 完成 PreprocessKV example 19, passages=10, rate=0.2: token_jaccard_mean=0.7377, block_jaccard_mean=0.9031, score_cosine_mean=1.0000, score_bad_value_total=0

### PreprocessKV 汇总: passages=10, rate=0.1

- 输出: `preprocesskv_batch_20260626_184239_20examples_10passages_rate0.1.json`
- 耗时: 199.2s
- token_jaccard mean over examples: 0.6906 (min=0.6407, max=0.7771)
- block_jaccard mean over examples: 0.8120 (min=0.7861, max=0.8621)
- score_cosine mean over examples: 1.0000 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1262 (min=0.0741, max=0.1960)

### PreprocessKV 汇总: passages=10, rate=0.2

- 输出: `preprocesskv_batch_20260626_184239_20examples_10passages_rate0.2.json`
- 耗时: 199.4s
- token_jaccard mean over examples: 0.7231 (min=0.6861, max=0.7882)
- block_jaccard mean over examples: 0.8991 (min=0.8726, max=0.9237)
- score_cosine mean over examples: 1.0000 (min=0.9999, max=1.0000)
- score_l2_rel mean over examples: 0.1262 (min=0.0741, max=0.1960)

### PreprocessKV 运行 20260626_184239 结束

- 全量输出: `preprocesskv_batch_20260626_184239_all_cases.json`
- 完成 DraftModel example 12, passages=10, rate=0.1: token_jaccard_mean=0.6926, block_jaccard_mean=0.7616, score_cosine_mean=0.9509, score_bad_value_total=0
- 完成 DraftModel example 12, passages=10, rate=0.2: token_jaccard_mean=0.7594, block_jaccard_mean=0.8774, score_cosine_mean=0.9509, score_bad_value_total=0
- 完成 DraftModel example 13, passages=10, rate=0.1: token_jaccard_mean=0.6733, block_jaccard_mean=0.7336, score_cosine_mean=0.9613, score_bad_value_total=0
- 完成 DraftModel example 13, passages=10, rate=0.2: token_jaccard_mean=0.6734, block_jaccard_mean=0.8182, score_cosine_mean=0.9613, score_bad_value_total=0
- 完成 DraftModel example 14, passages=10, rate=0.1: token_jaccard_mean=0.6386, block_jaccard_mean=0.7227, score_cosine_mean=0.9543, score_bad_value_total=0
- 完成 DraftModel example 14, passages=10, rate=0.2: token_jaccard_mean=0.6864, block_jaccard_mean=0.8143, score_cosine_mean=0.9543, score_bad_value_total=0
- 完成 DraftModel example 15, passages=10, rate=0.1: token_jaccard_mean=0.7185, block_jaccard_mean=0.7897, score_cosine_mean=0.9495, score_bad_value_total=0
- 完成 DraftModel example 15, passages=10, rate=0.2: token_jaccard_mean=0.7645, block_jaccard_mean=0.8708, score_cosine_mean=0.9495, score_bad_value_total=0
- 完成 DraftModel example 16, passages=10, rate=0.1: token_jaccard_mean=0.6297, block_jaccard_mean=0.7253, score_cosine_mean=0.8736, score_bad_value_total=0
- 完成 DraftModel example 16, passages=10, rate=0.2: token_jaccard_mean=0.6726, block_jaccard_mean=0.8081, score_cosine_mean=0.8736, score_bad_value_total=0
- 完成 DraftModel example 17, passages=10, rate=0.1: token_jaccard_mean=0.6601, block_jaccard_mean=0.7392, score_cosine_mean=0.9666, score_bad_value_total=0
- 完成 DraftModel example 17, passages=10, rate=0.2: token_jaccard_mean=0.6849, block_jaccard_mean=0.8110, score_cosine_mean=0.9666, score_bad_value_total=0
- 完成 DraftModel example 18, passages=10, rate=0.1: token_jaccard_mean=0.7064, block_jaccard_mean=0.7940, score_cosine_mean=0.9387, score_bad_value_total=0
- 完成 DraftModel example 18, passages=10, rate=0.2: token_jaccard_mean=0.7223, block_jaccard_mean=0.8364, score_cosine_mean=0.9387, score_bad_value_total=0
- 完成 DraftModel example 19, passages=10, rate=0.1: token_jaccard_mean=0.6815, block_jaccard_mean=0.7784, score_cosine_mean=0.9453, score_bad_value_total=0
- 完成 DraftModel example 19, passages=10, rate=0.2: token_jaccard_mean=0.7190, block_jaccard_mean=0.8465, score_cosine_mean=0.9453, score_bad_value_total=0

### DraftModel 汇总: passages=10, rate=0.1, mode=rrf

- 输出: `draft_batch_20260626_184238_20examples_10passages_rate0.1_rrf.json`
- 耗时: 234.7s
- token_jaccard mean over examples: 0.6745 (min=0.6029, max=0.7273)
- block_jaccard mean over examples: 0.7508 (min=0.6530, max=0.8269)
- score_cosine mean over examples: 0.9472 (min=0.8736, max=0.9666)
- score_l2_rel mean over examples: 0.2958 (min=0.2444, max=0.4415)

### DraftModel 汇总: passages=10, rate=0.2, mode=rrf

- 输出: `draft_batch_20260626_184238_20examples_10passages_rate0.2_rrf.json`
- 耗时: 235.0s
- token_jaccard mean over examples: 0.7076 (min=0.6400, max=0.7661)
- block_jaccard mean over examples: 0.8265 (min=0.7819, max=0.8774)
- score_cosine mean over examples: 0.9472 (min=0.8736, max=0.9666)
- score_l2_rel mean over examples: 0.2958 (min=0.2444, max=0.4415)

### DraftModel 运行 20260626_184238 结束

- 全量输出: `draft_batch_20260626_184238_all_cases.json`
