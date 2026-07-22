# WikiText-103 Delta-KV 预训练数据计划

## 1. 为什么需要预训练数据

当前 RAG residual 数据只有73个独立 main examples。即使全 token 评估很多，真正独立的 prefix/target context 仍少。平衡采样能改善训练，但不能替代 context 多样性。

仓库已有 draft selector 实验使用的严格 WikiText-103 split：

- train：`predictor_distill_wikitext/data/wikitext103_pairs_train_500k.jsonl`，来自原文前35%；
- validation：`predictor_distill_wikitext/data/wikitext103_pairs_val_50k.jsonl`，来自原文最后10%；
- 每条旧 pair 为384 doc tokens + 64 query tokens。

旧实验表明20k到500k、并换成真正独立WikiText语料后，selector recall@15可从约61.7%提升到76.9%。这证明独立语料规模值得验证，但旧 teacher score/logits 不是 Delta-KV 标签，不能直接复用。

## 2. 新的监督定义

每条 WikiText text slice 用 Qwen3 tokenizer 重新编码，构造：

```text
A = prefix tokens (256)
X = target tokens (128)

B_X = Qwen3-32B prefill(X) 中 X 的 raw/offline KV
T_X = Qwen3-32B prefill(A + X) 中 X 的 full KV
Delta_X = T_X - B_X
```

Key 将 full X 反向旋转256 positions后，在local/RoPE-aligned坐标与 offline Key 比较；Value直接比较。后置 query 不参与，因为它不能改变 X KV。

Tokenizer审计显示 Qwen2.5/Qwen3 vocab size为151665/151669，完整vocab不相同。正式构造从539MB WikiText原始文本按相同字符区间切分，并一次性生成Qwen3 int32 token cache；禁止直接复用旧 `doc_ids/query_ids`，也不采用Qwen2 decode再Qwen3 encode的过渡样本。

## 3. 紧凑存储

禁止为500k样本保存完整64层全tokenKV。每条样本只保存：

- 8个均匀target token的 offline K/V：`[64,8,8,256]` fp16；
- prefix all-mean和last-chunk mean：`[64,8,512]` fp16；
- K/V rank64 coefficient：`[64,8,8,128]` fp16；
- token positions、pair id、文本区间和 projection residual统计。

预计约3.5MB/sample：2k pilot约7GB，20k约70GB，500k约1.75TB。先2条 smoke，再64、2k、20k逐级扩大；只有20k预训练在严格RAG test上提升后才考虑更大规模。

## 4. Basis与两阶段训练

先在Qwen3-8B完成低成本验证。8B与32B层数不同（36 vs 64），必须独立学习basis和predictor；验证成功后迁移方法/训练流程，不直接复用8B Adapter权重。首轮在Wiki train学习per-layer/head rank64 basis，再用8B RAG residual fine-tune。

若Wiki projection residual过高，则改成：

1. Wiki+RAG train联合学习basis；
2. Wiki预训练context encoder；
3. 只用RAG train重新拟合/微调layer-head output basis与head；
4. RAG validation选checkpoint，11个RAG strict test只评一次。

## 5. Scaling gate

1. 2 samples：shape、RoPE、存储字段、可重建性。
2. 64 samples：吞吐、磁盘/sample、Wiki basis projection energy。
3. 2k train + 256 val：验证pretraining loss能下降且Wiki heldout改善。
4. 20k train + 2k val：在RAG上fine-tune，必须超过无预训练grouped predictor和Ridge。
5. 只有RAG Value explained energy明显超过18.09%，且至少两次seed一致，才扩大到100k/500k。

最终目标仍是formal preprocess-to-full Delta；Wiki raw Delta只用于初始化context encoder，不能替代RAG fine-tuning和端到端rate1对照。
