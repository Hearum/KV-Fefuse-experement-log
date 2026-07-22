# Qwen3-32B MuSiQue-v2 Hidden State Gap 计划

## 目标

在缓存 document KV 时额外保存每层输入 hidden state，并比较 document token 在不同上下文构造下的 hidden state 差异：

- `raw hidden`：`system + 当前 doc` 单独缓存时，当前 doc 的每层输入 hidden。
- `preprocess hidden`：按 BGE top-k preprocess KV 构造前缀后，重跑当前 doc 得到的每层输入 hidden。
- `full hidden`：`system + doc1 + doc2 + ... + docn` 完整前向时，每个 doc token 的每层输入 hidden。

重点回答：KV gap 较大的层是否对应 hidden state 本身发生较大变化；如果 hidden gap 也集中在少数层，说明未来 Adapter 可能要预测 hidden/update；如果 hidden gap 从浅层开始累积，说明跳过逐层递归会更难。

## 样本和口径

- 数据集：`musique-v2.jsonl`
- 模型：`Qwen3-32B`
- 初始 sanity：1-5 examples，小样本确认 shape、显存、定义。
- 后续扩大：如果显存和时间可控，再跑 50 examples。
- cache 复用：读取共享 setup-v2 Qwen3-32B cache；hidden cache 单独写入本实验 `hidden_cache/`，不混入共享 KV cache。

## 指标

- Relative L2：`||full - source|| / ||full||`
- Cosine：`cos(full, source)`
- Hidden delta energy share：每层 hidden gap 占总 gap 的比例。
- raw/preprocess hidden gap 比值。

## 注意

本实验分析 hidden state 变化，不直接做端到端 accuracy。`FUSIONRAG_SAVE_HIDDEN_CACHE=1` 已加入正式 KV 保存函数，但默认关闭，避免无意增大共享 cache。
