# Qwen3-32B MuSiQue-v2 Hidden State Gap

本实验验证：在 document KV cache 之外，额外缓存每层输入 hidden state，并比较 `raw/preprocess/full` 三种上下文下 document token hidden state 的变化。

详细计划见 `PLAN.md`，运行记录见 `EXPERIMENT_LOG.md`。

## 接口实现

已在 `ktransformers/util/utils.py` 增加可选 hidden cache 接口：

```bash
export FUSIONRAG_SAVE_HIDDEN_CACHE=1
export FUSIONRAG_HIDDEN_CACHE_DTYPE=bf16   # 可选：bf16/fp16/fp32
```

默认不保存 hidden cache，避免无意放大共享 cache。打开后，`prefill_and_save_kv_cache()` 和 `prefill_with_cache_and_save_preprocess()` 会额外保存：

```text
{example_id}_{chunk_id}_hidden.pt
```

保存内容是 `hidden_states[:-1]`，即每个 Transformer block 的输入 hidden state，shape 为：

```text
[num_layers, batch, seq_len, hidden_size]
```

选择这个定义是因为第 `l` 层 K/V 正是由第 `l` 层输入 hidden state 经过 `k_proj/v_proj` 得到。

## 2026-07-15 Sanity 结果

实验设置：

- model：`/mnt/qjhs-sh-lab-01/models/Qwen3-32B`
- dataset：`musique-v2.jsonl`
- samples：5 examples
- chunks：116 document chunks
- document tokens：84,634
- shared KV cache：`/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2`
- hidden tensor cache：本轮没有保存大 tensor，只保存聚合统计；`hidden_cache/` 当前为空。

### Global Hidden Gap

| source hidden | Relative L2 to full hidden | Cosine to full hidden | examples | chunks | doc tokens |
|---|---:|---:|---:|---:|---:|
| raw | 0.6942 | 0.7823 | 5 | 116 | 84,634 |
| preprocess | 0.7770 | 0.6962 | 5 | 116 | 84,634 |

### Top Gap Layers

raw hidden gap energy 最高的层：

| layer | Relative L2 | Energy Share | Cosine |
|---:|---:|---:|---:|
| 62 | 0.4766 | 0.0656 | 0.8895 |
| 61 | 0.5428 | 0.0627 | 0.8582 |
| 60 | 0.5873 | 0.0593 | 0.8366 |
| 59 | 0.6414 | 0.0563 | 0.8100 |
| 58 | 0.6885 | 0.0539 | 0.7853 |
| 57 | 0.7506 | 0.0521 | 0.7511 |
| 56 | 0.8293 | 0.0506 | 0.7061 |
| 55 | 0.9074 | 0.0494 | 0.6597 |

preprocess hidden gap energy 最高的层：

| layer | Relative L2 | Energy Share | Cosine |
|---:|---:|---:|---:|
| 63 | 0.4857 | 0.0888 | 0.8756 |
| 62 | 0.6202 | 0.0886 | 0.8003 |
| 61 | 0.6767 | 0.0777 | 0.7655 |
| 60 | 0.7091 | 0.0690 | 0.7422 |
| 59 | 0.7546 | 0.0622 | 0.7097 |
| 58 | 0.7874 | 0.0563 | 0.6855 |
| 57 | 0.8378 | 0.0518 | 0.6521 |
| 56 | 0.8991 | 0.0474 | 0.6053 |

## 初步结论

1. Hidden state 的上下文差异非常大。5-example 上 raw hidden 到 full hidden 的 relative L2 已达 `0.6942`，preprocess hidden 更大，为 `0.7770`。这说明 document KV 的 gap 不是只发生在投影后的 K/V 空间，Transformer block 输入 hidden 本身已经明显偏移。
2. Hidden gap 明显集中在后层，尤其 `56-63`。这和前面 MuSiQue-v2 KV gap 中 Value 后层能量集中的结论一致，说明后层 hidden/update 是 document context conditioning 的主要承载位置。
3. 这 5 个样本里 preprocess hidden 反而比 raw hidden 更远离 full hidden。这个结果和“preprocess KV 必然更接近 full KV”的直觉不一致，需要谨慎解释：preprocess 的前缀来自 BGE top-k 其他 chunk，不等价于真实 full prompt 中该 doc 之前的所有文档顺序；它可能改善某些端到端行为，但 hidden 几何上不一定更接近 full hidden。
4. 如果未来想省掉完整逐层重算，直接预测 K/V 可能不够；因为 K/V gap 背后有大的 hidden state drift。更合理的 Adapter 方向可能是预测后层 hidden correction 或后层 K/V correction，而不是试图从浅层开始完整替代 Transformer 递归。

## 输出文件

- Global summary：`results/hidden_global_summary.csv`
- Layer summary：`results/hidden_layer_summary.csv`
- JSON summary：`results/hidden_summary.json`
- Relative L2 图：`figures/hidden_layer_relative_l2.png`
- Energy share 图：`figures/hidden_layer_energy_share.png`
- 脚本：`scripts/collect_hidden_gap.py`



## 2026-07-16 50-example Scaling 结果

实验设置：

- model：`/mnt/qjhs-sh-lab-01/models/Qwen3-32B`
- dataset：`musique-v2.jsonl`
- samples：50 examples
- chunks：1,166 document chunks
- document tokens：827,074
- skipped：0
- 启动代码 commit：`0963c59`
- shared KV cache：`/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2`
- 运行方式：5 个 shard 并行，每个 shard 10 examples，qjy000 GPU0-4。

### Global Hidden Gap, 50 examples

| source hidden | Relative L2 to full hidden | Cosine to full hidden | examples | chunks | doc tokens |
|---|---:|---:|---:|---:|---:|
| raw | 0.7204 | 0.7648 | 50 | 1,166 | 827,074 |
| preprocess | 0.7431 | 0.7223 | 50 | 1,166 | 827,074 |

### Top Gap Layers, 50 examples

raw hidden gap energy 最高的层：

| layer | Relative L2 | Energy Share | Cosine |
|---:|---:|---:|---:|
| 62 | 0.4950 | 0.0649 | 0.8809 |
| 61 | 0.5620 | 0.0622 | 0.8479 |
| 60 | 0.6084 | 0.0590 | 0.8244 |
| 59 | 0.6648 | 0.0562 | 0.7952 |
| 58 | 0.7138 | 0.0539 | 0.7684 |
| 57 | 0.7775 | 0.0522 | 0.7316 |
| 56 | 0.8567 | 0.0507 | 0.6841 |
| 55 | 0.9367 | 0.0496 | 0.6339 |
| 54 | 1.0202 | 0.0487 | 0.5804 |
| 53 | 1.0984 | 0.0480 | 0.5289 |

preprocess hidden gap energy 最高的层：

| layer | Relative L2 | Energy Share | Cosine |
|---:|---:|---:|---:|
| 62 | 0.5868 | 0.0858 | 0.8226 |
| 63 | 0.4493 | 0.0817 | 0.8947 |
| 61 | 0.6408 | 0.0760 | 0.7906 |
| 60 | 0.6730 | 0.0678 | 0.7690 |
| 59 | 0.7174 | 0.0615 | 0.7383 |
| 58 | 0.7505 | 0.0560 | 0.7147 |
| 57 | 0.7999 | 0.0519 | 0.6815 |
| 56 | 0.8589 | 0.0479 | 0.6368 |
| 55 | 0.9181 | 0.0448 | 0.5913 |
| 54 | 0.9820 | 0.0424 | 0.5393 |

### 50-example 结论更新

1. 5-example sanity 的主要结论成立：hidden state 本身存在很大的 context-conditioned gap。50 examples 上 raw hidden 到 full hidden 的 relative L2 是 `0.7204`，preprocess hidden 是 `0.7431`。
2. preprocess hidden 比 5-example 时更接近 full hidden，但仍然弱于 raw hidden。这个现象说明：BGE top-k preprocess 前缀不是“更接近 full prompt hidden”的保证；它可能对某些 KV/端到端行为有帮助，但 hidden 几何距离上没有超过 raw。
3. hidden gap 的层分布非常稳定，raw 和 preprocess 都集中在后层。raw 的 top layers 是 `62,61,60,59,58,57,56,55,54,53`；preprocess 的 top layers 是 `62,63,61,60,59,58,57,56,55,54`。
4. 对 Adapter 设计的含义：如果目标是替代 document KV 重算，后层 correction 是更明确的方向。直接从 cached KV 推全层更新可能过难，因为 full prompt 对 document token 的 hidden trajectory 已经发生大幅漂移；更现实的方案是只预测后层 hidden/KV correction，或把后层 Value correction 作为第一阶段目标。

### 50-example 输出文件

- Merged global summary：`results/merged_50/hidden_global_summary.csv`
- Merged layer summary：`results/merged_50/hidden_layer_summary.csv`
- Merged JSON summary：`results/merged_50/hidden_summary.json`
- Merged relative L2 figure：`results/figures/hidden_layer_relative_l2_50.png`
- Merged energy share figure：`results/figures/hidden_layer_energy_share_50.png`
- Shard outputs：`results/shards_50/`

## 未完成事项

- 已完成 50-example scaling；如果要进一步确认跨数据集稳定性，下一步应在 2WikiQA-v2 或 HotpotQA-v2 上复现 hidden gap 统计。
- 当前未保存大 hidden tensor，只保存统计；如果要训练 Adapter，需要用 `--save-hidden-tensors` 或正式 `FUSIONRAG_SAVE_HIDDEN_CACHE=1` 生成样本。
- 需要进一步对齐 2WikiQA-v2 上的 hidden gap，确认后层集中是否跨数据集稳定。
