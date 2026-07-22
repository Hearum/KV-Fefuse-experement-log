# Motivation 问题扫描

## 背景

前面尝试过从 sparse recomputation kernel 的 pattern overhead 找 motivation，但这个方向说服力不够强。原因是：

1. `early_contiguous` 和 `late_contiguous` 的差异很大程度来自 causal attention 本身，后面的 query 本来就需要 attend 更长 prefix。
2. `random_sorted` 和 `strided_full_context` 的曲线几乎重合，说明这个对比没有提供额外信息。
3. `random_sorted` 和 `random_shuffled_same_set` 虽然能说明 query packing 会影响 kernel latency，但这个点偏 kernel implementation detail，和 FusionRAG 的核心目标关系不够直接。

因此，后续 motivation 应该从 FusionRAG 的主矛盾出发：**full document KV cache 太大，且在 GPU serving 中会持续消耗显存、带宽和 cache 恢复时间。**

## 更值得关注的问题

### 问题 1：Full document KV cache 显存随文档长度线性爆炸

当前 `StaticCache` 对每一层都预分配完整 K/V：

```text
key_cache[layer]:   [batch, kv_heads, max_cache_len, head_dim]
value_cache[layer]: [batch, kv_heads, max_cache_len, head_dim]
```

因此每个 token 的 KV cache 大小约为：

```text
num_layers * 2(K/V) * num_kv_heads * head_dim * bytes_per_element
```

FP16 下的估算如下：

| Model setting | Layers | KV heads | Head dim | 每 token KV | 1K | 2K | 4K | 8K | 16K | 32K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenPangu-like GQA | 34 | 8 | 128 | 136 KB | 0.13 GB | 0.27 GB | 0.53 GB | 1.06 GB | 2.12 GB | 4.25 GB |
| Mistral-like GQA | 32 | 8 | 128 | 128 KB | 0.12 GB | 0.25 GB | 0.50 GB | 1.00 GB | 2.00 GB | 4.00 GB |
| Qwen2 default MHA | 32 | 32 | 128 | 512 KB | 0.50 GB | 1.00 GB | 2.00 GB | 4.00 GB | 8.00 GB | 16.00 GB |
| Llama MHA | 32 | 32 | 128 | 512 KB | 0.50 GB | 1.00 GB | 2.00 GB | 4.00 GB | 8.00 GB | 16.00 GB |

这个问题比 sparse kernel pattern 更直接：RAG 文档越长，full document KV 越不可持续。对于 MHA 模型，单个 32K 文档的 KV 就可能达到 16GB；即使是 GQA，32K 也有 4GB 左右。batch 或并发一上来，显存很快被 KV cache 占满。

### 问题 2：KV cache reuse 不是免费，CPU->GPU 恢复 KV 本身有明显开销

现有 `load_kv_and_generate` 路径会先从磁盘加载 chunk KV 到 CPU，然后执行：

```text
chunk_key_cache = key_cache[idx].to(device)
chunk_value_cache = value_cache[idx].to(device)
for layer_idx in layers:
    past_key_values.key_cache[layer_idx].copy_(chunk_key_cache[layer_idx])
    past_key_values.value_cache[layer_idx].copy_(chunk_value_cache[layer_idx])
```

也就是说，cache reuse 的成本不是 0，它包含：

1. 磁盘读取；
2. CPU tensor 反序列化；
3. CPU->GPU 传输；
4. 每层 K/V copy 到 `StaticCache`；
5. 可能还有 RoPE revert 或 preprocess 相关操作。

我做了一个轻量 microbenchmark，模拟 OpenPangu-like GQA 配置：

```text
layers=34, kv_heads=8, head_dim=128, dtype=fp16
```

只测 CPU 上已有 stacked KV tensor 的 `.to(cuda)` 加逐层 copy 到目标 cache，不包含磁盘读取：

| Tokens | KV size | CPU->GPU + layer copy latency | Effective bandwidth |
|---:|---:|---:|---:|
| 512 | 0.066 GB | 4.828 ms | 14.77 GB/s |
| 1024 | 0.133 GB | 8.880 ms | 16.06 GB/s |
| 2048 | 0.266 GB | 17.532 ms | 16.27 GB/s |
| 4096 | 0.531 GB | 33.422 ms | 17.07 GB/s |
| 8192 | 1.062 GB | 66.582 ms | 17.13 GB/s |

这个结果说明：即使不算磁盘 IO，仅仅把 8K 文档 KV 从 CPU 恢复到 GPU 并写入每层 cache，也要约 66ms。多 chunk、多并发时，这个开销很容易抵消部分 cache reuse 的收益。

### 问题 3：Decode 阶段 full document KV 会持续消耗 HBM 带宽

full document KV 不只是 prefill 时贵。只要这些 document KV 留在 cache 里，decode 每生成一个 token 都要在 attention 中读历史 K/V。文档越长，每步 decode 读的 KV 越多。

这可以形成更强的 motivation：

```text
Full document KV:
  文档长 -> KV cache 大 -> 每步 decode 读更多 HBM -> latency / throughput 受限

Preprocess KV:
  用更紧凑的表示替代 full document KV -> decode 读更少 KV -> 降低显存和带宽压力
```

这个问题比 sparse recomputation pattern 更贴近 FusionRAG，因为 FusionRAG 的本质就是用 preprocess KV 替代 full attention document KV。

### 问题 4：Full KV 限制 batch size / concurrency

KV cache 是按 batch 线性放大的。一个请求 8K 文档在 GQA 模型上约 1GB KV，batch=8 就是 8GB；如果是 MHA 模型，8K 单请求约 4GB，batch=8 就是 32GB。

因此 full document KV 会直接限制：

1. 可支持的 batch size；
2. 同时在线请求数；
3. 长文档场景下的服务吞吐；
4. GPU 上其他 workspace / activation / model weights 的可用空间。

这个方向比“某个 sparse kernel 是否线性”更有系统意义。

## 更值得做的下一组实验

### 实验 A：KV size vs document length

目的：证明 full document KV cache 随文档长度线性增长，并且在长文档、多 batch 下不可持续。

横轴：

```text
document length = 1K, 2K, 4K, 8K, 16K, 32K
```

纵轴：

```text
KV cache memory footprint
```

曲线：

```text
full document KV
preprocess KV
raw top-k chunk KV
```

### 实验 B：KV restore latency vs document length

目的：证明 KV cache reuse 有 CPU->GPU 恢复成本，且这个成本随 KV size 线性增长。

横轴：

```text
document length / cached KV tokens
```

纵轴：

```text
CPU->GPU KV restore latency
```

当前初步结果已经显示：OpenPangu-like GQA 下，8K KV 恢复约 66ms，而且这还不含磁盘 IO。

### 实验 C：Decode latency vs KV length

目的：证明 full document KV 会拖慢每步 decode。

横轴：

```text
KV cache length = 1K, 2K, 4K, 8K, 16K, 32K
```

纵轴：

```text
single-token decode latency
```

对比：

```text
full document KV length
reduced / preprocess KV length
```

这个实验最适合支撑 FusionRAG 的核心 motivation。

### 实验 D：Batch/concurrency vs KV memory

目的：证明 full document KV 限制并发能力。

横轴：

```text
batch size / concurrent requests
```

纵轴：

```text
GPU memory usage / decode throughput / OOM boundary
```

对比：

```text
full document KV
preprocess KV
```

## 当前判断

应该放弃把 motivation 建在 sparse recomputation pattern overhead 上。更有价值的问题是：

1. full document KV cache 占用太大；
2. CPU/GPU 异构 cache reuse 有恢复成本；
3. decode 阶段 full KV 持续消耗 HBM 带宽；
4. full KV 限制 batch 和 concurrency；
5. preprocess KV 的价值在于用更小、更稳定的表示替代 full document KV，从而降低 memory footprint、KV restore latency 和 decode-time bandwidth pressure。

下一步最应该做的是 **Decode latency vs KV length**。如果这个实验能显示每步 decode latency 随 full KV length 明显增长，而 preprocess/reduced KV 显著降低这部分成本，那么 motivation 会比 sparse kernel pattern 清楚很多。

---

## 近期机制实验记录：query-stable update set 与 preprocess KV 偏移

这一节记录 2026-06-26 之后围绕“是否存在可离线缓存/可预测的重算或 preprocess 结构”做的实验。详细 JSON、图和脚本仍保存在各自实验目录里，但核心结论统一写在这里，方便后续继续追加。

### 实验 1：不同 query 触发的 token 选择是否稳定

目的：验证对于同一个文档，不同 query 触发的 online update token 是否高度重合。如果存在稳定集合，就可能说明文档天然存在一批 query-insensitive 的重要 token。

主要结果目录：

```text
docs/experiments/query_recompute_overlap_32q_preprocess/
docs/experiments/query_selection_frequency_stability_32q_preprocess/
```

实验设置：

```text
dataset = musique-pca-subset
examples = 20
passages = 10
queries per example = 32
selector = Target-QK / attention importance with preprocess KV
rates = 0.1, 0.2, 0.3, 0.5
```

关键结果：

| Rate | never selected | selected by all 32 queries | polarized set: selected 0 or 32 |
|---:|---:|---:|---:|
| 0.1 | 76.39% | 5.81% | 82.20% |
| 0.2 | 58.71% | 12.17% | 70.88% |
| 0.3 | 45.39% | 19.54% | 64.93% |
| 0.5 | 25.79% | 36.99% | 62.78% |

stable-set convergence，passages=10：

| Rate | 2 queries | 4 queries | 8 queries | 16 queries | 32 queries |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.7998 | 0.6984 | 0.6434 | 0.6084 | 0.5817 |
| 0.2 | 0.8249 | 0.7284 | 0.6725 | 0.6364 | 0.6086 |
| 0.3 | 0.8529 | 0.7664 | 0.7137 | 0.6786 | 0.6514 |
| 0.5 | 0.9003 | 0.8352 | 0.7927 | 0.7629 | 0.7399 |

结论：

```text
token 选择不是完全 query-specific；存在明显两极分化：
很多 token 在所有 query 下都不会被选中，也有一批 token 在大多数 query 下稳定被选中。
rate 越高，稳定交集越大。
```

这个现象支持一个可能方向：online update token 里存在 document-stable component，可以尝试离线预测或缓存。但它不是完整答案，因为仍有 query-specific component。

### 实验 2：online recompute 前后的 KV 差异

目的：确认 FusionRAG 的 online recompute 到底改变了什么，不把它和 preprocess KV 混淆。

主要结果目录：

```text
docs/experiments/online_recompute_kv_delta/
```

实验设置：

```text
examples = 1-3
max_passages = 10
rate = 0.1
preprocess cache = top10
对比对象 = 同一个 past_key_values 中 selected document tokens online recompute 前后的 KV
```

关键结果：

| Kind | selected relative L2 | cosine | token delta mean | p90 |
|---|---:|---:|---:|---:|
| key | 0.1972 | 0.9831 | 0.8028 | 1.0564 |
| value | 0.7154 | 0.7532 | 0.7917 | 1.2761 |

未选中的 token delta 为 0。

结论：

```text
online recompute 只改变 selected tokens；
value 被改得远大于 key；
如果要跳过 online recompute，最难的是预测 selected value KV 的变化。
```

### 实验 3：能否用简单 correction 逼近 online recompute 后的 KV

主要结果目录：

```text
docs/experiments/cached_kv_correction_probe/
```

实验设置：

```text
train examples = 1-4
test examples = 5-6
max_passages = 10
rate = 0.1
target = selected tokens 的 before-KV -> after-online-recompute-KV
```

测试方法：

```text
before_cache: 不做 correction
layer_head_dim_affine: 对每个 layer/head/dim 学 scale + shift
```

结果：

| Kind | Method | relative L2 | cosine |
|---|---|---:|---:|
| key | before_cache | 0.1956 | 0.9836 |
| key | layer_head_dim_affine | 0.1385 | 0.9925 |
| value | before_cache | 0.7056 | 0.7427 |
| value | layer_head_dim_affine | 0.6217 | 0.7833 |

结论：

```text
简单 channel-wise affine 对 key 有明显帮助，对 value 只有有限帮助。
value correction 需要 token/context-specific 信息，不能只靠全局 bias。
```

记录过但未展开的后续方法：

```text
low-rank delta
token-wise MLP correction
residual codebook
retrieval-based correction
position/chunk-aware correction
teacher distillation from online recompute
hybrid correction + residual recompute
```

### 实验 3.5：online recompute 的 K/V 写回消融

主要结果目录：

```text
MOTIVATION_EXPERIMENTS/recompute_v_only_ablation/
```

目的：验证 online recompute 阶段 selected document tokens 的重算 KV 是否必须写回 cache。比较三种方式：

```text
kv:
  原始行为，selected document tokens online recompute 后 K/V 都写回 past_key_values。

v_only:
  recompute 前备份 selected document tokens 的所有层 K；
  前向结束后恢复 K，只保留 V 的更新。
  query tokens 的 K/V 正常保留。

none:
  recompute 前备份 selected document tokens 的所有层 K/V；
  前向结束后恢复 K/V，相当于 document tokens 不写回重算结果。
  query tokens 的 K/V 正常保留。
```

实验设置：

```text
dataset = musique-pca-subset.jsonl
model = Qwen2.5-7B-Instruct
preprocess = True
topk = 10
rate = 0.1
reprocess_method = FusionRAG
revert_rope = True
samples = 20
```

结果：

| Mode | ROUGE | EM |
|---|---:|---:|
| kv | 0.2967 | 0.2000 |
| v_only | 0.2565 | 0.2000 |
| none | 0.2628 | 0.2000 |

样本级统计：

```text
samples = 20
v_only same as kv = 17
v_only changed from kv = 3
none same as kv = 15
none changed from kv = 5
kv avg words = 7.30
v_only avg words = 6.35
none avg words = 7.40
```

观察：

```text
完整 K/V 写回最好，但只比 v_only 高 0.0402 ROUGE，只比 none 高 0.0340 ROUGE。
在 rate=0.1 的小样本设置下，document token 的重算写回带来增量收益，但不是决定性差异。
none 略高于 v_only，说明只更新 V 并不一定比完全不写回 document KV 更好。
```

结论：

```text
如果目标是节省 document token KV 写回，none/v_only 可能是值得继续测的大方向；
但当前结果也说明只更新 V 没有明确优于完全不写回 document KV。
更稳妥的说法是：online recompute 的主要收益不一定完全来自 selected document KV 的持久写回，
也可能来自 query token 在稀疏上下文下重新 prefill 得到的状态。
```

注意：早先一次临时实现错误地恢复了 query tokens 的 K，导致 v_only 异常发散。当前结果已修正，只对 selected document tokens 做恢复，query tokens 的 K/V 保持新计算结果。

### 实验 4：不同 preprocess top-k prefix 对 KV 的影响

主要结果目录：

```text
docs/experiments/preprocess_prefix_kv_difference/
```

实验设置：

```text
cache_root = /raid/home/hming/fusionrag-pca-topk-cache-5/data
dataset = musique-pca-subset
top-k prefixes = 1, 3, 5, 10
common example/chunk pairs = 121
metric = symmetric relative L2 and cosine between KV tensors
```

整体差异：

| Kind | Pair | symmetric relative L2 | cosine |
|---|---|---:|---:|
| key | top1 vs top10 | 0.0405 | ~1.0 |
| value | top1 vs top10 | 0.2955 | 0.9521 |
| value | top3 vs top10 | 0.2073 | 0.9763 |
| value | top5 vs top10 | 0.1703 | 0.9839 |

结论：

```text
不同 preprocess prefix 对 key 的影响很小；
对 value 的影响明显更大；
随着 top-k 增加，value 逐渐靠近 top10/fuller-prefix 表示。
差异主要集中在中后层，尤其 layer 18/15/17/20/16/14。
```

进一步看 layer trend，不同 top-k pair 的 value layer-wise 曲线高度相关：

```text
多数 pair correlation > 0.98
最低约 0.9584
```

说明：

```text
不同 preprocess 设置改变的是 value 扰动强度，而不是主要作用位置。
```

### 实验 5：把 top-k 变化看成向量偏移，是否同向

主要结果目录：

```text
docs/experiments/preprocess_vector_direction/
```

目的：把 `KV_topK` 看成高维向量，检查随着 top-k 增加，是否沿同一方向移动。

核心指标：

```text
cosine(KV_top3 - KV_top1, KV_top10 - KV_top1)
cosine(KV_top5 - KV_top1, KV_top10 - KV_top1)
cosine(KV_top3 - KV_top1, KV_top5 - KV_top3)
```

关键结果：

| Kind | Delta A | Delta B | mean cosine |
|---|---|---|---:|
| value | 1->3 | 1->10 | 0.7022 |
| value | 1->5 | 1->10 | 0.7907 |
| value | 1->3 | 3->5 | -0.2307 |
| value | 3->5 | 5->10 | -0.2460 |
| key | 1->3 | 1->10 | 0.7229 |
| key | 1->5 | 1->10 | 0.8218 |

结论：

```text
随着 top-k 增加，KV 的确有一个中等偏强的整体偏移方向；
但它不是一条直线，不是每新增一批文档就沿同一 vector 继续前进。
更准确的说法是：整体朝 fuller-prefix/top10 表示靠近，但每批新增文档都会引入新的局部修正分量。
```

### 实验 6：固定 top-k，更换 preprocess 文档集合

主要结果目录：

```text
docs/experiments/preprocess_fixed_topk_prefix_swap/
```

目的：排除“只是 top-k 数量变化”的解释。固定 prefix 数量为 top5，但替换 prefix 文档内容，看目标 chunk 的 KV 偏移方向是否仍稳定。

实验设置：

```text
examples = 1, 2
target chunks = 6, 8
topk = 5
variants:
  retrieved_top5
  same_example_early
  same_example_late
  cross_next_early
  cross_prev_late
  mixed_same_cross
metric = cosine(KV_variant - raw_KV, KV_other_variant - raw_KV)
```

关键 value 结果：

| Pair | value delta-from-raw cosine |
|---|---:|
| retrieved_top5 vs same_example_early | 0.7588 |
| retrieved_top5 vs mixed_same_cross | 0.6499 |
| retrieved_top5 vs cross_next_early | 0.4080 |
| retrieved_top5 vs cross_prev_late | 0.3885 |

结论：

```text
固定 top-k 数量不保证 preprocess 方向一致；
prefix 文档内容会显著改变 value 的偏移方向；
同 example 内部文档更接近 retrieved top5，跨 example 文档方向明显变散。
```

这说明 preprocess KV 偏移包含两部分：

```text
1. 较稳定的上下文化 / fuller-prefix 方向；
2. 明显依赖 prefix 内容的语义分量。
```

### 当前综合判断

目前比较可靠的结论是：

```text
1. value KV 是 FusionRAG 里最敏感、最值得建模的部分；
2. online recompute 和 preprocess 都主要在 value 上产生大变化；
3. 存在稳定结构，但不是纯全局 bias；
4. top-k 增加会把 KV 朝 fuller-prefix 表示拉近，但路径不是线性的；
5. prefix 内容会明显改变 value 偏移方向，因此 correction/cache 方案需要考虑 prefix context。
```

如果要把它变成方法点，单纯“学一个全局 bias”不够。更合理的方向是：

```text
KV_corrected = KV_raw/preprocess + f(prefix_summary, token_state, layer, position)
```

其中 `f` 至少应该包含 prefix-aware 或 token-aware 信息，否则对 value 的修正能力有限。

### 补充：DeepSeek judge 评测 recompute K/V 写回

对 `recompute_v_only_ablation` 的三组输出，复用 `jybigdata` 中的 answer judge pipeline，以 DashScope OpenAI-compatible endpoint 调用 `deepseek-v3.2`，输入 `Question / Real Answer / Pred Answer`，输出 `Correct / Reason`。

结果文件：

```text
MOTIVATION_EXPERIMENTS/recompute_v_only_ablation/deepseek_accuracy/summary.csv
MOTIVATION_EXPERIMENTS/recompute_v_only_ablation/deepseek_accuracy/*.deepseek_judged.csv
```

| mode | correct / total | accuracy |
|---|---:|---:|
| kv | 6 / 20 | 0.300000 |
| v_only | 5 / 20 | 0.250000 |
| none | 5 / 20 | 0.250000 |

判断：语义 judge 下，完整 K/V 写回仍然最好；`v_only` 与 `none` 持平。这进一步支持当前结论：selected document tokens 的 online recompute 写回有收益，但“只更新 V”不是一个稳定可用的替代方案。

## 重要更正：真实 FusionRAG reflect pipeline sanity check

用户指出 `preprocess-exp/run_fusionrag.sh` 使用的是 `test_fusionrag_reflect.py`，而先前 `recompute_v_only_ablation` 使用的是当前分支的 `qwen_process_cache.py` 旧式 pipeline。排查后确认：两者不是同一套实验入口，之前 `kv/v_only/none` 的 20 样本结果不能代表 `preprocess-exp` 中真实 FusionRAG reflect pipeline。

### 对齐脚本

参考脚本：

```text
origin/preprocess-exp:run_fusionrag.sh
origin/preprocess-exp:test_fusionrag_reflect.py
```

关键差异：

| item | 旧 ablation pipeline | preprocess-exp 真实 pipeline |
|---|---|---|
| entry | `tools_run_recompute_v_only_ablation.py` -> `qwen_process_cache.py` | `run_fusionrag.sh` -> `test_fusionrag_reflect.py` |
| data | `musique-pca-subset.jsonl` | `data/result_reflect.json` |
| task unit | 单 question CSV | main question + intermediate sub-questions |
| evaluation | 后处理 CSV judge / EM / ROUGE | 生成后立即 judge，main question 要求所有 sub-question 正确 |
| result columns | `Question, Real Answer, Pred Answer` | `Main Question, Sub Question, Ground Truth, Predicted, Correct, F1, EM, Reason` |
| preprocess | 旧 `qwen_process_cache.py` preprocess | reflect 文档级 global corpus + BGE top-k + on-demand KV + FusionRAG preprocess |

### Sanity check 设置

为了验证真实 pipeline，提取 `origin/preprocess-exp:test_fusionrag_reflect.py` 到当前工作区临时文件：

```text
test_fusionrag_reflect_preprocess_exp.py
```

运行配置：

```text
model = /mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct
data = ./data/result_reflect.json
method = FusionRAG
preprocess = True
recall_method = bge
preprocess_scope = global
topk = 10
rate = 0.15
revert_rope = True
max_samples = 20
judge = DashScope OpenAI-compatible deepseek-v3.2
```

结果目录：

```text
MOTIVATION_EXPERIMENTS/reflect_pipeline_sanity/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/
```

结果文件：

```text
rate_0.15_revert_rope.csv
rate_0.15_revert_rope.txt
judge_cache_v2.json
```

### Sanity check 结果

```text
Main Questions Accuracy: 16/18 (0.8889)
Sub Questions Accuracy: 29/32 (0.9062)
Average F1 Score: 0.5296
Average EM Score: 0.2188
```

注意：`max_samples=20` 中有 2 个 main question 被原始数据标记为不可测/跳过，因此 main denominator 是 18；sub-question denominator 是 32。

### 结论

先前 `recompute_v_only_ablation` 的 30% DeepSeek judge accuracy 是旧 pipeline + 另一份数据格式下的结果，不应该用于评价真实 FusionRAG reflect pipeline。真实 `preprocess-exp` pipeline 在同样 20 个 main sample 范围内达到约 89% main accuracy / 91% sub-question accuracy，说明 pipeline 差异是主要原因。

下一步如果要比较 `kv / v_only / none`，必须把 K/V 写回 ablation 接到 `test_fusionrag_reflect.py` 真实入口上，而不是继续用 `qwen_process_cache.py` 的旧实验入口。

## 2026-06-29 K/V 单独更新实现 review

### 检查目标

怀疑 `只更新 K` / `只更新 V` 的实现可能有问题，因此重新检查真实 reflect pipeline 中 strict ablation 的写回逻辑。

### 代码路径

真实实验入口：

```text
test_fusionrag_reflect_preprocess_exp.py
```

当 `rate != 1` 时会进入：

```text
ktransformers/util/utils.py::load_kv_and_generate
```

当 `rate == 1` 时不会进入 K/V ablation，而是直接走 full recompute：

```text
prefill_and_generate(...)
```

因此 `rate=1` 只能作为 full attention/full recompute baseline，不能用来判断 `kv / v_only / k_only` 三种写回模式是否正确。

### Strict ablation 写回逻辑

当前 clean strict 实验使用：

```bash
FUSIONRAG_STRICT_REPROCESS_ABLATION=1
FUSIONRAG_CLEAN_STRICT_ABLATION=1
FUSIONRAG_REPROCESS_UPDATE_MODE={kv,v_only,k_only,none}
```

语义如下：

```text
kv     : 选中 token 的 K 和 V 都保留重算结果
v_only : 恢复原始 K，只保留重算后的 V
k_only : 恢复原始 V，只保留重算后的 K
none   : K 和 V 都恢复原始值
```

代码 review 结论：strict 分支中的 restore 方向没有写反。`v_only` 确实是只更新 V，`k_only` 确实是只更新 K。

### Sanity check

我临时加入了一个默认关闭的 debug 开关：

```bash
FUSIONRAG_DEBUG_RESTORE_CHECK=1
```

它会在 strict restore 之后，对 selected doc token 的 K/V cache 与重算前原始 K/V 做 max absolute diff 检查。最小样本验证命令使用 `max_samples=1, rate=0.3, topk=10, preprocess=True, reprocess_method=FusionRAG`。

日志目录：

```text
MOTIVATION_EXPERIMENTS/kv_update_restore_debug/
```

检查结果：

```text
mode=kv     selected=393 key_diff=15.875 value_diff=20.625
mode=v_only selected=393 key_diff=0      value_diff=20.625
mode=k_only selected=393 key_diff=15.875 value_diff=0
mode=none   selected=393 key_diff=0      value_diff=0
```

这个结果直接说明：

```text
v_only 没有更新 K，只更新了 V；
k_only 没有更新 V，只更新了 K；
none 确实没有保留任何重算后的 K/V；
kv 确实同时保留了重算后的 K/V。
```

### 仍需注意的问题

1. `rate=1` 是 full recompute baseline，不是 “select 100% token 后只更新 K/V” 的实验路径。因此表里 `rate=1` 三个 mode 一样是预期现象，不应作为 K/V ablation 的证据。
2. 非 strict 分支会把 “重算 doc token + query token” 放在一次 forward 里，再 restore selected doc K/V。这个路径不适合做干净的 K/V 单独更新结论。后续只使用 strict 分支解释实验。
3. FusionRAG selection forward 会先用 query 计算 importance，并把 query suffix 写入 cache。后续 strict 分支会用显式 `cache_position` 重写完整 query，且 sparse attention kernel 使用绝对位置做 causal mask，因此目前看不构成 K/V 写反问题；但实现上仍然不够干净，后续可以在 selection 后显式恢复/清理 query cache 和 `past_tokens`，降低解释风险。

### 当前结论

只更新 K/V 的 strict 实现本身没有发现写反或 restore 顺序错误。之前 “只更新 K 和只更新 V 效果接近/甚至 K-only 更好” 不太像是简单实现 bug 导致的；更可能来自实验任务、query-conditioned selection、生成随机性/评估波动，或者 K/V 在后续 query attention 中的作用不是只由重算前后 delta 大小决定。

### 追加观察：为什么 v_only 随 rate 变化很弱

进一步检查 `v_only` 不同 rate 的结果文件，发现它并不是完全没有生效。

不同 rate 的预测文本相同率：

```text
rate 0.15 vs 0.30: 173/250 = 69.2%
rate 0.15 vs 0.50: 152/250 = 60.8%
rate 0.15 vs 0.80: 139/250 = 55.6%
rate 0.30 vs 0.50: 181/250 = 72.4%
rate 0.30 vs 0.80: 157/250 = 62.8%
rate 0.50 vs 0.80: 182/250 = 72.8%
```

说明 rate 增大确实改变了不少生成结果，但这些变化没有稳定转化为 F1/EM 提升。

重算 token 数也确实随 rate 增大：

```text
v_only rate=0.30: avg prompt eval count = 454.1
v_only rate=0.50: avg prompt eval count = 739.6
v_only rate=0.80: avg prompt eval count = 1167.3
```

相对 rate=0 的 F1 变化分布：

```text
v_only rate=0.15: improved=41, worse=56, same=153, mean_delta=-0.0012
v_only rate=0.30: improved=38, worse=66, same=146, mean_delta=-0.0085
v_only rate=0.50: improved=47, worse=63, same=140, mean_delta=-0.0045
v_only rate=0.80: improved=51, worse=63, same=136, mean_delta=+0.0032
```

因此目前更准确的说法是：

```text
v_only 随 rate 增大会改变输出，但变化方向不稳定，平均指标被正负样本抵消。
```

可能原因：

1. `v_only` 保持 K 不变，因此 query 对文档 token 的 attention routing 基本仍由原始/preprocess K 决定。新增更新的 V 只有在对应 token 本来就被 query attend 到时才有明显作用。
2. rate 增大时新增的是 importance 排名更靠后的 token，这些 token 的 attention mass 可能较低；更新它们的 V 对最终 logits 影响有限。
3. K 的更新会改变后续 query 的匹配/路由，因此 `kv` 或 `k_only` 更可能随 rate 改变 retrieval-like 行为；只更新 V 更像是在固定路由下替换内容向量，收益会更弱。
4. 当前 `v_only/k_only` 是“完整重算 selected token 后，只控制最终 cache 写回哪些字段”的 ablation。也就是说内部重算过程仍然经过正常 attention，并不是数学上完全隔离的“只让 V 参与重算”。这个定义用于评估 cache writeback 是合理的，但解释时要说清楚。

后续如果要进一步确认机制，建议做一组 attention-mass 实验：对每个样本统计 query 对 selected token 的累计 attention mass，比较 top 15% / 30% / 50% / 80% 这些 token 实际承载了多少 query attention。如果 15% 以后新增 token 的 attention mass 很小，就能解释为什么 `v_only` 加 rate 不明显。

## 2026-06-29 Query Attention Mass vs Selected Token Rank

### 实验目的

验证 FusionRAG importance 排名前若干比例 token 是否已经覆盖大部分 query attention mass，从而解释 `v_only` 随 rate 增大平均 F1/EM 变化很弱的问题。

### 数据与设置

复用已有 score cache：

```text
MOTIVATION_EXPERIMENTS/query_recompute_overlap_32q_preprocess/details/
```

实验规模：

```text
20 examples
32 queries / example
640 query cases total
10 passages + system prompt
KV = preprocess top-10 KV cache
```

score 定义：

```text
对每个 query，用 preprocess KV 的 context K 计算 query-context QK softmax attention，
再对 layer/head/query token 求和。
排序和 mass 统计时排除 system tokens。
```

输出目录：

```text
MOTIVATION_EXPERIMENTS/query_attention_mass_by_selection_rank/
```

### 结果

| top rate | selected tokens mean | cumulative mass mean | mass p10 | mass p50 | mass p90 | incremental mass mean | incremental mass/token mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 367.4 | 0.3663 | 0.2725 | 0.3654 | 0.4557 | 0.3663 | 0.00100053 |
| 0.10 | 735.4 | 0.4632 | 0.3825 | 0.4623 | 0.5384 | 0.0969 | 0.00027297 |
| 0.15 | 1103.2 | 0.5331 | 0.4614 | 0.5320 | 0.5991 | 0.0699 | 0.00019659 |
| 0.20 | 1471.2 | 0.5898 | 0.5248 | 0.5893 | 0.6480 | 0.0568 | 0.00015961 |
| 0.30 | 2207.1 | 0.6812 | 0.6274 | 0.6799 | 0.7281 | 0.0914 | 0.00012849 |
| 0.50 | 3678.8 | 0.8155 | 0.7825 | 0.8146 | 0.8463 | 0.1343 | 0.00009443 |
| 0.80 | 5886.0 | 0.9491 | 0.9386 | 0.9493 | 0.9589 | 0.1336 | 0.00006274 |

### Observation

1. Attention/importance mass 有明显集中性，但不是极端集中。
   - top 5% token 覆盖约 36.6% mass。
   - top 15% token 覆盖约 53.3% mass。
   - top 30% token 覆盖约 68.1% mass。
   - top 50% token 覆盖约 81.6% mass。
   - top 80% token 覆盖约 94.9% mass。

2. 边际 token 的单位 attention mass 快速下降。
   - top 0-5% 区间平均每 token mass 约 `1.00e-3`。
   - 5-10% 区间降到 `2.73e-4`。
   - 10-15% 区间降到 `1.97e-4`。
   - 50-80% 区间只有 `6.27e-5`。

3. 这可以解释 `v_only` 的 rate 增大收益弱：
   - 只更新 V 时，K/routing 不变。
   - rate 后半段新增大量 token，但这些 token 被 query 实际读取的 mass/token 很低。
   - 因此输出会变，但质量提升不稳定，容易出现 improved/worse 抵消。

4. 这个结果同时支持 “selection 比 recompute rate 本身更关键”：
   - 如果新增 token 的 marginal mass 低，那么盲目提高 rate 不是好策略。
   - 更有价值的是预测/缓存高 mass token，或者减少低边际 token 的 online recomputation。

### 当前结论

这组实验没有证明 top 15% 已覆盖绝大部分 query attention，但证明了 attention mass 的显著长尾分布。它支持一个更稳妥的 motivation：

```text
FusionRAG 的收益主要来自高 importance / 高 attention-mass token；
rate 增大后新增 token 的边际贡献快速下降，因此只更新 V 的收益不稳定。
```

## 2026-06-29 Query Routing Shift after K/V Update

### 实验目的

进一步验证 `k_only` 为什么经常比 `v_only` 更有效：K update 是否真的改变 query 对文档 chunk 的 attention routing。

### 实验设置

选取 4 个代表性 case，均来自 `kv` 正确而 `v_only` 容易错误的样本：

```text
art_brut_lead_singer
national_dream_author
bartrams_bridge_water
first_african_american_candidate
```

对每个 case 构造 4 种 cache 状态：

```text
preprocess : 初始 preprocess KV，不做 online recompute
k_only     : 对 selected token 重算，但只保留 K
v_only     : 对 selected token 重算，但只保留 V
kv         : 对 selected token 重算，保留 K/V
```

统计 query attention mass：

```text
selected_mass       : query attention 落在 selected token 上的比例
gt_chunk_mass       : query attention 落在包含 gold answer 的 chunk 上的比例
wrong_chunk_mass    : query attention 落在包含错误预测实体的 chunk 上的比例
last-layer variants : 只看最后一层时的对应 mass
```

输出目录：

```text
MOTIVATION_EXPERIMENTS/query_routing_shift_after_k_update/
```

### 关键结果

#### case 1: Art Brut lead singer

标准答案是 `Eddie Argos`，`v_only` 错到 `Sxip Shirey`。

```text
mode        selected_mass   gt_chunk_mass   wrong_chunk_mass
preprocess  0.1758          0.0612          0.0335
k_only      0.2264          0.1051          0.0345
v_only      0.1625          0.0377          0.0367
kv          0.2299          0.1184          0.0310
```

Observation：

```text
k_only / kv 明显提高 gold chunk mass；
v_only 反而降低 gold chunk mass，并让 wrong chunk mass 略高。
```

这支持 “K update 改变 routing，V-only 不能修正错误 routing”。

#### case 2: Bartram's Covered Bridge water

标准答案是 `Crum Creek`，`v_only` 错到 `Mad River`。

```text
mode        selected_mass   gt_chunk_mass   wrong_chunk_mass
preprocess  0.1466          0.0815          0.0212
k_only      0.1184          0.1141          0.0188
v_only      0.1453          0.0812          0.0210
kv          0.1215          0.1168          0.0193
```

Observation：

```text
k_only / kv 将 gold chunk mass 从约 0.08 提高到约 0.115；
v_only 与 preprocess 几乎一致。
```

这是最清楚的 routing-shift case：只更新 K 已经能把 query 注意力往正确 chunk 拉。

#### case 3: National Dream author

标准答案是 `Pierre Berton`，`v_only` 错到 `Ludwig von Mises`。

```text
mode        selected_mass   gt_chunk_mass   wrong_chunk_mass
preprocess  0.1558          0.0295          0.1017
k_only      0.1672          0.0368          0.0892
v_only      0.1543          0.0275          0.1023
kv          0.1628          0.0351          0.0862
```

Observation：

```text
k_only / kv 小幅提高 gold chunk mass，并降低 wrong chunk mass；
v_only 基本不改变甚至略差。
```

这是中等强度支持。

#### case 4: First African American presidential candidate

标准答案是 `Frederick Douglass`，`v_only` 错到 `Barack Obama`。

```text
mode        selected_mass   gt_chunk_mass   wrong_chunk_mass
preprocess  0.1608          0.0521          0.1360
k_only      0.1888          0.0564          0.1363
v_only      0.1583          0.0518          0.1356
kv          0.1888          0.0572          0.1379
```

Observation：

```text
K update 提高 selected_mass，但没有明显把 mass 从 wrong chunk 转向 gold chunk。
```

这是负例。说明 K update 并不总能修正 routing，或者 chunk-level mass 太粗，没捕捉 token-level 细节。

### 当前结论

这组小样本机制实验支持一个更精确的说法：

```text
K update 的一个重要作用是改变 query routing；
当它能把 attention mass 推向 gold chunk 时，k_only/kv 明显优于 v_only。
但 K update 不是万能，部分样本中 chunk-level routing 没有明显修正。
```

因此，`k_only > v_only` 的现象不能简单解释成 “K 比 V 更重要”，更准确是：

```text
V update 只能改善被读取 token 的内容；
K update 有机会改变 query 读取哪些 token/chunk。
如果原始 routing 错了，只更新 V 很难救回来。
```

## 2026-06-29 Selection Method vs Attention Mass

### 实验目的

确认 “被选为重算 token 的位置是否承担主要 attention mass” 是否对不同 selection 方法都成立，并加入随机 selection 空白对照。

### 实验设置

评价 attention mass 的统一口径：

```text
主模型在 preprocess KV 上的 query-context QK softmax attention；
排序和统计时排除 system tokens。
```

比较的 selector：

```text
FusionRAG-QK(preprocess,32q): 主模型 QK selector，20 examples x 32 queries
FusionRAG-QK(preprocess,16q): 主模型 QK selector，20 examples x 16 queries
Draft-selector(raw,16q): draft model selector，评价仍使用主模型 preprocess attention mass
random_mass: 同样 token 数随机选取，64 次均值
stable_frequency_mass: 同一 example 内所有 query 的 selector top-k 投票得到的 query-agnostic stable set
oracle_attention_mass: 直接按评价 attention score top-k，作为上界
```

输出目录：

```text
MOTIVATION_EXPERIMENTS/selection_method_attention_mass_compare/
```

### 关键结果

#### FusionRAG-QK selector

主模型 QK selector 明显高于随机：

```text
rate=0.15: selector mass=0.5331, random=0.1498, enrichment=3.56x
rate=0.30: selector mass=0.6812, random=0.3000, enrichment=2.27x
rate=0.50: selector mass=0.8155, random=0.5004, enrichment=1.63x
```

这说明：

```text
FusionRAG 选中的 token 确实是主模型 query attention 高度集中的 token；
这个现象不是随机选 token 的自然结果。
```

#### Stable frequency set

query-agnostic stable set 也接近 FusionRAG-QK：

```text
rate=0.15: stable=0.5198 vs selector=0.5331
rate=0.30: stable=0.6689 vs selector=0.6812
rate=0.50: stable=0.8068 vs selector=0.8155
```

这说明同一 document 内确实存在比较稳定的高-attention token set。这个结果支持 “selection 可以部分缓存/离线化” 的方向。

#### Draft selector

Draft selector 在这个口径下几乎等于随机：

```text
rate=0.15: draft=0.1491, random=0.1497
rate=0.30: draft=0.2914, random=0.3000
rate=0.50: draft=0.4958, random=0.5003
```

这个结果很关键：

```text
并不是任意 selection 方法都会选中主模型真正 attend 的 token；
draft(raw) selector 至少在当前保存的这组 score 上，并没有对齐主模型 preprocess attention。
```

### 当前结论

这组实验把前面的 observation 说得更清楚：

```text
“selected token 承担主要 attention mass” 是 FusionRAG-QK selector 的性质；
随机 selection 不具备这个性质；
draft(raw) selector 在当前口径下也几乎不具备这个性质；
stable frequency set 却能保留大部分 FusionRAG-QK 的 attention mass。
```

因此，如果要减少 online selection 成本，目前最有价值的方向不是直接换 draft selector，而是：

```text
利用 document-level stable high-attention set，
离线缓存一批高概率会被 query attend 的 token，
在线只做少量 query-specific refinement。
```

## 实验：真实 FusionRAG pipeline 下，不同 selector 选中 token 的后续 attention mass

### 背景

前一组 `selection_method_attention_mass_compare` 使用的是保存下来的 score/attention cache，更多是在回答：

```text
selector 选出的 token，在该 selector/score 口径下是不是高分 token？
```

这个口径有价值，但它不等价于真实 online 流程中这些 token 是否真的被后续 query/decode 使用。因此补做一组真实 pipeline 实验。

### 实验设置

输出目录：

```text
MOTIVATION_EXPERIMENTS/real_pipeline_selector_attention_mass/
```

启动脚本：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
MAX_MAIN=80 MAX_SUB=160 ./run_real_pipeline_selector_attention_mass.sh \
  /raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/real_pipeline_selector_attention_mass
```

实际完成：

```text
15 组 = 3 selectors x 5 rates
每组 109 个 reflect 子问题
总明细 1635 条
```

真实流程：

```text
加载 offline preprocess KV
-> selector 按 rate 选 doc token
-> 对选中 doc token 做 online KV recompute
-> query prefill
-> 首 token decode
```

selectors：

```text
fusionrag_qk:
  FusionRAG 原本的主模型 QK/importance selector。

draft:
  3B draft model selector。
  使用后半层 query-to-doc attention，经 RRF(k=18)+smart selection。
  这不是 random，也不是裸 top-k。

random:
  在同样 doc token 范围内等数量随机选 token，作为空白对照。
```

rates：

```text
0.05, 0.15, 0.30, 0.50, 0.80
```

attention mass 定义：

```text
在 recompute 后的当前 KV 上，用 full-softmax 诊断重新计算 attention 分布，
统计 attention 落在 selector 选中 doc token 上的比例。
```

注意：

```text
这个统计不是 selector 自身的 score；
它回答的是：被选中并重算的 token，后续 query/decode 是否真的 attend 到它们。
```

统计两个阶段：

```text
query_mass:
  query prefill 阶段，query tokens 对 selected doc tokens 的 attention mass。

decode_mass:
  首个生成 token decode 阶段，对 selected doc tokens 的 attention mass。
```

### 关键结果

| selector | rate | n | selected | query mass mean | query last | decode mass mean | decode last | selection s | recompute s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| draft | 0.05 | 109 | 72.0 | 0.0778 | 0.0713 | 0.0782 | 0.0991 | 0.1398 | 0.0362 |
| draft | 0.15 | 109 | 217.0 | 0.1457 | 0.1340 | 0.1473 | 0.1806 | 0.1419 | 0.0536 |
| draft | 0.30 | 109 | 434.4 | 0.1910 | 0.1862 | 0.1935 | 0.2475 | 0.1404 | 0.0784 |
| draft | 0.50 | 109 | 724.5 | 0.2297 | 0.2349 | 0.2289 | 0.2970 | 0.1412 | 0.1081 |
| draft | 0.80 | 109 | 1159.2 | 0.2647 | 0.2852 | 0.2636 | 0.3679 | 0.1396 | 0.1542 |
| fusionrag_qk | 0.05 | 109 | 72.0 | 0.0476 | 0.0563 | 0.0403 | 0.0432 | 0.0304 | 0.0342 |
| fusionrag_qk | 0.15 | 109 | 217.0 | 0.1054 | 0.1182 | 0.0973 | 0.1152 | 0.0303 | 0.0524 |
| fusionrag_qk | 0.30 | 109 | 434.4 | 0.1541 | 0.1715 | 0.1455 | 0.1706 | 0.0314 | 0.0770 |
| fusionrag_qk | 0.50 | 109 | 724.5 | 0.1968 | 0.2137 | 0.1860 | 0.2208 | 0.0295 | 0.1072 |
| fusionrag_qk | 0.80 | 109 | 1159.2 | 0.2385 | 0.2530 | 0.2239 | 0.2899 | 0.0309 | 0.1541 |
| random | 0.05 | 109 | 72.0 | 0.0296 | 0.0392 | 0.0269 | 0.0425 | 0.0000 | 0.0357 |
| random | 0.15 | 109 | 217.0 | 0.0686 | 0.0822 | 0.0615 | 0.0904 | 0.0000 | 0.0540 |
| random | 0.30 | 109 | 434.4 | 0.1166 | 0.1261 | 0.1049 | 0.1343 | 0.0000 | 0.0788 |
| random | 0.50 | 109 | 724.5 | 0.1720 | 0.1740 | 0.1619 | 0.2065 | 0.0000 | 0.1082 |
| random | 0.80 | 109 | 1159.2 | 0.2437 | 0.2489 | 0.2381 | 0.2995 | 0.0000 | 0.1555 |

相对 random 的富集倍数：

| selector | rate | query mass / random | decode mass / random |
|---|---:|---:|---:|
| draft | 0.05 | 2.63x | 2.91x |
| draft | 0.15 | 2.12x | 2.40x |
| draft | 0.30 | 1.64x | 1.84x |
| draft | 0.50 | 1.34x | 1.41x |
| draft | 0.80 | 1.09x | 1.11x |
| fusionrag_qk | 0.05 | 1.61x | 1.50x |
| fusionrag_qk | 0.15 | 1.54x | 1.58x |
| fusionrag_qk | 0.30 | 1.32x | 1.39x |
| fusionrag_qk | 0.50 | 1.14x | 1.15x |
| fusionrag_qk | 0.80 | 0.98x | 0.94x |

### 观察

1. 在真实 query/decode full-softmax 诊断口径下，selector 的优势没有离线 score-cache 自评那么夸张。

这说明前面的离线实验更像是在验证 selector score 自身的集中性，而这组实验更接近真实后续 attention 使用情况。两者不能混为一谈。

2. draft selector 在这组真实 pipeline 口径下明显高于 random，尤其低 rate 更明显。

```text
rate=0.05:
  query mass 2.63x random
  decode mass 2.91x random

rate=0.15:
  query mass 2.12x random
  decode mass 2.40x random
```

这说明 draft selector 并不是随机；它在真实 KV recompute 后的后续 attention 上确实能覆盖更多被关注 token。

3. FusionRAG-QK 也高于 random，但在这组 full-softmax 诊断中弱于 draft selector。

这和前一组离线 score-cache 结果相反，提示需要进一步检查两个问题：

```text
FusionRAG-QK 的 saved importance 口径是否和 full-softmax diagnostic 完全一致；
真实 query/decode attention 是否更偏向 draft selector 捕获的 token 类型。
```

4. rate 越大，相对 random 的富集倍数越低。

这符合直觉：rate 越高，随机集合本身覆盖的 token 越多，selector 的边际优势会被稀释。到 rate=0.8 时，FusionRAG-QK 基本接近 random，说明高 rate 下 selection 的意义变小，主要是在“多重算”。

5. recompute 时间主要由 selected token 数决定，三种 selector 在同一 rate 下几乎一致。

例如：

```text
rate=0.05: recompute 约 0.034-0.036s
rate=0.80: recompute 约 0.154-0.156s
```

selector 类型主要影响 selection_time 和被选 token 的质量，不明显影响 recompute kernel 成本。

### 当前结论

这组实验回答了用户提出的问题：

```text
如果某些 token 被选为重算 token，后续 attention 是否主要分布到这些 token 上？
```

答案是：

```text
低 rate 下，是的，合理 selector 选中的 token 比 random 承接明显更多 attention；
但这种集中性没有离线 selector-score 自评那么强；
rate 越高，selector 和 random 的差距越小。
```

目前最值得继续查的是：

```text
为什么真实 full-softmax 诊断下 draft selector 比 FusionRAG-QK 更强；
这个现象是否会对应到最终 answer quality；
以及 FusionRAG-QK importance cache 和 full-softmax diagnostic 的统计口径是否存在偏差。
```

## 补充实验：attention mass 去向分解

### 目的

前面 `query_mass` 只统计了 selected doc token 的 attention 占比。为了回答 “剩余 attention 去了哪里”，补充把 full-softmax diagnostic 的 attention mass 拆成四个互斥部分：

```text
system:
  system prompt tokens。

selected_doc:
  selector 选中的 document tokens。

unselected_doc:
  没被 selector 选中的 document tokens。

query_or_current:
  query prefill 阶段是 query 内部可见 tokens；
  first-token decode 阶段是原 query tokens + 当前生成 token。
```

额外记录：

```text
doc_total = selected_doc + unselected_doc
system_first16 = system 前 16 个 token 的非互斥子集，用于观察 attention sink。
```

输出文件：

```text
MOTIVATION_EXPERIMENTS/real_pipeline_selector_attention_mass/breakdown_README.md
MOTIVATION_EXPERIMENTS/real_pipeline_selector_attention_mass/breakdown_summary.csv
```

### Query prefill breakdown

| selector | rate | system | selected doc | unselected doc | doc total | query/self | system first16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| draft | 0.05 | 0.4437 | 0.0778 | 0.1752 | 0.2530 | 0.3033 | 0.2981 |
| draft | 0.15 | 0.4427 | 0.1457 | 0.1179 | 0.2637 | 0.2937 | 0.2986 |
| draft | 0.30 | 0.4414 | 0.1910 | 0.0799 | 0.2710 | 0.2876 | 0.2995 |
| draft | 0.50 | 0.4386 | 0.2297 | 0.0473 | 0.2770 | 0.2844 | 0.2987 |
| draft | 0.80 | 0.4360 | 0.2647 | 0.0161 | 0.2808 | 0.2832 | 0.2976 |
| fusionrag_qk | 0.05 | 0.4588 | 0.0476 | 0.1891 | 0.2367 | 0.3045 | 0.3027 |
| fusionrag_qk | 0.15 | 0.4550 | 0.1054 | 0.1437 | 0.2491 | 0.2959 | 0.3029 |
| fusionrag_qk | 0.30 | 0.4508 | 0.1541 | 0.1045 | 0.2586 | 0.2906 | 0.3015 |
| fusionrag_qk | 0.50 | 0.4470 | 0.1968 | 0.0699 | 0.2668 | 0.2862 | 0.2999 |
| fusionrag_qk | 0.80 | 0.4431 | 0.2385 | 0.0350 | 0.2736 | 0.2833 | 0.2982 |
| random | 0.05 | 0.4490 | 0.0296 | 0.2158 | 0.2454 | 0.3056 | 0.2988 |
| random | 0.15 | 0.4432 | 0.0686 | 0.1901 | 0.2587 | 0.2981 | 0.2970 |
| random | 0.30 | 0.4382 | 0.1166 | 0.1542 | 0.2708 | 0.2911 | 0.2956 |
| random | 0.50 | 0.4347 | 0.1720 | 0.1076 | 0.2796 | 0.2857 | 0.2947 |
| random | 0.80 | 0.4329 | 0.2437 | 0.0417 | 0.2854 | 0.2817 | 0.2943 |

### 直接解释 draft rate=0.80

`draft rate=0.80` 的 query prefill：

```text
system:         43.60%
selected_doc:   26.47%
unselected_doc:  1.61%
query/self:     28.32%
```

因此，之前的 `selected_doc=26.47%` 不是说剩下 20% doc token 占了 73.53%。实际未选中的 20% doc token 只占 1.61%。剩余 attention 主要去了 system prompt 和 query/self，其中 system 前 16 个 token 单独占 29.76%，说明 attention sink 很强。

### 补充：document 内部归一化 selected/doc

已在完整 breakdown 表中增加 `selected/doc` 列：

```text
selected/doc = selected_doc / doc_total
```

完整更新表：

```text
MOTIVATION_EXPERIMENTS/real_pipeline_selector_attention_mass/breakdown_README.md
MOTIVATION_EXPERIMENTS/real_pipeline_selector_attention_mass/breakdown_summary.csv
```

关键 query prefill 结果：

| selector | rate=0.05 | rate=0.15 | rate=0.30 | rate=0.50 | rate=0.80 |
|---|---:|---:|---:|---:|---:|
| draft selected/doc | 0.3074 | 0.5528 | 0.7050 | 0.8293 | 0.9425 |
| fusionrag_qk selected/doc | 0.2010 | 0.4231 | 0.5960 | 0.7379 | 0.8719 |
| random selected/doc | 0.1207 | 0.2652 | 0.4306 | 0.6153 | 0.8538 |

这个指标说明：在 document attention 内部，draft selector 的 selected token 承接比例明显高于 random，FusionRAG-QK 也高于 random，但弱于 draft。以 `draft rate=0.80` 为例，selected token 占 doc attention 的 94.25%，未选中 doc token 只剩 5.75% 的 doc attention。

## 实验：Full-attention query->doc 分布作为 oracle selector

### 问题

进一步验证一个上界假设：

```text
如果我们默认知道真正 full attention 中 query 对前面 document tokens 的 attention 分布，
并按这个理想分布挑选需要 online recompute 的 preprocess KV tokens，
它是否会比 FusionRAG-QK selector 更好？
```

### 实验定义

`oracle_full_attention`：

```text
1. 对完整 context + query 跑一次真正 full-context forward；
2. 在每层、每个 head、每个 query token 上计算 full-softmax attention；
3. 取 query -> doc token attention 的 all-layer 平均分布；
4. 按 rate 从这个理想分布中取 top tokens；
5. 把这些 token 当成 FusionRAG 的 online recompute token；
6. 后续仍然加载 preprocess KV，并走同一套 FusionRAG recompute + generation pipeline。
```

对照：

```text
fusionrag_qk:
  当前 FusionRAG 在 preprocess KV 上的 QK/importance selector。

draft_selected_set:
  draft model selector 选出的 token 集合，只用于 selector 覆盖分析。

random:
  同样数量随机 token。
```

输出目录：

```text
MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare/
```

关键文件：

```text
selector_summary_all_rates.csv
answer_summary_all_rates.csv
README.md
```

### Selector 对 full-attention 理想分布的覆盖

`ideal mass` 表示 selector 选中的 token 覆盖了多少 full-attention query->doc 理想分布质量。

| rate | oracle ideal mass | draft ideal mass | fusionrag_qk ideal mass | random ideal mass |
|---:|---:|---:|---:|---:|
| 0.05 | 0.2377 | 0.0852 | 0.0644 | 0.0496 |
| 0.15 | 0.4306 | 0.2187 | 0.1864 | 0.1501 |
| 0.30 | 0.6002 | 0.3907 | 0.3491 | 0.2999 |
| 0.50 | 0.7566 | 0.5950 | 0.5565 | 0.4997 |
| 0.80 | 0.9241 | 0.8513 | 0.8456 | 0.7991 |

这个结果符合预期：

```text
oracle_full_attention 是理想分布 top-k，上界最高；
draft 和 FusionRAG-QK 都高于 random；
draft 在低/中 rate 下覆盖 full-attention 理想分布略高于 FusionRAG-QK。
```

### 真实生成指标

这里比较的是把 selector 选出的 token 喂回 FusionRAG preprocess KV pipeline 后，真实生成答案的 F1/EM。

| rate | fusionrag_qk F1 | oracle F1 | delta F1 | fusionrag_qk EM | oracle EM | delta EM |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.4421 | 0.4189 | -0.0232 | 0.1927 | 0.1743 | -0.0183 |
| 0.15 | 0.4764 | 0.4394 | -0.0369 | 0.2110 | 0.1927 | -0.0183 |
| 0.30 | 0.5170 | 0.4747 | -0.0423 | 0.2385 | 0.2202 | -0.0183 |
| 0.50 | 0.5386 | 0.4892 | -0.0493 | 0.2294 | 0.2110 | -0.0183 |
| 0.80 | 0.5183 | 0.5739 | +0.0556 | 0.2294 | 0.2661 | +0.0367 |

### 关键观察

1. oracle selector 对 full-attention 理想分布的覆盖显著更高。

这说明代码逻辑是符合定义的：如果目标就是 “覆盖 full attention query->doc attention mass”，oracle 是上界。

2. 但低/中 rate 下，oracle selector 的回答性能反而低于 FusionRAG-QK。

这说明：

```text
更接近 full-attention query attention 分布，
不等价于更适合作为 online recompute token selector。
```

可能原因：

```text
full attention 的 query->doc attention 反映的是原始完整上下文下 query 读取哪些 token；
FusionRAG-QK selector 反映的是 preprocess KV 状态下哪些 token 更需要被 online 更新；
二者目标并不完全一致。
```

换句话说，重算 token 的目标不是单纯复刻 full attention 的 query attention 分布，而是修正 preprocess KV 中对答案生成最敏感的 token。

3. rate=0.80 时 oracle 反而明显更好。

这说明当预算足够大时，oracle 的覆盖优势终于转化成回答质量提升：

```text
rate=0.80:
oracle F1 0.5739 vs fusionrag_qk F1 0.5183
oracle EM 0.2661 vs fusionrag_qk EM 0.2294
```

但这个 rate 下已经接近大规模重算，selection 的系统收益会下降。

### 当前 insight

这组实验给出的结论不是 “我们应该直接用 full-attention attention map 做 selector”。相反，更重要的 insight 是：

```text
selector 的优化目标不能只定义成 query attention distribution matching。
```

更合适的方向可能是：

```text
寻找哪些 token 的 KV 更新会最大程度改变最终答案 logits / answer quality，
而不是只寻找 full attention 中 query 最 attend 的 token。
```

这也解释了为什么之前 “K/V 更新变化量排序” 和 “attention selector 排序” 不完全一致：attention high token 不一定就是 recompute-sensitive token。

### 补充：加入 draft selector 的真实性能

上面的真实生成指标只比较了 `fusionrag_qk` 和 `oracle_full_attention`。之后补跑了 `draft_selected_set`，即 draft model selector 选 token 后，同样走 FusionRAG preprocess KV + online recompute + generation pipeline。

输出目录：

```text
MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare_with_draft_answer/
```

紧凑汇总：

```text
answer_summary_compact.csv
README_compact.md
```

真实生成结果：

| rate | draft F1 | fusionrag_qk F1 | oracle F1 | draft EM | fusionrag_qk EM | oracle EM |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.5199 | 0.4421 | 0.4189 | 0.2294 | 0.1927 | 0.1743 |
| 0.15 | 0.5383 | 0.4764 | 0.4394 | 0.2202 | 0.2110 | 0.1927 |
| 0.30 | 0.5619 | 0.5170 | 0.4747 | 0.2477 | 0.2385 | 0.2202 |
| 0.50 | 0.5799 | 0.5386 | 0.4892 | 0.2752 | 0.2294 | 0.2110 |
| 0.80 | 0.5718 | 0.5183 | 0.5739 | 0.2569 | 0.2294 | 0.2661 |

新观察：

```text
draft selector 在 rate=0.05~0.50 下都明显优于 FusionRAG-QK 和 oracle_full_attention；
rate=0.80 时 oracle F1 略高于 draft，但 draft 仍接近 oracle；
FusionRAG-QK 在这组实验中整体弱于 draft selector。
```

这进一步支持一个方向：

```text
draft model 的价值可能不在于替代主模型计算，
而在于作为更好的 online recompute token selector。
```

同时也进一步否定了一个简单假设：

```text
只要 selector 更接近 full-attention query->doc attention 分布，就一定生成更好。
```

因为 oracle 对理想分布覆盖最高，但低/中 rate 真实 F1 反而最低。

### 补充：使用 GLM-5.2 judge 评估 accuracy

对上一轮 `oracle_full_attention / fusionrag_qk / draft_selected_set` 的生成答案，额外使用 GLM-5.2 作为语义 judge 评估 accuracy。这里只对已有答案做离线 judge，没有重新生成。

启动命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  tools_judge_oracle_draft_results_with_glm.py \
  --workers 6 \
  --log_every 50
```

输出目录：

```text
MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare_with_draft_answer/glm_judge/
```

关键文件：

```text
README_glm_judge.md
glm_judge_summary.csv
all_glm_judged.csv
answer_detail_rate*.glm_judged.csv
glm_judge_cache.jsonl
```

GLM judge accuracy：

| rate | selector | n | GLM accuracy | F1 mean | EM mean |
|---:|---|---:|---:|---:|---:|
| 0.05 | draft_selected_set | 109 | 0.7890 | 0.5199 | 0.2294 |
| 0.05 | fusionrag_qk | 109 | 0.7064 | 0.4421 | 0.1927 |
| 0.05 | oracle_full_attention | 109 | 0.7064 | 0.4189 | 0.1743 |
| 0.15 | draft_selected_set | 109 | 0.8165 | 0.5383 | 0.2202 |
| 0.15 | fusionrag_qk | 109 | 0.7431 | 0.4764 | 0.2110 |
| 0.15 | oracle_full_attention | 109 | 0.6606 | 0.4394 | 0.1927 |
| 0.30 | draft_selected_set | 109 | 0.8532 | 0.5619 | 0.2477 |
| 0.30 | fusionrag_qk | 109 | 0.7982 | 0.5170 | 0.2385 |
| 0.30 | oracle_full_attention | 109 | 0.7248 | 0.4747 | 0.2202 |
| 0.50 | draft_selected_set | 109 | 0.8624 | 0.5799 | 0.2752 |
| 0.50 | fusionrag_qk | 109 | 0.8624 | 0.5386 | 0.2294 |
| 0.50 | oracle_full_attention | 109 | 0.7706 | 0.4892 | 0.2110 |
| 0.80 | draft_selected_set | 109 | 0.8532 | 0.5718 | 0.2569 |
| 0.80 | fusionrag_qk | 109 | 0.7890 | 0.5183 | 0.2294 |
| 0.80 | oracle_full_attention | 109 | 0.8257 | 0.5739 | 0.2661 |

观察：

```text
GLM judge 与 F1 趋势大体一致：
draft selector 在 rate=0.05/0.15/0.30 明显最好；
rate=0.50 时 draft 与 fusionrag_qk 的 GLM accuracy 持平，但 draft F1/EM 更高；
rate=0.80 时 oracle 的 F1/EM 最高，GLM accuracy 低于 draft 但高于 fusionrag_qk。
```

因此，如果用语义 judge 而不是 token-level F1，draft selector 仍然是低/中 rate 下最强的选择。

### 补充：selector 速度横向排版

为了更直观看速度和质量的 tradeoff，额外整理了横向速度表：

```text
MOTIVATION_EXPERIMENTS/oracle_full_attention_selector_compare_with_draft_answer/README_speed_comparison.md
```

这里的速度只统计 selector 选 token 的平均耗时，不包含后续 online recompute 和生成。

| rate | fusionrag_qk s | draft s | oracle_full_attention s | draft / qk | oracle / qk |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.0286 | 0.1381 | 0.3032 | 4.83x | 10.61x |
| 0.15 | 0.0267 | 0.1383 | 0.3026 | 5.17x | 11.32x |
| 0.30 | 0.0269 | 0.1386 | 0.3028 | 5.15x | 11.26x |
| 0.50 | 0.0270 | 0.1385 | 0.3031 | 5.12x | 11.21x |
| 0.80 | 0.0286 | 0.1388 | 0.3020 | 4.86x | 10.57x |

速度结论：

```text
FusionRAG-QK selector 最快，约 0.027-0.029s；
draft selector 约 0.138s，是 FusionRAG-QK 的约 5x；
oracle_full_attention 约 0.302-0.303s，是 FusionRAG-QK 的约 11x，而且需要 full attention，只能作为分析上界，不是在线可部署方法。
```

质量/速度 tradeoff：

```text
draft selector 明显更慢，但低/中 rate 下 F1 和 GLM accuracy 通常更好；
FusionRAG-QK selector 速度优势明显；
oracle_full_attention 最慢，并且低/中 rate 下质量反而不如 draft 和 FusionRAG-QK。
```

## 2026-06-30：Full Attention 下跨 Query 的文档热点 Token 稳定性

### 实验目的

之前我们观察到 FusionRAG/Draft selector 在不同 query 下会反复选中一批相似的 doc tokens。为了确认这是不是 selector 自身造成的偏置，这里改成在真实 full attention 条件下重新验证：

```text
固定同一个 example 的 system + all unique docs token 序列；
更换不同 query，包括 same-example related query 和 cross-example unrelated query；
对每个 query 跑 full attention；
统计 query tokens 对 doc tokens 的 attention score 分布；
按 attention score 从高到低选 top 15% doc tokens；
分析这些 top tokens 在不同 query 之间的集合重合度，以及每个 token 被选中的频率分布。
```

### 启动方式

脚本：

```text
tools_full_attention_query_anchor_stability.py
```

命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  tools_full_attention_query_anchor_stability.py \
  --max_examples 20 \
  --unrelated_queries 32 \
  --rate 0.15 \
  --output_dir MOTIVATION_EXPERIMENTS/full_attention_query_anchor_stability \
  --device cuda:0
```

### 实验设置

数据和模型：

```text
model: /mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct
data: ./data/result_reflect.json
examples: first 20 main questions
query pool: 每个 example 自己的 main/sub questions 作为 related query；
            从其他 example 抽取 32 个 main/sub questions 作为 unrelated query
doc sequence: 每个 example 的 system + all unique docs
top rate: 0.15
attention score: 对所有 layer/head 的 query-to-doc attention 做平均，再在 query token 维度聚合成每个 doc token 的 score
```

输出目录：

```text
MOTIVATION_EXPERIMENTS/full_attention_query_anchor_stability/
```

关键文件：

```text
README.md
example_summary.csv
query_detail.csv
stable_intersection_curve.csv
example_XXX_attention_distributions.npz
figures/example_XXX_frequency_hist.png
figures/aggregate_pairwise_jaccard.png
figures/stable_intersection_convergence.png
```

### 指标解释

```text
jaccard_all_mean:
  任意两个 query 的 top-15% doc token 集合的平均 Jaccard。

jaccard_related_related_mean:
  related query 之间的平均 Jaccard。

jaccard_unrelated_unrelated_mean:
  unrelated query 之间的平均 Jaccard。

jaccard_related_unrelated_mean:
  related query 和 unrelated query 之间的平均 Jaccard。

all_query_intersection_ratio:
  所有 query 共同选中的 doc tokens 数量 / 单个 query 的 top-15% token 数量。
  这个值越高，说明存在越强的跨 query 稳定 attention anchor set。

never_selected_ratio:
  从未被任何 query 选入 top-15% 的 doc tokens 比例。

always_selected_ratio:
  被所有 query 都选入 top-15% 的 doc tokens 比例，分母是全部 doc tokens。
```

### 主要结果

20 个 example、697 条 query 记录的 aggregate：

| metric | mean | median | min | max |
|---|---:|---:|---:|---:|
| jaccard_all_mean | 0.7452 | 0.7459 | 0.7062 | 0.7914 |
| jaccard_related_related_mean | 0.7579 | 0.7531 | 0.7116 | 0.8325 |
| jaccard_unrelated_unrelated_mean | 0.7469 | 0.7480 | 0.7069 | 0.7927 |
| jaccard_related_unrelated_mean | 0.7343 | 0.7283 | 0.7020 | 0.7808 |
| all_query_intersection_ratio | 0.7223 | 0.7176 | 0.6801 | 0.7722 |
| related_intersection_ratio | 0.8271 | 0.8197 | 0.7736 | 0.9000 |
| unrelated_intersection_ratio | 0.7262 | 0.7204 | 0.6801 | 0.7722 |
| never_selected_ratio | 0.6494 | 0.6406 | 0.6235 | 0.7079 |
| always_selected_ratio | 0.1081 | 0.1075 | 0.1019 | 0.1154 |
| selected_ge_50pct_ratio | 0.1336 | 0.1340 | 0.1284 | 0.1401 |
| selected_ge_80pct_ratio | 0.1202 | 0.1200 | 0.1153 | 0.1261 |

最不稳定的几个 example：

| example | doc len | queries | top count | all-query intersection / top-k | all-pair Jaccard |
|---:|---:|---:|---:|---:|---:|
| 19 | 2150 | 35 | 322 | 0.6801 | 0.7062 |
| 4 | 2605 | 35 | 390 | 0.6846 | 0.7228 |
| 14 | 2120 | 35 | 318 | 0.6887 | 0.7262 |
| 3 | 1243 | 35 | 186 | 0.6989 | 0.7461 |
| 16 | 2162 | 35 | 324 | 0.7068 | 0.7301 |

最稳定的几个 example：

| example | doc len | queries | top count | all-query intersection / top-k | all-pair Jaccard |
|---:|---:|---:|---:|---:|---:|
| 1 | 1205 | 34 | 180 | 0.7722 | 0.7914 |
| 15 | 1551 | 35 | 232 | 0.7586 | 0.7605 |
| 5 | 2727 | 35 | 409 | 0.7555 | 0.7475 |
| 12 | 2462 | 35 | 369 | 0.7507 | 0.7458 |
| 0 | 1171 | 34 | 175 | 0.7429 | 0.7783 |

### 当前 observation

```text
1. 在真实 full attention 下，query 对 doc tokens 的 attention hotspot 并不是完全 query-specific。
   即使换成 unrelated query，top-15% doc token 集合仍然高度重合。

2. 平均 pairwise Jaccard 约 0.745；
   related-related 是 0.758，related-unrelated 仍有 0.734。
   说明热点稳定性不只是由 query 语义相关性带来的。

3. 所有 query 的共同交集平均占单个 top-15% 集合的 72.2%。
   换句话说，如果每个 query 选 15% doc tokens，里面大约 10.8% 的全体 doc tokens 会被几乎所有 query 反复选中。

4. never-selected token 比例约 64.9%。
   这说明 full attention 的 doc-token 关注分布有明显两极分化：
   一批 token 长期作为 attention anchor，另一大批 token 几乎不会成为热点。

5. 这个现象支持一个新的系统方向：
   online 阶段不一定每次都从 query-specific selector 完整计算热点；
   可以考虑离线/半离线维护 document-level stable anchor set，
   online 只对 stable anchor 外的 query-sensitive residual tokens 做补充选择。
```

### 后续可做

```text
1. 把 stable anchor tokens 解码出来，检查它们是 chunk 边界、标点、标题、实体词，还是 attention sink 类型 token。
2. 分别统计 first layer / middle layer / last layer 的稳定性，确认稳定 anchor 来自全局 sink 还是语义层。
3. 对 rate=0.05/0.10/0.30 重复同一实验，画 stable intersection 随 rate 的曲线。
4. 用 stable anchor set 作为 selector 跑真实 FusionRAG pipeline，和 FusionRAG-QK / draft selector 对比 accuracy 与 latency。
```

### 补充：扩大 top token rate 后的稳定性

用户要求继续看更大的 top token 比例。这里不需要重新 forward，直接复用前面保存的 full-attention query-to-doc distributions，重新按不同 rate 取 top tokens 并统计稳定性。

输出目录：

```text
MOTIVATION_EXPERIMENTS/full_attention_query_anchor_stability_multirate/
```

启动方式：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python tools_analyze_full_attention_anchor_multirate.py
```

关键文件：

```text
README.md
aggregate_summary_by_rate.csv
example_summary_by_rate.csv
frequency_hist_by_rate.csv
stable_intersection_curve_by_rate.csv
figures/stability_metrics_vs_rate.png
figures/frequency_hist_rate_*.png
```

结果：

| rate | pairwise Jaccard | related-related | related-unrelated | all-query intersection/top-k | never selected | always selected | selected >=50% queries | selected >=80% queries |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 0.7452 | 0.7579 | 0.7343 | 0.7223 | 0.6494 | 0.1081 | 0.1336 | 0.1202 |
| 0.25 | 0.7854 | 0.7949 | 0.7769 | 0.7548 | 0.4930 | 0.1885 | 0.2304 | 0.2089 |
| 0.30 | 0.8111 | 0.8218 | 0.8032 | 0.7729 | 0.4434 | 0.2317 | 0.2811 | 0.2570 |
| 0.50 | 0.8860 | 0.8892 | 0.8797 | 0.8469 | 0.2847 | 0.4233 | 0.4854 | 0.4581 |
| 0.80 | 0.9503 | 0.9499 | 0.9471 | 0.9223 | 0.0915 | 0.7377 | 0.7945 | 0.7711 |

观察：

```text
1. 随着 top token rate 增大，集合重合度单调上升，这是预期现象。

2. 但更关键的是，在 rate=0.25/0.30 时，all-query intersection/top-k 已经达到 0.7548/0.7729。
   说明即使选择范围扩大到 25%-30%，大部分被选 token 仍然是跨 query 稳定的。

3. rate=0.30 时，related-unrelated Jaccard 仍有 0.8032。
   这进一步说明 full attention 的 doc-token hotspot 很大程度是 document-intrinsic 的，
   不只是由 query 语义相关性决定。

4. rate=0.80 时 always-selected ratio 达到 0.7377，never-selected ratio 只剩 0.0915。
   这个比例太大，更多反映集合选择范围本身变宽；真正有分析价值的区间更可能是 0.15-0.30。
```

## 2026-06-30：Full Attention Stable Anchor 解码分析

### 实验目的

前面的 full-attention 实验证明：固定同一个 example 的文档 token 序列后，不同 query 的 top-attention doc tokens 高度重合。这个现象还需要进一步解释：

```text
这些稳定 token 到底是什么？
它们是实体词/关键词，还是标点、空白、chunk 边界、文档末尾 token？
如果主要是语义 token，说明可能存在 document-intrinsic evidence anchors；
如果主要是结构 token 或末尾 token，则更像 attention sink / recency bias，需要谨慎作为创新点。
```

### 实验设置

不重新跑模型 forward，直接复用已有 full-attention query-to-doc attention distributions：

```text
source_dir:
  MOTIVATION_EXPERIMENTS/full_attention_query_anchor_stability

rate:
  0.15

stable anchor 定义:
  对每个 query 取 attention score 最高的 top 15% doc tokens；
  如果一个 doc token 被该 example 的所有 query 都选中，则记为 stable anchor。

解码方式:
  重新加载同一 tokenizer 和同一份 result_reflect.json；
  按原实验完全相同的方式重建 system + all unique docs token 序列；
  将 stable anchor 的 doc-token index 映射回 token id；
  用 tokenizer.decode 解码，并统计 token 类别和文档相对位置。
```

启动命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  tools_decode_full_attention_stable_anchors.py \
  --rate 0.15 \
  --freq_threshold 1.0 \
  --output_dir MOTIVATION_EXPERIMENTS/full_attention_stable_anchor_decode_rate015
```

输出目录：

```text
MOTIVATION_EXPERIMENTS/full_attention_stable_anchor_decode_rate015/
```

关键文件：

```text
README.md
stable_anchor_tokens.csv
stable_anchor_category_by_example.csv
stable_anchor_example_summary.csv
stable_anchor_examples.md
```

### 指标解释

```text
stable anchors / top-k:
  所有 query 都共同选中的 stable anchor 数量 / 单个 query top-15% token 数量。
  这个数值和前面 all-query intersection/top-k 对应。

stable anchors / all doc tokens:
  stable anchor 数量 / 全部 doc tokens 数量。
  表示稳定热点在整个文档 token 序列里的绝对占比。

category distribution:
  stable anchor token 解码后的粗粒度类型分布。

position distribution:
  stable anchor 在文档 token 序列中的相对位置分布。
```

### 结果

总体规模：

```text
examples: 20
total decoded stable anchors: 4385
stable anchors / top-k mean: 0.7223
stable anchors / all doc tokens mean: 0.1081
stable anchor mean relative position: 0.8840
```

类别分布：

| category | ratio | count |
|---|---:|---:|
| latin_word | 0.6091 | 2671 |
| punct_or_symbol | 0.2698 | 1183 |
| number | 0.0750 | 329 |
| space | 0.0353 | 155 |
| other_text | 0.0100 | 44 |
| newline_or_space | 0.0007 | 3 |

位置分布：

| document position bin | ratio | count |
|---|---:|---:|
| 00-10% | 0.0048 | 21 |
| 10-20% | 0.0046 | 20 |
| 20-30% | 0.0059 | 26 |
| 30-40% | 0.0080 | 35 |
| 40-50% | 0.0116 | 51 |
| 50-60% | 0.0239 | 105 |
| 60-70% | 0.0410 | 180 |
| 70-80% | 0.0787 | 345 |
| 80-90% | 0.1535 | 673 |
| 90-100% | 0.6680 | 2929 |

示例观察：

```text
Example 0 中，排名最前的 stable anchors 大量来自文档序列尾部：
  '.\n'
  ' the'
  'wards'
  ','
  ' north'
  ' Berkshire'
  ' village'
  ' Downs'

这些 token 既包含普通词和实体片段，也包含标点/换行。
更重要的是，它们在文档中的位置高度靠后。
```

### 结论

```text
1. stable anchors 不是纯空白或纯标点。
   约 60.9% 是 latin word，约 27.0% 是标点/符号，说明热点集合中确实包含大量正文 token。

2. 但是 stable anchors 有非常强的位置偏置。
   约 66.8% 的 stable anchors 位于文档 token 序列最后 10%，
   平均相对位置达到 0.884。

3. 因此，当前 full-attention stable hotspot observation 不能直接解释成“模型稳定关注文档语义证据”。
   更准确的说法是：
   full attention 下存在跨 query 稳定的 document-token hotspot，
   但这些 hotspot 很大程度混合了 recency bias / sequence-tail attention anchor。

4. 这个结果反而给下一步实验提供了明确方向：
   必须把位置偏置剥离掉，才能判断是否存在真正的语义 stable evidence anchors。
```

### 下一步实验建议

```text
1. 做 position-controlled stable anchor：
   在每个文档位置分桶内分别选 top attention token，避免全部被最后一个文档/尾部 token 占据。

2. 改成 per-document normalization：
   每个 retrieved document/chunk 内单独统计 top tokens，再合并 stable set，
   避免最后一个 document 因位置靠后天然占优。

3. 对比 reverse-document-order：
   把同一批 retrieved docs 的顺序反过来，再跑 full attention。
   如果 stable anchors 跟着序列末尾移动，说明主要是 recency；
   如果仍落在同一文档内容上，才说明更接近语义 anchor。

4. 做 chunk-level stable anchor：
   先统计哪些 chunk 被稳定关注，再在 chunk 内看 token；
   这样可以区分“稳定关注某个文档块”和“稳定关注文档序列末尾”。
```

## 2026-06-30：Per-Document Controlled Stable Anchor 实验

### 实验目的

上一组解码实验发现，full-attention stable anchors 有明显序列尾部偏置：约 66.8% 的 stable anchors 位于文档 token 序列最后 10%。因此，原始 global top-token 结果不能直接说明存在“语义稳定 evidence anchors”。

这一组实验要回答：

```text
如果控制文档顺序/文档末尾偏置，跨 query 稳定 hotspot 是否还存在？
如果仍然存在，说明 stable anchor 不完全是 recency bias；
如果显著消失，说明之前的现象主要来自序列位置。
```

### 实验设置

仍然不重新 forward，复用已经保存的 full-attention query-to-doc attention distribution。

对比两种选 token 方法：

```text
global_top:
  在整个拼接后的 all-doc token 序列上直接选 attention score 最高的 top x% tokens。
  这是之前实验使用的方法。

per_doc_top:
  在每个 retrieved document/chunk 内部分别选 attention score 最高的 top x% tokens，
  然后把每个文档内部选出的 token 合并。
  这样每个文档都会按自身长度获得固定比例名额，最后一个文档不能因为位置靠后独占 top tokens。
```

测试 rate：

```text
0.15, 0.30
```

启动命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
TMPDIR=/raid/home/hming/tmp /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  tools_analyze_full_attention_per_document_stability.py \
  --rates 0.15,0.30 \
  --output_dir MOTIVATION_EXPERIMENTS/full_attention_per_document_anchor_stability
```

输出目录：

```text
MOTIVATION_EXPERIMENTS/full_attention_per_document_anchor_stability/
```

关键文件：

```text
README.md
aggregate_summary.csv
example_summary.csv
stable_anchor_tokens.csv
stable_anchor_position_and_category.csv
doc_attention_mass.csv
doc_attention_mass_by_position.csv
```

### 指标解释

```text
Jaccard:
  任意两个 query 的 selected doc-token 集合平均重合度。

related-unrelated Jaccard:
  related query 和 unrelated query 的 selected token 集合平均重合度。
  这个指标更能说明稳定性是否独立于 query 语义。

all-query intersection/top-k:
  所有 query 共同选中的 token 数量 / 单个 query 的 top-k token 数量。
  表示最严格的 stable anchor 占比。

stable mean position:
  stable anchors 在拼接文档 token 序列中的平均相对位置。
  越接近 1，说明越集中在序列末尾。
```

### 结果

| rate | method | Jaccard | related-unrelated Jaccard | all-query intersection/top-k | stable/doc | never selected | stable mean position |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.15 | global_top | 0.7452 | 0.7343 | 0.7223 | 0.1081 | 0.6494 | 0.8840 |
| 0.15 | per_doc_top | 0.6291 | 0.6090 | 0.5028 | 0.0735 | 0.5731 | 0.5332 |
| 0.30 | global_top | 0.8111 | 0.8032 | 0.7729 | 0.2317 | 0.4434 | 0.7903 |
| 0.30 | per_doc_top | 0.7547 | 0.7432 | 0.6646 | 0.1971 | 0.4014 | 0.5089 |

文档位置上的 attention mass 分布：

| doc position bin | mean attention mass ratio | median |
|---|---:|---:|
| 00-10% | 0.0367 | 0.0351 |
| 10-20% | 0.0336 | 0.0334 |
| 20-30% | 0.0363 | 0.0359 |
| 30-40% | 0.0377 | 0.0297 |
| 40-50% | 0.0413 | 0.0359 |
| 50-60% | 0.0487 | 0.0393 |
| 60-70% | 0.0580 | 0.0447 |
| 70-80% | 0.0820 | 0.0766 |
| 80-90% | 0.0800 | 0.0789 |
| 90-100% | 0.1808 | 0.2066 |

### 结论

```text
1. 原始 global_top 的稳定性确实混入了很强的 recency bias。
   文档位置最后 10% 的 attention mass ratio 平均为 0.1808，
   明显高于前面大多数位置桶的 0.03-0.08。

2. per_doc_top 成功削弱了尾部偏置。
   rate=0.15 时，stable mean position 从 0.8840 降到 0.5332；
   rate=0.30 时，从 0.7903 降到 0.5089。
   这说明 per-document normalization 可以把 stable anchors 从序列末尾拉回到更均匀的位置。

3. 剥离位置偏置后，跨 query 稳定性仍然没有消失。
   rate=0.15 时，per_doc_top 的 related-unrelated Jaccard 仍有 0.6090，
   all-query intersection/top-k 仍有 0.5028。
   rate=0.30 时，related-unrelated Jaccard 仍有 0.7432，
   all-query intersection/top-k 仍有 0.6646。

4. 因此，更准确的 observation 是：
   full attention 下的稳定热点由两部分组成：
   一部分是强 recency/sequence-tail anchor；
   另一部分是在 per-document 控制后仍存在的 document-local stable anchors。

5. 这个结论比前一版更适合继续往系统设计走：
   可以把 selector 拆成 document-local stable anchors + query-sensitive residual tokens，
   而不是直接使用全局 stable anchors。
```

### 下一步

```text
1. 用 per_doc_top stable anchors 作为一个实际 selector 跑 FusionRAG pipeline，看 accuracy/latency。
2. 加一个 per_doc_stable + small query residual 的混合 selector。
3. 做 reverse-document-order full forward，进一步确认 recency bias 和内容 anchor 的边界。
```

## 2026-06-30：Offline Fixed Token Selector Phase 1

### 实验目的

这一组实验验证一个更接近系统优化的问题：

```text
能不能在 offline 阶段，不看当前真实 query，提前为每个 retrieved document sequence 固定选出一组 token；
在线阶段直接使用这个 fixed token set，减少 query-specific selector 的开销？
```

这里先不跑 answer pipeline，而是做 phase-1 机制验证：

```text
offline fixed set 是否能覆盖 held-out 当前 example query 的 online QK selected tokens？
```

如果 offline set 和 online selector 的 selected set 高度重合，说明 online selector 里存在大量可 offline 化的稳定成分。

### 防泄漏设置

本实验不使用当前 example 的真实问题来定义 offline set。

数据来自已有 Target-QK(preprocess KV) 32-query score cache：

```text
MOTIVATION_EXPERIMENTS/query_recompute_overlap_32q_preprocess/details/20260626_212157/
```

每个 example 有 32 个 query：

```text
1 个 native query
10 个 native_template query
21 个 control_other_example query
```

offline fixed set 构造只使用：

```text
control_other_example queries
```

评测时才和 held-out 当前 example queries 对比：

```text
native + native_template
```

因此：

```text
uses_current_example_question_for_offline_set = false
```

### Cache 设计

这次不直接保存某个 rate 的 selected set，而是复用/整理 rate-independent score cache：

```text
scores: 每个 calibration query 对每个 context token 的 QK importance score
labels: native / native_template / control_other_example
queries: query text
starts: system + document boundaries
system_len/context_len: 用于排除 system，只在 doc tokens 上选 token
```

后续 `0.10/0.15/0.30/0.50` 都从同一份 score cache 派生，不需要重新跑 selector。

### Offline Set 定义

对比三种方法：

```text
qk_calib_global_frequency:
  对每个 calibration query，在所有 doc tokens 上全局取 top-rate；
  统计 token 被选中频率；
  最终按频率选 top-rate。

qk_calib_per_doc_frequency:
  主方法。
  对每个 calibration query，在每个 retrieved document 内各自取 top-rate；
  统计 token 被选中频率；
  每个 document 内按频率选 top-rate。
  tie-break 使用 mean QK score。

random_per_doc:
  每个 document 内随机选同等比例 token，作为随机下界。
```

### 指标解释

```text
Jaccard:
  offline fixed set 与 held-out online QK selected set 的集合 Jaccard。

coverage online by offline:
  held-out online QK 选中的 tokens 中，有多少比例被 offline fixed set 覆盖。
  coverage = |online ∩ offline| / |online|

target_selection_mode:
  online QK selected set 的构造方式。
  global 表示全局 top-rate；
  per_doc 表示每个 document 内 top-rate。
```

### 启动命令

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
TMPDIR=/raid/home/hming/tmp /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  tools_offline_fixed_selector_phase1.py
```

输出目录：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/phase1_qk_score_cache_derived/
```

关键文件：

```text
README.md
score_cache_manifest.csv
fixed_set_manifest.csv
offline_vs_online_detail.csv
offline_vs_online_aggregate.csv
fixed_sets_npz/
```

### 主要结果

这里优先看 `heldout_all = native + native_template`。

| rate | target mode | offline method | Jaccard | coverage online by offline |
|---:|---|---|---:|---:|
| 0.10 | per_doc | qk_calib_per_doc_frequency | 0.6743 | 0.8040 |
| 0.10 | per_doc | random_per_doc | 0.0516 | 0.0981 |
| 0.15 | per_doc | qk_calib_per_doc_frequency | 0.6890 | 0.8145 |
| 0.15 | per_doc | random_per_doc | 0.0816 | 0.1508 |
| 0.30 | per_doc | qk_calib_per_doc_frequency | 0.7495 | 0.8559 |
| 0.30 | per_doc | random_per_doc | 0.1771 | 0.3009 |
| 0.50 | per_doc | qk_calib_per_doc_frequency | 0.8230 | 0.9025 |
| 0.50 | per_doc | random_per_doc | 0.3337 | 0.5004 |

global 对齐下，`qk_calib_global_frequency` 也很强：

| rate | target mode | offline method | Jaccard | coverage online by offline |
|---:|---|---|---:|---:|
| 0.15 | global | qk_calib_global_frequency | 0.6794 | 0.8075 |
| 0.30 | global | qk_calib_global_frequency | 0.7459 | 0.8534 |
| 0.50 | global | qk_calib_global_frequency | 0.8227 | 0.9022 |

### 结论

```text
1. offline calibration set 能非常强地预测 held-out 当前 query 的 online QK selected tokens。
   在主方法 qk_calib_per_doc_frequency@0.15 下，
   Jaccard = 0.6890，coverage online by offline = 0.8145。

2. 这个结果不是随机效应。
   同样 rate=0.15、per_doc 设置下，
   random_per_doc 的 Jaccard 只有 0.0816，coverage 只有 0.1508。

3. 因为 offline set 只使用 control_other_example queries 构造，
   没有使用当前 example 的 native/native_template query，
   所以这个结果支持“QK selector 存在可 offline 化的 query-stable 成分”。

4. rate 越大，offline set 对 online selector 的 coverage 越高：
   rate=0.10: coverage 0.8040
   rate=0.15: coverage 0.8145
   rate=0.30: coverage 0.8559
   rate=0.50: coverage 0.9025

5. 当前 phase-1 只证明 token selection overlap，不等价于 answer quality。
   下一步需要把 fixed set 接入真实 FusionRAG pipeline，验证 F1/EM/TTFT。
```

### 下一步

```text
1. 在 pipeline 里增加 fixed_offline selector，直接读取 fixed_sets_npz。
2. 先跑 qk_calib_per_doc_frequency@0.15 与 online_fusionrag_qk@0.15 的 answer quality。
3. 如果 fixed-only 稍差，再跑 fixed@0.10 + qk_residual@0.05。
```

## 2026-06-30：Offline Fixed Token Selector Phase 1b：Chunk-Level 修正

### 实验动机

用户指出一个关键部署问题：

```text
offline 阶段无法提前知道 RAG 在线时会召回哪些 chunks，也无法知道这些 chunks 的最终排列顺序。
```

因此，offline fixed set 不能定义成：

```text
某个固定 retrieved sequence 里的 global token positions
```

否则它只适用于这一个固定拼接顺序，不是可部署方案。

更合理的定义应该是：

```text
每个 chunk 在 offline 阶段独立获得自己的 fixed token set；
online RAG 召回任意 chunks 后，再把 chunk-local indices 映射到当前 prompt 的 global positions。
```

也就是：

```text
chunk_global_id -> selected_local_indices
```

### 实验设置

本实验复用 phase-1 的 Target-QK(preprocess KV) 32-query score cache，不重新跑模型：

```text
source:
  MOTIVATION_EXPERIMENTS/query_recompute_overlap_32q_preprocess/details/20260626_212157
```

offline set 构造仍然只使用：

```text
control_other_example queries
```

评测时再和 held-out 当前 example queries 比较：

```text
native + native_template
```

因此仍然满足：

```text
uses_current_example_question_for_offline_set = false
```

### Chunk-Level Offline Set 定义

主方法：

```text
qk_calib_per_chunk_frequency
```

对每个 chunk 独立执行：

```text
1. 对每个 calibration query，在该 chunk 内取 QK score top-rate token。
2. 统计 chunk 内每个 token 被 calibration queries 选中的频率。
3. 在该 chunk 内按 selected frequency 降序选择 top-rate local tokens。
4. tie-break 使用 mean QK score。
5. 保存 chunk-local token indices。
```

保存格式示例：

```text
chunk_global_id: musique200_example000_passage00
selected_local_indices: [0, 1, 2, 3, ...]
rate: 0.15
method: qk_calib_per_chunk_frequency
```

online 使用时：

```text
global_position = current_chunk_start + selected_local_index
```

### 启动命令

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
TMPDIR=/raid/home/hming/tmp /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  tools_offline_fixed_selector_phase1_chunk_level.py
```

输出目录：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/phase1b_chunk_level_qk/
```

关键文件：

```text
README.md
chunk_fixed_set_manifest.csv
offline_chunk_vs_online_detail.csv
offline_chunk_vs_online_aggregate.csv
chunk_fixed_sets_npz/
```

### 结果

这里的 online 对照是 held-out query 下的 online per-document QK selection。

| rate | method | Jaccard vs online per-doc QK | coverage online by offline |
|---:|---|---:|---:|
| 0.10 | qk_calib_per_chunk_frequency | 0.6743 | 0.8040 |
| 0.10 | random_per_chunk | 0.0532 | 0.1009 |
| 0.15 | qk_calib_per_chunk_frequency | 0.6890 | 0.8145 |
| 0.15 | random_per_chunk | 0.0801 | 0.1483 |
| 0.30 | qk_calib_per_chunk_frequency | 0.7495 | 0.8559 |
| 0.30 | random_per_chunk | 0.1766 | 0.3002 |
| 0.50 | qk_calib_per_chunk_frequency | 0.8230 | 0.9025 |
| 0.50 | random_per_chunk | 0.3329 | 0.4995 |

### 结论

```text
1. phase-1 的强 overlap 结果在 chunk-level 保存格式下仍然成立。
   rate=0.15 时，qk_calib_per_chunk_frequency 的 Jaccard = 0.6890，
   coverage online by offline = 0.8145。

2. 这个结果远高于 random_per_chunk：
   random_per_chunk@0.15 的 Jaccard = 0.0801，
   coverage = 0.1483。

3. 因此，offline fixed set 可以用可部署的 chunk-local 形式定义，
   而不是依赖某个 fixed retrieved sequence 的 global token positions。

4. 但这组实验仍然复用了一个固定 sequence 中得到的 QK scores。
   更严格的下一步是：
   对同一个 chunk 在不同上下文顺序/不同召回组合下采集 score，
   验证 chunk-local fixed set 是否对上下文顺序鲁棒。
```

### 下一步

```text
1. 在同一 musique-200 pipeline 中接入 chunk-local fixed selector，跑 answer quality。
2. 做 chunk order / retrieval context perturbation，测试 chunk-local set 的顺序鲁棒性。
3. 如果 fixed-only answer quality 不够，再做 fixed chunk anchors + online QK residual。
```

## 2026-06-30：Offline Fixed Token Selector Phase 1c：Chunk 顺序鲁棒性

### 实验目的

Phase 1b 已经把 offline fixed set 修正为 chunk-local 形式：

```text
chunk_global_id -> selected_local_indices
```

但还有一个问题需要验证：

```text
同一个 chunk 的 selected_local_indices 是否依赖它在 RAG prompt 中的相对顺序？
```

如果同一个 chunk 放在不同位置后，派生出来的 local fixed set 大幅变化，那么 offline chunk set 仍然不够稳健。  
如果变化很小，说明 chunk-local fixed set 更接近 chunk-intrinsic，可以离线保存并在不同召回顺序下复用。

### 实验设置

对同一个 example 的同一批 10 个 chunks，构造 4 种拼接顺序：

```text
original
reverse
shuffle_a
shuffle_b
```

每种顺序下：

```text
1. 只使用 control_other_example queries；
2. 重新计算 Target-QK(preprocess KV) score；
3. 在每个 chunk 内独立派生 qk_calib_per_chunk_frequency fixed set；
4. 对同一个 chunk，比较扰动顺序下的 selected_local_indices 与 original 顺序下的 selected_local_indices。
```

仍然不使用当前 example 的真实问题：

```text
no native/native_template query is used for deriving the sets
```

实验规模：

```text
examples: 0-4
chunks per example: 10
control queries per example: 21
rates: 0.10, 0.15, 0.30, 0.50
```

### 启动命令

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
TMPDIR=/raid/home/hming/tmp CUDA_VISIBLE_DEVICES=0 \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  tools_offline_fixed_selector_chunk_order_robustness.py \
  --num_examples 5 \
  --device cuda
```

输出目录：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/phase1c_chunk_order_robustness/
```

关键文件：

```text
README.md
chunk_order_score_meta.csv
chunk_order_jaccard_detail.csv
chunk_order_jaccard_aggregate.csv
```

### 指标解释

```text
Jaccard vs original order:
  对同一个 chunk，
  比较 original 顺序下派生的 selected_local_indices
  和 reverse/shuffle 顺序下派生的 selected_local_indices。

Jaccard 越接近 1，说明 chunk-local fixed set 越不依赖 chunk 在 prompt 中的位置。
```

### 结果

| rate | order | Jaccard mean | Jaccard median | min | max | n chunks |
|---:|---|---:|---:|---:|---:|---:|
| 0.10 | reverse | 0.9811 | 0.9792 | 0.9429 | 1.0000 | 50 |
| 0.10 | shuffle_a | 0.9855 | 0.9794 | 0.9556 | 1.0000 | 50 |
| 0.10 | shuffle_b | 0.9830 | 0.9792 | 0.9429 | 1.0000 | 50 |
| 0.15 | reverse | 0.9809 | 0.9857 | 0.8909 | 1.0000 | 50 |
| 0.15 | shuffle_a | 0.9827 | 0.9856 | 0.9000 | 1.0000 | 50 |
| 0.15 | shuffle_b | 0.9837 | 0.9859 | 0.8909 | 1.0000 | 50 |
| 0.30 | reverse | 0.9853 | 0.9867 | 0.9434 | 1.0000 | 50 |
| 0.30 | shuffle_a | 0.9872 | 0.9864 | 0.9487 | 1.0000 | 50 |
| 0.30 | shuffle_b | 0.9884 | 0.9876 | 0.9487 | 1.0000 | 50 |
| 0.50 | reverse | 0.9917 | 0.9916 | 0.9692 | 1.0000 | 50 |
| 0.50 | shuffle_a | 0.9915 | 0.9920 | 0.9667 | 1.0000 | 50 |
| 0.50 | shuffle_b | 0.9919 | 0.9915 | 0.9833 | 1.0000 | 50 |

合并所有扰动顺序：

| rate | Jaccard mean | Jaccard median | min | max | n chunks |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.9832 | 0.9792 | 0.9429 | 1.0000 | 150 |
| 0.15 | 0.9824 | 0.9858 | 0.8909 | 1.0000 | 150 |
| 0.30 | 0.9870 | 0.9867 | 0.9434 | 1.0000 | 150 |
| 0.50 | 0.9917 | 0.9916 | 0.9667 | 1.0000 | 150 |

### 结论

```text
1. chunk-local fixed set 对 chunk 顺序高度鲁棒。
   rate=0.15 时，所有 reverse/shuffle 扰动合并后的平均 Jaccard = 0.9824，
   median = 0.9858。

2. 即使最差 chunk 的 Jaccard 也有 0.8909。
   说明同一个 chunk 的 selected_local_indices 几乎不随 prompt 中的 chunk 顺序改变。

3. 这进一步支持可部署形式：
   offline 阶段可以为每个 chunk 预先保存 selected_local_indices；
   online RAG 召回任意顺序的 chunks 后，只需要做 local-to-global index mapping。

4. 这组实验仍然只覆盖前 5 个 examples。
   但信号非常强，下一步可以优先接真实 answer pipeline；
   如果 answer 结果有希望，再扩展到 20 examples 或更多数据。
```

### 下一步

```text
1. 接入真实 FusionRAG generation pipeline，比较：
   online_fusionrag_qk@0.15
   offline_chunk_fixed@0.15
   random_chunk@0.15

2. 如果 fixed-only 质量低于 online QK，再测试：
   offline_chunk_fixed@0.10 + online_qk_residual@0.05
```

## 2026-06-30：Offline chunk-fixed selector 接入真实生成流程

### 目的

验证“离线按 chunk 固定 selected token set”不只是集合重叠好看，而是在真实 FusionRAG 生成路径中是否能替代在线 QK selector。

核心对照：

```text
online_fusionrag_qk：
  在线用当前 query 计算 QK/attention importance，再选 rate=0.15 doc tokens 做 online update。

offline_chunk_qk：
  离线阶段为每个 chunk 保存 chunk-local fixed token set；
  在线只做 local-to-global index mapping，不再计算 selector。

random_chunk：
  同样按每个 chunk 固定选 rate=0.15 tokens，但 token 是随机选的，用作空白对照。
```

### 实验设置

```text
脚本：
  /raid/home/hming/FusionRAG-pca-analysis/tools_offline_fixed_selector_phase2_generation.py

启动命令：
  CUDA_VISIBLE_DEVICES=0 TMPDIR=/raid/home/hming/tmp \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  tools_offline_fixed_selector_phase2_generation.py \
    --model-path /mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct \
    --num-examples 20 \
    --rate 0.15 \
    --max-new-tokens 50

数据：
  musique-200 前 20 个 examples
  每个 example 使用 topk=10 passages

KV cache：
  /raid/home/hming/fusionrag-pca-top1-top10-cache-20/data/
  musique-pca-subset-preprocess-10-revert_rope-True/Qwen2.5-7B-Instruct

offline fixed set 来源：
  MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/
  phase1b_chunk_level_qk/chunk_fixed_sets_npz

泄漏控制：
  offline_chunk_qk 的 fixed set 来自 control_other_example calibration queries；
  不使用当前 example 的 native/native_template question。
```

### 结果

| selector | n | ROUGE-1 | EM | TTFT(s) | KV load(s) | selection(s) | update+query(s) | selected doc tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| online_fusionrag_qk | 20 | 0.2434 | 0.2000 | 0.4116 | 0.0599 | 0.1139 | 0.2378 | 1103.2 |
| offline_chunk_qk | 20 | 0.2492 | 0.2000 | 0.2706 | 0.0556 | 0.0000 | 0.2150 | 1098.6 |
| random_chunk | 20 | 0.2259 | 0.2000 | 0.2710 | 0.0559 | 0.0000 | 0.2151 | 1098.6 |

补充说明：

```text
1. 这里的 wall_time 包含后续 decode，不适合作为主要对比指标。
   offline_chunk_qk 和 random_chunk 在部分样本上生成更长回答，
   所以 wall_time 会被 decode 长度影响。

2. 当前关注的是 first-token profile：
   TTFT = KV load/copy + selection + online update/query prefill。

3. 这组实验无 error，共 60 条记录。
```

输出文件：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/
phase2_generation_rate015/answer_detail.csv

MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/
phase2_generation_rate015/summary.csv

MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/
phase2_generation_rate015/summary.json

MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/
phase2_generation_rate015/README.md
```

### 结论

```text
1. offline_chunk_qk 在 rate=0.15 下成功去掉了在线 selection 开销。
   selection 从 online QK 的 0.1139s 降到 0。

2. TTFT 从 0.4116s 降到 0.2706s，下降约 34.3%。
   这说明 stable chunk-local set 不只是统计现象，
   可以直接转化成在线路径收益。

3. offline_chunk_qk 的 ROUGE-1/EM 没有低于 online_fusionrag_qk：
   ROUGE-1 0.2492 vs 0.2434，EM 同为 0.20。
   但样本数只有 20，不能当作最终 accuracy 结论。

4. random_chunk 的 TTFT 接近 offline_chunk_qk，
   因为两者都跳过 selector 且更新 token 数相同；
   但 random_chunk 的 ROUGE-1 更低，说明固定 set 不能只是随机省 selection，
   需要有 calibration/QK 信号。

5. 下一步优先做 rate sweep：
   offline_chunk_qk vs online_fusionrag_qk vs random_chunk
   at rate=0.05/0.10/0.15/0.30/0.50。
   如果 offline_chunk_qk 在多个 rate 下稳定接近 online QK，
   这个方向可以成为第三个系统创新点。
```

## 2026-06-30：不同 selector 来源与 offline set 构造方法对比

### 背景

上一组实验只比较了：

```text
online_fusionrag_qk
offline_chunk_qk
random_chunk
```

这还不能说明到底是哪一部分重要：

```text
1. selector 来源重要？
   QK selector / draft selector / 位置启发式 / random

2. offline set 构造规则重要？
   frequency stable set / mean-score stable set
```

因此先不做 rate sweep，而是在固定 `rate=0.15` 下横向比较不同 offline fixed set 方法。

### Fixed set 构造

脚本：

```text
/raid/home/hming/FusionRAG-pca-analysis/tools_offline_fixed_selector_phase1d_set_methods.py
```

输出：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/
phase1d_selector_set_methods_rate015/
```

使用的 score cache：

```text
QK selector:
  MOTIVATION_EXPERIMENTS/query_recompute_overlap_32q_preprocess/
  details/20260626_212157/preprocesskv_example*_10passages_scores.npz

Draft selector:
  docs/experiments/query_recompute_overlap_detail_full/
  details/20260626_184238/draft_example*_10passages_scores.npz
```

泄漏控制：

```text
offline set 只用 control_other_example calibration queries 构造；
不使用当前 example 的 native / native_template queries。
```

方法定义：

```text
qk_frequency_per_chunk:
  QK 分数；每个 calibration query 在每个 chunk 内取 top 15%，
  对 token 被选中次数计数，最后选最高频 token。

qk_mean_score_per_chunk:
  QK 分数；对 calibration queries 的 score 取均值，
  每个 chunk 内选 mean score top 15%。

draft_frequency_per_chunk:
  draft selector 分数；frequency 规则。

draft_mean_score_per_chunk:
  draft selector 分数；mean-score 规则。

position_tail_per_chunk:
  每个 chunk 直接选最后 15% token。

position_boundary_per_chunk:
  每个 chunk 前后边界各选一部分，总量 15%。

random_per_chunk:
  每个 chunk 随机选 15% token。
```

### Set overlap 结果

这里用 held-out native/native_template queries 的 online QK selected set 作为参照。

| method | Jaccard vs online QK | coverage online by offline | n |
|---|---:|---:|---:|
| qk_frequency_per_chunk | 0.6960 | 0.8152 | 2200 |
| qk_mean_score_per_chunk | 0.6910 | 0.8120 | 2200 |
| draft_frequency_per_chunk | 0.4367 | 0.6032 | 2200 |
| draft_mean_score_per_chunk | 0.4306 | 0.5973 | 2200 |
| position_boundary_per_chunk | 0.2224 | 0.3578 | 2200 |
| random_per_chunk | 0.0794 | 0.1463 | 2200 |
| position_tail_per_chunk | 0.0703 | 0.1283 | 2200 |

### 真实生成对照

脚本：

```text
/raid/home/hming/FusionRAG-pca-analysis/tools_offline_fixed_selector_phase2_generation.py
```

启动命令：

```bash
CUDA_VISIBLE_DEVICES=0 TMPDIR=/raid/home/hming/tmp \
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
tools_offline_fixed_selector_phase2_generation.py \
  --model-path /mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct \
  --num-examples 20 \
  --rate 0.15 \
  --max-new-tokens 50 \
  --fixed-set-dir MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/phase1d_selector_set_methods_rate015/chunk_fixed_sets_npz \
  --out-dir MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/phase2_selector_methods_rate015 \
  --selectors online_fusionrag_qk offline_qk_freq offline_qk_mean offline_draft_freq offline_draft_mean position_boundary position_tail random_chunk
```

输出：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/
phase2_selector_methods_rate015/
```

结果：

| selector | n | ROUGE-1 | EM | TTFT(s) | selection(s) | update+query(s) |
|---|---:|---:|---:|---:|---:|---:|
| online_fusionrag_qk | 20 | 0.2434 | 0.2000 | 0.4113 | 0.1136 | 0.2374 |
| offline_qk_freq | 20 | 0.2492 | 0.2000 | 0.2720 | 0.0000 | 0.2153 |
| offline_qk_mean | 20 | 0.2493 | 0.2000 | 0.2717 | 0.0000 | 0.2148 |
| offline_draft_freq | 20 | 0.2492 | 0.2000 | 0.2716 | 0.0000 | 0.2149 |
| offline_draft_mean | 20 | 0.2459 | 0.2000 | 0.2722 | 0.0000 | 0.2150 |
| position_boundary | 20 | 0.1870 | 0.1500 | 0.2715 | 0.0000 | 0.2151 |
| position_tail | 20 | 0.2419 | 0.2000 | 0.2727 | 0.0000 | 0.2163 |
| random_chunk | 20 | 0.2259 | 0.2000 | 0.2723 | 0.0000 | 0.2156 |

### 当前结论

```text
1. QK-frequency 和 QK-mean 都能很好覆盖 online QK 的 selected set。
   coverage 分别是 0.8152 和 0.8120。
   frequency 略高，但差距很小。

2. Draft-based offline set 与 online QK 的集合重叠明显低于 QK-based set：
   coverage 约 0.60，而 QK-based 是 0.81。
   这说明 draft selector 学到的是部分相似但不完全一致的 token priority。

3. 真实生成上，draft_freq 在 20 个样本上没有明显掉分。
   但这不能直接说明 draft 和 QK 等价，因为样本数小，且答案指标不敏感；
   set overlap 表明二者选择机制确实不同。

4. position_boundary 明显差：
   ROUGE-1 0.1870，EM 0.15。
   说明只靠 chunk 边界启发式不够。

5. position_tail 的答案指标接近 online QK，但 set overlap 极低。
   这说明 tail/recency 可能在某些样本上碰巧有效，
   但不能解释 online QK selector 的稳定集合。

6. random_chunk 比 QK/draft 稳定方法差：
   ROUGE-1 0.2259。
   因此 offline fixed set 不能只是“省 selection 的随机重算”，
   必须来自 calibration selector signal。

7. 后续不应直接扫所有 rate。
   更合理的下一步是：
   固定 rate=0.15，扩大样本数到 100/200，
   先验证 QK-frequency、QK-mean、Draft-frequency 三个候选是否稳定。
```

## 2026-06-30：完整数据集重新跑 rate baseline（含 accuracy 与时间）

### 目的

按现有 `test_fusionrag_reflect_preprocess_exp.py` pipeline 重新跑完整数据集 baseline，不复用旧 CSV。

这组结果用于后续所有完整 accuracy 实验的对照，包括：

```text
rate=1.00:
  full attention / full recompute 空白对照。

rate=0.30:
  online FusionRAG-QK，重算 30% doc tokens。

rate=0.15:
  online FusionRAG-QK，重算 15% doc tokens。

rate=0.00:
  cache reuse，但不重算 doc tokens。
```

### 统一实验条件

```text
数据：
  data/result_reflect.json 完整 200 main questions。
  其中 llm_judge=False 或含问题答案的样本按原 pipeline 跳过。
  实际评测 135 main questions / 250 sub questions。

模型：
  Qwen2.5-7B-Instruct

检索：
  BGE-M3 + FAISS global topk=10

preprocess:
  preprocess=True
  preprocess_scope=global
  revert_rope=True

judge:
  GLM-5.2
  OpenAI-compatible API:
  http://36.150.226.221:32355/v1
```

### 输出目录

```text
rate=1:
  MOTIVATION_EXPERIMENTS/full_attention_rate1_baseline_rerun/

rate=0.30:
  MOTIVATION_EXPERIMENTS/full_accuracy_rerun_baselines/online_qk_rate030/

rate=0.15:
  MOTIVATION_EXPERIMENTS/full_accuracy_rerun_baselines/online_qk_rate015/

rate=0:
  MOTIVATION_EXPERIMENTS/full_accuracy_rerun_baselines/rate0_no_doc_recompute/

统一汇总：
  MOTIVATION_EXPERIMENTS/full_accuracy_rerun_baselines/full_accuracy_rerun_summary.csv
  MOTIVATION_EXPERIMENTS/full_accuracy_rerun_baselines/full_accuracy_rerun_summary.json
  MOTIVATION_EXPERIMENTS/full_accuracy_rerun_baselines/README.md
```

### 结果

| label | rate | Main Acc | Sub Acc | F1 | EM | prompt/full-prefill mean(s) | storage mean(s) | selection mean(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rate1_full_attention | 1.00 | 104/135 (77.04%) | 218/250 (87.20%) | 0.5666 | 0.2280 | 0.1167 | 0.0000 | 0.0000 |
| online_qk_rate030 | 0.30 | 94/135 (69.63%) | 199/250 (79.60%) | 0.5083 | 0.1920 | 0.2584 | 0.0253 | 0.1043 |
| online_qk_rate015 | 0.15 | 84/135 (62.22%) | 189/250 (75.60%) | 0.4878 | 0.2120 | 0.2311 | 0.0249 | 0.1032 |
| rate0_no_doc_recompute | 0.00 | 75/135 (55.56%) | 175/250 (70.00%) | 0.4510 | 0.1600 | 0.1011 | 0.0250 | 0.0000 |

说明：

```text
1. rate=1 是 full attention / full recompute baseline，不走 cache reuse selection。

2. prompt/full-prefill mean(s)：
   - rate=1：完整上下文 prefill 的平均时间。
   - FusionRAG 路径：脚本打印的 prompt eval duration，
     其中已包含该路径下的 online update/query prefill 口径。

3. rate=0.30 的 accuracy 明显高于 rate=0.15，
   说明随着重算 token 比例增加，质量确实提升。

4. rate=0 比 rate=0.15 明显低：
   Sub Acc 70.00% vs 75.60%，说明 online doc token recomputation 是有效的。

5. rate=1 full attention 质量最高：
   Sub Acc 87.20%，Main Acc 77.04%。
   这给后续 offline fixed selector / draft selector 提供最终上界。
```

### 附：offline fixed set 现有小样本效果

这里把前面 `rate=0.15` 的 offline fixed set 方法对比放在同一处，方便和完整 baseline 一起看。

输出目录：

```text
MOTIVATION_EXPERIMENTS/offline_fixed_token_selector/
phase2_selector_methods_rate015/
```

注意：

```text
这组目前是 20 examples 的真实生成小样本结果；
不是完整 data/result_reflect.json + GLM judge 的最终 accuracy。
最终结论还需要按完整数据集重跑 offline_qk_freq / offline_qk_mean / offline_draft_freq。
```

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

当前判断：

```text
offline fixed set 的核心收益已经在小样本中体现：
  selection(s): 0.1136 -> 0
  TTFT: 0.4113s -> 约 0.272s

但它的完整数据集 accuracy 还没跑完，不能和上面的 full/rate baseline 直接比较。
下一步应该跑：
  offline_qk_freq
  offline_qk_mean
  offline_draft_freq
on full data + GLM judge。
```
