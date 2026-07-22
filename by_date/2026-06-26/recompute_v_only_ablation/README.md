# Recompute KV Write-back Ablation

目标：验证 FusionRAG online recompute 阶段，selected document tokens 的重算 KV 是否必须写回 cache。对比原始 K/V 都更新、只更新 V、以及 K/V 都不更新。

## 实验设置

- dataset: `musique-pca-subset.jsonl`
- model: `/mnt/qjhs-sh-lab-01/models/Qwen2.5-7B-Instruct`
- cache_path: `/raid/home/hming/fusionrag-pca-topk-cache-5/`
- preprocess: `True`
- topk: `10`
- rate: `0.1`
- reprocess_method: `FusionRAG`
- revert_rope: `True`

## 实现方式

- `kv`: 原始行为，selected tokens online recompute 后 K/V 都写回 `past_key_values`。
- `v_only`: recompute 前备份 selected document tokens 的所有层 K，前向结束后恢复 K，只保留 V 的更新；query tokens 的 K/V 正常保留。
- `none`: recompute 前备份 selected document tokens 的所有层 K/V，前向结束后恢复 K/V；query tokens 的 K/V 正常保留。

## 结果

| mode | ROUGE | EM |
|---|---:|---:|
| kv | 0.296738 | 0.200000 |
| v_only | 0.256508 | 0.200000 |
| none | 0.262758 | 0.200000 |

## 差异

- v_only - kv ROUGE: `-0.040230`
- v_only - kv EM: `0.000000`
- none - kv ROUGE: `-0.033980`
- none - kv EM: `0.000000`

## 样本级现象

文件：

```text
sample_comparison.csv
sample_comparison_summary.json
```

统计结果：

| metric | value |
|---|---:|
| samples | 20 |
| v_only same as kv | 17 |
| v_only changed from kv | 3 |
| none same as kv | 15 |
| none changed from kv | 5 |
| kv avg words | 7.30 |
| v_only avg words | 6.35 |
| none avg words | 7.40 |

## 结论

在 `rate=0.1`、20 个 musique subset 样本上，完整 K/V 写回最好，但优势不大：

```text
kv      ROUGE 0.2967
none    ROUGE 0.2628
v_only  ROUGE 0.2565
```

这说明 selected document tokens 的 online recompute 写回确实有收益，但在这组小样本和低 rate 下，收益主要是增量式的，不是绝对必要条件。另一方面，`none` 略高于 `v_only`，说明“只更新 V、保留旧 K”不一定比“完全不写回 document token 的 K/V”更好；K/V 不一致可能抵消一部分 V 更新收益。

注意：早先一次临时实现把 query tokens 的 K 也恢复了，导致 `v_only` 异常发散。当前结果已修正：只对 selected document tokens 做恢复，query tokens 的 K/V 保持新计算结果。

## DeepSeek 语义正确性评测

复用了 `jybigdata` 的 `sources/iter_rag/run_test.py` 中的 answer judge 逻辑，使用 DashScope OpenAI-compatible endpoint：

```text
base_url = https://dashscope.aliyuncs.com/compatible-mode/v1
judge_model = deepseek-v3.2
input = Question / Real Answer / Pred Answer
output = Correct / Reason
```

结果目录：

```text
deepseek_accuracy/
```

| mode | correct / total | accuracy |
|---|---:|---:|
| kv | 6 / 20 | 0.300000 |
| v_only | 5 / 20 | 0.250000 |
| none | 5 / 20 | 0.250000 |

对应文件：

```text
deepseek_accuracy/summary.csv
deepseek_accuracy/summary.json
deepseek_accuracy/*.deepseek_judged.csv
deepseek_accuracy/judge_cache.jsonl
```

结论：DeepSeek judge 与 ROUGE/EM 的趋势一致，完整 K/V 写回仍然最好；`v_only` 和 `none` 在语义正确率上持平，说明只写回 V 没有提供稳定收益。
