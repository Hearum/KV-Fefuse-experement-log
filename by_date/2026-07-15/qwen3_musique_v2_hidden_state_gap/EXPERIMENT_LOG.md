# 实验日志

## 2026-07-15 建立实验

目的：在 KV cache 之外缓存每层输入 hidden state，比较 raw/preprocess/full hidden gap。先做小样本 sanity，不直接跑大规模。

代码变更：

- `ktransformers/util/utils.py` 新增 `FUSIONRAG_SAVE_HIDDEN_CACHE=1` 可选接口。
- hidden cache 文件命名：`{example_id}_{chunk_id}_hidden.pt`。
- hidden cache shape：`[num_layers, batch, seq_len, hidden_size]`，保存每层输入 hidden state。

结果待补。

## 2026-07-15 smoke: 1 example hidden gap

commit: 8bf7f4f

启动命令：

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/by_date/2026-07-15/qwen3_musique_v2_hidden_state_gap/scripts/collect_hidden_gap.py --device cuda:0 --max-examples 1 --max-full-seq-len 20000
```

## 2026-07-15 sanity: 5 examples hidden gap

commit: 8bf7f4f

启动命令：

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/by_date/2026-07-15/qwen3_musique_v2_hidden_state_gap/scripts/collect_hidden_gap.py --device cuda:0 --max-examples 5 --max-full-seq-len 20000
```


5-example sanity 已完成：

| source hidden | Relative L2 | Cosine | examples | chunks | doc tokens |
|---|---:|---:|---:|---:|---:|
| raw | 0.6942 | 0.7823 | 5 | 116 | 84,634 |
| preprocess | 0.7770 | 0.6962 | 5 | 116 | 84,634 |

观察：hidden gap 主要集中在后层。raw top layers 为 62/61/60/59/58/57/56/55；preprocess top layers 为 63/62/61/60/59/58/57/56。preprocess hidden 在这 5 个样本上比 raw hidden 更远离 full hidden，说明 BGE top-k preprocess 前缀并不等价于真实 full prompt 前序文档上下文。

输出：

- `results/hidden_global_summary.csv`
- `results/hidden_layer_summary.csv`
- `results/hidden_summary.json`
- `figures/hidden_layer_relative_l2.png`
- `figures/hidden_layer_energy_share.png`


## 2026-07-16 scaling: 50 examples hidden gap

目标：把 hidden gap sanity 从 5 examples 扩到 50 examples。为避免单卡长时间串行，按 5 个 shard 并行，每个 shard 10 examples，写入独立 output-dir，最后用 `scripts/merge_hidden_gap_shards.py` 合并。

启动前 commit：`0963c59`

共享 KV cache：

```text
/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2
```

分片启动命令模板：

```bash
CUDA_VISIBLE_DEVICES=<gpu> /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/by_date/2026-07-15/qwen3_musique_v2_hidden_state_gap/scripts/collect_hidden_gap.py   --device cuda:0   --start-example <start>   --max-examples 10   --max-full-seq-len 20000   --output-dir MOTIVATION_EXPERIMENTS/by_date/2026-07-15/qwen3_musique_v2_hidden_state_gap/results/shards_50/shard_<start>_<end>
```

计划分配：

| shard | examples | GPU |
|---|---:|---:|
| shard_00_10 | 0-9 | 0 |
| shard_10_20 | 10-19 | 1 |
| shard_20_30 | 20-29 | 2 |
| shard_30_40 | 30-39 | 3 |
| shard_40_50 | 40-49 | 4 |


50-example scaling 已完成并合并：

| source hidden | Relative L2 | Cosine | examples | chunks | doc tokens |
|---|---:|---:|---:|---:|---:|
| raw | 0.7204 | 0.7648 | 50 | 1,166 | 827,074 |
| preprocess | 0.7431 | 0.7223 | 50 | 1,166 | 827,074 |

合并命令：

```bash
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/by_date/2026-07-15/qwen3_musique_v2_hidden_state_gap/scripts/merge_hidden_gap_shards.py   --shard-root MOTIVATION_EXPERIMENTS/by_date/2026-07-15/qwen3_musique_v2_hidden_state_gap/results/shards_50   --output-dir MOTIVATION_EXPERIMENTS/by_date/2026-07-15/qwen3_musique_v2_hidden_state_gap/results/merged_50
```

结果：0 skipped。hidden gap 后层集中趋势稳定。preprocess hidden 在 50 examples 上仍比 raw hidden 更远离 full hidden，但差距小于 5-example sanity。


## 2026-07-16 full dataset scaling: 200 examples hidden gap

目标：在 `musique-v2` 完整 200 examples 上统计 hidden gap。按 8 个 shard 并行，每个 shard 25 examples，写入 `results/shards_full/`，最终合并到 `results/merged_full/`。

启动前 commit：`551ecd3`

计划分配：

| shard | examples | GPU |
|---|---:|---:|
| shard_000_025 | 0-24 | 0 |
| shard_025_050 | 25-49 | 1 |
| shard_050_075 | 50-74 | 2 |
| shard_075_100 | 75-99 | 3 |
| shard_100_125 | 100-124 | 4 |
| shard_125_150 | 125-149 | 5 |
| shard_150_175 | 150-174 | 6 |
| shard_175_200 | 175-199 | 7 |
