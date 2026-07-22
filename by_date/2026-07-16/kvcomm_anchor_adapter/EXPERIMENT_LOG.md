# 实验日志

## 2026-07-16：建立实验

- 目的：验证 KVCOMM-style prefix-matched Delta anchor 是否能替代 FusionRAG document recompute。
- 当前阶段：Phase A，复用 5 target × 40 train + 10 document-disjoint test contexts。
- 代码基线 commit：`191f82c`。
- 数据：`MOTIVATION_EXPERIMENTS/kv_lora/results/perdoc_context_deltas/strict_t{1..5}`。
- reflect cache：`/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/musique`。
- 注意：此前 offline/online attention score 对比已被标记为设计无效，本实验不使用其结论或输出。

计划命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
CUDA_VISIBLE_DEVICES=1 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/analyze_strict_anchor_transfer.py \
  --device cuda:0
```

### 首次 sanity 异常

- `strict_t{1..5}` 每个 source 实际有 `context_00..50` 共 51 个 tensor；manifest 的正式 40/10 split 只对应 `context_00..49`。
- `context_50` 是历史补采文件，不属于本轮 split。首版脚本因 glob 得到 51 个文件而在计数断言处退出，尚未产生统计结果。
- 已改为显式读取 `context_00..49`，保证补采文件不进入训练或测试。

## 2026-07-16：Phase A 完成

- 计划与首版脚本 commit：`f2bd5ac`。
- strict split 修正 commit：`9a9fabb`。
- shared cache 路径修正 commit：`e9ebc48`。
- 运行机器/GPU：`qjy000 / GPU1`；未使用或中断 qjy001 上的任务。
- 当前共享 reflect cache：`/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/musique`。
- 输出：`results/strict_anchor_transfer/{per_case_metrics.csv,summary.csv,summary.json,run.log}`。

正式命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/analyze_strict_anchor_transfer.py \
  --device cuda:0 \
  --targets 1,2,3,4,5 \
  --output-dir MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/results/strict_anchor_transfer
```

异常与修正：

- 旧 reflect cache 路径已迁移到统一 shared cache；本轮只改读取路径，没有重建 cache。
- target 4 存在真实零 Delta case，单 case recovery 为 0/0；记录为 NaN。正式 summary 先跨 50 cases 累加 norm²，再求 recovery/final error，不平均极小分母比例。
- 首轮 target1 暴露上述两类问题，只作为 smoke；正式结论只使用 `strict_anchor_transfer`。
- 结果：position top8 相对 mean 的 recovery error 改善约 K 8.1%、V 3.2%；oracle 相对 mean 也只有 K 15.8%、V 5.9%。
- 决策：未达到 Phase A 的 10% 门槛，因此本轮不修改主 pipeline、不启动 setup-v2 200-example dynamic anchor。
- 结果、脚本、图表归档 commit：`f0f994c`。


## 2026-07-16：setup-v2 position adapter smoke

本阶段开始把 Phase A 的 anchor 思路接到 setup-standard v2 端到端 pipeline，但仍按小样本 sanity 进行。

代码改动：

- `ktransformers/unified_process_cache.py`：把已有的 `static_key_bias_path/static_value_bias_path` 参数从 `main -> run_experiment -> load_kv_and_generate` 贯通，默认不启用，不影响原 pipeline。
- `MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py`：新增 `position_adapter_raw_rate0`、`position_adapter_preprocess_rate0` 方法，并支持静态 K/V bias 路径参数。
- `scripts/build_setup_v2_position_biases.py`：对同一个 setup-v2 example 的 document 做多个离线排列，用 full forward 观测每个 doc 的 Delta-KV，再以 `[1, prefix_token_ratio, doc_rank_ratio]` 做逐元素 ridge 拟合，输出原始顺序下的 `*_key_bias.pt` / `*_value_bias.pt`。

smoke 选择：

- 先尝试 `musique-v2` example 1，发现该样本包含 24 个 doc chunk，单 anchor 已很慢，因此停止，未产生有效资产。
- 改用 `musique-v2` 中较短的 example 69：`length=3440`，20 个 doc chunk，作为链路 smoke；这不是最终总体结论样本。

资产生成命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1   /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/build_setup_v2_position_biases.py   --source preprocess   --start 69 --end 69   --num-anchors 3   --output-dir /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/position_ridge_smoke_preprocess_e69_a3   --device cuda:0
```

输出：

- adapter cache：`/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/position_ridge_smoke_preprocess_e69_a3`
- log：`results/setup_v2_position_smoke/build_preprocess_e69_a3.log`
- 文件数：20 个 key bias、20 个 value bias、1 个 `metadata.json`。

初步观察：

- setup-v2 MuSiQue 的单样本 document 数可能达到 20+，多 anchor 离线构建成本不可忽略。
- `prepare_data(... preprocess=True)` 仍会触发一次 BGE embedding/召回准备；虽然共享 KV cache 复用，但 adapter 构建入口仍有额外固定开销。
- 下一步只在 example 69 上跑同源 `preprocess_rate0` 与 `position_adapter_preprocess_rate0`，验证生成质量是否至少不劣化。


### runner GPU 修正

首次并行启动 `preprocess_rate0` 与 `position_adapter_preprocess_rate0` 时发现：外部设置 `CUDA_VISIBLE_DEVICES=2/3` 后，setup-v2 runner 内部又执行 `os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)`，导致两个进程都落到同一张物理 GPU。该两次启动已停止，不计入结果。

修正：如果外部已经设置 `CUDA_VISIBLE_DEVICES`，runner 不再覆盖；如果外部未设置，仍使用 `--gpu`。这保证固定分配 GPU 的实验不会互相踩卡。


## 2026-07-16：setup-v2 endpoint smoke 结果

实验 commit：`96c6f59`。

同源 smoke 样本：

- example 69：较短样本，full_rate1 本身错误，只用于检查 adapter 稳定性。
- example 38：历史 full_rate1 与 online_draft rate0.15 正确，preprocess_rate0 与 online_qk rate0.15 错误，用于检查 adapter 是否能恢复 full 行为。

运行命令模板：

```bash
# baseline preprocess rate0
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1   /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py   --dataset musique-v2 --method preprocess_rate0 --rate 0.0   --start <example_id-1> --end <example_id> --gpu 0   --result-root MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/results/setup_v2_position_smoke/<run_name>

# position adapter preprocess rate0
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1   /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py   --dataset musique-v2 --method position_adapter_preprocess_rate0 --rate 0.0   --start <example_id-1> --end <example_id> --gpu 0   --result-root MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/results/setup_v2_position_smoke/<run_name>   --static-key-bias-path <adapter>/key   --static-value-bias-path <adapter>/value
```

主要结果：

- e38 full_rate1 historical：`TML Entertainment`，正确。
- e38 online_draft rate0.15 historical：`TML Entertainment`，正确。
- e38 preprocess_rate0 current：`Metalworks Institute`，错。
- e38 position adapter K+V scale=1：重复乱码，错。
- e38 position adapter V-only scale=0.1/0.25：仍为 `Metalworks Institute`，错。
- e38 position adapter K-only scale=0.1：仍为 `Metalworks Institute`，错。
- e69 full/current preprocess 均为 `Sire Records`，错；K+V scale=1 同样产生乱码，小 scale 不改变答案。

结论：当前 position-ridge full-rank bias 不能作为可用的 FusionRAG KV adapter。它要么不改变错误答案，要么在 scale=1 时破坏生成分布。下一步若继续，不应扩大当前版本，而应先做 layer/head/token gate 与 residual recompute 组合。


## 2026-07-16：oracle 写回排查

目的：确认 position adapter 乱码是否来自静态 bias 写回实现错误。

新增脚本：`scripts/build_setup_v2_oracle_biases.py`。

命令：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1   /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/build_setup_v2_oracle_biases.py   --source preprocess   --example 38   --output-dir /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/oracle_full_minus_preprocess_e38   --device cuda:0

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1   /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py   --dataset musique-v2   --method position_adapter_preprocess_rate0   --rate 0.0   --start 37 --end 38   --gpu 0   --result-root MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/results/setup_v2_position_smoke/oracle_e38   --static-key-bias-path /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/oracle_full_minus_preprocess_e38/key   --static-value-bias-path /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/oracle_full_minus_preprocess_e38/value   --static-key-bias-require-all   --static-value-bias-require-all
```

结果：example 38 oracle K+V 写回输出 `TML Entertainment`，正确，和 full_rate1 一致。runner 文本里 EM/ROUGE 显示 0.005 是因为该 runner 用完整 `tokens_data` 长度做分母；该单样本实际正确。

解释：静态 bias 写回通路不是根本问题；position-ridge 乱码来自近似 Delta 本身不可靠。当前结论要收窄为“position-ridge anchor adapter 不可用”，不能直接否定 KVCOMM-style anchor 方法。


## 2026-07-16：KVCOMM-like anchor matching 版本

背景：前一版 position-ridge 不是 KVCOMM 真实范式，因此不能用其乱码否定 KVCOMM。尝试 clone 官方仓库失败，错误为 GitHub TLS 中断；本轮按公开方法实现最小 KVCOMM-like：anchor pool + top-k matching + offset 加权。

构建命令：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1   /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/build_setup_v2_kvcomm_like_biases.py   --source preprocess   --example 38   --num-anchors 6   --topk-anchors 3   --matcher hybrid   --temperature 0.07   --output-dir /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/kvcomm_like_hybrid_e38_a6_top3   --device cuda:0
```

Endpoint 命令模式：

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1   /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py   --dataset musique-v2   --method position_adapter_preprocess_rate0   --rate 0.0   --start 37 --end 38   --gpu 0   --result-root MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/results/kvcomm_like_e38/<variant>   --static-key-bias-path /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/kvcomm_like_hybrid_e38_a6_top3/key   --static-value-bias-path /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/kvcomm_like_hybrid_e38_a6_top3/value
```

结果：

- full K+V scale=1：输出 `TML Entertainment`，正确，恢复 full。
- V-only scale=0.25/0.5：仍输出 `Metalworks Institute`，错。
- K+V scale=0.25：仍输出 `Metalworks Institute`，错。

解释：anchor top-k 真实 offset 加权明显优于 position-ridge，说明 KVCOMM-like 路线不能被前一版失败否定。但目前只是 e38 单样本 in-example anchor pool；还需验证多样本与跨 example anchor 可复用性。


## 2026-07-16：KVCOMM-like targeted gap cases 扩展

筛选：从 MuSiQue-v2 200 条中找 `full_rate1` exact-correct 且 `preprocess_rate0` wrong 的样本，共 9 个候选。候选表写入：`results/kvcomm_like_candidate_examples.csv`。

本轮跑最短 4 个 targeted gap cases：38、43、24、54。

构建命令模板：

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1   /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/build_setup_v2_kvcomm_like_biases.py   --source preprocess   --example <ex>   --num-anchors 4   --topk-anchors 2   --matcher hybrid   --temperature 0.07   --output-dir /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/kvcomm_like_hybrid_e<ex>_a4_top2   --device cuda:0
```

Endpoint 命令模板：

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1   /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py   --dataset musique-v2   --method position_adapter_preprocess_rate0   --rate 0.0   --start <ex-1> --end <ex>   --gpu 0   --result-root MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/results/kvcomm_like_batch/e<ex>_a4_top2_fullkv_scale1   --static-key-bias-path /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/kvcomm_like_hybrid_e<ex>_a4_top2/key   --static-value-bias-path /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/kvcomm_like_hybrid_e<ex>_a4_top2/value   --static-key-bias-require-all   --static-value-bias-require-all
```

结果：

- 38：恢复，`TML Entertainment`。
- 43：失败，仍为 think/incomplete。
- 24：恢复，`Francisco Guterres`。
- 54：恢复，`John D. Loudermilk`。

统计：targeted 4 cases 中 KVCOMM-like 3/4，online_draft 3/4，online_qk 1/4，preprocess_rate0 0/4。

结论：KVCOMM-like in-example anchor offset 有明确正信号，但还不是可部署方法；下一步必须验证 cross-example anchor reuse。


## 2026-07-16：Offline Prefix Bank 可部署性检查

目的：验证不使用当前 example doc set 的 offline anchor bank 是否能恢复 e38。

构建命令：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1   /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/build_setup_v2_offline_prefix_bank_biases.py   --source preprocess   --example 38   --num-anchors 3   --topk-anchors 2   --prefix-docs 3   --prefix-examples 1,2,3,4,5,6,7,8,9,10   --matcher hybrid   --temperature 0.07   --output-dir /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/offline_prefix_bank_e38_a3_top2_pdocs3   --device cuda:0
```

Endpoint 命令：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1   /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py   --dataset musique-v2   --method position_adapter_preprocess_rate0   --rate 0.0   --start 37 --end 38   --gpu 0   --result-root MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/results/offline_prefix_bank/e38_a3_top2_pdocs3_fullkv_scale1   --static-key-bias-path /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/offline_prefix_bank_e38_a3_top2_pdocs3/key   --static-value-bias-path /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/offline_prefix_bank_e38_a3_top2_pdocs3/value   --static-key-bias-require-all   --static-value-bias-require-all
```

结果：输出 `Equity Music Group`，错误。对比：in-example KVCOMM-like 输出 `TML Entertainment`，正确。

结论：当前 random/other-example offline prefix bank 不可用；in-example 正信号不能直接等价为可部署方案。后续需要用 retrieval-trace/BGE-neighbor prefix 构建 per-chunk anchor bank，并加低置信 fallback。

## 2026-07-16：BGE-neighbor Prefix Bank 可部署性检查

目的：random/other-example offline prefix bank 在 e38 失败后，继续验证更合理的离线 anchor 来源：每个目标 chunk 的全局 BGE top-neighbor chunks。

构建命令：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/build_setup_v2_offline_prefix_bank_biases.py \
  --source preprocess \
  --example 38 \
  --prefix-source bge_neighbors \
  --num-anchors 3 \
  --topk-anchors 2 \
  --prefix-docs 3 \
  --matcher hybrid \
  --temperature 0.07 \
  --output-dir /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/bge_neighbor_prefix_bank_e38_a3_top2_pdocs3 \
  --device cuda:0
```

Endpoint 命令：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
  /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py \
  --dataset musique-v2 \
  --method position_adapter_preprocess_rate0 \
  --rate 0.0 \
  --start 37 --end 38 \
  --gpu 0 \
  --result-root MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/results/offline_prefix_bank/e38_bge_neighbor_a3_top2_pdocs3_fullkv_scale1 \
  --static-key-bias-path /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/bge_neighbor_prefix_bank_e38_a3_top2_pdocs3/key \
  --static-value-bias-path /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/bge_neighbor_prefix_bank_e38_a3_top2_pdocs3/value \
  --static-key-bias-require-all \
  --static-value-bias-require-all
```

结果：输出 `Metalworks Institute`，错误；gold 为 `TML Entertainment`。这和 preprocess_rate0 的错误输出一致，未恢复 full。

结论：BGE-neighbor prefix bank 比 random-prefix 更可部署，但当前仍失败。in-example anchors 的有效性不能直接转化成“未知召回上下文下可用”的 adapter。后续应转向 retrieval-trace / same-query-distribution anchor bank，或者把 KVCOMM-like 作为 confidence-gated fallback 辅助，而不是静态 preprocess v2。

## 2026-07-16：BGE-neighbor Prefix Bank 9-case 扩展

目的：e38 单样本不能判断 BGE-neighbor prefix bank 是否可行，因此补跑全部 9 个 MuSiQue-v2 targeted gap candidates。

启动方式：8 张卡并行，每个样本固定一张卡；e38 已完成，剩余 e43/e24/e54/e14/e68/e66/e13/e63 同时拉起。命令模板：

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1   /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/build_setup_v2_offline_prefix_bank_biases.py   --source preprocess   --example <ex>   --prefix-source bge_neighbors   --num-anchors 3   --topk-anchors 2   --prefix-docs 3   --matcher hybrid   --temperature 0.07   --output-dir /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/bge_neighbor_prefix_bank_e<ex>_a3_top2_pdocs3   --device cuda:0

CUDA_VISIBLE_DEVICES=<gpu> PYTHONUNBUFFERED=1   /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python   MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py   --dataset musique-v2   --method position_adapter_preprocess_rate0   --rate 0.0   --start <ex-1> --end <ex>   --gpu 0   --result-root MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/results/offline_prefix_bank/e<ex>_bge_neighbor_a3_top2_pdocs3_fullkv_scale1   --static-key-bias-path /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/bge_neighbor_prefix_bank_e<ex>_a3_top2_pdocs3/key   --static-value-bias-path /mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/bge_neighbor_prefix_bank_e<ex>_a3_top2_pdocs3/value   --static-key-bias-require-all   --static-value-bias-require-all
```

结果：BGE-neighbor prefix bank 为 3/9 exact、4/9 contains；preprocess_rate0 为 1/9 contains；online_qk rate0.15 为 4/9 contains；online_draft rate0.15 为 7/9 contains。

结论：BGE-neighbor 不是完全无效，但目前只是弱信号，最多接近 online_qk，明显弱于 online_draft。要继续形成可部署 adapter，下一步应构造 retrieval-trace bank，而不是只靠 BGE semantic neighbors。
## 2026-07-16：Mixed Anchor Pool 扩池（待运行）

用户要求扩大每个目标文档的 anchor 池，并在 BGE 相似文档之外加入数据集随机文档。实现配置为 3 组 BGE anchor + 9 组跨 example 随机 anchor，每组 3 个文档，online top-4 融合。随机文档严格排除目标 example，固定 `random_seed=20260716`。

同时修正 online 计算口径：当前真实前缀和候选前缀的 matcher 特征直接从 preprocess Value cache 汇总；anchor Delta 的完整 Transformer forward 只发生在 offline。共享 cache 保持为 `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2`，adapter tensor 与实验结果继续分开保存。

计划先运行 examples `38,43,24,54,14,68,66,13,63`，达到 5/9 contains 后再扩完整 200 条。

### 实际运行与结果

代码提交：初始扩池实现 `4da6143`；固定 9-case launcher `1f6b349`；e63 chunk 分片接口 `a37abe0`。

主启动脚本：

```bash
nohup MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/run_mixed_anchor_9case.sh \
  > MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/results/mixed_anchor_pool_launcher.log 2>&1 &
```

e63 因单样本文档较长，将剩余 chunk 用 `--chunk-ids` 固定分到 GPU 0-7；完整参数保存在 `results/mixed_anchor_pool/logs/e63.build.gpu*.log` 对应进程命令和 adapter 的 `metadata_shard_gpu*.json`。

结果为 mixed 4/9 exact、4/9 contains；旧 BGE-only 为 3/9 exact、4/9 contains；online QK 为 4/9 contains；online Draft 为 7/9 contains。随机 anchor 占已记录 top-4 槽位 48.46%，平均融合权重 40.50%，不是“池扩大但随机项未命中”的问题，而是随机 Delta 的兼容性不可由当前长度 + cached Value mean cosine 稳定判断。

由于未达到事先规定的 5/9 contains 扩展门槛，本轮没有启动完整 200 条。结果写入 `results/mixed_anchor_pool/mixed_anchor_9case_summary.{csv,json}`。

## 2026-07-16：Oracle Compatibility 上界（计划）

新增 `--selection-mode oracle_value_l2|oracle_kv_l2`。Oracle 模式仅用于 offline 机制分析：为每个 chunk 计算真实 current-context Delta，记录全部候选的 K/V relative L2，并选择最接近真实 Value Delta 的 top-1。Online endpoint 仍只加载最终 bias tensor。

首轮固定使用与 mixed sanity 完全一致的 3 BGE + 9 random anchor pool、每组 3 docs、seed 20260716，运行相同 9 个 targeted cases。通过门槛设为至少 7/9 contains；达到后才进入跨 example gate 训练。

### Oracle top-1 实际结果

实现提交 `628429c`，两机固定分片 launcher 提交 `8965703`。启动命令：

```bash
# qjy000
nohup MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/run_oracle_value_build_host.sh \
  > MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/results/oracle_compatibility/oracle_build_qjy000.log 2>&1 &

# qjy003
nohup MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/run_oracle_value_build_host.sh \
  > MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/results/oracle_compatibility/oracle_build_qjy003.log 2>&1 &

# qjy000 endpoint coordinator
nohup MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/run_oracle_value_endpoint_9case.sh \
  > MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/results/oracle_compatibility/oracle_endpoint_coordinator.log 2>&1 &
```

结果为 4/9 exact、4/9 contains，未达到 7/9。192 个完整 candidate metadata chunk 上，best-anchor Value relative L2 均值 0.712，matcher top-1 为 0.809，matcher/oracle rank Spearman 为 0.347。结论是单 anchor 覆盖不足；下一轮改测整个 12-anchor Delta span 的 oracle ridge coefficient 上界。

## 2026-07-17：12-anchor Oracle Linear Span

实现提交 `a9ccaf8`，stage-1 launcher `8353f94`。首次 endpoint launcher 的 Bash `local` 同行赋值导致前三例错误使用外层 `ex=54` 计算 start offset；修复提交 `d2e6edf`，构建 bias 未受影响，四例 endpoint 已全部严格补跑。

启动脚本：

```bash
# qjy000 与 qjy003 分别启动
MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/run_oracle_kv_lstsq_stage1_build_host.sh

# qjy000 endpoint
MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/run_oracle_kv_lstsq_stage1_endpoint.sh
```

结果：e38/e54 正确，e43/e24 错误，2/4。90 chunks 的 Value relative L2 mean/median 为 0.599/0.652，Key 为 0.581/0.607；coefficient L2 mean 0.523，最大绝对系数 0.696。未达到 3/4 扩展门槛，下一轮只做 32-anchor reconstruction scaling probe，不直接扩 endpoint。

## 2026-07-17：32-anchor Scaling Probe

启动脚本 `scripts/run_oracle_kv_lstsq_a32_probe_host.sh`，提交 `3058d03`。qjy000/qjy003 共 16 张卡，固定测试 e38 `[1,6,12,18,24]`、e43 `[1,5,10,15,19]`、e24 `[1,6,11,17,22]`、e54 `[1,7,13,19,25]`，配置为 3 BGE + 29 random、joint K/V ridge `1e-4`。

同 20 chunks 对比：A12 -> A32 的 Value relative L2 为 0.5860 -> 0.5763（相对改善 1.66%），Key 为 0.5693 -> 0.5478（改善 3.78%）。扩池收益很小，不满足 residual 约 0.35 的 endpoint gate，因此没有启动 A32 endpoint。下一步改构造 retrieval-trace prefix anchors。

## 2026-07-17：Retrieval-trace last-3 Probe

实现 `prefix_source=retrieval_traces`，严格排除目标 example；commit `96b1031`。最初 `--topk 50` 错误联动 preprocess cache 路径，修复为 `topk=10` 加独立 `retrieval_topk=50`，commit `532ab4e`。启动脚本 `scripts/run_oracle_kv_lstsq_trace_probe_host.sh`，commit `3131df6`。

20 chunks 上 trace A12 的 Value/Key relative L2 为 0.5784/0.5570；mixed A12 为 0.5860/0.5693，mixed A32 为 0.5763/0.5478。last-3 trace 总体收益很小，但 e43/e24 改善、e38/e54 回退，提示前缀长度/位置分布是关键变量。下一轮使用完整 preceding-document trace。

## 2026-07-17：Retrieval-trace Full-prefix 与 e24 严格 Endpoint

实现 `--prefix-docs 0` 表示使用 retrieval occurrence 的全部 preceding documents，提交 `e9148b7`；e24 全 22 chunks 和 endpoint launcher 提交 `a120daf`。

20-chunk probe 启动脚本：

```bash
# qjy000 与 qjy003 分别固定运行各自 shard
MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/run_oracle_kv_lstsq_trace_fullprefix_probe_host.sh
```

e24 完整构建与 endpoint：

```bash
# qjy000 与 qjy003：共 16 卡补齐 22 个 chunks
MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/run_trace_fullprefix_e24_complete_host.sh

# qjy000：等待 22 个 key tensor 后自动运行 require-all endpoint
MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/run_trace_fullprefix_e24_endpoint.sh
```

固定配置：Qwen3-32B、MuSiQue-v2、preprocess source、retrieval-trace A12、full prefix、joint K/V oracle ridge `1e-4`、seed `20260716`。共享 preprocess cache 为 `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2`；adapter tensor 为 `/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-adapter/Qwen3-32B/musique-v2/oracle_kv_lstsq_retrieval_trace_a12_fullprefix_ridge1e-4_seed20260716_e24`。

Probe 的 Value/Key relative L2 为 0.5510/0.5038，相比 mixed A12 的 0.5860/0.5693 改善 5.98%/11.52%。e24 全量 endpoint 输出 `Jose Ramos-Horta`，gold `Francisco Guterres`，EM/ROUGE 为 0。初始 chunk 2 metadata 因同 GPU 串行任务使用同名文件而被覆盖，但 bias tensor 完整；metadata 命名修复和按层统计提交 `069e6bd`。

按层诊断启动命令：

```bash
nohup bash MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/run_trace_fullprefix_e24_layer_diagnostics.sh \
  > MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/results/oracle_span_trace_fullprefix_e24/layer_diagnostics/launcher.log 2>&1 &
```

补齐后完整 22 chunks 的 Value relative L2 mean/median 为 0.266/0.224，Key 为 0.217/0.165。代表 chunks 2/7/12/17/20/21 上，Value 的 48-63 层平均承载 96.17% target energy 和 97.17% error energy；Key 为 49.57%/57.81%。高误差 chunk 在所有层段同步升高，不支持“只修固定几层”这一解释。下一轮测试 layer-wise oracle ridge coefficient，验证当前 shared coefficient 是否是主要容量瓶颈。

## 2026-07-17：Layer-wise Oracle Coefficient Probe

实现提交 `067b102`。主启动命令：

```bash
nohup bash MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/run_trace_fullprefix_layerwise_probe.sh \
  > MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/results/oracle_span_trace_layerwise_probe/launcher.log 2>&1 &
```

主脚本固定运行 chunks 2/7/12/17/20/21；GPU 6/7 用同参数额外运行证据 chunks 4/10。每层独立解 12 维 joint K/V ridge coefficient，其他 anchor、cache、seed 均与 shared A12 相同。

8-chunk paired 结果：Value residual mean 0.3539 -> 0.3510，平均相对改善 0.75%；Key 0.2966 -> 0.2908，改善 1.87%。参数自由度从 12 增至 `64 x 12`，收益仍很小，因此不补完整 endpoint。结论是当前瓶颈为 retrieval-trace basis span，而不是跨层共享 coefficient。下一轮只做 trace A12 -> A32 scaling 上界。

## 2026-07-17：Full-prefix Retrieval-trace A32 Scaling

Probe launcher 提交 `552b784`，启动命令：

```bash
nohup bash MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/run_trace_fullprefix_a32_probe.sh \
  > MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/results/oracle_span_trace_a32_probe/launcher.log 2>&1 &
```

同 chunks 2/4/7/10/12/17/20/21 paired 比较 A12 与 A32。Value mean 0.3539 -> 0.3315（平均相对改善 6.32%），Key 0.2966 -> 0.2715（8.36%）。每个 chunk 的 32 条 prefix sequence 均唯一；平均 10.6 个 source examples。chunk 2/4 改善明显，chunk 10/7/12/17 接近饱和，说明 anchor 数量有局部收益但不是统一解。

由于证据 chunks 4/20 有改善，提交 `e25a917` 并在 qjy000/qjy003 各 7 卡补剩余 14 chunks，endpoint coordinator 等待 22 个 bias 后以 `require-all` 运行：

```bash
# qjy000、qjy003
nohup bash MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/run_trace_fullprefix_a32_e24_complete_host.sh > <host-build-log> 2>&1 &

# qjy000
nohup bash MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kvcomm_anchor_adapter/scripts/run_trace_fullprefix_a32_e24_endpoint.sh > <endpoint-coordinator-log> 2>&1 &
```

完整 22 chunks 结果：Value mean/median A12 0.2664/0.2236，A32 0.2547/0.2122；Key A12 0.2170/0.1648，A32 0.2023/0.1618。按 chunk 平均相对改善为 Value 3.76%、Key 5.66%。A32 coefficient L2 均值 0.594，最大单系数绝对值 0.790。

严格 endpoint 仍输出 `Jose Ramos-Horta`，gold 为 `Francisco Guterres`，EM/ROUGE 为 0。A32 未通过 gate，因此停止扩完整数据集和 coefficient predictor。Oracle coefficient 已使用真实 Delta；若 oracle span 都不能恢复 endpoint，可部署 matcher/predictor 不可能弥补该 basis 缺口。下一阶段只考虑从当前真实 preceding KV/hidden summary 生成新方向的 context-conditioned adapter。
