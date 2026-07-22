# Predictor Distillation on General Text

## 目标

验证一个不依赖现有 RAG 数据集的轻量 token selector predictor：用通用文本预训练数据构造 synthetic doc/query pair，用 full DraftModel selector 作为 teacher，训练一个只有少量层的小模型预测 document token importance。

## 数据集选择

- 数据集：WikiText-2 raw (`Salesforce/wikitext`, config `wikitext-2-raw-v1`)。
- 原因：通用语言建模文本，不包含当前 RAG pipeline 的 QA 标注；规模小，适合先跑通 teacher generation + distillation + held-out selector overlap。
- 扩展：后续可替换成 WikiText-103、OpenWebText-10k 或 RedPajama sample。

## 蒸馏设置

1. 从 WikiText 文本中切 synthetic pair：前半段作为 document，后半段作为 query continuation。
2. 用 Qwen2.5-3B DraftModel 的默认 full Draft selector 得到每个 doc token 的 teacher score。
3. 训练一个 tiny predictor：冻结 draft embedding，只训练投影层 + 少量 TransformerEncoder + token score head。
4. 主要训练目标是 KL(student distribution || teacher distribution)，不是固定 top-15 分类；因此同一个模型可以在推理时切换 5%/10%/15%/30% 等不同更新比例。
5. 验证指标：在多个 rate 下把 student top-r token set 和 teacher top-r token set 比较，报告 recall/Jaccard。

## 执行日志

- 2026-07-05：创建实验文件夹和计划。下一步生成 WikiText synthetic pairs、teacher scores，并训练 tiny predictor smoke。

## 框架选择

采用 Hugging Face Accelerate 作为训练框架。原因：本任务不是标准 causal LM fine-tuning，而是自定义 token-importance predictor；Accelerate 可以保留纯 PyTorch 训练循环，同时支持单卡/多卡、fp16 mixed precision 和后续 DDP 扩展。

## 当前实现

脚本目录：`MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/scripts/`

- `prepare_wikitext_pairs.py`：从 WikiText 或本地 text file 构造 prefix-suffix pair。远端当前无法访问 Hugging Face，因此增加了 `--text-file` fallback。
- `generate_teacher_scores.py`：用 Qwen2.5-3B DraftModel 的 full Draft selector 生成 teacher doc-token score 和 top-15% selected tokens。
- `predictor_distill_accelerate.py`：Accelerate 训练 tiny Transformer selector。输入 prefix+suffix，输出 prefix/doc token logits；默认使用 KL 蒸馏完整 teacher score distribution；rate 不写死在训练目标里，只在评估/推理时按 top-r 截断。

## Smoke 结果

由于远端无法直接下载 Hugging Face WikiText，本次 smoke 使用仓库中的普通文本/源码/Markdown 拼成 local corpus，仅用于验证训练链路，不作为方法效果结论。

执行：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/scripts/prepare_wikitext_pairs.py \
  --text-file MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/data/local_text_corpus.txt \
  --max-pairs 16 --doc-len 384 --query-len 64 \
  --out MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/data/text_pairs_smoke.jsonl

CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/scripts/generate_teacher_scores.py \
  --pairs MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/data/text_pairs_smoke.jsonl \
  --out-dir MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/teacher_cache_smoke \
  --limit 16 --device cuda:0

CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/scripts/predictor_distill_accelerate.py \
  --cache-dir MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/teacher_cache_smoke \
  --out-dir MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/checkpoints/smoke_accelerate_tiny_selector \
  --hidden 128 --layers 2 --heads 4 --batch-size 4 --epochs 3 --lr 3e-4
```

Smoke 训练日志：

| epoch | val KL | val BCE | top-15 recall | top-15 Jaccard |
|---:|---:|---:|---:|---:|
| 1 | 0.0510 | 0.4292 | 0.1316 | 0.0710 |
| 2 | 0.0243 | 0.4374 | 0.1140 | 0.0610 |
| 3 | 0.0157 | 0.4302 | 0.1491 | 0.0806 |

解释：loss 能下降，说明 Accelerate 训练链路、teacher cache、checkpoint 保存都可用；但 16 条 local-corpus smoke 数据太小，val recall 不代表真实能力。

## 下一步正式实验

1. 准备真实通用预训练文本文件，建议先放一个 OpenWebText/WikiText/RedPajama sample 到：
   `MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/data/pretrain_text.txt`
2. 生成 20k prefix-suffix pairs。
3. 8 卡并行生成 teacher cache，或先按 shard 生成 20k teacher files。
4. 用 Accelerate 训练 2-4 层 student。
5. 在 held-out teacher cache 上看 top-15 recall/Jaccard，目标至少超过 `middle partial Draft` 的 72% recall。
6. 如果 held-out recall 过线，再接入 FusionRAG pipeline 跑 `offline10 + student residual5`。

## Small-2K 正式实验

### 数据

远端无法直接下载 Hugging Face WikiText/OpenWebText，因此本轮先使用可访问的非 RAG 通用文本做 small 实验：

- local generic corpus: 仓库 Markdown / Python / 文档文本。
- sglang `benchmark/llm_judge/articles.jsonl`: 普通文章字符串，不是 FusionRAG RAG QA 数据。
- 合并文件：`data/pretrain_text_small.txt`，约 1.9MB。
- 生成方式：`doc_len=384`, `query_len=64`, `stride=64`。
- 样本数：2000 prefix-suffix pairs。

注意：这不是最终大规模预训练语料，只是为了完整验证 teacher generation + Accelerate distillation 是否能在非 RAG 文本上学到 selector policy。

### Teacher

- Teacher model：Qwen2.5-3B-Instruct DraftModel。
- Teacher signal：full Draft selector 默认策略，即后半层 query-to-prefix attention score + RRF，保存 doc token soft score 和 top-15% token set。
- Teacher cache：`teacher_cache_2k/`，共 2000 个 npz。
- 每条样本：doc 384 tokens，suffix 64 tokens，top-15% 为 57 tokens。

启动：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/scripts/prepare_wikitext_pairs.py \
  --text-file MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/data/pretrain_text_small.txt \
  --max-pairs 2000 --doc-len 384 --query-len 64 --stride 64 \
  --out MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/data/text_pairs_2k.jsonl

nohup MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/scripts/launch_teacher_2k.sh \
  > MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/logs/teacher_2k.launch.log 2>&1 &
```

### Student

- Framework：Hugging Face Accelerate。
- Model：tiny Transformer selector，4 layers，hidden 512，8 heads。
- Input：prefix + suffix token ids。
- Output：prefix/doc token logits。
- Loss：KL teacher soft distribution + top-k BCE variants。
- Split：最后约 15% teacher files 作为 held-out validation。

### 结果

| run | best epoch | val top-15 recall | val top-15 Jaccard | note |
|---|---:|---:|---:|---|
| `kl_bce0p25` | 7 | 51.72% | 35.81% | KL + light BCE，当前最好 |
| `bce2` | 7 | 51.43% | 35.64% | 增大 BCE 总权重，没有提升 |
| `posw5p67` | 5 | 51.27% | 35.50% | 正样本加权，没有提升 |

结果 CSV：`results/small2k_training_summary.csv`。

### 结论

1. Accelerate 训练框架、teacher cache、student checkpoint、held-out recall/Jaccard 评估链路已经完整跑通。
2. 2k 非 RAG 文本上，student 能学到明显高于随机的 selector 信号；top-15 recall 约 52%。
3. 当前结果仍明显低于 `middle partial Draft` 在 20-example selector alignment 上的 72.21% recall，因此还不值得接入 FusionRAG pipeline 跑 accuracy。
4. 单纯调 BCE 权重不能解决问题，瓶颈更可能是数据规模/质量、模型结构或 teacher label 形式。

### 下一步建议

- 换成真正大规模通用预训练文本，例如 OpenWebText / WikiText-103 / RedPajama sample，本地放到 `data/pretrain_text.txt`。
- 生成至少 20k-100k teacher cache，再训练同一 student。
- 尝试 listwise/ranking loss，而不是只用 KL+BCE。
- 如果 held-out recall 超过 70%，再把 student selector 接入 FusionRAG，跑 `offline10 + student residual5`。


## 20K KL-only Distribution Distillation

### 用户需求修正

本轮不再把目标固定成 top-15 token 分类。原因是 FusionRAG 的 recompute rate 后续需要动态调整，训练目标如果写死 top-15，会把 predictor 变成一个固定阈值分类器，泛化到 5%/10%/30% 时不自然。

因此本轮改成标准 selector 蒸馏：teacher 输出每个 doc token 的完整 importance score distribution，student 学这个分布；推理时按需要选择 top-r token。这里的 r 可以是 0.05、0.10、0.15、0.30 或其他比例。

### 数据和启动

- 样本数：20,000 prefix-suffix pairs。
- 文本来源：远端无法访问 Hugging Face，因此使用本地非 RAG 通用文本集合 `data/pretrain_text_small.txt`，来源包括仓库普通 Markdown/Python/docs 文本和 sglang `benchmark/llm_judge/articles.jsonl` 文章。这个数据只用于验证训练趋势，不是最终大规模预训练语料。
- 切分：`doc_len=384`, `query_len=64`, `stride=16`。
- Teacher：Qwen2.5-3B DraftModel full selector，保存每个 doc token 的 soft score distribution。
- Teacher cache：`teacher_cache_20k/`，共 20,000 个 npz。
- Student：4-layer Transformer selector，hidden 512，8 heads。
- Loss：KL only，`--bce-weight 0.0`。
- 评估：同一个 student distribution 在多个 rate 下取 top-r，和 teacher top-r 对齐。

启动命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/scripts/prepare_wikitext_pairs.py \
  --text-file MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/data/pretrain_text_small.txt \
  --max-pairs 20000 --doc-len 384 --query-len 64 --stride 16 \
  --out MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/data/text_pairs_20k.jsonl

nohup MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/scripts/launch_teacher_20k.sh \
  > MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/logs/teacher_20k.launch.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python \
  MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/scripts/predictor_distill_accelerate.py \
  --cache-dir MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/teacher_cache_20k \
  --out-dir MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/checkpoints/kl20k_h512_l4 \
  --hidden 512 --layers 4 --heads 8 --batch-size 32 --epochs 10 --lr 3e-4 \
  --temperature 2.0 --bce-weight 0.0 --eval-ratios 0.05,0.10,0.15,0.30
```

### 结果

完整逐 epoch 表：`results/kl20k_training_summary.csv`。

| epoch | val KL | recall@5% | Jaccard@5% | recall@10% | Jaccard@10% | recall@15% | Jaccard@15% | recall@30% | Jaccard@30% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0048 | 46.86% | 32.10% | 49.45% | 34.20% | 51.61% | 35.98% | 57.96% | 41.55% |
| 2 | 0.0044 | 48.78% | 33.75% | 51.40% | 35.87% | 53.40% | 37.54% | 59.34% | 42.83% |
| 3 | 0.0047 | 50.51% | 35.42% | 52.95% | 37.32% | 55.03% | 39.06% | 60.61% | 44.13% |
| 4 | 0.0040 | 51.87% | 36.59% | 54.16% | 38.36% | 56.22% | 40.14% | 61.74% | 45.29% |
| 5 | 0.0040 | 52.44% | 37.11% | 54.80% | 38.99% | 56.70% | 40.59% | 62.21% | 45.77% |
| 6 | 0.0039 | 53.24% | 37.92% | 55.43% | 39.60% | 57.38% | 41.26% | 62.81% | 46.38% |
| 7 | 0.0038 | 54.57% | 39.10% | 56.66% | 40.77% | 58.70% | 42.61% | 63.95% | 47.66% |
| 8 | 0.0038 | 55.29% | 39.84% | 57.62% | 41.73% | 59.60% | 43.51% | 65.04% | 48.84% |
| 9 | 0.0035 | 56.08% | 40.51% | 58.81% | 42.87% | 60.93% | 44.85% | 66.32% | 50.26% |
| 10 | 0.0034 | 56.86% | 41.28% | 59.59% | 43.64% | 61.72% | 45.64% | 67.28% | 51.32% |


Best by validation KL：epoch 10，val KL=0.0034。
Best by recall@15%：epoch 10，recall@15%=61.72%，Jaccard@15%=45.64%。

### 阶段结论

1. 20k KL-only 分布蒸馏比 2k top-15 辅助训练明显更好：rate=0.15 recall 从约 51.7% 提升到 61.7%。
2. 多个 rate 都随 epoch 稳定提升，说明模型学到的是 teacher importance distribution，而不是只记住某个固定比例的 token set。
3. 目前仍低于之前 `middle partial Draft` selector 的 72.21% token recall，因此暂时还不建议直接接入 FusionRAG accuracy pipeline 替换 online draft。
4. 下一步如果继续这条路线，应优先扩大到 100k 以上通用文本，并尝试更强 student 结构或直接用 draft model 的中间层初始化；单纯在 2k/20k 小数据上调 BCE 权重意义不大。


## 500K KL-only Scaling Run

### 启动记录

- 启动时间：2026-07-06 00:28 左右。
- 目的：把 20k KL-only selector distillation 扩大到 500k pair，验证 scaling 后 student 是否能接近 online/middle Draft selector。
- 样本数：500,000 prefix-suffix pairs。
- 数据来源：仍然是本地可访问的非 RAG 通用文本 `data/pretrain_text_small.txt`。由于远端无法访问 Hugging Face，本轮采用 `stride=1` 密集滑窗生成 500k pair。这个设置能显著增加 teacher/student 训练步数，但文本多样性不等价于真实 500k 独立语料；后续需要换真实大语料复核。
- 切分：`doc_len=384`, `query_len=64`, `stride=1`。
- Teacher：Qwen2.5-3B DraftModel full selector，8 卡并行生成，每卡 62,500 条。
- Teacher cache：`teacher_cache_500k/`。
- Teacher 启动脚本：`scripts/launch_teacher_500k.sh`。
- Student 计划：`scripts/launch_train_500k_kl.sh`，4-layer hidden-512 KL-only，20 epoch，batch size 64，评估 5%/10%/15%/30%。

启动命令：

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/scripts/prepare_wikitext_pairs.py \
  --text-file MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/data/pretrain_text_small.txt \
  --max-pairs 500000 --doc-len 384 --query-len 64 --stride 1 \
  --out MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/data/text_pairs_500k.jsonl

nohup MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/scripts/launch_teacher_500k.sh \
  > MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/logs/teacher_500k.launch.log 2>&1 &
```

### 500K 训练结果

完整逐 epoch 表：`results/kl500k_training_summary.csv`。模型 checkpoint：`checkpoints/kl500k_h512_l4/model.pt`。

| epoch | val KL | recall@5% | Jaccard@5% | recall@10% | Jaccard@10% | recall@15% | Jaccard@15% | recall@30% | Jaccard@30% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0045 | 48.72% | 33.65% | 51.16% | 35.50% | 53.71% | 37.72% | 60.47% | 44.06% |
| 2 | 0.0042 | 51.92% | 36.55% | 53.98% | 38.16% | 56.40% | 40.37% | 62.72% | 46.44% |
| 3 | 0.0039 | 54.13% | 38.68% | 56.20% | 40.24% | 58.29% | 42.15% | 63.74% | 47.42% |
| 4 | 0.0038 | 55.51% | 39.98% | 57.41% | 41.45% | 59.48% | 43.36% | 64.61% | 48.42% |
| 5 | 0.0037 | 56.84% | 41.34% | 58.43% | 42.50% | 60.29% | 44.22% | 65.09% | 48.98% |
| 6 | 0.0038 | 57.51% | 42.07% | 58.84% | 42.98% | 60.91% | 44.95% | 66.11% | 50.17% |
| 7 | 0.0036 | 58.20% | 42.76% | 59.53% | 43.71% | 61.56% | 45.63% | 66.45% | 50.52% |
| 8 | 0.0035 | 59.08% | 43.70% | 60.48% | 44.70% | 62.38% | 46.50% | 67.05% | 51.23% |
| 9 | 0.0034 | 59.66% | 44.36% | 61.14% | 45.41% | 62.97% | 47.14% | 67.75% | 52.05% |
| 10 | 0.0032 | 60.16% | 44.90% | 61.92% | 46.25% | 63.92% | 48.17% | 68.61% | 53.07% |
| 11 | 0.0032 | 60.64% | 45.40% | 62.38% | 46.74% | 64.42% | 48.72% | 69.45% | 54.03% |
| 12 | 0.0032 | 60.88% | 45.73% | 62.69% | 47.12% | 64.77% | 49.14% | 69.73% | 54.36% |
| 13 | 0.0032 | 61.26% | 46.13% | 63.03% | 47.52% | 65.12% | 49.55% | 70.18% | 54.94% |
| 14 | 0.0031 | 61.64% | 46.51% | 63.45% | 47.95% | 65.50% | 49.94% | 70.48% | 55.29% |
| 15 | 0.0031 | 61.79% | 46.76% | 63.68% | 48.25% | 65.78% | 50.30% | 70.66% | 55.50% |
| 16 | 0.0031 | 62.12% | 47.11% | 64.04% | 48.64% | 66.06% | 50.59% | 70.87% | 55.76% |
| 17 | 0.0031 | 62.14% | 47.15% | 64.05% | 48.70% | 66.05% | 50.68% | 71.25% | 56.25% |
| 18 | 0.0030 | 62.31% | 47.36% | 64.50% | 49.20% | 66.49% | 51.13% | 71.29% | 56.27% |
| 19 | 0.0030 | 62.29% | 47.39% | 64.46% | 49.20% | 66.60% | 51.31% | 71.45% | 56.51% |
| 20 | 0.0029 | 62.77% | 47.86% | 64.73% | 49.48% | 66.68% | 51.38% | 71.48% | 56.53% |

Best by recall@15%：epoch 20，recall@15%=66.68%，Jaccard@15%=51.38%。
Final epoch：epoch 20，val KL=0.0029，recall@5%=62.77%，recall@10%=64.73%，recall@15%=66.68%，recall@30%=71.48%。

### 500K 阶段结论

1. 增大到 500k pair 后，KL-only selector 继续明显提升：rate=0.15 recall 从 20k 的 61.72% 提升到 66.68%。
2. 高 rate 下更接近 teacher：rate=0.30 recall 达到 71.48%，已经接近之前 `middle partial Draft` 对 full Draft 的 72.21% token recall。
3. 但 rate=0.15 仍低于 online/middle Draft selector，所以如果目标是替代 15% online selector，还需要继续提高；如果系统允许更高 recompute rate，student selector 已经接近可测试 FusionRAG accuracy 的阶段。
4. 本轮 500k 来自同一小语料的 stride=1 密集滑窗，训练步数足够，但语料多样性不足。后续若换真实 500k-1M 通用语料，可能还有提升空间，也能排除重复窗口带来的过拟合风险。

## WikiText-103 Held-out Scaling Run

### 实验设计

本轮修正之前 500k 实验的主要问题：之前 500k 来自小语料 `stride=1` 密集滑窗，train/val 之间虽然按 pair id 分开，但语义和 token 窗口高度重叠。因此它只能说明训练量 scaling trend，不能说明真实泛化。

本轮改用真实 WikiText-103 train 文本：`data/wikitext103_train.txt`，文件大小约 515MB。实验切分如下：

- train pair 来源：WikiText-103 原始文本前 35%，最多生成 500,000 条 pair。
- held-out val pair 来源：WikiText-103 原始文本最后 10%，最多生成 50,000 条 pair。
- 中间 55% 文本不参与本轮，作为隔离带，避免 train/val 共享相邻滑窗。
- pair 构造：`doc_len=384`, `query_len=64`, `stride=64`。
- teacher：真实 Qwen2.5-3B full DraftModel selector，对每个 doc token 产生 full teacher importance distribution。
- student：4-layer hidden-512 Transformer selector，KL-only distribution distillation。
- 评估：在 held-out val cache 上，比较 student top-r token set 和 full DraftModel top-r token set 的 recall/Jaccard，r = 5%, 10%, 15%, 30%。

关键点：这里的 validation 不再来自训练 cache 的后 15%，而是独立 `teacher_cache_wikitext103_val_50k`。因此表中 token 差异可以更直接反映 student 对真实 full DraftModel selector 的泛化拟合能力。

启动脚本：

- pair generation：`scripts/prepare_text_slice_pairs.py`
- train teacher：`scripts/launch_teacher_wikitext103_500k_train.sh`
- val teacher：`scripts/launch_teacher_wikitext103_50k_val.sh`
- student training：`scripts/launch_train_wikitext103_500k_kl.sh`

### WikiText-103 500K 训练结果

完整逐 epoch 表：`results/kl_wikitext103_500k_training_summary.csv`。
模型 checkpoint：`checkpoints/kl_wikitext103_500k_h512_l4/model.pt`。

| epoch | val KL | recall@5% | Jaccard@5% | recall@10% | Jaccard@10% | recall@15% | Jaccard@15% | recall@30% | Jaccard@30% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0036 | 57.22% | 41.24% | 58.35% | 42.16% | 60.06% | 43.75% | 64.02% | 47.67% |
| 2 | 0.0033 | 58.96% | 42.99% | 60.47% | 44.29% | 62.41% | 46.17% | 66.19% | 50.04% |
| 3 | 0.0031 | 60.69% | 44.66% | 62.22% | 46.02% | 64.15% | 47.97% | 67.72% | 51.73% |
| 4 | 0.0029 | 62.00% | 46.03% | 63.38% | 47.26% | 65.27% | 49.17% | 68.80% | 52.93% |
| 5 | 0.0028 | 63.11% | 47.20% | 64.69% | 48.69% | 66.51% | 50.55% | 70.26% | 54.63% |
| 6 | 0.0025 | 64.25% | 48.40% | 65.90% | 49.97% | 67.73% | 51.90% | 71.52% | 56.13% |
| 7 | 0.0024 | 65.38% | 49.64% | 66.93% | 51.08% | 68.83% | 53.11% | 72.68% | 57.51% |
| 8 | 0.0021 | 67.95% | 52.46% | 69.64% | 54.13% | 71.48% | 56.20% | 75.05% | 60.46% |
| 9 | 0.0019 | 68.89% | 53.50% | 71.03% | 55.74% | 72.80% | 57.76% | 76.37% | 62.12% |
| 10 | 0.0018 | 69.71% | 54.47% | 71.87% | 56.76% | 73.62% | 58.78% | 77.13% | 63.13% |
| 11 | 0.0018 | 70.44% | 55.33% | 72.50% | 57.52% | 74.23% | 59.53% | 77.65% | 63.80% |
| 12 | 0.0017 | 70.91% | 55.89% | 72.99% | 58.12% | 74.74% | 60.18% | 78.14% | 64.45% |
| 13 | 0.0017 | 71.32% | 56.36% | 73.45% | 58.68% | 75.18% | 60.72% | 78.49% | 64.91% |
| 14 | 0.0016 | 71.70% | 56.83% | 73.78% | 59.09% | 75.50% | 61.14% | 78.85% | 65.41% |
| 15 | 0.0016 | 71.92% | 57.09% | 73.94% | 59.29% | 75.60% | 61.26% | 78.89% | 65.46% |
| 16 | 0.0016 | 72.08% | 57.29% | 74.24% | 59.66% | 75.87% | 61.61% | 79.11% | 65.75% |
| 17 | 0.0015 | 72.55% | 57.84% | 74.61% | 60.12% | 76.26% | 62.11% | 79.50% | 66.29% |
| 18 | 0.0015 | 72.75% | 58.09% | 74.77% | 60.32% | 76.45% | 62.36% | 79.72% | 66.57% |
| 19 | 0.0015 | 72.95% | 58.35% | 74.92% | 60.51% | 76.61% | 62.57% | 79.85% | 66.77% |
| 20 | 0.0015 | 73.24% | 58.69% | 75.25% | 60.92% | 76.90% | 62.94% | 80.08% | 67.07% |

Best by recall@15%：epoch 20，recall@15%=76.90%，Jaccard@15%=62.94%。
Final epoch：epoch 20，val KL=0.0015，recall@5%=73.24%，recall@10%=75.25%，recall@15%=76.90%，recall@30%=80.08%。

### WikiText-103 阶段结论

1. 这次验证集合是 WikiText-103 原始文本最后 10%，teacher cache 独立生成，不再复用训练 cache 的后 15%。因此指标更接近真实 held-out selector distillation。
2. 500k WikiText-103 训练后，student 对 full DraftModel selector 的 token set 复现显著提升：recall@15%=76.90%，Jaccard@15%=62.94%。
3. 该结果明显强于前一轮小语料 500k 的 recall@15%=66.68%，也超过之前 `middle partial Draft` 对 full Draft 的 72.21% recall 参考线。
4. 这说明用真实通用语料做 KL 分布蒸馏是有效路线，下一步值得接入 FusionRAG pipeline，测试 student selector 的真实 answer accuracy 和 selection latency。
