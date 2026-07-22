
## 2026-07-13 Qwen3-32B Rate=0 Cross-Dataset Baseline Launch

Purpose: fill missing `rate=0` / no document-token recompute baselines for Qwen3-32B cross-dataset comparison against online DraftModel `rate=0.15`.

Result root:

```text
/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_rate0_cross_dataset_baseline
```

Script:

```text
MOTIVATION_EXPERIMENTS/qwen3_rate0_cross_dataset_baseline/scripts/run_rate0_worker.py
MOTIVATION_EXPERIMENTS/qwen3_rate0_cross_dataset_baseline/scripts/summarize_rate0.py
```

Cache policy:

- Use existing cross-dataset preprocess/raw KV cache only:
  `/mnt/qjhs-sh-lab-04/lpl/hming/fusionrag-qwen3-attn-alpha-cross-cache-v2/worker_gpu{gpu}/{dataset}`
- HotpotQA rate=0 was assigned only to qjy001 GPU2-5, so it avoids the strict beta Hotpot run that was still using HotpotQA cache workers 0/1/6/7.
- 2WikiMQA and TriviaQA rate=0 were assigned to qjy003 GPU0-5. These use separate dataset cache directories and do not conflict with strict HotpotQA.
- MuSiQue is not included in this launch because qjy001 GPU0 is still running the single-writer MuSiQue cache task; previous MuSiQue rate=0 result already exists in `reflect_pipeline_full_kv_writeback_ablation/strict_summary_with_rate0.csv`.

Launch commands:

```bash
# qjy001, HotpotQA only
cd /home/hming/FusionRAG-pca-analysis
FUSIONRAG_RATE0_EXP_ROOT=/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_rate0_cross_dataset_baseline \
FUSIONRAG_RATE0_DATASETS=hotpotqa \
FUSIONRAG_RATE0_GPUS=2,3,4,5 \
setsid MOTIVATION_EXPERIMENTS/qwen3_rate0_cross_dataset_baseline/scripts/run_rate0_worker.py \
  > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_rate0_cross_dataset_baseline/logs/launcher_qjy001_hotpot.log 2>&1 < /dev/null &

# qjy003, 2WikiMQA + TriviaQA
cd /home/hming/FusionRAG-pca-analysis
FUSIONRAG_RATE0_EXP_ROOT=/mnt/qjhs-sh-lab-04/lpl/hming/qwen3_rate0_cross_dataset_baseline \
FUSIONRAG_RATE0_DATASETS=2wikimqa,triviaqa \
FUSIONRAG_RATE0_GPUS=0,1,2,3,4,5 \
setsid MOTIVATION_EXPERIMENTS/qwen3_rate0_cross_dataset_baseline/scripts/run_rate0_worker.py \
  > /mnt/qjhs-sh-lab-04/lpl/hming/qwen3_rate0_cross_dataset_baseline/logs/launcher_qjy003_2wiki_trivia.log 2>&1 < /dev/null &
```

Summarize:

```bash
cd /raid/home/hming/FusionRAG-pca-analysis
MOTIVATION_EXPERIMENTS/qwen3_rate0_cross_dataset_baseline/scripts/summarize_rate0.py
```
