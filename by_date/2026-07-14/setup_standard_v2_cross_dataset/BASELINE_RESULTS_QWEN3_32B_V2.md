# Qwen3-32B Setup-v2 Baseline Results

整理日期：2026-07-23

本文件记录 setup-standard v2 direct-QA pipeline 的稳定 baseline。数据集为 MuSiQue-v2、2WikiMQA-v2、HotpotQA-v2、TriviaQA-v2，不是旧 reflect/main-sub pipeline 的结果。

## 来源

- 实验目录：MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/
- 数据：data/musique-v2.jsonl（200）、data/2wikimqa-v2.jsonl（200）、data/hotpotqa-v2.jsonl（260）、data/triviaqa-v2.jsonl（270）
- runner：scripts/run_setup_v2_task.py
- EM/F1：setup_v2_summary.csv
- 合并表：setup_v2_summary_with_glm.csv
- GLM：各 rejudge_glm_clean_* 目录下的 rejudged_summary.csv
- GLM 口径：GLM-5.2，关闭 thinking，去除 think 标签内容和前导 Answer

所有列出的组均为完整数据集。EM、F1 和 GLM Acc 均为百分比。

## MuSiQue-v2（200）

| 方法 | Rate | EM | F1 | GLM Acc |
|---|---:|---:|---:|---:|
| full attention | 1.0 | 25.50 | 40.03 | 41.50 |
| raw KV | 0 | 15.00 | 27.77 | 29.50 |
| preprocess KV | 0 | 20.50 | 34.01 | 32.50 |
| online DraftModel | 0.05 | 20.00 | 33.80 | 36.00 |
| online QK | 0.05 | 20.50 | 33.90 | 33.50 |
| uniform alpha=0.1 | 0.05 | 22.00 | 35.94 | 38.00 |
| online DraftModel | 0.15 | 24.00 | 37.88 | 41.00 |
| online QK | 0.15 | 21.50 | 35.57 | 37.50 |
| uniform alpha=0.1 | 0.15 | 25.00 | 39.28 | 41.50 |
| offline3b_mean | 0.15 | 24.50 | 38.79 | 40.50 |
| offline3b_freq_boundary2 | 0.15 | 23.00 | 36.75 | 38.50 |
| offline3b_top2 | 0.15 | 24.50 | 38.47 | 40.50 |

## 2WikiMQA-v2（200）

| 方法 | Rate | EM | F1 | GLM Acc |
|---|---:|---:|---:|---:|
| full attention | 1.0 | 48.00 | 58.99 | 61.50 |
| raw KV | 0 | 32.50 | 42.96 | 42.00 |
| preprocess KV | 0 | 35.00 | 47.82 | 45.50 |
| online DraftModel | 0.05 | 41.50 | 53.90 | 51.50 |
| online QK | 0.05 | 39.50 | 51.52 | 51.00 |
| online DraftModel | 0.15 | 43.00 | 57.69 | 60.00 |
| online QK | 0.15 | 42.00 | 53.13 | 53.00 |
| uniform alpha=0.1 | 0.15 | 43.00 | 56.32 | 58.00 |
| offline3b_mean | 0.15 | 45.00 | 56.84 | 57.00 |
| offline3b_freq_boundary2 | 0.15 | 43.50 | 56.87 | 57.00 |
| offline3b_top2 | 0.15 | 43.50 | 55.92 | 56.50 |

本轮没有有效的 offline32b_top2 0.15 完整结果。

## HotpotQA-v2（260）

| 方法 | Rate | EM | F1 | GLM Acc |
|---|---:|---:|---:|---:|
| full attention | 1.0 | 70.38 | 80.78 | 91.15 |
| raw KV | 0 | 62.69 | 74.27 | 78.85 |
| preprocess KV | 0 | 68.08 | 77.33 | 85.38 |
| online DraftModel | 0.05 | 69.62 | 80.42 | 86.92 |
| online QK | 0.05 | 69.62 | 78.58 | 88.08 |
| online DraftModel | 0.15 | 70.77 | 81.12 | 88.85 |
| online QK | 0.15 | 66.54 | 77.09 | 85.77 |
| uniform alpha=0.1 | 0.15 | 70.38 | 80.94 | 89.23 |
| offline3b_mean | 0.15 | 69.23 | 78.95 | 87.31 |
| offline3b_freq_boundary2 | 0.15 | 68.46 | 78.33 | 86.15 |
| offline32b_top2 | 0.15 | 68.08 | 78.42 | 86.54 |

## TriviaQA-v2（270）

| 方法 | Rate | EM | F1 | GLM Acc |
|---|---:|---:|---:|---:|
| full attention | 1.0 | 66.30 | 76.93 | 90.00 |
| raw KV | 0 | 57.41 | 67.65 | 80.74 |
| preprocess KV | 0 | 64.44 | 74.59 | 87.78 |
| online DraftModel | 0.05 | 65.56 | 76.57 | 88.52 |
| online QK | 0.05 | 64.07 | 74.43 | 87.04 |
| online DraftModel | 0.15 | 66.67 | 77.12 | 90.00 |
| online QK | 0.15 | 64.81 | 75.14 | 87.41 |
| uniform alpha=0.1 | 0.15 | 65.93 | 76.46 | 89.63 |
| offline3b_mean | 0.15 | 64.81 | 75.69 | 87.78 |
| offline3b_freq_boundary2 | 0.15 | 64.07 | 75.15 | 86.67 |
| offline32b_top2 | 0.15 | 64.81 | 75.58 | 87.41 |

## 复现命令

    cd /raid/home/hming/FusionRAG-pca-analysis
    PY=/mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python
    EXP=MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset
    CACHE=/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2
    CUDA_VISIBLE_DEVICES=0 "$PY" "$EXP/scripts/run_setup_v2_task.py" --dataset musique-v2 --method online_draft --rate 0.15 --start 0 --end 200 --gpu 0 --cache-root "$CACHE"
    GLM_REJUDGE_WORKERS=10 "$PY" "$EXP/scripts/summarize_setup_v2.py" --glm-workers 10

后续 v2 实验引用本文件时，必须标注 setup-v2 pipeline、数据集版本、整理日期和来源 CSV。
