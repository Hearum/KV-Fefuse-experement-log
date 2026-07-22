# Dense/Sparse Attention × Alpha 正交实验结果

## 实验目的

区分两个因素：

1. attention support：dense causal attention 与 sparse block top-k attention；
2. KV write-back：alpha=0 与 alpha=1。

## 固定配置

- 模型：Qwen3-32B
- 数据集：MuSiQue-v2
- 样本：0--10，共10条，四组相同问题顺序
- cache：共享 preprocess KV
- FusionRAG selector：rate=0.15
- sparse：block size=64、top-k=8
- alpha：K/V 使用相同标量，分别为0和1
- 评测：runner 自动生成 EM、F1 和 GLM judge
- 机器：qjy003，GPU0--3
- 代码/实验文档提交：f4ec4a2

## 启动命令

四组均调用：

    /mnt/qjhs-sh-lab-01/wjh/FusionRAG/.venv/bin/python MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py --dataset musique-v2 --method METHOD --rate 0.15 --working-kv-alpha-k ALPHA --working-kv-alpha-v ALPHA --start 0 --end 10 --gpu GPU --result-root MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/results_alpha_support_ablation_10

实际组合：

| 方法 | alpha K/V | GPU | 结果路径 |
|---|---:|---:|---|
| dense attention + alpha=0 | 0/0 | 0 | results_alpha_support_ablation_10/dense_working_kv_preprocess/musique-v2/rate_0p15/alpha_k_0p0__alpha_v_0p0/seg_0_10 |
| dense attention + alpha=1 | 1/1 | 1 | results_alpha_support_ablation_10/dense_working_kv_preprocess/musique-v2/rate_0p15/alpha_k_1p0__alpha_v_1p0/seg_0_10 |
| sparse top-k=8 + alpha=0 | 0/0 | 2 | results_alpha_support_ablation_10/sparse_working_kv_preprocess/musique-v2/rate_0p15/alpha_k_0p0__alpha_v_0p0/seg_0_10 |
| sparse top-k=8 + alpha=1 | 1/1 | 3 | results_alpha_support_ablation_10/sparse_working_kv_preprocess/musique-v2/rate_0p15/alpha_k_1p0__alpha_v_1p0/seg_0_10 |

## 汇总结果

| 方法 | EM | F1 | GLM |
|---|---:|---:|---:|
| dense attention + alpha=0 | 0.3000 | 0.3558 | 4/10 = 40% |
| dense attention + alpha=1 | 0.2000 | 0.3028 | 4/10 = 40% |
| sparse top-k=8 + alpha=0 | 0.3000 | 0.3558 | 4/10 = 40% |
| sparse top-k=8 + alpha=1 | 0.3000 | 0.3558 | 4/10 = 40% |

## 逐题比较

- dense alpha=0 与 sparse alpha=0：10/10 预测完全一致。
- dense alpha=0 与 dense alpha=1：6/10 一致，4/10 改变。
- sparse alpha=0 与 sparse alpha=1：5/10 一致，5/10 改变。
- dense alpha=1 与 sparse alpha=1：5/10 一致，5/10 改变。

## 结论

### 1. alpha 确实生效

alpha=0 与 alpha=1 不是同一条输出路径。dense 情况改变4/10，sparse 情况改变5/10。因此之前 sparse 性能下降不能解释为“alpha 没打开”；本实验 alpha=1 已实际参与 KV write-back。

### 2. 这10条上 support 不是唯一因素

在 alpha=0 时，dense 与 sparse 输出10/10完全一致，且 EM/F1/GLM 完全相同。这说明对于 alpha=0 这条路径，本批样本的 sparse support 没有改变最终生成答案，或者 sparse alpha=0 的代码路径实际退化为与 dense 相同的有效计算路径。这个现象需要在更大样本上继续确认，不能直接外推为 sparse support 无影响。

### 3. alpha=1 的影响是非单调的

dense alpha=1 相比 dense alpha=0：EM 从0.30降到0.20，F1从0.3558降到0.3028，GLM保持4/10。

sparse alpha=1 相比 sparse alpha=0：三项 aggregate 指标在10条上相同，但逐题有5条改变，说明 aggregate coincidence 不能代表逐题等价。

这支持当前判断：sparse candidate hidden state 产生的 candidate KV 并不稳定地逼近 full recompute KV；alpha=1 写回可能改变答案，但不保证向 native 方向改善。

### 4. 当前不能据此交付稀疏重算

这只是10条 sanity check，不是完整数据集结论，也没有证明端到端加速。下一步应先扩展 alpha/support 对照到至少50条，并加入 native online_qk 同批 reference；只有当 sparse alpha=1 在逐题 accuracy 接近 native 且耗时下降时，才有交付意义。

## 三视角审核

### 模型语义审核

PASS。alpha=0/1、dense/sparse 四条路径分开统计；没有把 alpha 当作 attention support，也没有把 sparse candidate 当作 full recompute oracle。

### 系统实现审核

PASS。四组使用相同样本段、共享 preprocess cache、固定 rate=0.15，并记录了实际 GPU、命令模板、结果路径和自动评测产物。

### 实验科学审核

PASS。报告区分 aggregate 指标和逐题一致性；明确样本量只有10，alpha=0下 dense/sparse 完全一致的观察被标记为待扩展，而不是过度外推。

三视角审核均 PASS。

