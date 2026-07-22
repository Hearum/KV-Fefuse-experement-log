# MuSiQue GLM Judge 审核结论

## 总体不一致统计

- 总样本行数：2500
- 原始错误、GLM 改为正确：138
- 原始正确、GLM 改为错误：11
- false->true 中，原始 reason 明确提到预测为空/未提供的样本：120

## false->true 粗分类

- exact_norm: 91
- pred_subset_gold: 27
- low_overlap: 9
- gold_subset_pred: 7
- high_gold_token_coverage: 3
- partial_gold_token_coverage: 1

## 判断

GLM 的确比原始 judge 更宽，所有方法的 accuracy 都上升。但抽样和粗分类显示，主要原因不是全部误判变宽：false->true 中 91/138 是标准化后完全一致，另有 27/138 是预测是标准答案的子串，这些大多是原始 judge 将非空预测误判为空或过于严格。
同时，GLM 也存在少量宽判风险。典型案例包括：只回答 Henry III 的首次加冕年份 1216、额外多答 Middleton and Moston、把 Windjammer Communications 写成 Windjammer Cable 等。粗略看 low_overlap 类只有 9/138，但这些样本会放大 full 和部分 online 方法的收益。
因此，当前 GLM rerun 更适合作为“语义宽松 judge”的补充结果，不应直接和旧 judge 的绝对数值混用。写论文或主表时建议固定一种 judge，并在附录说明 GLM clean rerun 的判分尺度更宽。

详细抽样见 `glm_judge_review_samples.md`。
