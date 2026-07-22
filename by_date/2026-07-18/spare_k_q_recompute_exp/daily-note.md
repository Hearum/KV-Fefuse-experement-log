# Working-KV 实验每日记忆

## 2026-07-19

- 唯一任务规范：`goal.md`。目标是 layer-parallel 的 selected-token working-KV：每层 candidate 与 immutable base 融合后，必须先 scatter，再让本层 token attention 读取。
- 分支：`exp/sparse-kv-working-blend-20260719`；起点 `d2dcbd3`；首轮探索实现最后提交 `508d587`。
- setup-v2 实际模型文件是 `models/modeling_qwen3.py`，不是 `ktransformers/models/modeling_qwen3.py`。
- 共享 cache：`/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2`。禁止每 worker 生成独立 cache。
- 旧提交 50 样本矩阵在 qjy000/qjy001 运行；保留至完成，但只标为 exploratory。
- 首轮三 reviewer 全部返回。共同阻塞项：production base 语义、真实 attention 读取测试、cache length、多设备索引、dependency coverage、delta direction/oracle alpha、同源 alpha=0、验证/测试隔离、效率边界。
- 第二版修复已通过小模型测试：immutable selected-position base snapshots、pre-attention scatter、selected predecessor 实际读取、dependency stats、query isolation；working-KV 限定 SDPA，>32768 selected tokens fail-fast，`past_tokens` finally 恢复。
- 下一步顺序：提交修复 -> 32B 1-sample 两端回归 -> 完成同源 50 样本验证 -> 冻结 alpha -> 独立 test -> delta/oracle/dependency/效率分析 -> 详细报告 -> 三 reviewer 审核订正直至通过。
- 修复版 32B 端点通过：Dense/Sparse alpha0=`Mary Bono`，alpha1=`Salma Hayek`（Sparse 带 `Answer:` 前缀）。
- delta trace（N=5，每层前32 selected tokens）：raw K/V cosine 0.498/0.468，oracle residual 0.867/0.884；preprocess 0.545/0.562，oracle residual 0.839/0.827。global scalar 不足以恢复 dense delta。
- router stats（N=1、64层×64 heads）：dependency coverage 7.79%，mass recall 28.38%，480.6/9901.1 causal KV per query，support=4.85%。
- 单个 router example 的 preserve-all 静态估算需要125.5 blocks、8003.6 KV/query，达到dense support的80.83%；只能作为优先级证据，不能总体否决。
- 报告v1三审均NOT PASS：最终矩阵/test待完成；router因果措辞需降级；自动测试/静态保障/32B sanity需分开；补manifest和多设备边界。
- validation 前50中间：preprocess alpha0→.75 GLM 32%→34%；raw 26%→40%。只能用于选 alpha，待全矩阵与独立 test。
- 旧矩阵 Dense CSV 运行中曾出现临时/重复行，禁止作为正式结果；修复版汇总只接收 judge 后恰好 N=50 的 metrics。
- 修复版 validation 20/20 条件完成。按预先规则冻结 Dense raw alpha=0.75 和 Sparse raw alpha=0.75；preprocess 只保留 alpha=0 参考。
- Frozen test rows 51--200 已完成，每方法 150 个唯一问题。Full GLM=41.33%；Dense raw 0.75=33.33%，比 full 低8pp；Sparse raw 0.75=27.33%，比 full 低14pp。
- Dense raw 0.75 相对 raw0 在 test 仅 GLM +2.67pp，paired CI 覆盖0；Sparse raw 0.75 相对 raw0 为 -3.33pp。validation 的 +14pp 趋势未在 test 复现，不能识别为单一原因。
- 最终方法判断：Working-KV 语义保留；Dense blend 是不省 attention 的诊断上界；Top-K=8 MoBA + global alpha 不可用，且当前 Python prototype 无端到端加速证据。
- 完整性汇总 commit `28adeb9`；最终报告需包含 validation/test、delta N=5、router N=5、限制和复现命令，然后进入三 reviewer 循环。
- 首轮最终报告三审均 NOT PASS；核心语义和逐样本数值获确认。订正重点：cached K 写清 k_norm+RoPE、降级 Dense/delta 因果措辞、未来新增 dev/holdout、冻结完整逐样本证据、增加 cache read-only 与 GPU 绑定 guard、扩充 manifest。
- 第二轮：因果与科研 reviewer PASS，系统 reviewer NOT PASS。发现 cache read-only 尝试未覆盖 FAISS/目录写入，已删除该不完整接口并如实保留回填风险；validation 汇总器改为整行冲突检测、严格 N/question-set；manifest 分开标注 Working-KV semantic guards 与 runner GPU binding commit。需第三轮三方全审。
- 2026-07-20 rate=0.15 主线：目标改为低于原生 online_qk rate=0.15 的重算成本，同时保持其性能；full rate=1 仅作上界。
- MuSiQue-v2 前10条严格校验：native online_qk 与 strict doc/query 两阶段 10/10 答案一致；Working-KV alpha=1 与 native 逐题输出不一致；当前 alpha 分支不能当作 native rate=0.15 等价实现。
- 已生成 WORKING_KV_RATE015_DELIVERY_PLAN.md 和 WORKING_KV_RATE015_EXECUTIVE_REPORT.md；计划 commit 2f7f495。报告三视角审核均 PASS，后续先修正 oracle 等价性，再测试 sparse/predictor。

## 2026-07-20（继续验证）

- 当前代码语义 guard 已直接运行通过：Working-KV endpoints、immutable base snapshot、pre-attention scatter、selected-predecessor read、working-K routing、dependency stats、query isolation、past-token restoration，以及 setup-v2 GPU binding 均 PASS。
- 正在运行的两个候选验证均复用共享 preprocess cache：qjy000 GPU6 为 sparse block top-k=64、query chunk=32、0--10；qjy001 GPU0 为 sparse block top-k=32、query chunk=64、0--10。
- 截至本记录时两任务仍在 GPU 执行，尚未产生完整 metrics.csv；不得提前把其作为 accuracy 结果。已有 top-k32 小样本结果为 GLM 4/10、答案不一致 5/10；top-k8 的50条结果为 GLM 16/50、相同批次 native 为20/50、答案不一致22/50。
- 当前代码提交链：native alpha=1 等价修正 1ba0b7b；sparse gather 内存修正 85ad6f0；报告与分片启动器 4dfb016。正式结果完成后再追加结果 commit。
- top-k64 验证于约21分钟只生成5/10条，前5条 exact=3/5，未生成 metrics.csv，随后停止；不能计为完整 accuracy。V4 报告记录该执行失败边界，提交 cc3451e。
- 三视角审核（模型语义、系统实现、实验科学）对 V4 均 PASS：确认没有把 partial prediction 当完整指标，也没有把 aggregate GLM 或 top-k64 前5条当作交付结论。
- 完成 SUPPORT_UPPER_BOUND_ANALYSIS.md：5个example的router trace显示 top-k=8 support=4.80%、attention mass recall=29.74%、dependency coverage=7.42%；若强制保留全部 selected predecessor dependency，support约80.89%。这给出了继续扩大top-k的计算上界约束。分析提交 21ffabe，三视角审核均 PASS。
- sparse block representative 已从逐 block Python loop 改成 padding + reshape + sum/length 批量计算，随机 context=128/130/32768 的结果逐元素等价测试 PASS，代码提交 a587724。
- qjy001 GPU0 vectorized top-k32 smoke（0--1）完成：GLM 0/1，EM/F1 0；预测 Maggie Gyllenhaal，与旧 top-k32 smoke 的同一条预测一致，说明本次优化没有改变 routing/output。命令与结果根记录于 results_vectorized_smoke/，自动 GLM 已由 runner 同步执行。
- 本 smoke 只验证实现等价和性能优化方向，不外推完整 accuracy；下一步若继续扩大应先测 wall-clock，再做 10 条质量对照。
- V5 报告已完成并通过三视角审核，记录 vectorized representative 的代码提交 a587724、smoke 结果路径、自动 GLM 结果和前序对照；报告提交 3a4f9be。下一步是同一配置 wall-clock 回归，再决定是否扩大10条。
- vectorized block representative microbenchmark（GPU0，H=64,C=32768,D=128,B=64，20次同步平均）：旧Python loop 25.54ms，新实现1.53ms，约16.6x，max_abs/mean_abs均0。V6报告提交304a388，三视角审核均PASS。
- 额外 benchmark（qjy003 GPU0，H=8,Q=128,N=1024,D=128）：手写 bmm/softmax 1.667ms，SDPA 2.057ms，max_abs=4.88e-4；当前形状下 SDPA并未更快，因此暂不替换现有 attention 实现。该结果为优化探针，不改变实验代码。
- vectorized top-k32 10条回归运行8分52秒后仍停在3/10，无新增第4条，未生成metrics.csv，已停止；3条中exact=2/3。该结果确认block mean向量化并未解决端到端gather瓶颈，status.log已记录，不能作为完整accuracy。
- 按用户要求暂停kernel方向，优先补完整accuracy。chunk probe（head32/query256）因申请8GiB OOM，无有效结果；已改回安全的head8/query64，启动top-k32完整0--10评测，结果根results_accuracy_top32_10，PID3909600，自动EM/F1/GLM随runner执行。
- 按要求启动 alpha/support 正交实验，qjy003 GPU0--3，MuSiQue-v2 0--10，preprocess KV、rate=0.15、共享cache；dense alpha=0/GPU0 PID3448192，dense alpha=1/GPU1 PID3448193，sparse top-k=8 block64 alpha=0/GPU2 PID3448194，sparse top-k=8 block64 alpha=1/GPU3 PID3448195。结果根 results_alpha_support_ablation_10，自动EM/F1/GLM由runner执行。
- alpha/support四组10条结果完成并记录于 ALPHA_SUPPORT_ABLATION_RESULTS.md（提交04292fc）：dense a0 EM/F1=.30/.3558 GLM4/10；dense a1=.20/.3028 GLM4/10；sparse topk8 a0=.30/.3558 GLM4/10；sparse topk8 a1=.30/.3558 GLM4/10。dense a0与sparse a0逐题10/10一致；alpha确实生效，但alpha1不保证向native改善。三视角审核PASS。
- 重新按 CACHE_ANCHORED_SPARSE_RECOMPUTE_PLAN.md 启动阶段A：dense alpha=.25/.5/.75（GPU0/1/2，PID3656265/66/67），sparse top-k8 alpha=.25/.5/.75（GPU3/4/5，PID3656268/69/3658235），全部MuSiQue-v2 0--10、rate=.15、preprocess KV、自动EM/F1/GLM。首次GPU5路径写错已修正，失败进程无有效结果。
- 阶段A六组完成，结果报告 CACHE_ANCHORED_SPARSE_PLAN_A_RESULTS.md（提交4a169c0，三视角PASS）。native前10 EM/F1=.20/.3028 GLM4/10；dense alpha .25/.5/.75分别 .30/.3558/.3694 GLM均4/10；sparse topk8 alpha .25/.5/.75分别 .30/.3558/.3865/.3917 GLM均4/10。下一步冻结sparse alpha=.75，扩展独立50条并加入topk32。
- 阶段B启动：冻结alpha K/V=.75，在MuSiQue-v2完整0--50样本运行sparse top-k8（GPU0 PID3678710）与top-k32（GPU1 PID3678711），rate=.15、block64、共享preprocess cache；结果根results_frozen_alpha075_50，native同批参考为已有results_rate015_50_fixed。
- 发现并发写同一result-root导致results_frozen_alpha075_50冲突，已停止PID3678710/3678711并将目录改名results_frozen_alpha075_50_collision_invalid保留异常证据（不使用、不删除）。已用独立结果根重启：top8 GPU0 PID3725109 -> results_frozen_alpha075_50_top8；top32 GPU1 PID3725110 -> results_frozen_alpha075_50_top32。
