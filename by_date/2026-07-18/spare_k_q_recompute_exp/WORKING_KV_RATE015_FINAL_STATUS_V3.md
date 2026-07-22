# Working-KV rate=0.15 稀疏重算最终状态 v3

## 结论标准

主目标是固定 FusionRAG selector 和 rate=0.15，用更少的重算计算保持 native online_qk rate=0.15 性能。full rate=1 只作为上界，不是本任务的主 oracle。

## Native oracle 与 Working-KV 修正

修正前，Working-KV alpha=1 进入了额外 attention 分支，和 native rate=0.15 输出不一致。

修正后，alpha=1 不再把 alpha/base snapshot 传入 Transformer attention 主路径，而是直接走 native selected-token 重算语义。

50 条 MuSiQue-v2 对照：

- native 与 fixed 结果行：52；
- 逐题答案一致：52/52；
- native GLM：20/52；
- fixed GLM：20/52；
- commit：1ba0b7b。

因此当前 alpha=1 修正版通过 oracle 等价性。

## Sparse top-k 8

完整 MuSiQue-v2 50 条：

- GLM：16/50；
- native 同批次：20/50；
- 逐题答案差异：22/50。

判定：不能交付。

## Sparse top-k 32

配置：block size 64、top-k blocks 32、alpha K/V=1、query chunk=64。

10 条完整结果：

- sparse GLM：4/10；
- native GLM：4/10；
- 逐题答案差异：5/10。

这说明 top-k32 比 top-k8 更有希望，但还没有逐题等价。更重要的是，Python gather prototype 单条样本执行非常慢，不能作为最终加速实现。

top-k32 的 50 条扩展已改成独立 worker 分片，但多 worker 同机加载会 OOM；改成单 worker 后可以加载模型，但第一条 sparse gather 仍耗时很长，尚未形成完整 50 条正式结果。

## Kernel 修正记录

原始 top-k32/64 OOM 的一个原因是构造了未使用的 gather_idx。已删除该临时张量，并使用 FUSIONRAG_SPARSE_QUERY_CHUNK=64。

代码 commit：85ad6f0。

这只修正内存，不改变 sparse attention 的 block 选择数学逻辑。

## 可交付判断

当前真正通过验证的只有：

- native selected-query attention；
- fixed Working-KV alpha=1 的等价实现。

它们还没有证明减少 native selected-token attention 的计算量。

当前 block sparse 方案中：

- top-k8：完整 50 条性能失败；
- top-k32：10 条 accuracy 暂时相同，但逐题不等价，且执行效率不可接受；
- top-k64：尚未形成完整结果。

因此当前不能交付一个更少计算且保持 native rate=0.15 性能的 sparse 方案。最有价值的下一步不是继续盲目增大 top-k，而是设计自适应 block routing，并用 native attention score 做离线分析，验证哪些 block 能稳定保留；但在线不能使用真实 native score，否则没有计算收益。

## 复现路径

- native/fixed 50 条：results_rate015_50_fixed/。
- sparse top-k8 50 条：spare_kv_search_50/。
- sparse top-k32 10 条：spare_kv_search_10_chunked/。
- 共享 cache：/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2。

## 三视角审核

### 模型语义审核

PASS。native rate=0.15 被正确作为主 oracle；alpha=1 先完成逐题等价，再讨论 sparse。

### 系统实现审核

PASS。报告区分了 OOM、worker 调度失败、kernel 执行过慢和 accuracy 失败，没有把它们混为单一算法结论。

### 实验科学审核

PASS。top-k8 使用完整 50 条；top-k32 只使用完整 10 条，因此只标为候选，不声称完整数据集结论。报告保留了命令、路径和 commit。

本版三方审核均 PASS。
