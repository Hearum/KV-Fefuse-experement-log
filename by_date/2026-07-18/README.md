# 2026-07-18 实验目录

本目录保存 2026-07-18 创建或启动的 KV 稀疏重算、稀疏 adapter 和搜索实现实验。以下索引逐个说明目录用途；详细命令、commit、结果和异常以各实验目录内文档为准。

## 实验列表

- `delta_kv_sparse_adapter`：在原始或 preprocess KV 上研究 sparse attention 产生的 K/V 与缓存 KV 的比例插值 adapter，计划文件已定义 alpha、rate 和重算路径。
- `spare_k_q_recompute_exp`：在 MuSiQue-v2 setup-standard pipeline 上比较 sparse K/Q recompute 与 full/dense baseline，包含结果汇总和后续实验计划。
- `spare_kv_search_10`：历史 top-10 稀疏 KV 搜索实验，保留 smoke、中间结果和原始启动记录。
- `spare_kv_search_10_chunked`：历史 top-10 chunked 搜索变体，记录分块执行对稀疏 KV 搜索的影响。
- `spare_kv_search_10_chunked_logs`：`spare_kv_search_10_chunked` 的运行日志归档目录。
- `spare_kv_search_10_fixedkernel`：历史 top-10 fixed-kernel 实现变体，用于排查固定 kernel 路径的行为和性能。
- `spare_kv_search_10_fixedkernel_logs`：`spare_kv_search_10_fixedkernel` 的运行日志归档目录。
- `spare_kv_search_10_logs`：`spare_kv_search_10` 的运行日志归档目录。
- `spare_kv_search_10_retry_logs`：top-10 搜索重试任务的日志归档，用于定位失败、超时和重跑原因。
- `spare_kv_search_10_top64chunk`：top-10 搜索配合 top-64 chunk 候选范围的历史变体。
- `spare_kv_search_10_top64chunk_logs`：`spare_kv_search_10_top64chunk` 的运行日志归档目录。
- `spare_kv_search_50`：历史 top-50 稀疏 KV 搜索实验，用于与 top-10 候选范围比较。
- `spare_kv_search_50_chunked`：历史 top-50 chunked 搜索变体，关注分块实现和候选覆盖范围。
- `spare_kv_search_50_chunked_logs`：`spare_kv_search_50_chunked` 的运行日志归档目录。
- `spare_kv_search_50_logs`：`spare_kv_search_50` 的运行日志归档目录。
- `spare_kv_search_50_sharded`：历史 top-50 sharded 搜索变体，关注跨 GPU/分片执行方式。
- `spare_kv_search_50_sharded_logs`：`spare_kv_search_50_sharded` 的运行日志归档目录。

每个实验目录必须包含 `plan.md`、`README.md` 或 `EXPERIMENT_LOG.md`，并记录主项目 branch/commit、实验仓库 commit、原始数据、模型/cache、结果路径和复现命令。
