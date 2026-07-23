# KV-Fefuse Experiment Log Agent Guide

本仓库只维护 FusionRAG 的实验计划、实验记录、分析脚本、图表和小型结果。主项目代码规则以父仓库的 AGENTS.md 为准；本文件只补充实验日志仓库的具体规则。

## 仓库边界

- 只提交 Markdown、实验脚本、图表和小型 CSV/JSON 汇总。
- 不提交模型、KV cache、checkpoint、大型 tensor、完整生成中间结果或临时 worker 输出。
- 大型 artifact 统一放在 /raid/home/hming/experiment-artifacts/，并在 ARTIFACT_LOCATIONS 文档记录。
- cache 按 MODEL/DATASET 复用，不按实验、GPU 或 worker 复制。

## Canonical 目录

实验实体只能保存于：

by_date/YYYY-MM-DD/实验名/

- YYYY-MM-DD 是实验首次创建或启动日期。
- 根目录不得创建实验实体目录，也不保留实验软链接。
- 每个日期目录必须有 README.md。
- 每个实验目录必须先创建 plan.md，再启动实验；完成后维护 README.md 和 EXPERIMENT_LOG.md。
- 旧实验路径只作为历史参考，不得新建或恢复根目录别名。
- 所有实验的记录都是用中文

每个日期目录的 `README.md` 还必须提供该日期下所有实验文件夹的中文索引，不能只列目录名或使用占位符。每个实验文件夹至少用一段中文说明其研究目的/方法、数据或配置、当前状态和结果或日志入口；纯日志目录也要说明它对应哪一组实验。

新增、重命名、迁移或补跑实验时，必须同步更新所在日期目录的 `README.md`。日期 README 是 agent 了解当天实验全貌的第一入口，实验目录内的 README/EXPERIMENT_LOG.md 用于保存更详细的过程和复现信息。

## 实验启动记录

每次实验必须记录以下内容：
- 实验的plan.md内容，和开发过程的EXPERIMENT_LOG.md，记录每一轮对话/处理结果/当前状态的关键内容，可以帮助维护上下文memory。和一个交付README.md，记录本组实验的交付文档。
- 对于最终结果的交付文档README.md，详细说明回答和做出回答的理由过程。每完成一版，都使用三个子智能体审核，检查做出回答的理由和回答的正确性本身。每个子智能体有自己的偏好，但也不是自扫门前雪，都会基于全局给出。可能好的回答。审核-订正循环，直到全通过。
- 实验目的、假设、变量、对照组、样本范围和验收标准。
- 主项目仓库路径、branch、完整 commit hash 和工作区状态。
- 本实验日志仓库 commit hash。
- 原始数据集、绝对路径、版本、过滤规则和样本切分。
- 模型路径、dtype、依赖环境和关键参数。
- raw/preprocess/full KV 定义和 MODEL/DATASET cache 路径。
- 完整启动命令、环境变量、工作目录。
- 结果、日志、生成答案和测评元数据路径。
- 开始/结束时间、退出状态、有效/失败/跳过样本数和异常。

## 结果和复现

结果必须同时记录原始文件位置和汇总指标，包括 EM、F1、GLM judge accuracy，以及适用时的延迟、重算 token 数和计算量。结论要区分事实、推断和未验证假设。

每个完成实验必须提供最小复现步骤：checkout 主项目 branch/commit，初始化本 submodule 并 checkout 对应 commit，准备或复用原始数据/cache，运行单 GPU smoke test，再扩展到完整数据集。

切换主项目分支前，先提交并推送本仓库实验记录，再更新父仓库 submodule 指针。禁止用 checkout/reset 覆盖未提交记录。补跑、重判或 pipeline 修复必须新增日志条目并记录新的 branch/commit。
