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

## 实验启动记录

每次实验必须记录以下内容：

- 实验目的、假设、变量、对照组、样本范围和验收标准。
- 主项目仓库路径、branch、完整 commit hash 和工作区状态。
- 本实验日志仓库 commit hash。
- 原始数据集、绝对路径、版本、过滤规则和样本切分。
- 模型路径、dtype、依赖环境和关键参数。
- raw/preprocess/full KV 定义和 MODEL/DATASET cache 路径。
- 完整启动命令、环境变量、工作目录、主机、GPU、PID 和 worker 分片。
- 结果、日志、生成答案和测评元数据路径。
- 开始/结束时间、退出状态、有效/失败/跳过样本数和异常。

## 结果和复现

结果必须同时记录原始文件位置和汇总指标，包括 EM、F1、GLM judge accuracy，以及适用时的延迟、重算 token 数和计算量。结论要区分事实、推断和未验证假设。

每个完成实验必须提供最小复现步骤：checkout 主项目 branch/commit，初始化本 submodule 并 checkout 对应 commit，准备或复用原始数据/cache，运行单 GPU smoke test，再扩展到完整数据集。

切换主项目分支前，先提交并推送本仓库实验记录，再更新父仓库 submodule 指针。禁止用 checkout/reset 覆盖未提交记录。补跑、重判或 pipeline 修复必须新增日志条目并记录新的 branch/commit。
