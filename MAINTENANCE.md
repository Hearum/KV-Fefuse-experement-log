# KV-Fefuse Experiment Log

本仓库保存 FusionRAG 的机制分析、消融实验、脚本、图表和可复现记录。模型、KV cache 和大型中间张量不上传。

## 维护规则

- 每个实验目录先写 plan.md，再启动实验。
- 每次运行追加 EXPERIMENT_LOG.md，记录日期、commit、启动命令、模型、数据集、cache 路径、GPU、结果和异常。
- 实验结果与 cache 分离；cache 只按 MODEL/DATASET 复用。
- 主代码大改或消融实验先创建 commit/分支。
- 不上传模型、KV cache、*.pt、*.safetensors、*.npz 等大文件。
- Markdown 使用中文，命令和路径保持原样。
- 切换主代码分支不得覆盖实验记录；实验日志仓库独立维护。

## 与主项目分支切换的同步规则

本仓库只保存过滤后的文档、实验脚本、图表和小型结果文件。模型、KV cache、大型 tensor、完整生成中间文件和其他大文件不进入本仓库。

每次切换 FusionRAG 主项目分支前，必须先完成当前实验记录：

1. 将本次实验的 plan、启动命令、原始数据路径、模型/cache 路径、结果路径、分支名和完整 commit 写入 EXPERIMENT_LOG.md。
2. 提交实验日志仓库的修改并推送远端。
3. 在主项目提交对应的 submodule commit 指针。
4. 再切换主项目分支，并执行 git submodule update --init --recursive。

禁止通过 checkout、reset 或分支切换覆盖未提交的实验记录。
