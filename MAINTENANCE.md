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
