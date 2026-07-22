# Qwen3-235B-A22B FusionRAG Smoke Test

## 目的

验证当前 FusionRAG 实验 pipeline 能否把主模型从现有 Qwen3 dense/32B 配置切换到 `/home/hming/models/Qwen3-235B-A22B`，优先测试：

- `online_qk_rate015`: FusionRAG-QK selector, rate=0.15
- `online_draft_rate015`: DraftModel selector, rate=0.15

## 环境

- 机器：qjy001 / qjhs-sh-lab-02
- GPU：8 x NVIDIA H20, 每卡约 97GB，测试开始时空闲
- 代码路径：`/home/hming/FusionRAG-pca-analysis`
- 模型路径：`/home/hming/models/Qwen3-235B-A22B`
- 测试脚本：`MOTIVATION_EXPERIMENTS/qwen3_235b_smoke_run.sh`

## 已执行的 smoke

启动命令：

```bash
ssh qjy001 tmux new-session -d -s qwen3_235b_smoke /home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/qwen3_235b_smoke_run.sh
```

脚本配置：

- dataset: `data/result_reflect.json` / `musique`
- sample: `[0, 1)`，只跑 1 个样本
- topk: 10
- rate: 0.15
- preprocess: true
- preprocess_scope: global
- recall_method: bge
- use_multi_gpu: true

## 当前结论

当前 FusionRAG runner 暂不支持直接运行 `Qwen3-235B-A22B`。

原因是 `Qwen3-235B-A22B` 的配置为：

- `architectures = ["Qwen3MoeForCausalLM"]`
- `model_type = "qwen3_moe"`
- `num_hidden_layers = 94`
- `num_experts = 128`
- `num_experts_per_tok = 8`

但当前 `test_fusionrag_reflect_preprocess_exp.py` 对 `model_type=qwen3` 强制加载的是 `/mnt/qjhs-sh-lab-01/wjh/FusionRAG/ktransformers/models/modeling_qwen3.py` 中的 dense Qwen3 兼容实现：

```python
from models.modeling_qwen3 import Qwen2ForCausalLM as Qwen3ForCausalLM
```

该自定义实现包含 FusionRAG 所需的 `reprocess_method / passages_len / importance_cache` 逻辑，但它不是 `Qwen3MoeForCausalLM`，没有 MoE expert/gating 结构。HF transformers 4.53.3 虽然支持原生 `Qwen3MoeForCausalLM`，但原生类没有当前 FusionRAG 修改过的 attention/recompute/importance-cache 逻辑，不能直接替换。

因此：

- `online_qk_rate015`：需要主模型自定义 attention 写入 QK importance，当前 MoE 主模型不支持。
- `online_draft_rate015`：即使 token selection 来自 3B draft，主模型 online recompute/generation 仍依赖当前 FusionRAG 改过的 cache/forward 路径，当前 MoE 主模型不支持。
- offline fixed-set 方法：selection 可以复用已有离线 token set，但推理阶段仍需要 235B 主模型走 FusionRAG cache/recompute/generate 路径，所以同样被 MoE loader 阻塞。

## 后续若要支持 235B

需要给 `qwen3_moe` 单独适配 FusionRAG 模型类，而不是只改启动参数。最小改造范围包括：

1. 基于 transformers 的 `modeling_qwen3_moe.py` 或 ktransformers 版本新增 FusionRAG-aware `Qwen3MoeForCausalLM`。
2. 在 MoE attention forward 中补齐当前 dense qwen3 的 `reprocess_method/passages_len/importance_cache` 行为。
3. 确认 `StaticCache` 与 94 层、GQA heads、MoE layer 的 KV shape/device_map 兼容。
4. 再重新做 1-sample smoke：先 `online_qk_rate015`，再 `online_draft_rate015`，最后才跑 offline fixed-set。

## 2026-07-09 适配进展

已在修改前创建本地 commit：

```text
4728d07 checkpoint before qwen3 moe adapter
```

随后新增独立适配文件：

```text
ktransformers/models/modeling_qwen3_moe.py
```

该文件从 transformers 4.53.3 的 `transformers.models.qwen3_moe.modeling_qwen3_moe.py` 复制而来，没有覆盖现有 dense `modeling_qwen3.py`。当前改动原则：保留 HF MoE router/expert 主体逻辑，只补 FusionRAG 需要的接口。

当前已完成：

- 将新文件的相对导入改为 `transformers.*` 绝对导入，便于作为仓库内独立模型文件加载。
- 在 `Qwen3MoeAttention.forward` 中补充 `FusionRAG` / `Cache-Craft` / `speculative_prefill` 的 `importance_cache` 写入逻辑。
- 将 `Qwen3MoeForCausalLM.forward` 默认 `logits_to_keep` 改为 1，避免长 prefill 时计算全序列 logits。
- 在 `test_fusionrag_reflect_preprocess_exp.py` 增加独立 `model_type in ('qwen3_moe', 'qwen3moe')` 分支。
- 将 smoke 脚本的 `--model_type` 改为 `qwen3_moe`。

已做轻量验证：

```text
python -m py_compile ktransformers/models/modeling_qwen3_moe.py test_fusionrag_reflect_preprocess_exp.py
```

通过。

使用 tiny Qwen3-MoE config + 仓库 `StaticCache` 做 CPU forward 验证：

```text
normal logits: (1, 1, 128)
fusion logits: (1, 1, 128)
importance_cache shape: (4, 8)
importance_sum: 15.9990
nonzero entries: 8
```

这说明新增 MoE 文件至少能接受 FusionRAG 的额外 forward 参数，并能在构造样例中写入 `importance_cache`。下一步才是重新启动 235B 真实权重 smoke。
