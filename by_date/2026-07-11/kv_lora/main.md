# KV-LoRA 阶段研究目标

本阶段的核心目标是研究：在真实 RAG 推理过程中，FusionRAG 对已有 document KV cache 的重算，本质上产生了什么样的结构化变化，以及这种变化能否被一种类似 LoRA 的轻量更新机制近似，从而替代昂贵的完整重算。

具体而言，需要分别从两类缓存起点出发：

* Raw KV cache；
* Preprocess KV cache。

对于同一个真实 RAG example，在不同 query、不同重算比例和不同文档上下文下，观察 reprocess 前后 KV 的变化：

[
\Delta KV
=========

## KV_{\mathrm{after\ reprocess}}

KV_{\mathrm{before\ reprocess}}.
]

研究重点不是预设 (\Delta KV) 一定低秩，而是探索它是否具有可利用的稳定结构，例如：

* 低秩或谱集中；
* token、layer、head 或 chunk 局部性；
* raw 和 preprocess 起点下不同的更新模式；
* 不同 query 下共享的更新子空间；
* 可以由少量系数、basis 或稀疏更新表示的规律。

最终需要回答的核心问题是：

> FusionRAG 的重算是否可以被重新表述为对已有 KV cache 的结构化残差更新，而不必重新运行完整模型计算？

如果实验表明这些更新可以由少量共享 basis 和输入相关系数组成，则可以进一步设计类似 LoRA 的 KV 更新方法：

[
KV_{\mathrm{updated}}
=====================

KV_{\mathrm{cached}}
+
\Delta KV,
]

其中：

[
\Delta KV
\approx
A(x)B
]

或：

[
\Delta KV
\approx
\sum_{i=1}^{r}\alpha_i(x)B_i.
]

这里：

* (B) 或 (B_i) 是可复用的低维更新 basis；
* (A(x)) 或 (\alpha_i(x)) 由当前 query、document、prefix context、selector support 和 cache 类型决定；
* 在线阶段只生成少量更新参数，而不是重新计算完整 document KV。

因此，本阶段最终不是单纯证明“KV Delta 是低秩的”，而是判断：

1. 真实 reprocess Delta 是否具有可压缩结构；
2. Raw KV 和 Preprocess KV 是否需要不同的更新模型；
3. Query 主要改变更新位置、更新系数，还是整个更新子空间；
4. 是否能够用轻量、条件化的 KV Adapter 替代部分或全部 reprocess；
5. 这种替代是否能在保持 RAG 质量的同时显著降低重算开销。

如果低秩假设不成立，也应继续寻找更合适的结构，例如稀疏更新、分层更新、token-selective 更新或低秩与稀疏结合，而不是强行套用 LoRA。
