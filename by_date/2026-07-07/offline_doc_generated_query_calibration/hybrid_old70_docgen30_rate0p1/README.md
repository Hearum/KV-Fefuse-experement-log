# Hybrid Old70 + Docgen30 Fixed Set

Method: `hybrid_old70_docgen30_frequency_global`

构造方式：以旧版 `draft_smart_frequency_global` offline10 为主，每个 chunk 保留旧 offline set 的前 70% token，然后用 docgen 的完整 rank 补足到原来的 10% offline 预算。

实现细节：按每个 example 的实际 chunk keys 动态生成，支持 `chunk00...chunkXX`。如果某个 chunk 在旧 offline set 中缺失，则退化为 docgen set。

输出目录：`chunk_fixed_sets_npz/`

文件数：135
chunk 行数：2098
旧 offline 缺失 chunk 数：0
