from pathlib import Path
import csv

root = Path("MOTIVATION_EXPERIMENTS/oracle_delta_selection_pipeline")
base_csv = Path("MOTIVATION_EXPERIMENTS/reflect_pipeline_strict_kv/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv")
rate0_csv = Path("MOTIVATION_EXPERIMENTS/reflect_pipeline_full_rate0/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.0_revert_rope.csv")
runs = {
    "importance_strict_kv_rate0.15": base_csv,
    "oracle_value_delta_rate0.15": root / "oracle_value_rate015/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv",
    "oracle_key_delta_rate0.15": root / "oracle_key_rate015/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv",
    "oracle_kv_delta_rate0.15": root / "oracle_kv_rate015/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.csv",
    "true_rate0_no_doc_recompute": rate0_csv,
}


def load(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


all_rows = {k: load(p) for k, p in runs.items()}
base = all_rows["importance_strict_kv_rate0.15"]
base_f1 = sum(float(r["F1"]) for r in base) / len(base)
base_em = sum(float(r["EM"]) for r in base) / len(base)

summary = []
for name, rows in all_rows.items():
    f1 = [float(r["F1"]) for r in rows]
    em = [float(r["EM"]) for r in rows]
    summary.append(
        {
            "run": name,
            "rows": len(rows),
            "avg_f1": sum(f1) / len(f1),
            "avg_em": sum(em) / len(em),
            "delta_f1_vs_importance": (sum(f1) / len(f1)) - base_f1,
            "delta_em_vs_importance": (sum(em) / len(em)) - base_em,
        }
    )

comparisons = []
for name in [
    "oracle_value_delta_rate0.15",
    "oracle_key_delta_rate0.15",
    "oracle_kv_delta_rate0.15",
    "true_rate0_no_doc_recompute",
]:
    rows = all_rows[name]
    better = worse = same = pred_diff = 0
    deltas = []
    for b, r in zip(base, rows):
        bf = float(b["F1"])
        rf = float(r["F1"])
        deltas.append(rf - bf)
        if rf > bf + 1e-12:
            better += 1
        elif rf < bf - 1e-12:
            worse += 1
        else:
            same += 1
        if b["Predicted"] != r["Predicted"]:
            pred_diff += 1
    comparisons.append(
        {
            "run": name,
            "pred_diff_count": pred_diff,
            "f1_better_rows": better,
            "f1_worse_rows": worse,
            "f1_same_rows": same,
            "mean_f1_delta": sum(deltas) / len(deltas),
        }
    )

with (root / "oracle_delta_summary.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

with (root / "oracle_delta_row_comparison_vs_importance.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(comparisons[0].keys()))
    w.writeheader()
    w.writerows(comparisons)

md = []
md.append("# Oracle Delta Selection 实验记录\n\n")
md.append("## 实验问题\n\n")
md.append("原始 FusionRAG pipeline 是根据 query-to-document importance score 排序，然后按 rate 选 token 做 online recomputation。本实验验证一个理想条件：如果提前知道每个文档 token 在线重算前后的 K/V 变化大小，直接优先更新变化最大的 token，是否能提升回答质量。\n\n")
md.append("## 实验设置\n\n")
md.append("- 数据：`data/result_reflect.json`，完整 200 main questions；实际评测 135 main / 250 sub questions。\n")
md.append("- 模型：`Qwen2.5-7B-Instruct`。\n")
md.append("- KV：`preprocess=True`，`preprocess_scope=global`，`topk=10`，BGE recall。\n")
md.append("- rate：`0.15`，与之前 strict FusionRAG 主实验对齐。\n")
md.append("- 写回语义：开启 strict two-stage KV writeback，`FUSIONRAG_STRICT_REPROCESS_ABLATION=1`，`FUSIONRAG_CLEAN_STRICT_ABLATION=1`，`FUSIONRAG_REPROCESS_UPDATE_MODE=kv`。\n")
md.append("- Oracle 打分方式：载入 preprocess KV 后，先对所有 document token 做一次全量 online recomputation，比较重算前后的 K/V delta；随后恢复 cache，再按 delta top-rate 选择 token，走正常 selected-token recomputation + query generation。\n")
md.append("- 三种排序：`value_delta`、`key_delta`、`kv_delta`。其中 `kv_delta` 使用 K/V 合并相对 L2 变化，`value_delta` 和 `key_delta` 是消融。\n")
md.append("- 评测限制：当前 DeepSeek/DashScope judge key 不可用或余额不足，因此本轮不使用 LLM judge 的 Main/Sub accuracy；只比较脚本本地计算的 F1/EM。\n\n")
md.append("## 汇总结果\n\n")
md.append("| run | rows | Avg F1 | Avg EM | dF1 vs importance | dEM vs importance |\n")
md.append("|---|---:|---:|---:|---:|---:|\n")
for s in summary:
    md.append(f"| `{s['run']}` | {s['rows']} | {s['avg_f1']:.4f} | {s['avg_em']:.4f} | {s['delta_f1_vs_importance']:+.4f} | {s['delta_em_vs_importance']:+.4f} |\n")
md.append("\n## 行级对比\n\n")
md.append("| run | pred diff vs importance | F1 better | F1 worse | F1 same | mean dF1 |\n")
md.append("|---|---:|---:|---:|---:|---:|\n")
for c in comparisons:
    md.append(f"| `{c['run']}` | {c['pred_diff_count']} | {c['f1_better_rows']} | {c['f1_worse_rows']} | {c['f1_same_rows']} | {c['mean_f1_delta']:+.4f} |\n")
md.append("\n## 结论\n\n")
md.append("1. 直接用 K/V 变化大小做 token selection 没有提升质量，反而低于原始 query-conditioned importance selection。\n")
md.append("2. 三个 oracle 中 `value_delta` 最好，但 Avg F1 仍只有 0.4581，低于 importance baseline 的 0.4960；`key_delta` 和 `kv_delta` 更接近 rate=0。\n")
md.append("3. 这说明“变化大”不等价于“对当前 query 有用”。K/V delta 更像 context correction magnitude，而 FusionRAG 的 importance score 捕获的是 query relevance。\n")
md.append("4. 因此，如果要优化 selection，不应该只预测 K/V delta；更合理的是预测 query-conditioned benefit，例如 `importance × expected_delta`、或者训练一个直接预测 answer-quality gain / recompute benefit 的 scorer。\n\n")
md.append("## 输出文件\n\n")
md.append("- `oracle_delta_summary.csv`：汇总 F1/EM。\n")
md.append("- `oracle_delta_row_comparison_vs_importance.csv`：逐 sub-question 与 importance baseline 的文本/F1 差异。\n")
md.append("- `oracle_value_rate015/`、`oracle_key_rate015/`、`oracle_kv_rate015/`：三组原始输出。\n")
(root / "README.md").write_text("".join(md), encoding="utf-8")

print(root / "README.md")
print("SUMMARY")
for s in summary:
    print(s)
print("COMPARISONS")
for c in comparisons:
    print(c)
