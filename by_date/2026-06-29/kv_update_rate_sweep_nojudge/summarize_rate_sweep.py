from pathlib import Path
import csv

root = Path("MOTIVATION_EXPERIMENTS/kv_update_rate_sweep_nojudge")
prior = Path("MOTIVATION_EXPERIMENTS/reflect_pipeline_full_kv_writeback_ablation/strict_clean_kv_v_k_writeback_summary.csv")
paths = {}

with prior.open(newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        run = r["run"]
        if run == "strict_kv_rate0.15":
            paths[("kv", "0.15")] = Path(r["path"])
        elif run == "strict_v_only_clean_rate0.15":
            paths[("v_only", "0.15")] = Path(r["path"])
        elif run == "strict_k_only_clean_rate0.15":
            paths[("k_only", "0.15")] = Path(r["path"])
        elif run == "true_rate0_no_doc_recompute":
            for mode in ["kv", "v_only", "k_only"]:
                paths[(mode, "0.0")] = Path(r["path"])

for mode in ["kv", "v_only", "k_only"]:
    paths[(mode, "1.0")] = root / "kv_rate1p0/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_1.0_revert_rope.csv"

for rate in ["0.3", "0.5", "0.8"]:
    for mode in ["kv", "v_only", "k_only"]:
        paths[(mode, rate)] = root / f"{mode}_rate{rate.replace('.', 'p')}/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_{rate}_revert_rope.csv"


def metrics(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    f1 = [float(r["F1"]) for r in rows]
    em = [float(r["EM"]) for r in rows]
    return {"rows": len(rows), "avg_f1": sum(f1) / len(f1), "avg_em": sum(em) / len(em), "path": str(path)}


rates = ["0.0", "0.15", "0.3", "0.5", "0.8", "1.0"]
modes = ["kv", "v_only", "k_only"]
records = []
for rate in rates:
    for mode in modes:
        m = metrics(paths[(mode, rate)])
        records.append({"rate": rate, "mode": mode, **m})

with (root / "kv_update_rate_sweep_summary.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["rate", "mode", "rows", "avg_f1", "avg_em", "path"])
    w.writeheader()
    w.writerows(records)

by = {(r["rate"], r["mode"]): r for r in records}
with (root / "kv_update_rate_sweep_pivot_f1.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["rate"] + modes)
    for rate in rates:
        w.writerow([rate] + [f"{by[(rate, mode)]['avg_f1']:.6f}" for mode in modes])

with (root / "kv_update_rate_sweep_pivot_em.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["rate"] + modes)
    for rate in rates:
        w.writerow([rate] + [f"{by[(rate, mode)]['avg_em']:.6f}" for mode in modes])

md = []
md.append("# K/V 写回模式 Rate Sweep（无 LLM judge）\n\n")
md.append("## 实验设置\n\n")
md.append("- 数据：data/result_reflect.json 完整 200 main questions，实际输出 135 main / 250 sub questions。\n")
md.append("- 模型：Qwen2.5-7B-Instruct。\n")
md.append("- KV：preprocess=True，preprocess_scope=global，topk=10，BGE recall。\n")
md.append("- FusionRAG selection：原始 query-conditioned importance selection。\n")
md.append("- 写回语义：strict two-stage ablation，先重算 selected document tokens，再按模式保留 K/V，然后单独 forward query。\n")
md.append("- 模式：kv 同时更新 K/V；v_only 只保留 V 更新；k_only 只保留 K 更新。\n")
md.append("- rate：0.0, 0.15, 0.3, 0.5, 0.8, 1.0。rate=0 和 rate=1 与写回模式无关，因此三列共享同一结果。\n")
md.append("- 本轮设置 FUSIONRAG_SKIP_LLM_JUDGE=1，不看 LLM judge accuracy，只看本地 F1/EM。\n\n")
md.append("## Avg F1\n\n")
md.append("| rate | kv | v_only | k_only |\n|---:|---:|---:|---:|\n")
for rate in rates:
    md.append(f"| {rate} | {by[(rate, 'kv')]['avg_f1']:.4f} | {by[(rate, 'v_only')]['avg_f1']:.4f} | {by[(rate, 'k_only')]['avg_f1']:.4f} |\n")

md.append("\n## Avg EM\n\n")
md.append("| rate | kv | v_only | k_only |\n|---:|---:|---:|---:|\n")
for rate in rates:
    md.append(f"| {rate} | {by[(rate, 'kv')]['avg_em']:.4f} | {by[(rate, 'v_only')]['avg_em']:.4f} | {by[(rate, 'k_only')]['avg_em']:.4f} |\n")

md.append("\n## 初步观察\n\n")
md.append("- kv 随 rate 增加整体提升，并在 rate=1.0 达到最高 F1/EM，说明完整 K/V online recomputation 是有效的。\n")
md.append("- v_only 在 rate=0.15 到 0.3 下降，随后随 rate 增加恢复；到 rate=0.8 F1 接近 kv@0.8，但 EM 仍低。\n")
md.append("- k_only 在 rate=0.3/0.5 的 F1 接近或略高于 v_only，但到 rate=0.8 明显低于 v_only 和 kv，说明只更新 K 的收益不稳定。\n")
md.append("- 这组结果和之前 K/V delta 观察并不矛盾：Value 的数值变化更大，但只更新 Value 不能稳定替代完整 K/V；Key 的数值变化小，仍会影响 attention routing，因此 K/V 最好一起保留。\n\n")
md.append("## 输出文件\n\n")
md.append("- kv_update_rate_sweep_summary.csv：长表。\n")
md.append("- kv_update_rate_sweep_pivot_f1.csv：F1 透视表。\n")
md.append("- kv_update_rate_sweep_pivot_em.csv：EM 透视表。\n")
md.append("- 各 run 原始 CSV 位于对应 mode_rate 子目录。\n")
(root / "README.md").write_text("".join(md), encoding="utf-8")

print(root / "README.md")
print("F1")
for rate in rates:
    print(rate, [f"{by[(rate, mode)]['avg_f1']:.4f}" for mode in modes])
print("EM")
for rate in rates:
    print(rate, [f"{by[(rate, mode)]['avg_em']:.4f}" for mode in modes])
