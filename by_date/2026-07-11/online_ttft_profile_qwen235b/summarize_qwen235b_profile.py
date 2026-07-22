#!/usr/bin/env python3
import csv
import json
from pathlib import Path


EXP = Path(__file__).resolve().parent


def load_jsons(root: Path):
    rows = []
    for path in sorted(root.glob("*_summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"file": str(path), "error": repr(exc)})
            continue
        cfg = data.get("config", {})
        row = {
            "rate": cfg.get("rate"),
            "num_measured": data.get("num_measured"),
            "end_to_end_ttft_s_mean": data.get("end_to_end_ttft_s_mean"),
            "end_to_end_ttft_s_p50": data.get("end_to_end_ttft_s_p50"),
            "end_to_end_ttft_s_p95": data.get("end_to_end_ttft_s_p95"),
            "kv_load_copy_s_mean": data.get("kv_load_copy_s_mean"),
            "online_update_query_prefill_s_mean": data.get("online_update_query_prefill_s_mean"),
            "extra_impl_overhead_s_mean": data.get("extra_impl_overhead_s_mean"),
            "selected_doc_tokens_mean": data.get("fusion_selected_doc_tokens_mean", data.get("selected_doc_tokens_mean")),
            "reprocess_prefill_tokens_mean": data.get("fusion_reprocess_prefill_tokens_mean", data.get("reprocess_prefill_tokens_mean")),
            "full_tokens_mean": data.get("full_tokens_mean"),
            "summary_json": str(path),
        }
        if "fusion_selection_time_s_mean" in data:
            row["selector"] = "qk"
            row["selection_s_mean"] = data.get("fusion_selection_time_s_mean")
            row["score_forward_s_mean"] = data.get("fusion_selection_score_forward_time_s_mean")
        else:
            row["selector"] = "draft"
            row["selection_s_mean"] = data.get("draft_selection_s_mean")
            row["score_forward_s_mean"] = data.get("draft_score_forward_s_mean")
        rows.append(row)
    rows.sort(key=lambda r: (r.get("selector", ""), float(r.get("rate") or 0.0)))
    return rows


def write_csv(rows, path):
    fields = [
        "selector",
        "rate",
        "num_measured",
        "end_to_end_ttft_s_mean",
        "end_to_end_ttft_s_p50",
        "end_to_end_ttft_s_p95",
        "kv_load_copy_s_mean",
        "selection_s_mean",
        "score_forward_s_mean",
        "online_update_query_prefill_s_mean",
        "extra_impl_overhead_s_mean",
        "selected_doc_tokens_mean",
        "reprocess_prefill_tokens_mean",
        "full_tokens_mean",
        "summary_json",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def fmt(x):
    if x is None or x == "":
        return ""
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def main():
    rows = []
    rows += load_jsons(EXP / "qk_rate_sweep")
    rows += load_jsons(EXP / "draft_rate_sweep")
    out_csv = EXP / "qwen235b_profile_summary.csv"
    write_csv(rows, out_csv)
    md = EXP / "README.md"
    lines = [
        "# Qwen3-235B Online TTFT Profile",
        "",
        "本实验把主模型替换为 `Qwen3-235B-A22B`，在 MuSiQue reflect 数据上复跑 online TTFT profile。离线 raw/preprocess KV 不计入 TTFT。",
        "",
        "- QK cache root: `/home/hming/fusionrag-qwen3-235b-three-groups-unified-prompt-cache/online_qk_rate015`",
        "- Draft cache root: `/home/hming/fusionrag-qwen3-235b-three-groups-unified-prompt-cache/online_draft_rate015`",
        "- 主模型加载：`model_type=qwen3_moe`, `use_multi_gpu=true`, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`",
        "- Draft selector：`Qwen2.5-3B-Instruct`, RRF-18 + smart selection，和原 profile 语义对齐。",
        "",
        "## Summary",
        "",
        "| selector | rate | measured | e2e mean | kv load | selection | score forward | update+query | overhead | selected | full tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("error"):
            continue
        lines.append(
            "| {selector} | {rate} | {num_measured} | {e2e} | {kv} | {sel} | {score} | {upd} | {oh} | {selected} | {full} |".format(
                selector=row.get("selector", ""),
                rate=row.get("rate", ""),
                num_measured=row.get("num_measured", ""),
                e2e=fmt(row.get("end_to_end_ttft_s_mean")),
                kv=fmt(row.get("kv_load_copy_s_mean")),
                sel=fmt(row.get("selection_s_mean")),
                score=fmt(row.get("score_forward_s_mean")),
                upd=fmt(row.get("online_update_query_prefill_s_mean")),
                oh=fmt(row.get("extra_impl_overhead_s_mean")),
                selected=fmt(row.get("selected_doc_tokens_mean")),
                full=fmt(row.get("full_tokens_mean")),
            )
        )
    lines += [
        "",
        "## Files",
        "",
        f"- aggregate CSV: `{out_csv}`",
        "- detail/summary JSON: `qk_rate_sweep/`, `draft_rate_sweep/`",
        "- logs: `logs/`",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
