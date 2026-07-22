#!/usr/bin/env python3
"""
Measure how often offline preprocess top-k document recall overlaps with the
documents used by online reflect/RAG sub-questions.

Default mode is read-only over existing run.log files. Use --compute-bge only
when logs do not contain complete "Retrieved similar docs" lines; this computes
document embeddings with BGE/FAISS but still does not load the LLM or mutate KV
caches.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


DocId = Tuple[int, int]  # (example_id 0-based, chunk_id 1-based)


DATASET_DEFAULTS = {
    "musique": {
        "data": "data/result_reflect.json",
        "run_log": "MOTIVATION_EXPERIMENTS/qwen3_235b_param_scaling_cross_dataset/results/musique/offline32b_top2_rate015/full_0_200/run.log",
    },
    "2wikimqa": {
        "data": "MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/2wikimqa_reflect.json",
        "run_log": "MOTIVATION_EXPERIMENTS/qwen3_235b_param_scaling_cross_dataset/results/2wikimqa/offline32b_top2_rate015/full_0_200/run.log",
    },
    "hotpotqa": {
        "data": "MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/hotpotqa_reflect.json",
        "run_log": "MOTIVATION_EXPERIMENTS/qwen3_235b_param_scaling_cross_dataset/results/hotpotqa/offline32b_top2_rate015/full_0_260/run.log",
    },
    "triviaqa": {
        "data": "MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization/data/triviaqa_reflect.json",
        "run_log": None,
    },
}


def load_reflect_dataset(path: Path, limit: Optional[int] = None) -> Tuple[List[dict], List[int]]:
    with path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    if limit is not None:
        dataset = dataset[:limit]

    questions = []
    corpus_lens = []
    for example_id, item in enumerate(dataset):
        question_docs: List[str] = []
        doc_to_idx: Dict[str, int] = {}
        sub_questions = []
        should_test = True

        for sub_q in item.get("intermediate_context", []):
            doc_chunk_ids = []
            for doc in sub_q.get("retrieve docs", []):
                if doc not in doc_to_idx:
                    question_docs.append(doc)
                    doc_to_idx[doc] = len(question_docs)
                doc_chunk_ids.append(doc_to_idx[doc])

            query = str(sub_q.get("query", ""))
            if query.startswith("Intermediate query") and ":" in query:
                query = query.split(":", 1)[1].strip()

            answer = str(sub_q.get("answer", ""))
            if answer.startswith("Intermediate answer") and ":" in answer:
                answer = answer.split(":", 1)[1].strip()
            if "No relevant information found" in answer or "没有相关信息" in answer:
                should_test = False

            sub_questions.append(
                {
                    "query": query,
                    "answer": answer,
                    "chunk_ids": doc_chunk_ids,
                }
            )

        questions.append(
            {
                "example_id": example_id,
                "main_question": item.get("question", ""),
                "docs": question_docs,
                "sub_questions": sub_questions,
                "should_test": should_test,
            }
        )
        corpus_lens.append(len(question_docs))
    return questions, corpus_lens


def parse_run_log(path: Path, topk: int) -> Dict[DocId, List[DocId]]:
    offline: Dict[DocId, List[DocId]] = {}
    if not path or not path.exists():
        return offline

    main_re = re.compile(r"Main Question\s+(\d+)/")
    prep_re = re.compile(r"Preprocessing document\s+(\d+)/")
    qdoc_re = re.compile(r"Q(\d+)-Doc(\d+)")

    current_example: Optional[int] = None
    current_chunk: Optional[int] = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = main_re.search(line)
            if m:
                current_example = int(m.group(1)) - 1
                current_chunk = None
                continue
            m = prep_re.search(line)
            if m:
                current_chunk = int(m.group(1))
                continue
            if "Retrieved similar docs:" not in line:
                continue
            if current_example is None or current_chunk is None:
                continue
            docs = []
            for qm in qdoc_re.finditer(line):
                docs.append((int(qm.group(1)) - 1, int(qm.group(2))))
            offline[(current_example, current_chunk)] = docs[:topk]
    return offline


def find_group_and_index(corpus_lens: Sequence[int], global_idx: int) -> DocId:
    acc = 0
    for example_id, n_docs in enumerate(corpus_lens):
        if global_idx < acc + n_docs:
            return example_id, global_idx - acc + 1
        acc += n_docs
    raise IndexError(global_idx)


def compute_bge_topk(
    questions: Sequence[dict],
    corpus_lens: Sequence[int],
    bge_model_path: str,
    topk: int,
    scope: str,
    batch_size: int,
) -> Dict[DocId, List[DocId]]:
    import numpy as np
    import faiss
    from FlagEmbedding import FlagModel

    corpus = [doc for q in questions for doc in q["docs"]]
    model = FlagModel(bge_model_path, use_fp16=False)
    offline: Dict[DocId, List[DocId]] = {}

    if scope == "per_example":
        offset = 0
        for q in questions:
            docs = q["docs"]
            if not docs:
                continue
            emb = model.encode(docs, batch_size=batch_size).astype(np.float32)
            query_emb = model.encode_queries(docs, batch_size=batch_size).astype(np.float32)
            index = faiss.index_factory(emb.shape[-1], "Flat", faiss.METRIC_INNER_PRODUCT)
            index.train(emb)
            index.add(emb)
            _, idx = index.search(query_emb, k=min(topk, len(docs)))
            for local_i, row in enumerate(idx):
                src = (q["example_id"], local_i + 1)
                offline[src] = [find_group_and_index(corpus_lens, offset + int(j)) for j in row[:topk]]
            offset += len(docs)
        return offline

    emb = model.encode(corpus, batch_size=batch_size).astype(np.float32)
    query_emb = model.encode_queries(corpus, batch_size=batch_size).astype(np.float32)
    index = faiss.index_factory(emb.shape[-1], "Flat", faiss.METRIC_INNER_PRODUCT)
    index.train(emb)
    index.add(emb)
    _, idx = index.search(query_emb, k=topk)
    for global_i, row in enumerate(idx):
        src = find_group_and_index(corpus_lens, global_i)
        offline[src] = [find_group_and_index(corpus_lens, int(j)) for j in row[:topk]]
    return offline


def uniq(seq: Iterable[int]) -> List[int]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def ratio(num: int, den: int) -> float:
    return float(num) / den if den else 0.0


def summarize(values: Sequence[float]) -> Dict[str, float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {k: math.nan for k in ["mean", "p25", "p50", "p75", "p90", "p95"]}
    def pct(p: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * p
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)
    return {
        "mean": statistics.fmean(vals),
        "p25": pct(0.25),
        "p50": pct(0.50),
        "p75": pct(0.75),
        "p90": pct(0.90),
        "p95": pct(0.95),
    }


def metric_row(offline_set: Set[DocId], online_set: Set[DocId]) -> Dict[str, float]:
    inter = offline_set & online_set
    union = offline_set | online_set
    return {
        "offline_n": len(offline_set),
        "online_n": len(online_set),
        "hit_n": len(inter),
        "jaccard": ratio(len(inter), len(union)),
        "offline_covered_by_online": ratio(len(inter), len(offline_set)),
        "online_covered_by_offline": ratio(len(inter), len(online_set)),
    }


def build_rows(
    dataset_name: str,
    questions: Sequence[dict],
    offline: Dict[DocId, List[DocId]],
    topks: Sequence[int],
    include_skipped: bool,
    effective_exclude_self: bool,
) -> Tuple[List[dict], List[dict]]:
    subq_rows = []
    chunk_rows = []
    for q in questions:
        example_id = int(q["example_id"])
        if (not include_skipped) and (not q["should_test"]):
            continue
        for sub_idx, subq in enumerate(q["sub_questions"]):
            online_doc_ids = {(example_id, cid) for cid in uniq(subq["chunk_ids"])}
            if not online_doc_ids:
                continue
            for k in topks:
                union_offline: Set[DocId] = set()
                available_chunks = 0
                for src in online_doc_ids:
                    if src not in offline:
                        continue
                    available_chunks += 1
                    cur = list(offline[src])[:k]
                    if effective_exclude_self:
                        cur = [d for d in cur if d != src]
                    union_offline.update(cur)

                    crow = metric_row(set(cur), online_doc_ids)
                    crow.update(
                        {
                            "dataset": dataset_name,
                            "level": "chunk",
                            "topk": k,
                            "example_id": example_id,
                            "sub_q_idx": sub_idx,
                            "source_chunk_id": src[1],
                            "online_chunk_ids": " ".join(str(x[1]) for x in sorted(online_doc_ids)),
                            "offline_doc_ids": " ".join(f"Q{x[0]+1}-Doc{x[1]}" for x in cur),
                            "offline_available": 1,
                        }
                    )
                    chunk_rows.append(crow)

                srow = metric_row(union_offline, online_doc_ids)
                srow.update(
                    {
                        "dataset": dataset_name,
                        "level": "sub_question",
                        "topk": k,
                        "example_id": example_id,
                        "sub_q_idx": sub_idx,
                        "online_chunk_ids": " ".join(str(x[1]) for x in sorted(online_doc_ids)),
                        "available_online_chunks": available_chunks,
                        "total_online_chunks": len(online_doc_ids),
                        "offline_available_ratio": ratio(available_chunks, len(online_doc_ids)),
                    }
                )
                subq_rows.append(srow)
    return subq_rows, chunk_rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: Sequence[dict], metrics: Sequence[str]) -> List[dict]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["level"], r["topk"])].append(r)
    out = []
    for (dataset, level, topk), group in sorted(grouped.items()):
        base = {"dataset": dataset, "level": level, "topk": topk, "n": len(group)}
        for metric in metrics:
            stat = summarize([float(r[metric]) for r in group])
            for name, val in stat.items():
                base[f"{metric}_{name}"] = val
        out.append(base)
    return out


def format_float(x: float) -> str:
    return "nan" if x != x else f"{x:.4f}"


def write_markdown(
    path: Path,
    repo_root: Path,
    dataset_summaries: Sequence[dict],
    aggregate_rows: Sequence[dict],
    settings: dict,
) -> None:
    lines = []
    lines.append("# Offline preprocess top-k 与 online RAG 文档命中率")
    lines.append("")
    lines.append("## 实验设置")
    lines.append("")
    lines.append(f"- 仓库路径：`{repo_root}`；qjy001 常用路径是 `/home/hming/FusionRAG-pca-analysis`，qjy000 也可能有 `/raid/home/hming/FusionRAG-pca-analysis`。")
    lines.append(f"- offline top-k 来源：`{settings['offline_source']}`。")
    lines.append(f"- top-k 统计点：`{', '.join(map(str, settings['topks']))}`。")
    lines.append(f"- self-recall 口径：`{'exclude' if settings['effective_exclude_self'] else 'include'}`；主 runner 在非 `REPEAT_SELF` preprocess 中会跳过 self-recall，因此默认报告使用 exclude。")
    lines.append(f"- skipped example：`{'included' if settings['include_skipped'] else 'excluded'}`。")
    lines.append("")
    lines.append("启动命令：")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /home/hming/FusionRAG-pca-analysis  # 或 /raid/home/hming/FusionRAG-pca-analysis")
    lines.append("python3 MOTIVATION_EXPERIMENTS/offline_preprocess_online_hit_rate/analyze_offline_preprocess_online_hit_rate.py \\")
    lines.append("  --datasets musique 2wikimqa hotpotqa triviaqa")
    lines.append("```")
    lines.append("")
    lines.append("如现有 `run.log` 不完整，可显式追加 `--compute-bge --bge-model-path /mnt/qjhs-sh-lab-01/models/bge-m3`；这只跑 BGE/FAISS embedding 检索，不启动 235B。")
    lines.append("")
    lines.append("## 代码定位")
    lines.append("")
    lines.append("- `test_fusionrag_reflect_preprocess_exp.py::prepare_reflect_data`：读取 `intermediate_context[*]['retrieve docs']`，在每个 example 内按文档文本去重，映射为 1-indexed `chunk_id`；同时构建全局 `global_corpus` 和 `corpus_lens`。")
    lines.append("- 同函数中 BGE/FAISS 分支生成 `context_rank`，形状为 `[total_docs, topk]`，元素是全局 doc index；`find_group_and_index(corpus_lens, idx)` 可还原为 `(example_id, local_doc_idx)`，在线打印为 `Q{example+1}-Doc{chunk_id}`。")
    lines.append("- preprocess 生成 KV 时，对当前 doc 的 `global_doc_idx = sum(corpus_lens[:example_id]) + doc_idx` 读取 `context_rank[global_doc_idx][:topk]`，加载这些相似文档 KV 后再把当前 doc 追加进去，保存到 `preprocess_save_path/{example_id}_{chunk_id}_key/value.pt`。")
    lines.append("- online RAG 生成时，每个 sub-question 使用 `doc_chunk_ids = sub_q_info['chunk_ids']`，`kv_chunk_ids = [0] + doc_chunk_ids`，随后 `load_kv_and_generate(..., chunk_ids=kv_chunk_ids)` 载入 system chunk 0 和这些实际检索/使用的文档 chunk。")
    lines.append("")
    lines.append("## 统计口径")
    lines.append("")
    lines.append("- `Jaccard = |offline_topk ∩ online_docs| / |offline_topk ∪ online_docs|`。")
    lines.append("- `offline_covered_by_online = |intersection| / |offline_topk|`：offline 预取的 top-k 中有多少被 online 实际用到。")
    lines.append("- `online_covered_by_offline = |intersection| / |online_docs|`：online 实际文档中有多少落在 offline top-k 内。")
    lines.append("- `chunk` 层：以 online sub-question 中每个 source chunk 为单位，比较该 chunk 的 offline top-k 和同一 sub-question 的 online docs。")
    lines.append("- `sub_question` 层：把该 sub-question 所有 online chunks 的 offline top-k 取并集，再与 online docs 比较。")
    lines.append("")
    lines.append("## 数据覆盖")
    lines.append("")
    lines.append("| dataset | examples | docs | sub_questions_used | offline_rows | offline_coverage | source |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for s in dataset_summaries:
        lines.append(
            f"| {s['dataset']} | {s['examples']} | {s['docs']} | {s['sub_questions_used']} | "
            f"{s['offline_rows']} | {format_float(s['offline_coverage'])} | {s['offline_source']} |"
        )
    lines.append("")
    lines.append("## 主要结果")
    lines.append("")
    lines.append("| dataset | level | topk | n | Jaccard mean/p50/p90 | offline覆盖 mean/p50/p90 | online覆盖 mean/p50/p90 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in aggregate_rows:
        if int(r["topk"]) not in (1, 2, 5, 10):
            continue
        lines.append(
            f"| {r['dataset']} | {r['level']} | {r['topk']} | {r['n']} | "
            f"{format_float(r['jaccard_mean'])}/{format_float(r['jaccard_p50'])}/{format_float(r['jaccard_p90'])} | "
            f"{format_float(r['offline_covered_by_online_mean'])}/{format_float(r['offline_covered_by_online_p50'])}/{format_float(r['offline_covered_by_online_p90'])} | "
            f"{format_float(r['online_covered_by_offline_mean'])}/{format_float(r['online_covered_by_offline_p50'])}/{format_float(r['online_covered_by_offline_p90'])} |"
        )
    lines.append("")
    lines.append("## 初步结论")
    lines.append("")
    lines.append("- 以当前口径看，offline preprocess 的 BGE top-k 相似文档与 online query 实际使用文档的重合通常偏低；`offline_covered_by_online` 直接表示预取 KV 中真正被在线 RAG 用到的比例。")
    lines.append("- `sub_question` 层的 `online_covered_by_offline` 高于 `chunk` 层时，说明一个 query 的多个 source chunks 的 offline top-k 并集能覆盖更多在线 docs，但单个 chunk 的预取仍较分散。")
    lines.append("- 如果某个 dataset 的 `offline_coverage` 明显小于 1，说明只能基于现有日志中打印过的 preprocess rows 统计；需要完整结论时请用 `--compute-bge` 补齐。")
    lines.append("")
    lines.append("详细明细见：`sub_question_detail.csv`、`chunk_detail.csv`、`aggregate_summary.csv`、`dataset_summary.json`。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", help="FusionRAG-pca-analysis root")
    parser.add_argument("--out-dir", default="MOTIVATION_EXPERIMENTS/offline_preprocess_online_hit_rate")
    parser.add_argument("--datasets", nargs="+", default=["musique", "2wikimqa", "hotpotqa", "triviaqa"])
    parser.add_argument("--topks", nargs="+", type=int, default=[1, 2, 3, 5, 10])
    parser.add_argument("--include-skipped", action="store_true")
    parser.add_argument("--include-self", action="store_true", help="Include self-recall in offline top-k metrics")
    parser.add_argument("--compute-bge", action="store_true")
    parser.add_argument("--bge-model-path", default="/mnt/qjhs-sh-lab-01/models/bge-m3")
    parser.add_argument("--preprocess-scope", choices=["global", "per_example"], default="global")
    parser.add_argument("--bge-batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_subq_rows: List[dict] = []
    all_chunk_rows: List[dict] = []
    dataset_summaries = []

    for dataset in args.datasets:
        if dataset not in DATASET_DEFAULTS:
            raise ValueError(f"Unknown dataset {dataset}. Known: {sorted(DATASET_DEFAULTS)}")
        defaults = DATASET_DEFAULTS[dataset]
        data_path = repo_root / defaults["data"]
        if not data_path.exists():
            dataset_summaries.append(
                {
                    "dataset": dataset,
                    "examples": 0,
                    "docs": 0,
                    "sub_questions_used": 0,
                    "offline_rows": 0,
                    "offline_coverage": 0.0,
                    "offline_source": "missing_data",
                }
            )
            continue

        questions, corpus_lens = load_reflect_dataset(data_path, limit=args.limit)
        total_docs = sum(corpus_lens)
        log_path = repo_root / defaults["run_log"] if defaults["run_log"] else None
        offline = parse_run_log(log_path, max(args.topks)) if log_path else {}
        offline_source = f"log:{log_path.relative_to(repo_root)}" if offline else "none"

        if args.compute_bge and len(offline) < total_docs:
            offline = compute_bge_topk(
                questions,
                corpus_lens,
                args.bge_model_path,
                max(args.topks),
                args.preprocess_scope,
                args.bge_batch_size,
            )
            offline_source = f"computed_bge:{args.bge_model_path}:{args.preprocess_scope}"

        subq_rows, chunk_rows = build_rows(
            dataset,
            questions,
            offline,
            args.topks,
            args.include_skipped,
            effective_exclude_self=not args.include_self,
        )
        all_subq_rows.extend(subq_rows)
        all_chunk_rows.extend(chunk_rows)

        used_subqs = sum(
            1
            for q in questions
            if args.include_skipped or q["should_test"]
            for sq in q["sub_questions"]
            if sq["chunk_ids"]
        )
        dataset_summaries.append(
            {
                "dataset": dataset,
                "examples": len(questions),
                "docs": total_docs,
                "sub_questions_used": used_subqs,
                "offline_rows": len(offline),
                "offline_coverage": ratio(len(offline), total_docs),
                "offline_source": offline_source,
            }
        )

    metrics = ["jaccard", "offline_covered_by_online", "online_covered_by_offline"]
    aggregate_rows = aggregate(all_subq_rows + all_chunk_rows, metrics)

    write_csv(out_dir / "sub_question_detail.csv", all_subq_rows)
    write_csv(out_dir / "chunk_detail.csv", all_chunk_rows)
    write_csv(out_dir / "aggregate_summary.csv", aggregate_rows)
    (out_dir / "dataset_summary.json").write_text(json.dumps(dataset_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    write_markdown(
        out_dir / "README.md",
        repo_root,
        dataset_summaries,
        aggregate_rows,
        {
            "offline_source": "existing run.log first; optional BGE/FAISS recompute with --compute-bge",
            "topks": args.topks,
            "effective_exclude_self": not args.include_self,
            "include_skipped": args.include_skipped,
        },
    )

    print(f"Wrote {out_dir}")
    for s in dataset_summaries:
        print(
            f"{s['dataset']}: examples={s['examples']} docs={s['docs']} "
            f"offline_rows={s['offline_rows']} coverage={s['offline_coverage']:.3f} source={s['offline_source']}"
        )


if __name__ == "__main__":
    main()
