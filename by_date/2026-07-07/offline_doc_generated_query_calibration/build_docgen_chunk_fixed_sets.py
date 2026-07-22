#!/usr/bin/env python3
"""Build chunk-local DocGen Draft fixed sets from reusable generated questions."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
REAL_PROJ = "/mnt/qjhs-sh-lab-01/wjh/FusionRAG"
sys.path = [
    p
    for p in sys.path
    if p not in {REAL_PROJ, str(ROOT), str(Path(REAL_PROJ) / "ktransformers"), str(ROOT / "ktransformers")}
]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, REAL_PROJ)

from ktransformers.util.utils import _fusionrag_compute_draft_doc_attention_scores  # noqa: E402
from test_fusionrag_reflect_preprocess_exp import PreprocessScope, RecallMethod, _rate_tag, prepare_reflect_data  # noqa: E402


def finite(x):
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def qwen3_question_tokens(tokenizer, query):
    text = (
        "<|im_end|>\n<|im_start|>user\n/no_think\n"
        f"Question: {query}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    )
    return torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)


def smart_query_selection_local(attention_scores, target_ratio, threshold_factor=0.5):
    attention_scores = finite(attention_scores)
    doc_len = int(attention_scores.shape[0])
    target_count = int(doc_len * target_ratio)
    if target_count <= 0:
        return np.array([], dtype=np.int64)
    target_count = min(target_count, doc_len)

    threshold = float(np.mean(attention_scores)) + threshold_factor * float(np.std(attention_scores))
    high_positions = np.where(attention_scores > threshold)[0].astype(int).tolist()

    components, current = [], []
    for pos in sorted(high_positions):
        if not current or pos - current[-1] <= 2:
            current.append(pos)
        else:
            components.append(current)
            current = [pos]
    if current:
        components.append(current)

    component_scores = [(comp, float(sum(attention_scores[p] for p in comp))) for comp in components]
    component_scores.sort(key=lambda x: x[1], reverse=True)

    selected = set()
    for comp, _score in component_scores:
        extended = set()
        for p in comp:
            for offset in (-1, 0, 1):
                q = p + offset
                if 0 <= q < doc_len:
                    extended.add(q)
        new_positions = extended - selected
        if len(selected) + len(new_positions) <= target_count * 1.1:
            selected.update(extended)

    if len(selected) < target_count:
        for pos in np.argsort(attention_scores)[::-1]:
            selected.add(int(pos))
            if len(selected) >= target_count:
                break

    while len(selected) > target_count:
        selected.remove(min(selected, key=lambda p: attention_scores[p]))
    return np.array(sorted(selected), dtype=np.int64)


def rank_chunk(scores, rate, threshold_factor):
    scores = finite(scores)
    n_queries, doc_len = scores.shape
    target_count = max(1, int(rate * doc_len))
    per_query_selected = [smart_query_selection_local(s, rate, threshold_factor) for s in scores]
    freq = np.zeros(doc_len, dtype=np.int32)
    for idx in per_query_selected:
        freq[idx] += 1
    mean_score = scores.mean(axis=0)
    order = sorted(range(doc_len), key=lambda i: (-int(freq[i]), -float(mean_score[i]), int(i)))
    selected = np.asarray(sorted(order[:target_count]), dtype=np.int64)
    rank = np.asarray(order, dtype=np.int64)
    return selected, rank, freq, mean_score


def load_docgen_queries(path: Path):
    grouped = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped[(int(row["example_id"]), int(row["doc_id"]))].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda r: int(r["query_id"]))
    return grouped


@torch.no_grad()
def capture_chunk_scores(draft_model, tokenizer, system_tensor, doc_tensor, rows, args):
    context_tokens = [system_tensor, doc_tensor]
    system_len = int(system_tensor.shape[0])
    context_len = int(system_tensor.shape[0] + doc_tensor.shape[0])
    doc_len = int(doc_tensor.shape[0])
    scores, labels, queries = [], [], []
    for row in rows[: args.queries_per_doc]:
        q = row["question"]
        query_tensor = qwen3_question_tokens(tokenizer, q)
        full_input = torch.cat(context_tokens + [query_tensor]).unsqueeze(0).to(args.device)
        score = _fusionrag_compute_draft_doc_attention_scores(
            draft_model,
            full_input,
            query_start=context_len,
            doc_start=system_len,
            doc_len=doc_len,
            device=args.device,
            rrf_k=args.rrf_k,
            score_layers=args.score_layers,
            stop_after_score_layers=False,
        )
        scores.append(np.nan_to_num(score.float().detach().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0))
        labels.append(f"docgen_doc{int(row['doc_id']):02d}_q{int(row['query_id']):02d}")
        queries.append(q)
        del full_input, score
    torch.cuda.empty_cache()
    return np.stack(scores).astype(np.float32), labels, queries


def write_csv(path, rows):
    if not rows:
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def process_example(args, draft_model, tokenizer, questions_data, system_tensor, docgen, example_id, out_dir):
    q_data = questions_data[example_id]
    if not q_data.get("should_test", True):
        return [{"example_id": example_id, "status": "skipped_should_test_false"}]

    method = args.method
    rates = [float(x) for x in args.rates.split(",")]
    fixed_payloads = {rate: {} for rate in rates}
    manifest_rows = []
    score_dir = out_dir / "chunk_score_cache_npz"
    score_dir.mkdir(parents=True, exist_ok=True)

    for doc_id, doc_tensor in enumerate(q_data["doc_tensors"]):
        score_path = score_dir / f"example{example_id:03d}_chunk{doc_id:02d}_docgen_scores.npz"
        rows = docgen.get((example_id, doc_id), [])
        if score_path.exists() and not args.overwrite_scores:
            data = np.load(score_path, allow_pickle=True)
            scores = np.asarray(data["scores"], dtype=np.float32)
            labels = [str(x) for x in data["labels"]]
            queries = [str(x) for x in data["queries"]]
            score_status = "cached"
        else:
            if len(rows) < args.queries_per_doc:
                manifest_rows.append({
                    "example_id": example_id,
                    "chunk_id": doc_id,
                    "status": "missing_queries",
                    "n_queries": len(rows),
                })
                continue
            scores, labels, queries = capture_chunk_scores(draft_model, tokenizer, system_tensor, doc_tensor, rows, args)
            np.savez_compressed(
                score_path,
                scores=scores,
                labels=np.asarray(labels, dtype=object),
                queries=np.asarray(queries, dtype=object),
                chunk_len=np.asarray(int(doc_tensor.shape[0]), dtype=np.int32),
                example_id=np.asarray(example_id, dtype=np.int32),
                chunk_id=np.asarray(doc_id, dtype=np.int32),
                query_source=np.asarray("docgen_chunk_local", dtype=object),
                score_layers=np.asarray(str(args.score_layers), dtype=object),
                rrf_k=np.asarray(args.rrf_k, dtype=np.int32),
            )
            score_status = "ok"

        for rate in rates:
            selected, rank, freq, mean_score = rank_chunk(scores, rate, args.threshold_factor)
            fixed_payloads[rate][f"chunk{doc_id:02d}_{method}"] = selected
            fixed_payloads[rate][f"chunk{doc_id:02d}_{method}_rank"] = rank
            fixed_payloads[rate][f"chunk{doc_id:02d}_{method}_freq"] = freq
            fixed_payloads[rate][f"chunk{doc_id:02d}_{method}_mean_score"] = mean_score.astype(np.float32)
            manifest_rows.append({
                "example_id": example_id,
                "chunk_id": doc_id,
                "method": method,
                "rate": rate,
                "chunk_len": int(doc_tensor.shape[0]),
                "selected_count": int(selected.shape[0]),
                "n_queries": int(scores.shape[0]),
                "score_status": score_status,
            })
        print(f"example={example_id} chunk={doc_id} queries={len(queries)} status={score_status}", flush=True)

    for rate, payload in fixed_payloads.items():
        fixed_dir = out_dir / f"rate_{_rate_tag(rate)}" / "chunk_fixed_sets_npz"
        fixed_dir.mkdir(parents=True, exist_ok=True)
        outp = fixed_dir / f"example{example_id:03d}_rate{_rate_tag(rate)}_chunk_local_sets.npz"
        np.savez_compressed(outp, **payload)
    return manifest_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/mnt/qjhs-sh-lab-01/models/Qwen3-32B")
    ap.add_argument("--draft-model-path", default="/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct")
    ap.add_argument("--data-path", default="./data/result_reflect.json")
    ap.add_argument("--bge-model-path", default="/mnt/qjhs-sh-lab-01/models/bge-m3")
    ap.add_argument("--docgen-query-jsonl", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--example-start", type=int, default=0)
    ap.add_argument("--example-end", type=int, default=200)
    ap.add_argument("--queries-per-doc", type=int, default=32)
    ap.add_argument("--rates", default="0.10,0.15")
    ap.add_argument("--method", default="docgen_draft_smart_frequency_global")
    ap.add_argument("--rrf-k", type=int, default=18)
    ap.add_argument("--score-layers", default=None)
    ap.add_argument("--threshold-factor", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite-scores", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docgen = load_docgen_queries(Path(args.docgen_query_jsonl))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    questions_data, system_tensor, _context_rank, _corpus_lens = prepare_reflect_data(
        args.data_path,
        tokenizer,
        args.bge_model_path,
        "qwen3",
        10,
        None,
        False,
        RecallMethod.BGE,
        42,
        0,
        PreprocessScope.GLOBAL,
    )

    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device)
    draft_model.eval()

    all_rows = []
    end = min(args.example_end, len(questions_data))
    for example_id in range(args.example_start, end):
        print(f"process example {example_id}/{end}", flush=True)
        rows = process_example(args, draft_model, tokenizer, questions_data, system_tensor, docgen, example_id, out_dir)
        all_rows.extend(rows)
        write_csv(out_dir / f"manifest_{args.example_start}_{end}.partial.csv", all_rows)

    write_csv(out_dir / f"manifest_{args.example_start}_{end}.csv", all_rows)
    (out_dir / f"README_{args.example_start}_{end}.md").write_text(
        "# DocGen Chunk-Local Draft Fixed Sets\n\n"
        f"- examples: {args.example_start}..{end - 1}\n"
        f"- query source: `{args.docgen_query_jsonl}`\n"
        f"- queries per doc: {args.queries_per_doc}\n"
        f"- rates: {args.rates}\n"
        f"- method: `{args.method}`\n"
        f"- score layers: `{args.score_layers}`\n"
        f"- rrf_k: {args.rrf_k}\n"
        "- scoring unit: system + single document chunk + generated question\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
