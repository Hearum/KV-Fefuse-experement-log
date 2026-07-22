#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def as_text_list(value):
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                for key in ("text", "contents", "document", "doc", "passage"):
                    if key in item and isinstance(item[key], str):
                        out.append(item[key])
                        break
        return out
    if isinstance(value, str):
        return [value]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/result_reflect.json")
    parser.add_argument("--model-path", default="/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--out", required=True)
    parser.add_argument("--doc-len", type=int, default=384)
    parser.add_argument("--query-len", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    data = json.loads(Path(args.data_path).read_text(encoding="utf-8"))
    rows = []
    for ex_id, item in enumerate(data):
        docs = as_text_list(item.get("retrieved_results")) or as_text_list(item.get("gold_docs"))
        if not docs:
            continue
        doc_text = "\n".join(docs)
        queries = [item.get("question", "")]
        for sub in item.get("intermediate_context") or []:
            q = sub.get("query") if isinstance(sub, dict) else None
            if q:
                queries.append(q)
        doc_ids_full = tokenizer.encode(doc_text, add_special_tokens=False)
        if len(doc_ids_full) < 32:
            continue
        doc_ids = doc_ids_full[: args.doc_len]
        for sub_idx, query in enumerate(queries):
            query_ids = tokenizer.encode(str(query), add_special_tokens=False)[: args.query_len]
            if len(query_ids) < 4:
                continue
            rows.append(
                {
                    "pair_id": len(rows),
                    "example_id": ex_id,
                    "sub_q_idx": sub_idx,
                    "doc_ids": doc_ids,
                    "query_ids": query_ids,
                    "text_preview": tokenizer.decode((doc_ids + query_ids)[:96]),
                }
            )
            if args.max_pairs and len(rows) >= args.max_pairs:
                break
        if args.max_pairs and len(rows) >= args.max_pairs:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} pairs to {out}")


if __name__ == "__main__":
    main()
