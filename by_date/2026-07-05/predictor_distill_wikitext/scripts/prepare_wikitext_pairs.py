#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--out", default="MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/data/wikitext_pairs.jsonl")
    parser.add_argument("--text-file", default=None, help="Optional local plain-text file. If set, do not download WikiText.")
    parser.add_argument("--max-pairs", type=int, default=80)
    parser.add_argument("--doc-len", type=int, default=384)
    parser.add_argument("--query-len", type=int, default=64)
    parser.add_argument("--min-len", type=int, default=520)
    parser.add_argument("--stride", type=int, default=0, help="If >0, create overlapping windows from token stream.")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.text_file:
        text_iter = Path(args.text_file).read_text(encoding="utf-8", errors="ignore").splitlines()
    else:
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        text_iter = ((item.get("text") or "") for item in ds)

    rows = []
    window_len = args.doc_len + args.query_len
    if args.stride > 0:
        all_text = "\n".join(t for t in text_iter if (t or "").strip())
        all_ids = tokenizer.encode(all_text, add_special_tokens=False)
        for start in range(0, max(0, len(all_ids) - window_len + 1), args.stride):
            ids = all_ids[start : start + window_len]
            doc_ids = ids[: args.doc_len]
            query_ids = ids[args.doc_len : args.doc_len + args.query_len]
            rows.append(
                {
                    "pair_id": len(rows),
                    "doc_ids": doc_ids,
                    "query_ids": query_ids,
                    "text_preview": tokenizer.decode(ids[:96]),
                }
            )
            if len(rows) >= args.max_pairs:
                break
    else:
        buffer = ""
        for raw_text in text_iter:
            text = (raw_text or "").strip()
            if not text or text.startswith("="):
                continue
            buffer = (buffer + "\n" + text).strip()
            ids = tokenizer.encode(buffer, add_special_tokens=False)
            if len(ids) < args.min_len:
                continue
            doc_ids = ids[: args.doc_len]
            query_ids = ids[args.doc_len : args.doc_len + args.query_len]
            if len(doc_ids) == args.doc_len and len(query_ids) >= 16:
                rows.append(
                    {
                        "pair_id": len(rows),
                        "doc_ids": doc_ids,
                        "query_ids": query_ids,
                        "text_preview": tokenizer.decode(ids[:96]),
                    }
                )
                buffer = ""
            if len(rows) >= args.max_pairs:
                break

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} pairs to {out}")


if __name__ == "__main__":
    main()
