#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def read_text_slice(path: Path, start_frac: float, end_frac: float) -> str:
    data = path.read_text(encoding="utf-8", errors="ignore")
    n = len(data)
    start = int(n * start_frac)
    end = int(n * end_frac)
    if start < 0 or end > n or start >= end:
        raise ValueError(f"invalid slice: start_frac={start_frac}, end_frac={end_frac}, chars={n}")
    # Move to line boundaries so train/val do not split a line mid-way.
    if start > 0:
        nl = data.find("\n", start)
        start = n if nl < 0 else nl + 1
    if end < n:
        nl = data.rfind("\n", 0, end)
        end = 0 if nl < 0 else nl
    return data[start:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-pairs", type=int, required=True)
    parser.add_argument("--doc-len", type=int, default=384)
    parser.add_argument("--query-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--start-frac", type=float, default=0.0)
    parser.add_argument("--end-frac", type=float, default=1.0)
    parser.add_argument("--pair-id-offset", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    text = read_text_slice(Path(args.text_file), args.start_frac, args.end_frac)
    text = "\n".join(line for line in text.splitlines() if line.strip() and not line.strip().startswith("="))
    ids = tokenizer.encode(text, add_special_tokens=False)

    window_len = args.doc_len + args.query_len
    rows = 0
    with out.open("w", encoding="utf-8") as f:
        for start in range(0, max(0, len(ids) - window_len + 1), args.stride):
            span = ids[start : start + window_len]
            row = {
                "pair_id": args.pair_id_offset + rows,
                "doc_ids": span[: args.doc_len],
                "query_ids": span[args.doc_len : args.doc_len + args.query_len],
                "text_slice": [args.start_frac, args.end_frac],
                "token_start_in_slice": start,
                "text_preview": tokenizer.decode(span[:96]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
            if rows >= args.max_pairs:
                break

    print(
        f"wrote {rows} pairs to {out}; slice=({args.start_frac},{args.end_frac}); "
        f"tokens_in_slice={len(ids)}; stride={args.stride}"
    )


if __name__ == "__main__":
    main()
