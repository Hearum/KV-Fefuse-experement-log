#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def load_records(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    if text.lstrip().startswith("["):
        return json.loads(text)
    out = []
    for line in text.splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def split_passages(context: str):
    parts = re.split(r"(?=Passage\s+\d+\n)", context)
    docs = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part = re.sub(r"^Passage\s+\d+\n", "", part).strip()
        if part:
            docs.append(part)
    return docs


def get_answer(row):
    if row.get("answers"):
        ans = row["answers"]
        return str(ans[0] if isinstance(ans, list) else ans)
    out = row.get("output")
    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict) and "answer" in first:
            return str(first["answer"])
        if isinstance(first, str):
            return first
    return ""


def get_docs(row):
    out = row.get("output")
    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict) and isinstance(first.get("document"), list):
            return [str(x) for x in first["document"] if str(x).strip()]
    if isinstance(row.get("context"), str):
        return split_passages(row["context"])
    return []


def convert(records, max_examples=None):
    out = []
    for i, row in enumerate(records[: max_examples or len(records)]):
        question = str(row.get("input") or row.get("question") or "")
        answer = get_answer(row)
        docs = get_docs(row)
        if not question or not answer or not docs:
            continue
        out.append(
            {
                "question": question,
                "answer": answer,
                "intermediate_context": [
                    {
                        "query": question,
                        "answer": answer,
                        "retrieve docs": docs[:10],
                    }
                ],
                "llm_judge": True,
                "source_id": str(row.get("id") or row.get("_id") or i),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-examples", type=int, default=None)
    args = ap.parse_args()
    records = load_records(Path(args.input))
    converted = convert(records, args.max_examples)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"converted {len(converted)}/{len(records)} -> {args.output}")


if __name__ == "__main__":
    main()
