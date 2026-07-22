#!/usr/bin/env python3
"""Generate reusable GLM questions for each retrieved document chunk."""

import argparse
import csv
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from openai import OpenAI


DEFAULT_BASE_URL = "http://36.150.226.221:32355/v1"
DEFAULT_API_KEY = os.environ.get(
    "FUSIONRAG_JUDGE_API_KEY",
    "api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS",
)
DEFAULT_MODEL = "GLM-5.2"


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_prompt(prompt_path: Path):
    text = prompt_path.read_text(encoding="utf-8")
    sys_m = re.search(r"## System Prompt\n\n(.*?)\n\n## User Prompt Template", text, re.S)
    user_m = re.search(r"## User Prompt Template\n\n(.*)", text, re.S)
    if not sys_m or not user_m:
        raise ValueError(f"Cannot parse prompt file: {prompt_path}")
    return sys_m.group(1).strip(), user_m.group(1).strip()


def iter_example_docs(data_path: Path):
    data = json.loads(data_path.read_text(encoding="utf-8"))
    for example_id, item in enumerate(data):
        doc_to_id = {}
        docs = []
        for sub_q in item.get("intermediate_context", []):
            for doc in sub_q.get("retrieve docs", []):
                if doc not in doc_to_id:
                    doc_to_id[doc] = len(docs)
                    docs.append(doc)
        for doc_id, doc in enumerate(docs):
            yield example_id, doc_id, doc


def parse_questions(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    obj = json.loads(raw)
    qs = obj["questions"] if isinstance(obj, dict) else obj
    out = []
    seen = set()
    for q in qs:
        q = str(q).strip()
        if not q:
            continue
        key = re.sub(r"\s+", " ", q.lower())
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


def call_glm(client, model, system_prompt, user_template, doc, temperature, max_retries):
    user_prompt = user_template.replace("{document_chunk}", doc)
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                stream=False,
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = resp.choices[0].message.content
            qs = parse_questions(content)
            if len(qs) >= 32:
                return qs[:32], content, None
            last_err = f"only {len(qs)} valid questions"
        except Exception as exc:
            last_err = repr(exc)
        time.sleep(min(2 ** attempt, 8))
    return [], "", last_err


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_existing_query_rows(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_existing_manifest(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dedup_query_rows(rows):
    keyed = {}
    for r in rows:
        key = (int(r["example_id"]), int(r["doc_id"]), int(r["query_id"]))
        keyed[key] = r
    return [keyed[k] for k in sorted(keyed)]


def dedup_manifest(rows):
    keyed = {}
    for r in rows:
        key = (int(r["example_id"]), int(r["doc_id"]))
        keyed[key] = r
    return [keyed[k] for k in sorted(keyed)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/result_reflect.json")
    ap.add_argument("--prompt-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api-key", default=DEFAULT_API_KEY)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt-version", default="glm_docgen_prompt_v2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--start-example", type=int, default=0)
    ap.add_argument("--end-example", type=int, default=200)
    ap.add_argument("--limit-docs", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_path = out_dir / "docgen_queries.jsonl"
    manifest_path = out_dir / "docgen_query_manifest.csv"
    cache_path = out_dir / "docgen_query_cache.json"

    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    cache_lock = Lock()

    system_prompt, user_template = load_prompt(Path(args.prompt_path))
    all_docs = [
        (ex, doc_id, doc)
        for ex, doc_id, doc in iter_example_docs(Path(args.data_path))
        if args.start_example <= ex < args.end_example
    ]
    if args.limit_docs is not None:
        all_docs = all_docs[: args.limit_docs]

    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)
    rows = load_existing_query_rows(query_path)
    manifest = load_existing_manifest(manifest_path)

    def one(item):
        ex, doc_id, doc = item
        h = sha_text(doc)
        key = f"{h}:{args.prompt_version}:{args.model}"
        with cache_lock:
            cached = cache.get(key)
        if cached and len(cached.get("questions", [])) >= 32:
            qs = cached["questions"][:32]
            status, err, raw = "cached", "", ""
        else:
            qs, raw, err = call_glm(
                client,
                args.model,
                system_prompt,
                user_template,
                doc,
                args.temperature,
                args.max_retries,
            )
            status = "ok" if len(qs) >= 32 else "failed"
            if status == "ok":
                with cache_lock:
                    cache[key] = {
                        "questions": qs,
                        "source_doc_hash": h,
                        "prompt_version": args.prompt_version,
                        "glm_model": args.model,
                    }
        return ex, doc_id, doc, h, key, qs, status, err, raw

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, item) for item in all_docs]
        for fut in as_completed(futures):
            ex, doc_id, doc, h, key, qs, status, err, _raw = fut.result()
            done += 1
            for qid, q in enumerate(qs[:32]):
                rows.append(
                    {
                        "example_id": ex,
                        "doc_id": doc_id,
                        "query_id": qid,
                        "question": q,
                        "source_doc_hash": h,
                        "prompt_version": args.prompt_version,
                        "glm_model": args.model,
                    }
                )
            manifest.append(
                {
                    "example_id": ex,
                    "doc_id": doc_id,
                    "source_doc_hash": h,
                    "status": status,
                    "n_questions": len(qs),
                    "error": err or "",
                }
            )
            if done % 20 == 0:
                with cache_lock:
                    cache_snapshot = dict(cache)
                cache_path.write_text(json.dumps(cache_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
                write_jsonl(query_path, dedup_query_rows(rows))
                write_csv(manifest_path, dedup_manifest(manifest))
                print(f"done_docs={done}/{len(all_docs)} ok_questions={len(rows)}", flush=True)

    rows = dedup_query_rows(rows)
    manifest = dedup_manifest(manifest)
    with cache_lock:
        cache_snapshot = dict(cache)
    cache_path.write_text(json.dumps(cache_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(query_path, rows)
    write_csv(manifest_path, manifest)

    sample = out_dir / "sample_queries.md"
    with sample.open("w", encoding="utf-8") as f:
        f.write("# Sample Generated Queries\n\n")
        shown = 0
        for r in rows:
            if r["query_id"] == 0:
                if shown >= 10:
                    break
                f.write(f"## example {r['example_id']} doc {r['doc_id']}\n\n")
                subset = [
                    x for x in rows
                    if x["example_id"] == r["example_id"] and x["doc_id"] == r["doc_id"]
                ][:8]
                for x in subset:
                    f.write(f"- {x['question']}\n")
                f.write("\n")
                shown += 1
    print(f"wrote {len(rows)} query rows for {len(manifest)} docs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
