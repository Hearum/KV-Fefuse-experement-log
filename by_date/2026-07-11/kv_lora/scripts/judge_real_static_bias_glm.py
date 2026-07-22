#!/usr/bin/env python3
import argparse, glob, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/raid/home/hming/FusionRAG-pca-analysis")
from openai import OpenAI
from test_fusionrag_reflect_preprocess_exp import (
    DEFAULT_JUDGE_API_KEY,
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    judge_answer_with_openai,
)

p = argparse.ArgumentParser()
p.add_argument("--workers", type=int, default=6)
a = p.parse_args()
root = Path("/raid/home/hming/FusionRAG-pca-analysis/MOTIVATION_EXPERIMENTS/kv_lora")
outdir = root / "results/real_static_bias_50/glm_judge"
outdir.mkdir(parents=True, exist_ok=True)
outfile = outdir / "judged.jsonl"
done = set()
if outfile.exists():
    for line in outfile.read_text().splitlines():
        z = json.loads(line); done.add((z["example"], z["subquery"], z["method"]))
rows = []
for path in glob.glob(str(root / "results/real_static_bias_50/shard_*.jsonl")):
    rows += [json.loads(x) for x in open(path)]
methods = ["full_rate1", "raw_rate0", "random_cross_example_static_bias", "topk_preprocess_rate0"]
tasks = [(z, m) for z in rows for m in methods if z.get(m) is not None and (z["example"], z["subquery"], m) not in done]

def work(item):
    z, method = item
    client = OpenAI(api_key=DEFAULT_JUDGE_API_KEY, base_url=DEFAULT_JUDGE_BASE_URL)
    ok, reason = judge_answer_with_openai(client, DEFAULT_JUDGE_MODEL, z["question"], z[method], z["answer"])
    return dict(example=z["example"], subquery=z["subquery"], method=method, correct=bool(ok), reason=reason,
                question=z["question"], answer=z["answer"], prediction=z[method])

with ThreadPoolExecutor(max_workers=a.workers) as pool:
    futures = [pool.submit(work, x) for x in tasks]
    for i, fut in enumerate(as_completed(futures), 1):
        z = fut.result()
        with outfile.open("a") as f: f.write(json.dumps(z, ensure_ascii=False) + "\n")
        if i % 20 == 0: print(i, len(tasks), flush=True)

all_rows = [json.loads(x) for x in outfile.read_text().splitlines()]
summary = {}
for method in methods:
    x = [z for z in all_rows if z["method"] == method]
    summary[method] = {"n": len(x), "correct": sum(z["correct"] for z in x),
                       "accuracy": sum(z["correct"] for z in x) / len(x) if x else None}
(outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
