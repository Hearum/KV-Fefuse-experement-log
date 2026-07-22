#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import hashlib
import json
import re
import threading
import time
import urllib.request
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
EXP = ROOT / "MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization"
OUT = EXP / "rejudge_glm_clean_20260714" / "rerun_20260714_1555"
MODEL_NAME = "Qwen3-32B"

BASE_URL = "http://36.150.226.221:32355/v1"
API_KEY = "api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS"
JUDGE_MODEL = "GLM-5.2"
PROMPT_VERSION = "cross_dataset_clean_think_answer_v1_rerun_20260714_1555"

DATASETS = OrderedDict([
    ("2wikimqa", 200),
    ("hotpotqa", 260),
    ("triviaqa", 270),
])

RUNS = OrderedDict([
    ("full_rate1", {"rate": 1.0, "subdir": "FusionRAG_global_topk10_bge"}),
    ("online_qk_rate015", {"rate": 0.15, "subdir": "FusionRAG_global_topk10_bge"}),
    ("online_draft_rate015", {"rate": 0.15, "subdir": "DraftModel_global_topk10_bge"}),
    ("offline3b_mean", {"rate": 0.15, "subdir": "FusionRAG_global_topk10_bge"}),
    ("offline3b_freq_boundary2", {"rate": 0.15, "subdir": "FusionRAG_global_topk10_bge"}),
    ("offline32b_top2", {"rate": 0.15, "subdir": "FusionRAG_global_topk10_bge"}),
])

CACHE = OUT / "judge_cache_glm52_clean.jsonl"
ROWS_CSV = OUT / "rejudged_rows.csv"
SUMMARY_CSV = OUT / "rejudged_summary.csv"
SUMMARY_JSON = OUT / "rejudged_summary.json"


def segments(n: int, step: int = 25):
    for start in range(0, n, step):
        yield start, min(n, start + step)


def csv_candidates(label: str, dataset: str, start: int, end: int, subdir: str, rate: float):
    base = EXP / "results" / label / dataset / f"seg_{start}_{end}" / MODEL_NAME / dataset / subdir
    if not base.exists():
        return []
    patterns = [
        str(base / f"rate_{rate}*_revert_rope.csv"),
        str(base / f"rate_{rate}*.csv"),
        str(base / "rate_*_revert_rope.csv"),
        str(base / "rate_*.csv"),
    ]
    seen = []
    for pat in patterns:
        for path in glob.glob(pat):
            if path not in seen:
                seen.append(path)
    return [Path(path) for path in seen]


def clean_pred(text: str | None) -> str:
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^\s*(?:answer|final answer|答案)\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def make_prompt(question: str, gold: str, pred: str) -> str:
    return f"""你是一个答案评估专家。你的任务是判断预测答案是否正确地回答了问题。

问题: {question}

标准答案: {gold}

预测答案: {pred}

请判断预测答案是否正确回答了问题。判断标准：
1. 预测答案包含了标准答案的关键信息。
2. 预测答案与标准答案在语义上等价。
3. 允许措辞上的细微差异，只要意思保持一致即可。
4. 如果预测答案为空、只包含思考标记、或没有给出可核验答案，判断为错误。

请严格按照以下格式回答：
判断: [正确/错误]
原因: [详细说明为什么正确或错误，至少30字]"""


def parse_judge(text: str):
    text = (text or "").strip()
    correct = False
    reason = text
    for i, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if (stripped.startswith("判断") or stripped.lower().startswith("judgment")) and (":" in stripped or "：" in stripped):
            value = re.split("[:：]", stripped, maxsplit=1)[-1].strip()
            if "正确" in value or value.lower().startswith("yes"):
                correct = True
            elif "错误" in value or value.lower().startswith("no"):
                correct = False
        if stripped.startswith("原因") or stripped.lower().startswith("reason"):
            value = re.split("[:：]", stripped, maxsplit=1)[-1].strip() if (":" in stripped or "：" in stripped) else ""
            tail = "\n".join(text.splitlines()[i + 1:]).strip()
            reason = (value + "\n" + tail).strip() if tail else value
            break
    return correct, reason or text


def call_glm(prompt: str, retries: int = 4, timeout: int = 90):
    payload = {
        "model": JUDGE_MODEL,
        "stream": False,
        "temperature": 0,
        "thinking": {"type": "disabled"},
        "max_tokens": 300,
        "messages": [
            {"role": "system", "content": "你是一个专业的答案评估专家。"},
            {"role": "user", "content": prompt},
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                BASE_URL.rstrip() + "/chat/completions",
                data=data,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                obj = json.loads(response.read().decode("utf-8"))
            raw = obj["choices"][0]["message"]["content"].strip()
            correct, reason = parse_judge(raw)
            return correct, reason, raw
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2 ** attempt, 10))
    return False, f"GLM judge failed: {last_error}", ""


def cache_key(question: str, gold: str, pred: str) -> str:
    raw = json.dumps({"q": question, "gold": gold, "pred": pred, "model": JUDGE_MODEL, "prompt": PROMPT_VERSION}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache():
    cache = {}
    if CACHE.exists():
        with CACHE.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    cache[item["cache_key"]] = item
    return cache


def append_cache(item: dict, lock: threading.Lock):
    with lock:
        with CACHE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_all_rows():
    rows = []
    missing = []
    for dataset, expected in DATASETS.items():
        for label, cfg in RUNS.items():
            for start, end in segments(expected):
                candidates = csv_candidates(label, dataset, start, end, cfg["subdir"], cfg["rate"])
                if not candidates:
                    missing.append(f"{dataset}/{label}/seg_{start}_{end}")
                    continue
                path = candidates[0]
                with path.open("r", newline="", encoding="utf-8") as handle:
                    for row_id, row in enumerate(csv.DictReader(handle)):
                        out = dict(row)
                        out["dataset"] = dataset
                        out["method"] = label
                        out["rate"] = cfg["rate"]
                        out["source_csv"] = str(path)
                        out["segment"] = f"seg_{start}_{end}"
                        out["source_row"] = row_id
                        rows.append(out)
    return rows, missing


def is_true(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "correct", "正确"}


def summarize_group(rows: list[dict], correct_key: str):
    by_sub = OrderedDict()
    for row in rows:
        key = (row.get("Main Question", "").strip(), row.get("Sub Question", "").strip())
        by_sub[key] = row
    groups = defaultdict(list)
    for (main_q, _sub_q), row in by_sub.items():
        groups[main_q].append(is_true(row.get(correct_key)))
    sub_total = len(by_sub)
    sub_correct = sum(is_true(row.get(correct_key)) for row in by_sub.values())
    main_total = len(groups)
    main_correct = sum(1 for values in groups.values() if values and all(values))
    return main_correct, main_total, sub_correct, sub_total


def write_csv(path: Path, rows: list[dict]):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, missing = read_all_rows()
    cache = load_cache()
    lock = threading.Lock()
    print(f"loaded_rows={len(rows)} cache={len(cache)} missing_segments={len(missing)}", flush=True)

    def work(index_row):
        index, row = index_row
        question = (row.get("Sub Question") or row.get("Main Question") or "").strip()
        gold = (row.get("Ground Truth") or "").strip()
        pred_clean = clean_pred(row.get("Predicted"))
        key = cache_key(question, gold, pred_clean)
        item = cache.get(key)
        if item is None:
            correct, reason, raw = call_glm(make_prompt(question, gold, pred_clean))
            item = {"cache_key": key, "correct": correct, "reason": reason, "raw": raw}
            append_cache(item, lock)
        out = dict(row)
        out["Predicted_Clean"] = pred_clean
        out["Rejudge_Correct"] = "True" if item["correct"] else "False"
        out["Rejudge_Reason"] = item.get("reason", "")
        out["Rejudge_Raw"] = item.get("raw", "")
        return index, out

    out_rows = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(work, item) for item in enumerate(rows)]
        for done, future in enumerate(as_completed(futures), 1):
            index, row = future.result()
            out_rows[index] = row
            if done % 100 == 0 or done == len(rows):
                print(f"judged={done}/{len(rows)}", flush=True)

    out_rows = [row for row in out_rows if row is not None]
    write_csv(ROWS_CSV, out_rows)

    summary_rows = []
    for dataset in DATASETS:
        for method, cfg in RUNS.items():
            group = [row for row in out_rows if row["dataset"] == dataset and row["method"] == method]
            old_main_c, old_main_t, old_sub_c, old_sub_t = summarize_group(group, "Correct")
            new_main_c, new_main_t, new_sub_c, new_sub_t = summarize_group(group, "Rejudge_Correct")
            flips_false_to_true = sum((not is_true(row.get("Correct"))) and is_true(row.get("Rejudge_Correct")) for row in group)
            flips_true_to_false = sum(is_true(row.get("Correct")) and (not is_true(row.get("Rejudge_Correct"))) for row in group)
            summary_rows.append({
                "dataset": dataset,
                "method": method,
                "rate": cfg["rate"],
                "rows": len(group),
                "old_main_correct": old_main_c,
                "old_main_total": old_main_t,
                "old_main_acc": old_main_c / old_main_t if old_main_t else 0.0,
                "glm_main_correct": new_main_c,
                "glm_main_total": new_main_t,
                "glm_main_acc": new_main_c / new_main_t if new_main_t else 0.0,
                "old_sub_correct": old_sub_c,
                "old_sub_total": old_sub_t,
                "old_sub_acc": old_sub_c / old_sub_t if old_sub_t else 0.0,
                "glm_sub_correct": new_sub_c,
                "glm_sub_total": new_sub_t,
                "glm_sub_acc": new_sub_c / new_sub_t if new_sub_t else 0.0,
                "false_to_true_rows": flips_false_to_true,
                "true_to_false_rows": flips_true_to_false,
            })

    write_csv(SUMMARY_CSV, summary_rows)
    SUMMARY_JSON.write_text(json.dumps({"missing_segments": missing, "summary": summary_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
