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
EXP = ROOT / "MOTIVATION_EXPERIMENTS"
OUT = EXP / "cross_dataset_offline_generalization/rejudge_glm_clean_20260714/musique_rerun_20260714_1718"
MODEL_NAME = "Qwen3-32B"
DATASET = "musique"

BASE_URL = "http://36.150.226.221:32355/v1"
API_KEY = "api_aK7mP9xQ2vL4wR8sT1yU5zC3bN6dE0fG7hI2jK4lM8nO1pQ3rS"
JUDGE_MODEL = "GLM-5.2"
PROMPT_VERSION = "musique_clean_think_answer_v1_rerun_20260714_1718"

RUNS = OrderedDict([
    ("full_rate1", {
        "rate": "1.0",
        "root": EXP / "qwen3_hybrid70_online_baselines/full_rate1",
        "subdir": "FusionRAG_global_topk10_bge",
    }),
    ("online_qk_rate015", {
        "rate": "0.15",
        "root": EXP / "qwen3_rate015_online_offline/online_qk_rate015",
        "subdir": "FusionRAG_global_topk10_bge",
    }),
    ("online_draft_rate015", {
        "rate": "0.15",
        "root": EXP / "qwen3_rate015_online_offline/online_draft_rate015",
        "subdir": "DraftModel_global_topk10_bge",
    }),
    ("offline_hybrid70_rate015", {
        "rate": "0.15",
        "root": EXP / "qwen3_rate015_online_offline/offline_hybrid70_rate015",
        "subdir": "FusionRAG_global_topk10_bge",
    }),
    ("draft_smart_mean_score_global", {
        "rate": "0.15",
        "root": EXP / "qwen3_draft_smart_global_rate015/draft_smart_mean_score_global",
        "subdir": "FusionRAG_global_topk10_bge",
    }),
    ("draft_smart_freq_boundary0p02_global", {
        "rate": "0.15",
        "root": EXP / "offline_boundary_mix_rate015/accuracy_runs/draft_smart_freq_boundary0p02_global",
        "subdir": "FusionRAG_global_topk10_bge",
    }),
    ("draft32b_smart_top2_mean_global", {
        "rate": "0.15",
        "root": EXP / "offline_draft32b_teacher_rate015/accuracy_runs_control_qwen3_32b/draft32b_smart_top2_mean_global",
        "subdir": "FusionRAG_global_topk10_bge",
    }),
    ("offline10_draft005", {
        "rate": "0.1",
        "root": EXP / "qwen3_fair_budget_offline_vs_residual/offline10_draft005",
        "subdir": "DraftModel_global_topk10_bge",
    }),
    ("offline10_hybrid_old70_docgen30_draft005", {
        "rate": "0.15",
        "root": EXP / "offline_doc_generated_query_calibration/qwen3_docgen_fair_budget/offline10_hybrid_old70_docgen30_draft005",
        "subdir": "DraftModel_global_topk10_bge",
    }),
    ("offline20_only", {
        "rate": "0.2",
        "root": EXP / "qwen3_fair_budget_offline_vs_residual/offline20_only",
        "subdir": "DraftModel_global_topk10_bge",
    }),
])

CACHE = OUT / "judge_cache_glm52_clean.jsonl"
ROWS_CSV = OUT / "rejudged_rows.csv"
SUMMARY_CSV = OUT / "rejudged_summary.csv"
SUMMARY_JSON = OUT / "rejudged_summary.json"


def csv_candidates(cfg: dict) -> list[Path]:
    root = cfg["root"]
    subdir = cfg["subdir"]
    rate = cfg["rate"]
    pattern = root / "seg_*" / MODEL_NAME / DATASET / subdir / f"rate_{rate}*_revert_rope.csv"
    paths = [Path(p) for p in glob.glob(str(pattern))]
    if not paths:
        pattern = root / "seg_*" / MODEL_NAME / DATASET / subdir / "rate_*_revert_rope.csv"
        paths = [Path(p) for p in glob.glob(str(pattern))]
    return sorted(paths, key=lambda p: str(p))


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
    for label, cfg in RUNS.items():
        paths = csv_candidates(cfg)
        if not paths:
            missing.append(label)
            continue
        for path in paths:
            segment = next((part for part in path.parts if part.startswith("seg_")), "")
            with path.open("r", newline="", encoding="utf-8") as handle:
                for row_id, row in enumerate(csv.DictReader(handle)):
                    out = dict(row)
                    out["dataset"] = DATASET
                    out["method"] = label
                    out["rate"] = cfg["rate"]
                    out["source_csv"] = str(path)
                    out["segment"] = segment
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
    print(f"loaded_rows={len(rows)} cache={len(cache)} missing_methods={len(missing)}", flush=True)
    if missing:
        print("missing=" + ",".join(missing), flush=True)

    def work(index_row):
        index, row = index_row
        question = (row.get("Sub Question") or row.get("Main Question") or "").strip()
        gold = (row.get("Ground Truth") or "").strip()
        pred_clean = clean_pred(row.get("Predicted"))
        key = cache_key(question, gold, pred_clean)
        if key in cache:
            item = cache[key]
            correct = bool(item["glm_correct"])
            reason = item.get("glm_reason", "")
            raw = item.get("glm_raw", "")
        else:
            prompt = make_prompt(question, gold, pred_clean)
            correct, reason, raw = call_glm(prompt)
            item = {
                "cache_key": key,
                "glm_correct": correct,
                "glm_reason": reason,
                "glm_raw": raw,
                "question": question,
                "gold": gold,
                "pred_clean": pred_clean,
            }
            append_cache(item, lock)
        out = dict(row)
        out["Predicted_clean"] = pred_clean
        out["glm_correct"] = str(correct)
        out["glm_reason"] = reason
        out["glm_raw"] = raw
        out["row_index"] = index
        return index, out

    judged = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(work, item) for item in enumerate(rows)]
        for done, fut in enumerate(as_completed(futures), 1):
            index, out = fut.result()
            judged[index] = out
            if done % 100 == 0 or done == len(rows):
                print(f"judged={done}/{len(rows)}", flush=True)

    judged = [row for row in judged if row is not None]
    write_csv(ROWS_CSV, judged)

    summary = []
    for method in RUNS:
        group_rows = [row for row in judged if row.get("method") == method]
        if not group_rows:
            continue
        old_main_c, old_main_t, old_sub_c, old_sub_t = summarize_group(group_rows, "Correct")
        glm_main_c, glm_main_t, glm_sub_c, glm_sub_t = summarize_group(group_rows, "glm_correct")
        summary.append({
            "dataset": DATASET,
            "method": method,
            "rate": RUNS[method]["rate"],
            "rows": len(group_rows),
            "old_main_correct": old_main_c,
            "old_main_total": old_main_t,
            "old_main_acc": old_main_c / old_main_t if old_main_t else 0,
            "glm_main_correct": glm_main_c,
            "glm_main_total": glm_main_t,
            "glm_main_acc": glm_main_c / glm_main_t if glm_main_t else 0,
            "old_sub_correct": old_sub_c,
            "old_sub_total": old_sub_t,
            "old_sub_acc": old_sub_c / old_sub_t if old_sub_t else 0,
            "glm_sub_correct": glm_sub_c,
            "glm_sub_total": glm_sub_t,
            "glm_sub_acc": glm_sub_c / glm_sub_t if glm_sub_t else 0,
            "false_to_true_rows": sum((not is_true(row.get("Correct"))) and is_true(row.get("glm_correct")) for row in group_rows),
            "true_to_false_rows": sum(is_true(row.get("Correct")) and (not is_true(row.get("glm_correct"))) for row in group_rows),
        })

    write_csv(SUMMARY_CSV, summary)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(SUMMARY_CSV, flush=True)


if __name__ == "__main__":
    main()
