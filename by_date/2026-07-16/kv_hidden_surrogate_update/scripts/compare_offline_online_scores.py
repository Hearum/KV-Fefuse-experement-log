#!/usr/bin/env python3
"""Compare offline draft attention scores with online main-model FusionRAG scores.

The offline score cache stores per-document-token scores from a small draft model
and control queries. The online score is the main Qwen3-32B score that FusionRAG
computes from the real query before selecting recompute tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer


ROOT_CANDIDATES = [Path("/raid/home/hming/FusionRAG-pca-analysis"), Path("/home/hming/FusionRAG-pca-analysis")]
ROOT = next((p for p in ROOT_CANDIDATES if p.exists()), ROOT_CANDIDATES[0])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "ktransformers") not in sys.path:
    sys.path.insert(0, str(ROOT / "ktransformers"))

from ktransformers.models.custom_cache import StaticCache  # noqa: E402
from ktransformers.unified_process_cache import load_model  # noqa: E402
from ktransformers.util.utils import prepare_data, rotate_half  # noqa: E402


EXP_SETUP = ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset"
DEFAULT_MODEL = "/mnt/qjhs-sh-lab-01/models/Qwen3-32B"
DEFAULT_BGE = "/mnt/qjhs-sh-lab-01/models/bge-m3"
DEFAULT_CACHE_ROOT = "/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2"
DEFAULT_SCORE_ROOT = EXP_SETUP / "score_cache_full_3b_20260715"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="musique-v2", choices=["musique-v2", "2wikimqa-v2", "hotpotqa-v2", "triviaqa-v2"])
    p.add_argument("--max-examples", type=int, default=5)
    p.add_argument("--start-example", type=int, default=0, help="0-based example index")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    p.add_argument("--score-root", default=str(DEFAULT_SCORE_ROOT))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--revert-rope", action="store_true", default=True)
    p.add_argument("--no-revert-rope", dest="revert_rope", action="store_false")
    p.add_argument("--rates", default="0.05,0.10,0.15,0.30")
    p.add_argument("--max-cache-len", type=int, default=32768)
    return p.parse_args()


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a = a[mask]
    b = b[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_rankdata(np.asarray(a)), _rankdata(np.asarray(b)))


def _top_indices(x: np.ndarray, k: int) -> np.ndarray:
    k = max(0, min(int(k), len(x)))
    if k == 0:
        return np.array([], dtype=np.int64)
    part = np.argpartition(-x, k - 1)[:k]
    return part[np.argsort(-x[part])]


def _ndcg_at_k(order_scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
    k = max(1, min(int(k), len(relevance)))
    rel = np.asarray(relevance, dtype=np.float64)
    rel = rel - np.nanmin(rel)
    denom = np.nanmax(rel)
    if denom > 0:
        rel = rel / denom
    picked = _top_indices(order_scores, k)
    ideal = _top_indices(rel, k)
    discount = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float(np.sum(rel[picked] * discount))
    idcg = float(np.sum(rel[ideal] * discount))
    return dcg / idcg if idcg > 0 else float("nan")


def _score_variants(scores: np.ndarray, rates: Iterable[float]) -> Dict[str, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    variants: Dict[str, np.ndarray] = {
        "offline3b_mean": scores.mean(axis=0),
        "offline3b_top2_mean": np.sort(scores, axis=0)[-2:].mean(axis=0),
    }
    # Frequency proxy: how often a token appears in each control query top-k set.
    freq = np.zeros(scores.shape[1], dtype=np.float64)
    for row in scores:
        k = max(1, int(round(0.15 * len(row))))
        freq[_top_indices(row, k)] += 1.0
    variants["offline3b_freq_top15"] = freq / max(1, scores.shape[0])
    return variants


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    return (x - mu) / sd if sd > 0 else x * 0.0


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else x * 0.0


def _load_setup_data(args: argparse.Namespace, tokenizer):
    data_path = str(EXP_SETUP / "data") + "/"
    return prepare_data(
        "Qwen3-32B",
        data_path,
        f"{args.dataset}.jsonl",
        str(Path(args.cache_root)) + "/",
        tokenizer,
        args.topk,
        args.revert_rope,
        False,
        DEFAULT_BGE,
    )


def _fill_cache_and_online_score(model, tokenizer, past_key_values, passages, load_path: Path, example_id_1based: int, args: argparse.Namespace) -> np.ndarray:
    device = args.device
    passages_len = [int(p.shape[0]) for p in passages]
    system_len = passages_len[0]
    past_len = 0
    key_cache = []
    value_cache = []

    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0
        if past_key_values.importance_cache:
            past_key_values.importance_cache[layer_idx].zero_()

    for chunk_id, passage in enumerate(passages[:-1]):
        key = torch.load(load_path / f"{example_id_1based}_{chunk_id}_key.pt", weights_only=True, map_location="cpu")
        value = torch.load(load_path / f"{example_id_1based}_{chunk_id}_value.pt", weights_only=True, map_location="cpu")
        key_cache.append(key)
        value_cache.append(value)

    for chunk_id, passage in enumerate(passages[:-1]):
        passage_len = int(passage.shape[0])
        chunk_key_cache = key_cache[chunk_id].to(device)
        chunk_value_cache = value_cache[chunk_id].to(device)
        if args.revert_rope and chunk_id > 0:
            position_ids = torch.full((1, passage_len), past_len - system_len, device=device)
            try:
                cos, sin = model.model.rotary_emb(chunk_key_cache[0], position_ids)
            except AttributeError:
                cos, sin = model.model.layers[0].self_attn.rotary_emb(chunk_key_cache[0], position_ids)
            cos = cos.to(chunk_key_cache.device).unsqueeze(1)
            sin = sin.to(chunk_key_cache.device).unsqueeze(1)
            chunk_key_cache = (chunk_key_cache * cos) + (rotate_half(chunk_key_cache) * sin)

        for layer_idx in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[layer_idx].narrow(2, past_len, passage_len).copy_(chunk_key_cache[layer_idx])
            past_key_values.value_cache[layer_idx].narrow(2, past_len, passage_len).copy_(chunk_value_cache[layer_idx])
            past_key_values.past_tokens[layer_idx] += passage_len
        past_len += passage_len

    query_text = tokenizer.decode(passages[-1])
    query_prefix_len = len(tokenizer.encode(query_text.split("Question: ")[0]))
    if query_prefix_len >= len(passages[-1]):
        query_prefix_len = len(tokenizer.encode(query_text.split("Question：")[0])) + 1
    query_tokens = passages[-1][query_prefix_len:].unsqueeze(0).to(device)
    cache_position = torch.arange(past_len, past_len + query_tokens.shape[1], device=device)
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(query_tokens).to(device)
        model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            reprocess_method="FusionRAG",
            return_dict=False,
            use_cache=True,
            passages_len=passages_len,
            history_key_cache=key_cache,
        )
    k_sum = torch.sum(past_key_values.importance_cache[-1], dim=0)[:past_len]
    return k_sum[system_len:past_len].detach().float().cpu().numpy()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rates = [float(x) for x in args.rates.split(",") if x.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    model = load_model("qwen3", args.model_path, config).to(args.device).eval()
    prompt_data, tokens_data, _question_list, _answer_list, _stop_id, raw_path, preprocess_path, _csv_path, *_rest = _load_setup_data(args, tokenizer)
    load_path = Path(preprocess_path)
    score_npz_dir = Path(args.score_root) / args.dataset / "score_cache_npz"

    rows: List[dict] = []
    diag_rows: List[dict] = []
    for example_idx in range(args.start_example, min(args.start_example + args.max_examples, len(tokens_data))):
        example_id = example_idx + 1
        score_path = score_npz_dir / f"setup_v2_{args.dataset}_example{example_idx:03d}_scores.npz"
        if not score_path.exists():
            print(f"[skip] missing offline score {score_path}")
            continue
        offline = np.load(score_path, allow_pickle=True)
        online = _fill_cache_and_online_score(model, tokenizer, past_key_values=StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=args.max_cache_len,
            device=args.device,
            dtype=model.dtype,
            passage_len=args.max_cache_len,
        ), passages=tokens_data[example_idx], load_path=load_path, example_id_1based=example_id, args=args)
        scores = np.asarray(offline["scores"], dtype=np.float64)
        if scores.shape[1] != online.shape[0]:
            raise ValueError(f"length mismatch example={example_idx}: offline={scores.shape[1]} online={online.shape[0]}")

        variants = _score_variants(scores, rates)
        online_z = _zscore(online)
        online_mm = _minmax(online)
        diag_rows.append({
            "dataset": args.dataset,
            "example_idx": example_idx,
            "doc_tokens": int(len(online)),
            "offline_controls": int(scores.shape[0]),
            "online_mean": float(np.mean(online)),
            "online_std": float(np.std(online)),
            "online_min": float(np.min(online)),
            "online_max": float(np.max(online)),
            "offline_mean_mean": float(np.mean(variants["offline3b_mean"])),
            "offline_mean_std": float(np.std(variants["offline3b_mean"])),
        })
        for method, offline_score in variants.items():
            offline_z = _zscore(offline_score)
            offline_mm = _minmax(offline_score)
            base = {
                "dataset": args.dataset,
                "example_idx": example_idx,
                "method": method,
                "doc_tokens": int(len(online)),
                "pearson_z": _pearson(offline_z, online_z),
                "pearson_minmax": _pearson(offline_mm, online_mm),
                "spearman_rank": _spearman(offline_score, online),
            }
            for rate in rates:
                k = max(1, int(round(rate * len(online))))
                off_top = set(_top_indices(offline_score, k).tolist())
                on_top = set(_top_indices(online, k).tolist())
                inter = len(off_top & on_top)
                union = len(off_top | on_top)
                rows.append({
                    **base,
                    "rate": rate,
                    "k": k,
                    "topk_overlap": inter / k,
                    "topk_jaccard": inter / union if union else float("nan"),
                    "ndcg_at_k": _ndcg_at_k(offline_score, online, k),
                })
        print(f"[done] example={example_idx} doc_tokens={len(online)}")

    per_path = out_dir / "per_example_method_metrics.csv"
    with per_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    diag_path = out_dir / "score_scale_diagnostics.csv"
    with diag_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()) if diag_rows else [])
        writer.writeheader()
        writer.writerows(diag_rows)

    summary = []
    if rows:
        keys = sorted({(r["method"], r["rate"]) for r in rows})
        metrics = ["pearson_z", "pearson_minmax", "spearman_rank", "topk_overlap", "topk_jaccard", "ndcg_at_k"]
        for method, rate in keys:
            part = [r for r in rows if r["method"] == method and r["rate"] == rate]
            rec = {"dataset": args.dataset, "method": method, "rate": rate, "examples": len({r["example_idx"] for r in part})}
            for m in metrics:
                rec[m + "_mean"] = float(np.nanmean([r[m] for r in part]))
                rec[m + "_std"] = float(np.nanstd([r[m] for r in part]))
            summary.append(rec)
    sum_path = out_dir / "summary_metrics.csv"
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
        writer.writeheader()
        writer.writerows(summary)

    meta = {
        "dataset": args.dataset,
        "start_example": args.start_example,
        "max_examples": args.max_examples,
        "model_path": args.model_path,
        "cache_root": args.cache_root,
        "score_root": args.score_root,
        "load_path": str(load_path),
        "rates": rates,
        "outputs": [str(per_path), str(diag_path), str(sum_path)],
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
