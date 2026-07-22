#!/usr/bin/env python3
"""Calibrate global/layer-wise gates for predicted-K functional adapter.

The K predictor is trained from offline own_kv -> DeltaK on train records.
Gate scales are selected on calibration functional spans and evaluated on a
separate test range. Full-context K/V/hidden are used only as metric teachers.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

for candidate in (
    Path("/raid/home/hming/FusionRAG-pca-analysis"),
    Path("/home/hming/FusionRAG-pca-analysis"),
):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        ROOT = candidate
        break
else:
    ROOT = Path.cwd()

SCRIPT_DIR = ROOT / "MOTIVATION_EXPERIMENTS/structured_kv_adapter/scripts"
spec = importlib.util.spec_from_file_location("pred_k_probe", SCRIPT_DIR / "analyze_qwen3_8b_predicted_k_functional_probe.py")
pred_k_probe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pred_k_probe)

from ktransformers.models.custom_cache import StaticCache
from test_fusionrag_reflect_preprocess_exp import load_model

L, H_KV, F = pred_k_probe.L, pred_k_probe.H_KV, pred_k_probe.F


def empty_totals(variants):
    return {name: {key: 0.0 for key in ("logit_err", "logit_base", "out_err", "out_base")} for name in variants}


def add_metric(total, logits, out, logits_full, out_full):
    total["logit_err"] += float((logits - logits_full).square().sum())
    total["logit_base"] += float(logits_full.square().sum())
    total["out_err"] += float((out - out_full).square().sum())
    total["out_base"] += float(out_full.square().sum())


def rel(total):
    return {
        "logit_rel_error": math.sqrt(total["logit_err"] / max(total["logit_base"], 1e-30)),
        "output_rel_error": math.sqrt(total["out_err"] / max(total["out_base"], 1e-30)),
    }


def summarize_totals(totals):
    return {name: rel(total) for name, total in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--token-cache", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model-path", default="/home/hming/models/Qwen3-8B")
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--stride", type=int, default=416)
    parser.add_argument("--calib-start", type=int, default=96)
    parser.add_argument("--calib-count", type=int, default=32)
    parser.add_argument("--test-start", type=int, default=128)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--scales", default="0,0.03,0.1,0.2,0.3,0.45,0.6,0.8,1.0")
    args = parser.parse_args()
    scales = [float(x) for x in args.scales.split(",") if x.strip()]

    train_paths = sorted(Path(args.train_dir).glob("sample*.pt"))[: args.train_count]
    if not train_paths:
        raise FileNotFoundError(args.train_dir)
    predictors, train_rows = pred_k_probe.train_k_predictors(train_paths, args.rank, args.alpha)

    device = "cuda:0"
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"
    model, _ = load_model("qwen3", args.model_path, cfg, device, False)
    model.eval()
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=512, device=device, dtype=model.dtype, passage_len=512, model=model)
    captures = {}
    handles = []

    def make_v_hook(layer_idx):
        def hook(module, inputs, output):
            captures[layer_idx] = inputs[0].detach().cpu().half()
        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.v_proj.register_forward_hook(make_v_hook(layer_idx)))

    def capture(ids):
        captures.clear()
        for layer in range(len(cache.key_cache)):
            cache.past_tokens[layer] = 0
        ids = ids.unsqueeze(0).to(device)
        pos = torch.arange(ids.shape[1], device=device)
        with torch.no_grad():
            model(input_ids=ids, cache_position=pos, past_key_values=cache, use_cache=True, return_dict=False)
        keys = torch.stack([x[:, :, : ids.shape[1], :].detach().cpu().half() for x in cache.key_cache])
        values = torch.stack([x[:, :, : ids.shape[1], :].detach().cpu().half() for x in cache.value_cache])
        h_in = torch.stack([captures[i][0] for i in range(len(model.model.layers))]).half()
        return keys, values, h_in

    ids = torch.load(args.token_cache, map_location="cpu", weights_only=True)
    groups = cfg.num_attention_heads // cfg.num_key_value_heads
    base_variants = ["cachedK_cachedV", "fullK_cachedV", "cachedK_fullV", "fullK_fullV"]
    pred_variants = [f"predK_s{scale:g}_cachedV" for scale in scales]
    variants = base_variants + pred_variants

    def eval_range(start, count, tag):
        totals = empty_totals(variants)
        layer_totals = [{name: {key: 0.0 for key in ("logit_err", "logit_base", "out_err", "out_base")} for name in variants} for _ in range(cfg.num_hidden_layers)]
        processed = 0
        for sample in range(start, start + count):
            begin = sample * args.stride
            span = ids[begin : begin + 416].long()
            if len(span) < 416:
                break
            prefix, target, query = span[:256], span[256:384], span[384:416]
            offline_k, offline_v, _ = capture(target)
            full_k, full_v, full_h = capture(torch.cat([prefix, target, query]))
            full_x_k = full_k[:, :, :, 256:384, :].float()
            full_x_v = full_v[:, :, :, 256:384, :].float()
            cached_x_k = pred_k_probe.rotate(model, offline_k[:, :, :, :128, :], 256, device).float()
            cached_x_v = offline_v[:, :, :, :128, :].float()
            own_kv_all = torch.cat([offline_k[:, 0, :, :128, :], offline_v[:, 0, :, :128, :]], -1).float()
            pred_delta_k = pred_k_probe.predict_delta_k(predictors, own_kv_all)
            offline_target_k = offline_k[:, :, :, :128, :].float()
            pred_x_k_by_scale = {}
            for scale in scales:
                pred_offline_k = offline_target_k + scale * pred_delta_k.unsqueeze(1)
                pred_x_k_by_scale[scale] = pred_k_probe.rotate(model, pred_offline_k, 256, device).float()
            q_pos = torch.arange(384, 416)
            for layer_idx in range(cfg.num_hidden_layers):
                q = pred_k_probe.q_for_positions(model, full_h[layer_idx, 384:416].float(), q_pos, layer_idx, device).unsqueeze(0)
                k_full = pred_k_probe.repeat_kv(full_x_k[layer_idx], groups)
                k_cached = pred_k_probe.repeat_kv(cached_x_k[layer_idx], groups)
                v_full = pred_k_probe.repeat_kv(full_x_v[layer_idx], groups)
                v_cached = pred_k_probe.repeat_kv(cached_x_v[layer_idx], groups)
                logits_full = torch.matmul(q, k_full.transpose(-1, -2)) / math.sqrt(q.shape[-1])
                logits_cached = torch.matmul(q, k_cached.transpose(-1, -2)) / math.sqrt(q.shape[-1])
                p_full = torch.softmax(logits_full, dim=-1)
                p_cached = torch.softmax(logits_cached, dim=-1)
                out_full = torch.matmul(p_full, v_full)
                tensors = {
                    "cachedK_cachedV": (logits_cached, torch.matmul(p_cached, v_cached)),
                    "fullK_cachedV": (logits_full, torch.matmul(p_full, v_cached)),
                    "cachedK_fullV": (logits_cached, torch.matmul(p_cached, v_full)),
                    "fullK_fullV": (logits_full, out_full),
                }
                for scale in scales:
                    k_pred = pred_k_probe.repeat_kv(pred_x_k_by_scale[scale][layer_idx], groups)
                    logits_pred = torch.matmul(q, k_pred.transpose(-1, -2)) / math.sqrt(q.shape[-1])
                    p_pred = torch.softmax(logits_pred, dim=-1)
                    tensors[f"predK_s{scale:g}_cachedV"] = (logits_pred, torch.matmul(p_pred, v_cached))
                for name, (logits, out) in tensors.items():
                    add_metric(totals[name], logits, out, logits_full, out_full)
                    add_metric(layer_totals[layer_idx][name], logits, out, logits_full, out_full)
            processed += 1
            print(json.dumps({"split": tag, "sample": sample, "processed": processed}), flush=True)
        return processed, totals, layer_totals

    calib_n, calib_totals, calib_layer_totals = eval_range(args.calib_start, args.calib_count, "calib")
    test_n, test_totals, test_layer_totals = eval_range(args.test_start, args.test_count, "test")

    global_best = min(scales, key=lambda s: rel(calib_totals[f"predK_s{s:g}_cachedV"])["output_rel_error"])
    layer_best = []
    for layer_idx in range(cfg.num_hidden_layers):
        best = min(scales, key=lambda s: rel(calib_layer_totals[layer_idx][f"predK_s{s:g}_cachedV"])["output_rel_error"])
        layer_best.append(best)

    layer_gate_total = {key: 0.0 for key in ("logit_err", "logit_base", "out_err", "out_base")}
    for layer_idx, scale in enumerate(layer_best):
        src = test_layer_totals[layer_idx][f"predK_s{scale:g}_cachedV"]
        for key in layer_gate_total:
            layer_gate_total[key] += src[key]

    summary = {
        "calib": summarize_totals(calib_totals),
        "test": summarize_totals(test_totals),
        "test_selected": {
            "predK_global_calib_best": {"scale": global_best, **rel(test_totals[f"predK_s{global_best:g}_cachedV"])},
            "predK_layerwise_calib_best": {"layer_scales": layer_best, **rel(layer_gate_total)},
        },
    }
    base = summary["test"]["cachedK_cachedV"]["output_rel_error"]
    oracle_k = summary["test"]["fullK_cachedV"]["output_rel_error"]
    for item in summary["test_selected"].values():
        item["recovery_vs_cached_to_fullK"] = (base - item["output_rel_error"]) / max(base - oracle_k, 1e-30)

    for handle in handles:
        handle.remove()

    result = {
        "task": "calibrated gates for functional predicted-K adapter",
        "train_dir": args.train_dir,
        "train_count": len(train_paths),
        "train_k_predictor_mean_explained": sum(r["train_explained"] for r in train_rows) / len(train_rows),
        "rank": args.rank,
        "alpha": args.alpha,
        "scales": scales,
        "calib_start": args.calib_start,
        "calib_count_requested": args.calib_count,
        "calib_processed": calib_n,
        "test_start": args.test_start,
        "test_count_requested": args.test_count,
        "test_processed": test_n,
        "summary": summary,
        "notes": "Global/layer-wise scales are chosen only on calibration split and then evaluated on separate test split. Full K/V/hidden are teacher references only.",
    }
    Path(args.output_json).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
