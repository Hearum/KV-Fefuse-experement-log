#!/usr/bin/env python3
"""Small-sample Delta-KV mechanism analysis for FusionRAG.

This script intentionally starts from a tiny slice: one or a few examples and
sub-questions. It compares document-token KV from:
  offline/raw: system + document
  full:        system + selected documents + query

The document span in the full prompt is recovered from exact token offsets.
"""

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


def _repo_imports(repo: Path):
    sys.path.insert(0, str(repo))
    from transformers import AutoConfig, AutoTokenizer
    from ktransformers.models.custom_cache import StaticCache
    from test_fusionrag_reflect_preprocess_exp import (
        PreprocessScope,
        RecallMethod,
        load_model,
        prepare_reflect_data,
    )

    return AutoConfig, AutoTokenizer, StaticCache, PreprocessScope, RecallMethod, load_model, prepare_reflect_data


def _question_tensor(tokenizer, model_type: str, query: str) -> torch.Tensor:
    is_qwen3_family = model_type in ("qwen3", "qwen3_moe", "qwen3moe")
    if is_qwen3_family:
        text = (
            f"<|im_end|>\n<|im_start|>user\nQuestion: {query}\n/no_think"
            f"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: "
        )
    else:
        text = f"<|im_end|>\n<|im_start|>user\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    return torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)


def _capture_kv(model, static_cache, input_ids: torch.Tensor, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    for layer_idx in range(len(static_cache.key_cache)):
        static_cache.past_tokens[layer_idx] = 0
    input_ids = input_ids.unsqueeze(0).to(device)
    seq_len = int(input_ids.shape[1])
    cache_position = torch.arange(seq_len, device=device)
    static_cache.cur_idx = cache_position
    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_ids).to(device)
        _ = model(
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            past_key_values=static_cache,
            return_dict=False,
            use_cache=True,
        )
    past_len = int(static_cache.past_tokens[0])
    keys = torch.stack([x[:, :, :past_len, :].detach().cpu() for x in static_cache.key_cache])
    values = torch.stack([x[:, :, :past_len, :].detach().cpu() for x in static_cache.value_cache])
    return keys, values


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_position_delta(model, key: torch.Tensor, position_delta: int, device: str) -> torch.Tensor:
    """Apply the same relative RoPE shift used by FusionRAG revert_rope loading.

    key shape is [layers, batch, kv_heads, tokens, head_dim].
    """
    if position_delta == 0:
        return key.clone()
    shifted = []
    for layer_idx in range(key.shape[0]):
        layer_key = key[layer_idx].to(device)
        position_ids = torch.full((1, layer_key.shape[2]), int(position_delta), device=layer_key.device)
        try:
            cos, sin = model.model.rotary_emb(layer_key, position_ids)
        except AttributeError:
            cos, sin = model.model.layers[layer_idx].self_attn.rotary_emb(layer_key, position_ids)
        cos = cos.to(layer_key.device).unsqueeze(1)
        sin = sin.to(layer_key.device).unsqueeze(1)
        shifted.append(((layer_key * cos) + (_rotate_half(layer_key) * sin)).detach().cpu())
    return torch.stack(shifted)


def _energy_rank(s: torch.Tensor, frac: float) -> int:
    if s.numel() == 0:
        return 0
    energy = s.float().pow(2)
    total = energy.sum()
    if float(total) <= 0:
        return 0
    cum = torch.cumsum(energy, dim=0) / total
    return int(torch.searchsorted(cum, torch.tensor(frac)).item() + 1)


def _svd_metrics(delta_layer: torch.Tensor, base_layer: torch.Tensor, max_svd_rank: int) -> Dict[str, float]:
    # layer tensor shape: [batch=1, heads, tokens, head_dim].
    delta_matrix = delta_layer[0].permute(1, 0, 2).reshape(delta_layer.shape[2], -1).float()
    base_matrix = base_layer[0].permute(1, 0, 2).reshape(base_layer.shape[2], -1).float()
    if delta_matrix.numel() == 0:
        return {}
    s = torch.linalg.svdvals(delta_matrix)
    base_s = torch.linalg.svdvals(base_matrix)
    s_keep = s[:max_svd_rank].cpu().numpy()
    fro = torch.linalg.norm(delta_matrix).item()
    top = float(s[0].item()) if s.numel() else 0.0
    stable_rank = (fro * fro / (top * top)) if top > 0 else 0.0
    prob = s.float() / (s.float().sum() + 1e-12)
    effective_rank = float(torch.exp(-(prob * torch.log(prob + 1e-12)).sum()).item())
    base_fro = torch.linalg.norm(base_matrix).item()
    base_top = float(base_s[0].item()) if base_s.numel() else 0.0
    base_stable_rank = (base_fro * base_fro / (base_top * base_top)) if base_top > 0 else 0.0
    return {
        "rank90": _energy_rank(s, 0.90),
        "rank95": _energy_rank(s, 0.95),
        "rank99": _energy_rank(s, 0.99),
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
        "base_stable_rank": base_stable_rank,
        "base_rank90": _energy_rank(base_s, 0.90),
        "base_rank95": _energy_rank(base_s, 0.95),
        "singular_values": s_keep.tolist(),
    }


def _basic_metrics(base: torch.Tensor, full: torch.Tensor, max_svd_rank: int) -> Tuple[Dict, List[Dict], List[Dict], List[Dict]]:
    delta = full.float() - base.float()
    base_f = base.float()
    flat_delta = delta.reshape(-1)
    flat_base = base_f.reshape(-1)
    diff_norm = torch.linalg.norm(flat_delta).item()
    base_norm = torch.linalg.norm(flat_base).item()
    cosine = torch.nn.functional.cosine_similarity(full.float().reshape(1, -1), base_f.reshape(1, -1)).item()
    rms = float(torch.sqrt(torch.mean(flat_delta.pow(2))).item())
    summary = {
        "shape": list(base.shape),
        "diff_l2": diff_norm,
        "base_l2": base_norm,
        "relative_l2": diff_norm / (base_norm + 1e-12),
        "cosine_full_vs_offline": cosine,
        "delta_abs_mean": float(flat_delta.abs().mean().item()),
        "delta_abs_p50": float(torch.quantile(flat_delta.abs(), 0.50).item()),
        "delta_abs_p90": float(torch.quantile(flat_delta.abs(), 0.90).item()),
        "delta_abs_p99": float(torch.quantile(flat_delta.abs(), 0.99).item()),
        "near_zero_abs_1e-6": float((flat_delta.abs() < 1e-6).float().mean().item()),
        "near_zero_abs_1pct_rms": float((flat_delta.abs() < 0.01 * rms).float().mean().item()),
    }

    layer_rows, head_rows, token_rows = [], [], []
    total_delta_sq = float(delta.pow(2).sum().item()) + 1e-12
    for layer_idx in range(delta.shape[0]):
        d_l = delta[layer_idx]
        b_l = base_f[layer_idx]
        layer_diff = torch.linalg.norm(d_l.reshape(-1)).item()
        layer_base = torch.linalg.norm(b_l.reshape(-1)).item()
        svd = _svd_metrics(d_l, b_l, max_svd_rank)
        row = {
            "layer": layer_idx,
            "diff_l2": layer_diff,
            "base_l2": layer_base,
            "relative_l2": layer_diff / (layer_base + 1e-12),
            "energy_share": float(d_l.pow(2).sum().item() / total_delta_sq),
        }
        row.update({k: v for k, v in svd.items() if k != "singular_values"})
        row["singular_values"] = svd.get("singular_values", [])
        layer_rows.append(row)

        for head_idx in range(delta.shape[2]):
            d_h = d_l[:, head_idx, :, :]
            b_h = b_l[:, head_idx, :, :]
            head_rows.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "diff_l2": torch.linalg.norm(d_h.reshape(-1)).item(),
                    "base_l2": torch.linalg.norm(b_h.reshape(-1)).item(),
                    "relative_l2": torch.linalg.norm(d_h.reshape(-1)).item()
                    / (torch.linalg.norm(b_h.reshape(-1)).item() + 1e-12),
                    "energy_share": float(d_h.pow(2).sum().item() / total_delta_sq),
                }
            )

    token_delta = torch.linalg.norm(delta.permute(3, 0, 1, 2, 4).reshape(delta.shape[3], -1), dim=1)
    token_base = torch.linalg.norm(base_f.permute(3, 0, 1, 2, 4).reshape(delta.shape[3], -1), dim=1)
    token_energy = token_delta.pow(2)
    token_total = float(token_energy.sum().item()) + 1e-12
    for token_idx in range(delta.shape[3]):
        token_rows.append(
            {
                "token": token_idx,
                "all_layer_delta_l2": float(token_delta[token_idx].item()),
                "all_layer_base_l2": float(token_base[token_idx].item()),
                "all_layer_relative_l2": float(token_delta[token_idx].item() / (token_base[token_idx].item() + 1e-12)),
                "all_layer_energy_share": float(token_energy[token_idx].item() / token_total),
            }
        )
    return summary, layer_rows, head_rows, token_rows


def _write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_spectrum(path: Path, rows: List[Dict], title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for row in rows:
        s = row.get("singular_values") or []
        if s:
            x = np.arange(1, len(s) + 1)
            plt.semilogy(x, np.asarray(s), marker="o", markersize=3, linewidth=1, label=f"layer {row['layer']}")
    plt.xlabel("singular value index")
    plt.ylabel("singular value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _append_log(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/raid/home/hming/FusionRAG-pca-analysis")
    parser.add_argument("--model_path", default="/mnt/qjhs-sh-lab-01/models/Qwen3-32B")
    parser.add_argument("--model_type", default="qwen3")
    parser.add_argument("--model_name", default="Qwen3-32B")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--bge_model_path", default="/mnt/qjhs-sh-lab-01/models/bge-m3")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--end_sample", type=int, default=1)
    parser.add_argument("--max_subquestions", type=int, default=1)
    parser.add_argument("--max_docs_per_subquestion", type=int, default=2)
    parser.add_argument("--max_cache_len", type=int, default=8192)
    parser.add_argument("--max_svd_rank", type=int, default=64)
    parser.add_argument("--include_rope_aligned_key", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo)
    task_dir = repo / "MOTIVATION_EXPERIMENTS" / "kv_lora"
    output_dir = Path(args.output_dir) if args.output_dir else task_dir
    data_path = Path(args.data_path) if args.data_path else repo / "data" / "result_reflect.json"
    AutoConfig, AutoTokenizer, StaticCache, PreprocessScope, RecallMethod, load_model, prepare_reflect_data = _repo_imports(repo)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    model, _ = load_model(args.model_type, args.model_path, config, args.device, False)
    model.eval()

    questions_data, system_tensor, _, _ = prepare_reflect_data(
        str(data_path),
        tokenizer,
        args.bge_model_path,
        args.model_type,
        topk=1,
        max_main_questions=args.end_sample,
        preprocess=False,
        recall_method=RecallMethod.BGE,
        preprocess_scope=PreprocessScope.GLOBAL,
    )

    static_cache = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=args.max_cache_len,
        device=args.device,
        dtype=model.dtype,
        passage_len=args.max_cache_len,
        model=model,
    )

    records = []
    layer_rows_all, head_rows_all, token_rows_all = [], [], []
    manifest = {
        "model_path": args.model_path,
        "model_name": args.model_name,
        "data_path": str(data_path),
        "start_sample": args.start_sample,
        "end_sample": args.end_sample,
        "max_subquestions": args.max_subquestions,
        "max_docs_per_subquestion": args.max_docs_per_subquestion,
        "kv_shape_order": "[layers, batch, kv_heads, tokens, head_dim]",
        "offline_definition": "forward(system + document), slice document span after system tokens",
        "full_definition": "forward(system + selected_documents + query), slice exact document offsets",
    }

    for example_id in range(args.start_sample, min(args.end_sample, len(questions_data))):
        q_data = questions_data[example_id]
        if not q_data.get("should_test", True):
            continue
        doc_tensors = q_data["doc_tensors"]
        for sub_q_idx, sub_q in enumerate(q_data["sub_questions"][: args.max_subquestions]):
            doc_chunk_ids = sub_q["chunk_ids"][: args.max_docs_per_subquestion]
            selected_docs = [doc_tensors[cid - 1] for cid in doc_chunk_ids]
            question_tensor = _question_tensor(tokenizer, args.model_type, sub_q["query"])
            full_input = torch.cat([system_tensor] + selected_docs + [question_tensor])
            if int(full_input.shape[0]) > args.max_cache_len:
                raise ValueError(f"full prompt length {full_input.shape[0]} exceeds max_cache_len={args.max_cache_len}")
            full_k, full_v = _capture_kv(model, static_cache, full_input, args.device)
            offset = int(system_tensor.shape[0])
            for doc_rank, (chunk_id, doc_tensor) in enumerate(zip(doc_chunk_ids, selected_docs)):
                offline_input = torch.cat([system_tensor, doc_tensor])
                if int(offline_input.shape[0]) > args.max_cache_len:
                    raise ValueError(f"offline prompt length {offline_input.shape[0]} exceeds max_cache_len={args.max_cache_len}")
                off_k, off_v = _capture_kv(model, static_cache, offline_input, args.device)
                doc_len = int(doc_tensor.shape[0])
                off_slice = slice(int(system_tensor.shape[0]), int(system_tensor.shape[0]) + doc_len)
                full_slice = slice(offset, offset + doc_len)
                comparisons = [("key", off_k, full_k), ("value", off_v, full_v)]
                if args.include_rope_aligned_key:
                    aligned_key = off_k.clone()
                    aligned_key[:, :, :, off_slice, :] = _apply_rope_position_delta(
                        model,
                        off_k[:, :, :, off_slice, :],
                        int(offset - system_tensor.shape[0]),
                        args.device,
                    )
                    comparisons.append(("key_rope_aligned", aligned_key, full_k))
                for kind, off_all, full_all in comparisons:
                    base = off_all[:, :, :, off_slice, :]
                    full = full_all[:, :, :, full_slice, :]
                    summary, layer_rows, head_rows, token_rows = _basic_metrics(base, full, args.max_svd_rank)
                    rec_prefix = {
                        "example_id": example_id,
                        "sub_q_idx": sub_q_idx,
                        "doc_rank": doc_rank,
                        "chunk_id": chunk_id,
                        "kind": kind,
                        "doc_len": doc_len,
                        "full_prompt_len": int(full_input.shape[0]),
                    }
                    records.append({**rec_prefix, **{k: v for k, v in summary.items() if k != "shape"}, "shape": json.dumps(summary["shape"])})
                    for row in layer_rows:
                        layer_rows_all.append({**rec_prefix, **row, "singular_values": json.dumps(row.get("singular_values", []))})
                    for row in head_rows:
                        head_rows_all.append({**rec_prefix, **row})
                    for row in token_rows:
                        token_rows_all.append({**rec_prefix, **row})
                offset += doc_len

    result_dir = output_dir / "results"
    figure_dir = output_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    rope_tag = "_ropekey" if args.include_rope_aligned_key else ""
    tag = f"{args.model_name}_ex{args.start_sample}_{args.end_sample}_sub{args.max_subquestions}_docs{args.max_docs_per_subquestion}{rope_tag}"
    _write_csv(result_dir / f"delta_kv_summary_{tag}.csv", records)
    _write_csv(result_dir / f"delta_kv_layer_metrics_{tag}.csv", layer_rows_all)
    _write_csv(result_dir / f"delta_kv_head_metrics_{tag}.csv", head_rows_all)
    _write_csv(result_dir / f"delta_kv_token_metrics_{tag}.csv", token_rows_all)
    with (result_dir / f"delta_kv_manifest_{tag}.json").open("w", encoding="utf-8") as f:
        json.dump({**manifest, "num_records": len(records)}, f, ensure_ascii=False, indent=2)

    # Plot spectra for first key/value record and the max-energy layer for each.
    plot_rows = []
    for kind in ("key", "value"):
        rows = [r for r in layer_rows_all if r["kind"] == kind]
        if rows:
            best = max(rows, key=lambda r: float(r["diff_l2"]))
            best = dict(best)
            best["singular_values"] = json.loads(best["singular_values"])
            plot_rows.append(best)
    _plot_spectrum(figure_dir / f"singular_spectrum_sanity_{tag}.png", plot_rows, "Delta-KV singular spectrum sanity")

    _append_log(
        output_dir / "EXPERIMENT_LOG.md",
        f"""
## Sanity run: {tag}

- 命令由 `scripts/analyze_delta_kv_sanity.py` 生成，范围 `example_id=[{args.start_sample}, {args.end_sample})`，每个问题最多 `{args.max_subquestions}` 个 sub-question、每个 sub-question 最多 `{args.max_docs_per_subquestion}` 个 doc。
- 模型：`{args.model_path}`；数据：`{data_path}`；device：`{args.device}`。
- KV shape 顺序确认：`[layers, batch, kv_heads, tokens, head_dim]`。document token 维度为第 3 维；切片方式为 offline `system_len:system_len+doc_len`，full 按 `system + docs + query` 的累计 offset。
- 输出：`results/delta_kv_summary_{tag}.csv`、`results/delta_kv_layer_metrics_{tag}.csv`、`results/delta_kv_head_metrics_{tag}.csv`、`results/delta_kv_token_metrics_{tag}.csv`、`figures/singular_spectrum_sanity_{tag}.png`。
""",
    )
    print(json.dumps({"tag": tag, "records": len(records), "result_dir": str(result_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
