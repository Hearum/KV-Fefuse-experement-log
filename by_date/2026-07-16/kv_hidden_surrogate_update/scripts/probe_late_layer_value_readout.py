#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path('/raid/home/hming/FusionRAG-pca-analysis')
if not ROOT.exists():
    ROOT = Path('/home/hming/FusionRAG-pca-analysis')
sys.path.insert(0, str(ROOT))

from transformers import AutoConfig, AutoTokenizer
from ktransformers.models.custom_cache import StaticCache
from ktransformers.unified_process_cache import load_model
from ktransformers.util.utils import prepare_data, rotate_half


def parse_layers(spec: str, max_layers: int) -> list[int]:
    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    out = sorted(set(x for x in out if 0 <= x < max_layers))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Probe whether cheap sparse Value readout explains late-layer hidden/KV delta.')
    p.add_argument('--model-path', default='/mnt/qjhs-sh-lab-01/models/Qwen3-32B')
    p.add_argument('--bge-model-path', default='/mnt/qjhs-sh-lab-01/models/bge-m3')
    p.add_argument('--cache-root', default='/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2')
    p.add_argument('--data-path', default=str(ROOT / 'MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/data'))
    p.add_argument('--data-name', default='musique-v2.jsonl')
    p.add_argument('--output-dir', default=str(ROOT / 'MOTIVATION_EXPERIMENTS/by_date/2026-07-16/kv_hidden_surrogate_update'))
    p.add_argument('--start-example', type=int, default=0)
    p.add_argument('--max-examples', type=int, default=1)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-cache-len', type=int, default=32768)
    p.add_argument('--max-full-seq-len', type=int, default=20000)
    p.add_argument('--topk', type=int, default=10)
    p.add_argument('--revert-rope', action='store_true', default=True)
    p.add_argument('--layers', default='56-62', help='Layer l whose attention branch predicts x^{l+1}; layer 63 has no captured x^{64}.')
    p.add_argument('--top-ms', default='8,16,32,64')
    p.add_argument('--score-modes', default='uniform,recency,value_norm')
    p.add_argument('--max-target-tokens-per-chunk', type=int, default=64)
    p.add_argument('--seed', type=int, default=1)
    return p.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def capture_layer_inputs_with_cache(model, input_ids: torch.Tensor, cache: StaticCache | None, device: str, max_cache_len: int):
    captured = []
    hooks = []

    def make_hook(layer_idx):
        def hook(_module, inputs):
            captured.append((layer_idx, inputs[0].detach().to('cpu', dtype=torch.float32)))
        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_pre_hook(make_hook(layer_idx)))
    if cache is not None:
        for layer_idx in range(len(cache.key_cache)):
            cache.past_tokens[layer_idx] = 0
    try:
        with torch.no_grad():
            ids = input_ids.unsqueeze(0).to(device)
            if cache is None:
                model(input_ids=ids, use_cache=False, return_dict=False)
            else:
                pos = torch.arange(ids.shape[1], device=device)
                model(input_ids=ids, cache_position=pos, past_key_values=cache, use_cache=True, return_dict=False)
    finally:
        for h in hooks:
            h.remove()
    captured.sort(key=lambda x: x[0])
    hidden = torch.stack([h for _, h in captured])
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return hidden


def rotate_chunk_key(model, chunk_key_cache: torch.Tensor, offset: int, device: str) -> torch.Tensor:
    if offset == 0:
        return chunk_key_cache
    position_ids = torch.full((1, chunk_key_cache.shape[3]), int(offset), device=device)
    try:
        cos, sin = model.model.rotary_emb(chunk_key_cache[0].to(device), position_ids)
    except AttributeError:
        cos, sin = model.model.layers[0].self_attn.rotary_emb(chunk_key_cache[0].to(device), position_ids)
    cos = cos.to(chunk_key_cache.device).unsqueeze(1)
    sin = sin.to(chunk_key_cache.device).unsqueeze(1)
    return (chunk_key_cache * cos) + (rotate_half(chunk_key_cache) * sin)


def load_raw_combined_kv(model, raw_path: Path, ex_id: int, passages, system_len: int, device: str, revert_rope: bool):
    keys = []
    vals = []
    past_len = 0
    for chunk_id, passage in enumerate(passages[:-1]):
        k = torch.load(raw_path / f'{ex_id}_{chunk_id}_key.pt', map_location='cpu', weights_only=True)
        v = torch.load(raw_path / f'{ex_id}_{chunk_id}_value.pt', map_location='cpu', weights_only=True)
        if revert_rope and chunk_id > 0:
            k = rotate_chunk_key(model, k, past_len - system_len, device)
        keys.append(k)
        vals.append(v)
        past_len += int(passage.shape[0])
    return torch.cat(keys, dim=3), torch.cat(vals, dim=3)


def repeat_kv_tensor(v: torch.Tensor, n_rep: int) -> torch.Tensor:
    # [kv_heads, seq, dim] -> [heads, seq, dim]
    if n_rep == 1:
        return v
    return v.repeat_interleave(n_rep, dim=0)


def sample_targets(start: int, n: int, max_targets: int, rng: random.Random) -> list[int]:
    indices = list(range(start, start + n))
    if len(indices) <= max_targets:
        return indices
    return sorted(rng.sample(indices, max_targets))


def weights_for_prefix(values: torch.Tensor, target_abs: int, top_m: int, mode: str, tau: float = 1.0):
    # values: [heads, seq, head_dim], prefix is [:target_abs]
    if target_abs <= 0:
        return None, None
    prefix = values[:, :target_abs, :]
    seq = prefix.shape[1]
    m = min(top_m, seq)
    if mode == 'uniform':
        idx = torch.arange(seq - m, seq, dtype=torch.long)
        logits = torch.zeros(m, dtype=torch.float32)
    elif mode == 'recency':
        idx = torch.arange(seq - m, seq, dtype=torch.long)
        dist = torch.arange(m, 0, -1, dtype=torch.float32)
        logits = -torch.log1p(dist)
    elif mode == 'value_norm':
        score = prefix.float().pow(2).sum(dim=(0, 2))
        idx = torch.topk(score, k=m).indices.sort().values.cpu()
        logits = score[idx].float().cpu()
    else:
        raise ValueError(f'unknown score mode: {mode}')
    w = torch.softmax(logits / max(tau, 1e-6), dim=0)
    return idx, w


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    den = torch.linalg.norm(af) * torch.linalg.norm(bf)
    if float(den) == 0.0:
        return float('nan')
    return float(torch.dot(af, bf) / den)


def scalar_fit_metrics(feature: torch.Tensor, target: torch.Tensor):
    x = feature.float().flatten()
    y = target.float().flatten()
    xx = torch.dot(x, x)
    yy = torch.dot(y, y)
    if float(xx) == 0.0 or float(yy) == 0.0:
        return {'scalar': float('nan'), 'rel_l2': float('nan'), 'r2_zero': float('nan')}
    scalar = torch.dot(x, y) / xx
    pred = scalar * x
    err = torch.sum((pred - y) ** 2)
    rel = torch.sqrt(err / yy)
    r2 = 1.0 - float(err / yy)
    return {'scalar': float(scalar), 'rel_l2': float(rel), 'r2_zero': r2}


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    result_dir = out / 'results'
    fig_dir = out / 'figures'
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    _, tokens_data, _, _, _, raw_path_s, _, _, _, _, _, _ = prepare_data(
        'Qwen3-32B', args.data_path.rstrip('/') + '/', args.data_name,
        args.cache_root.rstrip('/') + '/', tokenizer, args.topk, args.revert_rope,
        False, args.bge_model_path,
    )
    raw_path = Path(raw_path_s)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config._attn_implementation = 'sdpa'
    model = load_model('qwen3', args.model_path, config).to(args.device)
    model.eval()
    num_layers = int(model.config.num_hidden_layers)
    layers = parse_layers(args.layers, num_layers - 1)
    top_ms = [int(x) for x in args.top_ms.split(',') if x.strip()]
    score_modes = [x.strip() for x in args.score_modes.split(',') if x.strip()]
    n_rep = int(model.config.num_attention_heads // model.config.num_key_value_heads)

    rows = []
    examples = []
    skipped = []
    end = min(len(tokens_data), args.start_example + args.max_examples)
    for ex_idx in range(args.start_example, end):
        ex_id = ex_idx + 1
        passages = tokens_data[ex_idx]
        system = passages[0]
        docs = passages[1:-1]
        full_ids = torch.cat([system] + docs)
        if int(full_ids.numel()) > args.max_full_seq_len:
            skipped.append({'example': ex_id, 'reason': 'full_seq_too_long', 'seq_len': int(full_ids.numel())})
            continue
        missing = []
        for chunk_id in range(0, len(passages) - 1):
            if not (raw_path / f'{ex_id}_{chunk_id}_key.pt').exists() or not (raw_path / f'{ex_id}_{chunk_id}_value.pt').exists():
                missing.append(chunk_id)
        if missing:
            skipped.append({'example': ex_id, 'reason': 'missing_raw_kv', 'chunks': missing[:5]})
            continue

        full_cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=args.max_cache_len, device=args.device, dtype=model.dtype, passage_len=args.max_cache_len)
        full_hidden = capture_layer_inputs_with_cache(model, full_ids, full_cache, args.device, args.max_cache_len)
        raw_key, raw_value = load_raw_combined_kv(model, raw_path, ex_id, passages, int(system.shape[0]), args.device, args.revert_rope)
        full_value = torch.stack([full_cache.value_cache[l][:, :, :full_ids.numel(), :].detach().cpu() for l in range(num_layers)])

        doc_start = int(system.shape[0])
        for chunk_id, doc in enumerate(docs, start=1):
            n = int(doc.shape[0])
            target_positions = sample_targets(doc_start, n, args.max_target_tokens_per_chunk, rng)
            raw_hidden_doc = capture_layer_inputs_with_cache(model, torch.cat([system, doc]), None, args.device, args.max_cache_len)
            raw_hidden_by_abs = {}
            for abs_pos in target_positions:
                local = abs_pos - doc_start
                raw_hidden_by_abs[abs_pos] = raw_hidden_doc[:, :, int(system.shape[0]) + local, :]

            for layer in layers:
                layer_module = model.model.layers[layer]
                raw_v_layer = repeat_kv_tensor(raw_value[layer, 0], n_rep)  # [heads, seq, dim]
                full_v_layer = repeat_kv_tensor(full_value[layer, 0], n_rep)
                for top_m in top_ms:
                    for mode in score_modes:
                        cos_hidden = []
                        fit_hidden_rel = []
                        fit_hidden_r2 = []
                        cos_value = []
                        fit_value_rel = []
                        fit_value_r2 = []
                        used = 0
                        for abs_pos in target_positions:
                            idx, w = weights_for_prefix(raw_v_layer, abs_pos, top_m, mode)
                            if idx is None:
                                continue
                            prefix_v = raw_v_layer[:, idx, :].float()  # [heads, m, dim]
                            z = torch.einsum('hmf,m->hf', prefix_v, w).reshape(1, 1, -1).to(args.device, dtype=model.dtype)
                            with torch.no_grad():
                                attn_feature = layer_module.self_attn.o_proj(z).detach().cpu().float()[0, 0]
                            # Attention branch at layer l contributes to x^{l+1}; compare with old->full next-layer hidden delta.
                            full_next = full_hidden[layer + 1, :, abs_pos, :].float()[0]
                            old_next = raw_hidden_by_abs[abs_pos][layer + 1].float()[0]
                            delta_next = full_next - old_next
                            cos_hidden.append(cosine(attn_feature, delta_next))
                            fm = scalar_fit_metrics(attn_feature, delta_next)
                            fit_hidden_rel.append(fm['rel_l2'])
                            fit_hidden_r2.append(fm['r2_zero'])

                            # Value branch sanity: compare readout vector with local full-old repeated value delta flattened.
                            raw_v_i = raw_v_layer[:, abs_pos, :].float().reshape(-1)
                            full_v_i = full_v_layer[:, abs_pos, :].float().reshape(-1)
                            delta_v = full_v_i - raw_v_i
                            z_flat = z.detach().cpu().float().reshape(-1)
                            cos_value.append(cosine(z_flat, delta_v))
                            vm = scalar_fit_metrics(z_flat, delta_v)
                            fit_value_rel.append(vm['rel_l2'])
                            fit_value_r2.append(vm['r2_zero'])
                            used += 1
                        if used:
                            def mean(xs):
                                vals = [x for x in xs if not math.isnan(float(x))]
                                return sum(vals) / len(vals) if vals else float('nan')
                            rows.append({
                                'example': ex_id,
                                'chunk_id': chunk_id,
                                'layer': layer,
                                'top_m': top_m,
                                'score_mode': mode,
                                'tokens': used,
                                'mean_cos_o_proj_to_delta_hidden_next': mean(cos_hidden),
                                'mean_scalar_fit_rel_l2_hidden_next': mean(fit_hidden_rel),
                                'mean_scalar_fit_r2_hidden_next': mean(fit_hidden_r2),
                                'mean_cos_readout_to_delta_value': mean(cos_value),
                                'mean_scalar_fit_rel_l2_value': mean(fit_value_rel),
                                'mean_scalar_fit_r2_value': mean(fit_value_r2),
                            })
            doc_start += n
            del raw_hidden_doc
            print(f'processed example={ex_id} chunk={chunk_id} sampled_tokens={len(target_positions)}', flush=True)
        examples.append(ex_id)
        del full_hidden, full_cache, raw_key, raw_value, full_value
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()

    write_csv(result_dir / 'probe_readout_predictability_rows.csv', rows)
    # Aggregate over examples/chunks.
    groups = {}
    for row in rows:
        key = (row['layer'], row['top_m'], row['score_mode'])
        slot = groups.setdefault(key, {'tokens': 0, 'rows': 0, 'metrics': {}})
        slot['tokens'] += int(row['tokens'])
        slot['rows'] += 1
        for k, v in row.items():
            if k.startswith('mean_'):
                slot['metrics'].setdefault(k, []).append(float(v))
    summary = []
    for (layer, top_m, mode), slot in sorted(groups.items()):
        outrow = {'layer': layer, 'top_m': top_m, 'score_mode': mode, 'rows': slot['rows'], 'tokens': slot['tokens']}
        for k, vals in slot['metrics'].items():
            vals = [x for x in vals if not math.isnan(x)]
            outrow[k] = sum(vals) / len(vals) if vals else float('nan')
        summary.append(outrow)
    write_csv(result_dir / 'probe_readout_predictability_summary.csv', summary)
    payload = {'args': vars(args), 'examples': examples, 'skipped': skipped, 'rows': len(rows), 'summary': summary}
    (result_dir / 'probe_readout_predictability.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'examples': examples, 'skipped': skipped, 'rows': len(rows), 'summary_head': summary[:10]}, indent=2))


if __name__ == '__main__':
    main()
