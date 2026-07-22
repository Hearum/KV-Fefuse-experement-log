#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', default='/mnt/qjhs-sh-lab-01/models/Qwen3-32B')
    p.add_argument('--bge-model-path', default='/mnt/qjhs-sh-lab-01/models/bge-m3')
    p.add_argument('--cache-root', default='/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2')
    p.add_argument('--data-path', default=str(ROOT / 'MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/data'))
    p.add_argument('--data-name', default='musique-v2.jsonl')
    p.add_argument('--output-dir', default=str(ROOT / 'MOTIVATION_EXPERIMENTS/by_date/2026-07-15/qwen3_musique_v2_hidden_state_gap'))
    p.add_argument('--max-examples', type=int, default=2)
    p.add_argument('--start-example', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-cache-len', type=int, default=32768)
    p.add_argument('--max-full-seq-len', type=int, default=20000)
    p.add_argument('--topk', type=int, default=10)
    p.add_argument('--revert-rope', action='store_true', default=True)
    p.add_argument('--save-hidden-tensors', action='store_true')
    return p.parse_args()


def find_group_and_index(corpus_lens, global_index: int):
    acc = 0
    for group_idx, n in enumerate(corpus_lens):
        if global_index < acc + n:
            return group_idx, global_index - acc
        acc += n
    raise IndexError(f'global corpus index out of range: {global_index}')


def empty_stats(num_layers: int):
    return {
        source: {
            'diff_sq': torch.zeros(num_layers, dtype=torch.float64),
            'full_sq': torch.zeros(num_layers, dtype=torch.float64),
            'source_sq': torch.zeros(num_layers, dtype=torch.float64),
            'dot': torch.zeros(num_layers, dtype=torch.float64),
        }
        for source in ('raw', 'preprocess')
    }


def add_hidden_stat(acc, full: torch.Tensor, source: torch.Tensor) -> None:
    f = full.float()
    s = source.float()
    d = f - s
    reduce_dims = tuple(range(1, f.ndim))
    acc['diff_sq'] += (d * d).sum(dim=reduce_dims).double().cpu()
    acc['full_sq'] += (f * f).sum(dim=reduce_dims).double().cpu()
    acc['source_sq'] += (s * s).sum(dim=reduce_dims).double().cpu()
    acc['dot'] += (f * s).sum(dim=reduce_dims).double().cpu()


def summarize(stats, processed_examples, processed_chunks, processed_tokens):
    global_rows = []
    layer_rows = []
    for source, acc in stats.items():
        total_diff = float(acc['diff_sq'].sum().item())
        total_full = float(acc['full_sq'].sum().item())
        total_source = float(acc['source_sq'].sum().item())
        total_dot = float(acc['dot'].sum().item())
        global_rows.append({
            'source': source,
            'kind': 'hidden_layer_input',
            'relative_l2': math.sqrt(total_diff / total_full) if total_full else float('nan'),
            'diff_norm': math.sqrt(total_diff),
            'full_norm': math.sqrt(total_full),
            'source_norm': math.sqrt(total_source),
            'cosine_source_full': total_dot / math.sqrt(total_source * total_full) if total_source and total_full else float('nan'),
            'num_examples': processed_examples,
            'num_chunks': processed_chunks,
            'num_tokens': processed_tokens,
        })
        for layer in range(len(acc['diff_sq'])):
            diff = float(acc['diff_sq'][layer].item())
            full = float(acc['full_sq'][layer].item())
            src = float(acc['source_sq'][layer].item())
            dot = float(acc['dot'][layer].item())
            layer_rows.append({
                'source': source,
                'kind': 'hidden_layer_input',
                'layer': layer,
                'relative_l2': math.sqrt(diff / full) if full else float('nan'),
                'diff_norm': math.sqrt(diff),
                'full_norm': math.sqrt(full),
                'source_norm': math.sqrt(src),
                'cosine_source_full': dot / math.sqrt(src * full) if src and full else float('nan'),
                'delta_energy_share': diff / total_diff if total_diff else float('nan'),
            })
    return global_rows, layer_rows


def write_csv(path: Path, rows) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot(layer_rows, fig_dir: Path) -> None:
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {'raw': '#4C78A8', 'preprocess': '#F58518'}
    for metric, ylabel, name in [
        ('relative_l2', 'Relative L2 to Full Hidden', 'hidden_layer_relative_l2.png'),
        ('delta_energy_share', 'Hidden Delta Energy Share', 'hidden_layer_energy_share.png'),
    ]:
        plt.figure(figsize=(9.5, 4.8))
        for source in ('raw', 'preprocess'):
            sub = sorted([r for r in layer_rows if r['source'] == source], key=lambda r: int(r['layer']))
            plt.plot([int(r['layer']) for r in sub], [float(r[metric]) for r in sub], label=source, color=colors[source], linewidth=2)
        plt.xlabel('Layer')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.25)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(fig_dir / name, dpi=180)
        plt.close()


def forward_with_layer_input_capture(model, forward_fn, device: str) -> torch.Tensor:
    captured = []
    hooks = []

    def make_hook(layer_idx):
        def hook(_module, inputs):
            hidden = inputs[0].detach().to('cpu', dtype=torch.bfloat16)
            captured.append((layer_idx, hidden))
        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_pre_hook(make_hook(layer_idx)))
    try:
        with torch.no_grad():
            forward_fn()
    finally:
        for hook in hooks:
            hook.remove()
    if not captured:
        raise RuntimeError('No layer input hidden states were captured by hooks')
    captured.sort(key=lambda x: x[0])
    hidden = torch.stack([h for _, h in captured])
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return hidden


def forward_hidden(model, input_ids: torch.Tensor, device: str) -> torch.Tensor:
    def run():
        return model(
            input_ids=input_ids.unsqueeze(0).to(device),
            use_cache=False,
            return_dict=False,
        )
    return forward_with_layer_input_capture(model, run, device)


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


def fill_cache_from_raw(model, cache, raw_path: Path, tokens_data, corpus_lens, example_idx: int, target_chunk_id: int, context_rank, system_len: int, device: str, revert_rope: bool):
    for layer_idx in range(len(cache.key_cache)):
        cache.past_tokens[layer_idx] = 0
    past_len = 0

    def copy_chunk(corpus_i_1based: int, c_id: int):
        nonlocal past_len
        key = torch.load(raw_path / f'{corpus_i_1based}_{c_id}_key.pt', map_location='cpu', weights_only=True)
        value = torch.load(raw_path / f'{corpus_i_1based}_{c_id}_value.pt', map_location='cpu', weights_only=True)
        length = int(value.shape[3])
        if revert_rope and c_id > 0:
            key = rotate_chunk_key(model, key, past_len - system_len, device)
        for layer_idx in range(len(cache.key_cache)):
            cache.key_cache[layer_idx].narrow(2, past_len, length).copy_(key[layer_idx].to(cache.key_cache[layer_idx].device))
            cache.value_cache[layer_idx].narrow(2, past_len, length).copy_(value[layer_idx].to(cache.value_cache[layer_idx].device))
            cache.past_tokens[layer_idx] += length
        past_len += length

    copy_chunk(example_idx + 1, 0)
    rank_offset = sum(corpus_lens[:example_idx]) + target_chunk_id - 1
    for corpus_id in context_rank[rank_offset]:
        corpus_i, c_id0 = find_group_and_index(corpus_lens, int(corpus_id))
        corpus_i_1based = corpus_i + 1
        c_id = c_id0 + 1
        if corpus_i == example_idx and c_id == target_chunk_id:
            continue
        copy_chunk(corpus_i_1based, c_id)
    return past_len


def preprocess_hidden_for_chunk(model, cache, raw_path, tokens_data, corpus_lens, ex_idx, chunk_id, context_rank, system_len, device, revert_rope):
    past_len = fill_cache_from_raw(model, cache, raw_path, tokens_data, corpus_lens, ex_idx, chunk_id, context_rank, system_len, device, revert_rope)
    doc_tokens = tokens_data[ex_idx][chunk_id].unsqueeze(0).to(device)
    cache_position = torch.arange(past_len, past_len + doc_tokens.shape[1], device=device)
    def run():
        return model(
            inputs_embeds=model.model.embed_tokens(doc_tokens),
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
            return_dict=False,
        )
    return forward_with_layer_input_capture(model, run, device)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    result_dir = out / 'results'
    hidden_dir = out / 'hidden_cache'
    result_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    _, tokens_data, _, _, _, raw_path_s, _, _, _, _, context_rank, corpus_lens = prepare_data(
        'Qwen3-32B', args.data_path.rstrip('/') + '/', args.data_name,
        args.cache_root.rstrip('/') + '/', tokenizer, args.topk, args.revert_rope,
        True, args.bge_model_path,
    )
    raw_path = Path(raw_path_s)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config._attn_implementation = 'sdpa'
    model = load_model('qwen3', args.model_path, config).to(args.device)
    model.eval()
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=args.max_cache_len, device=args.device, dtype=model.dtype, passage_len=args.max_cache_len)

    stats = empty_stats(model.config.num_hidden_layers)
    skipped = []
    processed_examples = 0
    processed_chunks = 0
    processed_tokens = 0
    end = min(len(tokens_data), args.start_example + args.max_examples)

    for ex_idx in range(args.start_example, end):
        passages = tokens_data[ex_idx]
        system = passages[0]
        docs = passages[1:-1]
        full_ids = torch.cat([system] + docs)
        if int(full_ids.numel()) > args.max_full_seq_len:
            skipped.append({'example': ex_idx + 1, 'reason': 'full_seq_too_long', 'seq_len': int(full_ids.numel())})
            continue
        missing = []
        for chunk_id in range(0, len(passages) - 1):
            if not (raw_path / f'{ex_idx + 1}_{chunk_id}_key.pt').exists() or not (raw_path / f'{ex_idx + 1}_{chunk_id}_value.pt').exists():
                missing.append(chunk_id)
        if missing:
            skipped.append({'example': ex_idx + 1, 'reason': 'missing_raw_kv', 'chunks': missing[:5]})
            continue

        full_hidden = forward_hidden(model, full_ids, args.device)
        start = int(system.shape[0])
        for chunk_id, doc in enumerate(docs, start=1):
            n = int(doc.shape[0])
            full_doc_hidden = full_hidden[:, :, start:start + n, :]
            raw_hidden = forward_hidden(model, torch.cat([system, doc]), args.device)[:, :, int(system.shape[0]):, :]
            pre_hidden = preprocess_hidden_for_chunk(
                model, cache, raw_path, tokens_data, corpus_lens, ex_idx, chunk_id, context_rank, int(system.shape[0]), args.device, args.revert_rope
            )
            add_hidden_stat(stats['raw'], full_doc_hidden, raw_hidden)
            add_hidden_stat(stats['preprocess'], full_doc_hidden, pre_hidden)
            if args.save_hidden_tensors:
                torch.save(raw_hidden, hidden_dir / f'{ex_idx + 1}_{chunk_id}_raw_hidden.pt')
                torch.save(pre_hidden, hidden_dir / f'{ex_idx + 1}_{chunk_id}_preprocess_hidden.pt')
                torch.save(full_doc_hidden, hidden_dir / f'{ex_idx + 1}_{chunk_id}_full_hidden.pt')
            start += n
            processed_chunks += 1
            processed_tokens += n
            print(f'processed example={ex_idx + 1} chunk={chunk_id} tokens={n}', flush=True)
            del full_doc_hidden, raw_hidden, pre_hidden
        processed_examples += 1
        del full_hidden
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()

    global_rows, layer_rows = summarize(stats, processed_examples, processed_chunks, processed_tokens)
    write_csv(result_dir / 'hidden_global_summary.csv', global_rows)
    write_csv(result_dir / 'hidden_layer_summary.csv', layer_rows)
    payload = {
        'args': vars(args),
        'raw_path': str(raw_path),
        'processed_examples': processed_examples,
        'processed_chunks': processed_chunks,
        'processed_tokens': processed_tokens,
        'skipped': skipped,
        'global_summary': global_rows,
        'top_layers_by_energy': {
            source: sorted([r for r in layer_rows if r['source'] == source], key=lambda r: float(r['delta_energy_share']), reverse=True)[:10]
            for source in ('raw', 'preprocess')
        },
    }
    (result_dir / 'hidden_summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    plot(layer_rows, out / 'figures')
    print(json.dumps(payload['global_summary'], indent=2))
    print(f'skipped={len(skipped)}')


if __name__ == '__main__':
    main()
