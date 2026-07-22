#!/usr/bin/env python3
import argparse
import csv
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class TeacherCacheDataset(Dataset):
    def __init__(self, cache_dir, limit=0):
        self.paths = sorted(Path(cache_dir).glob("pair*.npz"))
        if limit and limit > 0:
            self.paths = self.paths[: int(limit)]
        if not self.paths:
            raise FileNotFoundError(f"no pair*.npz under {cache_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        return {
            "pair_id": int(data["pair_id"]),
            "doc_ids": data["doc_ids"].astype(np.int64),
            "query_ids": data["query_ids"].astype(np.int64),
            "teacher_scores": data["teacher_scores"].astype(np.float32),
        }


def collate(batch):
    max_doc = max(len(x["doc_ids"]) for x in batch)
    max_query = max(len(x["query_ids"]) for x in batch)
    bsz = len(batch)
    input_ids = torch.zeros(bsz, max_doc + max_query, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    teacher_scores = torch.zeros(bsz, max_doc, dtype=torch.float32)
    doc_mask = torch.zeros(bsz, max_doc, dtype=torch.bool)
    doc_lens = torch.zeros(bsz, dtype=torch.long)
    query_lens = torch.zeros(bsz, dtype=torch.long)
    pair_ids = []
    for i, item in enumerate(batch):
        d = torch.from_numpy(item["doc_ids"])
        q = torch.from_numpy(item["query_ids"])
        s = torch.from_numpy(item["teacher_scores"])
        input_ids[i, : len(d)] = d
        input_ids[i, max_doc : max_doc + len(q)] = q
        attention_mask[i, : len(d)] = True
        attention_mask[i, max_doc : max_doc + len(q)] = True
        teacher_scores[i, : len(d)] = s
        doc_mask[i, : len(d)] = True
        doc_lens[i] = len(d)
        query_lens[i] = len(q)
        pair_ids.append(item["pair_id"])
    return {
        "pair_ids": pair_ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "teacher_scores": teacher_scores,
        "doc_mask": doc_mask,
        "doc_lens": doc_lens,
        "query_lens": query_lens,
        "max_doc": max_doc,
    }


class NativeLayer4AttentionSelector(nn.Module):
    def __init__(self, model_path, layers=4, dtype=torch.bfloat16, score_mode="attn_prob"):
        super().__init__()
        if score_mode not in {"attn_prob", "qk_logits", "hidden_head"}:
            raise ValueError(f"unsupported score_mode: {score_mode}")
        self.score_mode = score_mode
        base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.config = base.config
        self.embed_tokens = base.model.embed_tokens
        self.layers = nn.ModuleList(list(base.model.layers[: int(layers)]))
        self.rotary_emb = getattr(base.model, "rotary_emb", None)
        self.num_heads = int(base.config.num_attention_heads)
        self.num_kv_heads = int(base.config.num_key_value_heads)
        self.head_dim = int(base.config.hidden_size // base.config.num_attention_heads)
        hidden_size = int(base.config.hidden_size)
        self.score_doc_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.score_query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.score_out = nn.Linear(hidden_size, 1, bias=False)
        del base

    def forward(self, input_ids, attention_mask, doc_lens):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            use_global_rope = True
        else:
            cos = sin = None
            use_global_rope = False

        key_positions = torch.arange(seq_len, device=device).unsqueeze(0)
        query_start = int(doc_lens.max().item())
        query_positions = torch.arange(query_start, seq_len, device=device).unsqueeze(1)
        causal_mask = key_positions > query_positions
        pad_mask = ~attention_mask.unsqueeze(1).unsqueeze(1)
        doc_scores = None

        for layer_idx, layer in enumerate(self.layers):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            q_len = normed.shape[1]
            query_states = layer.self_attn.q_proj(normed)
            key_states = layer.self_attn.k_proj(normed)
            value_states = layer.self_attn.v_proj(normed)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            if not use_global_rope:
                cos, sin = layer.self_attn.rotary_emb(value_states, position_ids)
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            query_states = (query_states * cos) + (rotate_half(query_states) * sin)
            key_states = (key_states * cos) + (rotate_half(key_states) * sin)
            n_rep = self.num_heads // self.num_kv_heads
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)

            if layer_idx == len(self.layers) - 1:
                query_subset = query_states[:, :, query_start:, :]
                attn_logits = torch.matmul(query_subset.float(), key_states.float().transpose(2, 3)) / math.sqrt(self.head_dim)
                attn_logits = attn_logits.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
                attn_logits = attn_logits.masked_fill(pad_mask, float("-inf"))
                if self.score_mode == "qk_logits":
                    doc_scores = attn_logits[:, :, :, :query_start].mean(dim=(1, 2))
                attn = F.softmax(attn_logits, dim=-1)
                if self.score_mode == "attn_prob":
                    doc_scores = attn[:, :, :, :query_start].mean(dim=(1, 2))
                attn_output_subset = torch.matmul(attn.to(value_states.dtype), value_states)
                attn_output_prefix = F.scaled_dot_product_attention(
                    query_states[:, :, :query_start, :],
                    key_states[:, :, :query_start, :],
                    value_states[:, :, :query_start, :],
                    is_causal=True,
                )
                attn_output = torch.cat([attn_output_prefix, attn_output_subset], dim=2)
            else:
                attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True)

            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            hidden_states = residual + layer.self_attn.o_proj(attn_output)
            residual = hidden_states
            hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))

        if self.score_mode == "hidden_head":
            query_hidden = hidden_states[:, query_start:, :].float()
            query_mask = attention_mask[:, query_start:].to(query_hidden.dtype).unsqueeze(-1)
            query_pool = (query_hidden * query_mask).sum(dim=1) / query_mask.sum(dim=1).clamp_min(1.0)
            doc_hidden = hidden_states[:, :query_start, :].float()
            fused = torch.tanh(self.score_doc_proj(doc_hidden) + self.score_query_proj(query_pool).unsqueeze(1))
            doc_scores = self.score_out(fused).squeeze(-1).float()

        return doc_scores


def masked_kl(student_scores, teacher_scores, doc_mask, temperature):
    neg_inf = torch.finfo(student_scores.dtype).min
    s_logp = F.log_softmax(student_scores.masked_fill(~doc_mask, neg_inf) / temperature, dim=-1)
    t_prob = F.softmax(teacher_scores.masked_fill(~doc_mask, neg_inf) / temperature, dim=-1)
    return F.kl_div(s_logp, t_prob, reduction="batchmean") * (temperature * temperature)


@torch.no_grad()
def eval_metrics(model, dataloader, accelerator, ratios, temperature):
    model.eval()
    totals = {"kl": 0.0, "n": 0}
    ratio_stats = {r: {"recall": 0.0, "jaccard": 0.0} for r in ratios}
    for batch in dataloader:
        scores = model(batch["input_ids"], batch["attention_mask"], batch["doc_lens"])
        doc_mask = batch["doc_mask"]
        kl = masked_kl(scores, batch["teacher_scores"], doc_mask, temperature)
        n = scores.shape[0]
        totals["kl"] += float(accelerator.gather_for_metrics(kl.detach()).mean().item()) * n
        totals["n"] += n
        for i in range(n):
            dlen = int(batch["doc_lens"][i].item())
            for ratio in ratios:
                k = max(1, int(dlen * ratio))
                pred = torch.topk(scores[i, :dlen], k).indices
                teacher = torch.topk(batch["teacher_scores"][i, :dlen], k).indices
                ps = set(pred.detach().cpu().tolist())
                ts = set(teacher.detach().cpu().tolist())
                inter = len(ps & ts)
                ratio_stats[ratio]["recall"] += inter / max(1, len(ts))
                ratio_stats[ratio]["jaccard"] += inter / max(1, len(ps | ts))
    out = {"val_kl": totals["kl"] / max(1, totals["n"])}
    for ratio in ratios:
        tag = str(ratio).replace(".", "p")
        out[f"val_recall_r{tag}"] = ratio_stats[ratio]["recall"] / max(1, totals["n"])
        out[f"val_jaccard_r{tag}"] = ratio_stats[ratio]["jaccard"] / max(1, totals["n"])
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--val-cache-dir", required=True)
    parser.add_argument("--musique-val-cache-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--score-mode", choices=["attn_prob", "qk_logits", "hidden_head"], default="attn_prob")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--head-lr", type=float, default=0.0, help="Optional scorer head learning rate; <=0 uses --lr.")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--eval-ratios", default="0.05,0.10,0.15,0.30")
    parser.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--resume-from", default=None, help="Checkpoint file or directory to resume from.")
    parser.add_argument("--save-every-epoch", action="store_true", help="Keep one checkpoint file per epoch.")
    parser.add_argument("--log-every", type=int, default=50, help="Progress bar update interval in steps.")
    args = parser.parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    ratios = [float(x) for x in args.eval_ratios.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TeacherCacheDataset(args.cache_dir, args.train_limit)
    val_ds = TeacherCacheDataset(args.val_cache_dir, args.val_limit)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2)
    musique_dl = None
    if args.musique_val_cache_dir:
        musique_ds = TeacherCacheDataset(args.musique_val_cache_dir, 0)
        musique_dl = DataLoader(musique_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2)

    model = NativeLayer4AttentionSelector(args.model_path, args.layers, score_mode=args.score_mode)
    if args.score_mode == "hidden_head" and args.head_lr > 0:
        head_param_ids = {id(p) for m in (model.score_doc_proj, model.score_query_proj, model.score_out) for p in m.parameters()}
        head_params = [p for p in model.parameters() if id(p) in head_param_ids]
        backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params, "lr": args.head_lr},
        ], weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    history = []
    start_epoch = 0
    resume_path = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.is_dir():
            resume_path = resume_path / "training_state_latest.pt"
            if not resume_path.exists():
                resume_path = Path(args.resume_from) / "native_layer4_selector.pt"
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        state = torch.load(resume_path, map_location="cpu")
        model.embed_tokens.load_state_dict(state["embed_tokens"], strict=True)
        model.layers.load_state_dict(state["layers"], strict=True)
        if "score_head" in state:
            missing, unexpected = model.load_state_dict(state["score_head"], strict=False)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        history = list(state.get("history", []))
        if history:
            start_epoch = max(int(row.get("epoch", 0)) for row in history)
        else:
            start_epoch = int(state.get("epoch", 0))
        if accelerator.is_main_process:
            print(f"Resumed from {resume_path}; start_epoch={start_epoch}; optimizer={'optimizer' in state}", flush=True)

    model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)
    if musique_dl is not None:
        musique_dl = accelerator.prepare(musique_dl)

    if start_epoch == 0 and not history:
        before = eval_metrics(model, val_dl, accelerator, ratios, args.temperature)
        before = {f"wiki_{k}": v for k, v in before.items()}
        if musique_dl is not None:
            before.update({f"musique_{k}": v for k, v in eval_metrics(model, musique_dl, accelerator, ratios, args.temperature).items()})
        before.update({"epoch": 0, "train_loss": 0.0})
        history.append(before)
        if accelerator.is_main_process:
            print(before, flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        seen = 0
        progress = None
        if accelerator.is_main_process and tqdm is not None:
            progress = tqdm(total=len(train_dl), desc=f"epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        for step, batch in enumerate(train_dl, start=1):
            scores = model(batch["input_ids"], batch["attention_mask"], batch["doc_lens"])
            loss = masked_kl(scores, batch["teacher_scores"], batch["doc_mask"], args.temperature)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach().item()) * scores.shape[0]
            seen += scores.shape[0]
            if progress is not None:
                if step % max(1, args.log_every) == 0 or step == len(train_dl):
                    progress.set_postfix(loss=f"{running / max(1, seen):.6f}")
                progress.update(1)
        if progress is not None:
            progress.close()
        metrics = eval_metrics(model, val_dl, accelerator, ratios, args.temperature)
        metrics = {f"wiki_{k}": v for k, v in metrics.items()}
        if musique_dl is not None:
            metrics.update({f"musique_{k}": v for k, v in eval_metrics(model, musique_dl, accelerator, ratios, args.temperature).items()})
        metrics.update({"epoch": epoch + 1, "train_loss": running / max(1, seen)})
        history.append(metrics)
        if accelerator.is_main_process:
            print(metrics, flush=True)
            unwrapped = accelerator.unwrap_model(model)
            raw_optimizer = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
            state = {
                "embed_tokens": unwrapped.embed_tokens.state_dict(),
                "layers": unwrapped.layers.state_dict(),
                "score_head": unwrapped.state_dict(),
                "optimizer": raw_optimizer.state_dict(),
                "epoch": epoch + 1,
                "args": vars(args),
                "history": history,
            }
            latest_path = out_dir / "training_state_latest.pt"
            torch.save(state, latest_path)
            if args.save_every_epoch:
                torch.save(state, out_dir / f"training_state_epoch{epoch + 1:03d}.pt")

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(
            {
                "embed_tokens": unwrapped.embed_tokens.state_dict(),
                "layers": unwrapped.layers.state_dict(),
                "score_head": unwrapped.state_dict(),
                "optimizer": (optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer).state_dict(),
                "epoch": args.epochs,
                "args": vars(args),
                "history": history,
            },
            out_dir / "native_layer4_selector.pt",
        )
        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            fields = sorted({k for row in history for k in row})
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)


if __name__ == "__main__":
    main()
