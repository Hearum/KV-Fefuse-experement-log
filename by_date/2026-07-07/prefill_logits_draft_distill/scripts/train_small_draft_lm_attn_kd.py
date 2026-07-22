#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


class BlockDataset(Dataset):
    def __init__(self, blocks):
        self.blocks = blocks
    def __len__(self):
        return int(self.blocks.shape[0])
    def __getitem__(self, idx):
        return self.blocks[idx]


def is_dist():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def rank0():
    return int(os.environ.get("RANK", "0")) == 0

def setup_dist():
    if is_dist() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank

def barrier():
    if dist.is_initialized():
        dist.barrier()


def build_token_cache(tokenizer, text_path, cache_path, seq_len, max_sequences):
    cache_path = Path(cache_path)
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    if cache_path.exists():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    eos = tokenizer.eos_token_id
    buf, blocks = [], []
    with Path(text_path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line, add_special_tokens=False)
            if not ids:
                continue
            buf.extend(ids + ([eos] if eos is not None else []))
            while len(buf) >= seq_len + 1:
                blocks.append(buf[: seq_len + 1])
                buf = buf[seq_len + 1:]
                if max_sequences and len(blocks) >= max_sequences:
                    break
            if max_sequences and len(blocks) >= max_sequences:
                break
    if not blocks:
        raise RuntimeError(f"no blocks built from {text_path}")
    tensor = torch.tensor(blocks, dtype=torch.long)
    torch.save(tensor, cache_path)
    meta_path.write_text(json.dumps({
        "text_path": str(text_path),
        "seq_len": seq_len,
        "num_sequences": int(tensor.shape[0]),
        "max_sequences": int(max_sequences),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def trim_to_first_layers(model, layers):
    model.model.layers = torch.nn.ModuleList(list(model.model.layers[: int(layers)]))
    model.config.num_hidden_layers = int(layers)
    if hasattr(model.model, "config"):
        model.model.config.num_hidden_layers = int(layers)
    return model


def logits_kd_loss(student_logits, teacher_logits, labels, temperature, kl_weight, ce_weight):
    s = student_logits[:, :-1, :].float()
    t = teacher_logits[:, :-1, :].float()
    y = labels[:, 1:].contiguous()
    temp = float(temperature)
    s_logp = F.log_softmax(s / temp, dim=-1)
    t_logp = F.log_softmax(t / temp, dim=-1)
    t_prob = t_logp.exp()
    kl = (t_prob * (t_logp - s_logp)).sum(dim=-1).mean() * (temp * temp)
    ce = F.cross_entropy(s.reshape(-1, s.shape[-1]), y.reshape(-1))
    return kl_weight * kl + ce_weight * ce, kl.detach(), ce.detach()


def mean_attention(attentions, mode, last_n):
    if mode == "last":
        selected = attentions[-1:]
    elif mode == "all":
        selected = attentions
    elif mode == "lastn":
        selected = attentions[-int(last_n):]
    else:
        raise ValueError(f"bad attention mode: {mode}")
    acc = None
    for a in selected:
        # [B, H, L, L] -> [B, L, L]
        m = a.float().mean(dim=1)
        acc = m if acc is None else acc + m
    return acc / max(1, len(selected))


def attention_row_kl(student_attn, teacher_attn, eps=1e-8):
    # Both [B, L, L] already causal attention probabilities. Compare each query row.
    # Skip position 0 because it has no meaningful history beyond itself.
    s = student_attn[:, 1:, :].clamp_min(eps)
    t = teacher_attn[:, 1:, :].clamp_min(eps)
    t = t / t.sum(dim=-1, keepdim=True).clamp_min(eps)
    s = s / s.sum(dim=-1, keepdim=True).clamp_min(eps)
    return (t * (t.log() - s.log())).sum(dim=-1).mean()


def save_checkpoint(out_dir, student, optimizer, step, epoch, args, tag="latest"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = student.module if hasattr(student, "module") else student
    torch.save({
        "step": int(step),
        "epoch": int(epoch),
        "args": vars(args),
        "student_state": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": raw.config.to_dict(),
    }, out / f"training_state_{tag}.pt")


def append_history(out_dir, row):
    with (Path(out_dir) / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-model", default="/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct")
    p.add_argument("--student-init-model", default="/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct")
    p.add_argument("--text-path", default="MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/data/wikitext103_train.txt")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-path", required=True)
    p.add_argument("--student-layers", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--max-sequences", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--ce-weight", type=float, default=0.1)
    p.add_argument("--attn-weight", type=float, default=1.0)
    p.add_argument("--teacher-attn-mode", choices=["last", "lastn", "all"], default="lastn")
    p.add_argument("--teacher-attn-last-n", type=int, default=18)
    p.add_argument("--student-attn-mode", choices=["last", "all"], default="last")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--resume-from", default=None)
    args = p.parse_args()

    local_rank = setup_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    if rank0():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if rank0():
        build_token_cache(tokenizer, args.text_path, args.cache_path, args.seq_len, args.max_sequences)
    barrier()
    blocks = torch.load(args.cache_path, map_location="cpu")
    ds = BlockDataset(blocks)
    sampler = DistributedSampler(ds, shuffle=True) if dist.is_initialized() else None
    dl = DataLoader(ds, batch_size=args.per_device_batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=2, pin_memory=True, drop_last=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=dtype, trust_remote_code=True, attn_implementation="eager").to(device).eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    student = AutoModelForCausalLM.from_pretrained(args.student_init_model, torch_dtype=dtype, trust_remote_code=True, attn_implementation="eager")
    student = trim_to_first_layers(student, args.student_layers).to(device).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    global_step = 0
    start_epoch = 0
    if args.resume_from:
        state = torch.load(args.resume_from, map_location="cpu")
        missing, unexpected = student.load_state_dict(state["student_state"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        global_step = int(state.get("step", 0))
        start_epoch = int(state.get("epoch", 0))
        if rank0():
            print(
                json.dumps(
                    {
                        "event": "resume",
                        "resume_from": args.resume_from,
                        "step": global_step,
                        "epoch": start_epoch,
                        "missing": len(missing),
                        "unexpected": len(unexpected),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if dist.is_initialized():
        student = DDP(student, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    total_updates = args.max_steps if args.max_steps > 0 else math.ceil(len(dl) * args.epochs / max(1, args.grad_accum))
    pbar = tqdm(total=total_updates, initial=global_step, desc="small-draft attn-kd", dynamic_ncols=True) if rank0() and tqdm is not None else None
    accum = 0
    sums = {"loss":0.0, "logits":0.0, "kl":0.0, "ce":0.0, "attn":0.0, "n":0}
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dl:
            ids = batch.to(device, non_blocking=True)
            with torch.no_grad():
                tout = teacher(input_ids=ids, use_cache=False, output_attentions=True)
                teacher_attn = mean_attention(tout.attentions, args.teacher_attn_mode, args.teacher_attn_last_n).detach()
            sout = student(input_ids=ids, use_cache=False, output_attentions=True)
            student_attn = mean_attention(sout.attentions, args.student_attn_mode, 1)
            logits_loss, kl, ce = logits_kd_loss(sout.logits, tout.logits, ids, args.temperature, args.kl_weight, args.ce_weight)
            attn_loss = attention_row_kl(student_attn, teacher_attn)
            loss = logits_loss + args.attn_weight * attn_loss
            (loss / args.grad_accum).backward()
            accum += 1
            sums["loss"] += float(loss.detach().item())
            sums["logits"] += float(logits_loss.detach().item())
            sums["kl"] += float(kl.item())
            sums["ce"] += float(ce.item())
            sums["attn"] += float(attn_loss.detach().item())
            sums["n"] += 1
            if accum % args.grad_accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if pbar is not None:
                pbar.update(1)
            if rank0() and (global_step == 1 or global_step % args.log_every == 0):
                n = max(1, sums.pop("n"))
                row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "epoch": epoch, "step": global_step, "world_size": int(os.environ.get("WORLD_SIZE", "1"))}
                row.update({k: v / n for k, v in sums.items()})
                print(json.dumps(row, ensure_ascii=False), flush=True)
                append_history(out_dir, row)
                sums = {"loss":0.0, "logits":0.0, "kl":0.0, "ce":0.0, "attn":0.0, "n":0}
            if rank0() and args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(out_dir, student, optimizer, global_step, epoch, args, tag=f"step{global_step:06d}")
                save_checkpoint(out_dir, student, optimizer, global_step, epoch, args, tag="latest")
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
    if rank0():
        save_checkpoint(out_dir, student, optimizer, global_step, epoch, args, tag="latest")
        save_checkpoint(out_dir, student, optimizer, global_step, epoch, args, tag=f"final_step{global_step:06d}")
    if pbar is not None:
        pbar.close()
    barrier()

if __name__ == "__main__":
    main()
