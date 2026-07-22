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
    buf = []
    blocks = []
    text_path = Path(text_path)
    with text_path.open("r", encoding="utf-8", errors="ignore") as f:
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
                buf = buf[seq_len + 1 :]
                if max_sequences and len(blocks) >= max_sequences:
                    break
            if max_sequences and len(blocks) >= max_sequences:
                break
    if not blocks:
        raise RuntimeError(f"no token blocks built from {text_path}")
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


def distill_loss(student_logits, teacher_logits, labels, temperature, kl_weight, ce_weight):
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


def save_checkpoint(out_dir, student, optimizer, step, epoch, args, tag="latest"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_student = student.module if hasattr(student, "module") else student
    ckpt = {
        "step": int(step),
        "epoch": int(epoch),
        "args": vars(args),
        "student_state": raw_student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": raw_student.config.to_dict(),
    }
    torch.save(ckpt, out / f"training_state_{tag}.pt")


def append_history(out_dir, row):
    path = Path(out_dir) / "history.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-model", default="/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct")
    p.add_argument("--student-init-model", default="/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct")
    p.add_argument("--text-path", default="MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/data/wikitext103_train.txt")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-path", default="MOTIVATION_EXPERIMENTS/causal_lm_distill_draftmodel_local/data/wikitext103_qwen_blocks_s256_n20000.pt")
    p.add_argument("--student-layers", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--max-sequences", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--ce-weight", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=200)
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
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=dtype, trust_remote_code=True).to(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    student = AutoModelForCausalLM.from_pretrained(args.student_init_model, torch_dtype=dtype, trust_remote_code=True)
    student = trim_to_first_layers(student, args.student_layers).to(device).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    start_epoch = 0
    if args.resume_from:
        state = torch.load(args.resume_from, map_location="cpu")
        missing, unexpected = student.load_state_dict(state["student_state"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", 0))
        start_epoch = int(state.get("epoch", 0))
        if rank0():
            print(f"resumed from {args.resume_from}; missing={len(missing)} unexpected={len(unexpected)} step={start_step}", flush=True)

    if dist.is_initialized():
        student = DDP(student, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    global_step = start_step
    accum = 0
    run_loss = run_kl = run_ce = 0.0
    seen_updates = 0
    total_updates = args.max_steps if args.max_steps > 0 else math.ceil(len(dl) * args.epochs / max(1, args.grad_accum))
    pbar = None
    if rank0() and tqdm is not None:
        pbar = tqdm(total=total_updates, initial=global_step, desc="FusionRAG-CD train", dynamic_ncols=True)

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dl:
            ids = batch.to(device, non_blocking=True)
            with torch.no_grad():
                teacher_logits = teacher(input_ids=ids, use_cache=False).logits
            student_logits = student(input_ids=ids, use_cache=False).logits
            loss, kl, ce = distill_loss(student_logits, teacher_logits, ids, args.temperature, args.kl_weight, args.ce_weight)
            (loss / args.grad_accum).backward()
            accum += 1
            run_loss += float(loss.detach().item())
            run_kl += float(kl.item())
            run_ce += float(ce.item())
            if accum % args.grad_accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            seen_updates += 1
            if pbar is not None:
                pbar.update(1)
            if rank0() and (global_step % args.log_every == 0 or global_step == 1):
                denom = max(1, seen_updates * args.grad_accum)
                row = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": epoch,
                    "step": global_step,
                    "loss": run_loss / denom,
                    "kl": run_kl / denom,
                    "ce": run_ce / denom,
                    "lr": optimizer.param_groups[0]["lr"],
                    "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                }
                print(json.dumps(row, ensure_ascii=False), flush=True)
                append_history(out_dir, row)
                run_loss = run_kl = run_ce = 0.0
                seen_updates = 0
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
