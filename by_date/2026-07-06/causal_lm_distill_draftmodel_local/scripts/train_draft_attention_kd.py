#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader

ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
import sys

sys.path.insert(0, str(ROOT / "MOTIVATION_EXPERIMENTS/native_initialized_layer4_distill/scripts"))
from train_native_layer4_selector import (  # noqa: E402
    NativeLayer4AttentionSelector,
    TeacherCacheDataset,
    collate,
    eval_metrics,
)


def masked_log_softmax(scores, mask, temperature):
    neg_inf = torch.finfo(scores.dtype).min
    return F.log_softmax(scores.masked_fill(~mask, neg_inf) / temperature, dim=-1)


def masked_softmax(scores, mask, temperature):
    return masked_log_softmax(scores, mask, temperature).exp()


def teacher_topk_mask(teacher_scores, doc_mask, ratios):
    out = torch.zeros_like(doc_mask)
    for i in range(teacher_scores.shape[0]):
        dlen = int(doc_mask[i].sum().item())
        if dlen <= 0:
            continue
        for ratio in ratios:
            k = max(1, int(dlen * ratio))
            idx = torch.topk(teacher_scores[i, :dlen], k).indices
            out[i, idx] = True
    return out & doc_mask


def kdflow_style_attention_loss(student_scores, teacher_scores, doc_mask, temperature, topk_ratios, reverse_kl_weight, topk_mass_weight):
    t_prob = masked_softmax(teacher_scores.float(), doc_mask, temperature)
    s_logp = masked_log_softmax(student_scores.float(), doc_mask, temperature)
    loss = F.kl_div(s_logp, t_prob, reduction="batchmean") * (temperature * temperature)

    if reverse_kl_weight > 0:
        s_prob = s_logp.exp()
        t_logp = masked_log_softmax(teacher_scores.float(), doc_mask, temperature)
        reverse = F.kl_div(t_logp, s_prob, reduction="batchmean") * (temperature * temperature)
        loss = loss + reverse_kl_weight * reverse

    if topk_mass_weight > 0:
        top_mask = teacher_topk_mask(teacher_scores, doc_mask, topk_ratios)
        s_prob = s_logp.exp()
        selected_mass = (s_prob * top_mask.to(s_prob.dtype)).sum(dim=-1).clamp_min(1e-8)
        target_mass = (top_mask & doc_mask).sum(dim=-1).float() / doc_mask.sum(dim=-1).float().clamp_min(1.0)
        # Encourage the student distribution to put more probability mass than random selection on teacher top tokens.
        mass_loss = -torch.log(selected_mass).mean() * target_mass.mean().detach().clamp_min(0.05)
        loss = loss + topk_mass_weight * mass_loss

    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--val-cache-dir", required=True)
    parser.add_argument("--musique-val-cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--reverse-kl-weight", type=float, default=0.2)
    parser.add_argument("--topk-mass-weight", type=float, default=1.0)
    parser.add_argument("--topk-ratios", default="0.05,0.10,0.15")
    parser.add_argument("--eval-ratios", default="0.05,0.10,0.15,0.30")
    parser.add_argument("--train-limit", type=int, default=20000)
    parser.add_argument("--val-limit", type=int, default=2000)
    parser.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every-epoch", action="store_true")
    args = parser.parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    out_dir = ROOT / args.out_dir
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)

    eval_ratios = [float(x) for x in args.eval_ratios.split(",") if x.strip()]
    topk_ratios = [float(x) for x in args.topk_ratios.split(",") if x.strip()]
    train_ds = TeacherCacheDataset(ROOT / args.cache_dir, args.train_limit)
    val_ds = TeacherCacheDataset(ROOT / args.val_cache_dir, args.val_limit)
    musique_ds = TeacherCacheDataset(ROOT / args.musique_val_cache_dir, 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2)
    musique_dl = DataLoader(musique_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2)

    model = NativeLayer4AttentionSelector(args.model_path, args.layers, score_mode="attn_prob")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model, optimizer, train_dl, val_dl, musique_dl = accelerator.prepare(model, optimizer, train_dl, val_dl, musique_dl)

    history = []
    base = eval_metrics(model, val_dl, accelerator, eval_ratios, temperature=2.0)
    base = {f"wiki_{k}": v for k, v in base.items()}
    base.update({f"musique_{k}": v for k, v in eval_metrics(model, musique_dl, accelerator, eval_ratios, temperature=2.0).items()})
    base.update({"epoch": 0, "train_loss": 0.0})
    history.append(base)
    if accelerator.is_main_process:
        print(base, flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for step, batch in enumerate(train_dl, start=1):
            scores = model(batch["input_ids"], batch["attention_mask"], batch["doc_lens"])
            loss = kdflow_style_attention_loss(
                scores,
                batch["teacher_scores"],
                batch["doc_mask"],
                args.temperature,
                topk_ratios,
                args.reverse_kl_weight,
                args.topk_mass_weight,
            )
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach().item()) * scores.shape[0]
            seen += scores.shape[0]
            if accelerator.is_main_process and (step % args.log_every == 0 or step == len(train_dl)):
                print({"epoch": epoch, "step": step, "train_loss": running / max(1, seen)}, flush=True)

        metrics = eval_metrics(model, val_dl, accelerator, eval_ratios, temperature=2.0)
        metrics = {f"wiki_{k}": v for k, v in metrics.items()}
        metrics.update({f"musique_{k}": v for k, v in eval_metrics(model, musique_dl, accelerator, eval_ratios, temperature=2.0).items()})
        metrics.update({"epoch": epoch, "train_loss": running / max(1, seen)})
        history.append(metrics)
        if accelerator.is_main_process:
            print(metrics, flush=True)
            unwrapped = accelerator.unwrap_model(model)
            state = {
                "embed_tokens": unwrapped.embed_tokens.state_dict(),
                "layers": unwrapped.layers.state_dict(),
                "epoch": epoch,
                "args": vars(args),
                "history": history,
            }
            torch.save(state, out_dir / "training_state_latest.pt")
            if args.save_every_epoch:
                torch.save(state, out_dir / f"training_state_epoch{epoch:03d}.pt")

    if accelerator.is_main_process:
        fields = sorted({k for row in history for k in row})
        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(history)
        torch.save({"history": history, "args": vars(args)}, out_dir / "final_metrics.pt")


if __name__ == "__main__":
    main()
