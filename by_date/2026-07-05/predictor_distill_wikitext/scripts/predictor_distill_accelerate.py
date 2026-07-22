#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset


class TeacherCacheDataset(Dataset):
    def __init__(self, cache_dir, split="train", val_fraction=0.15, use_all=False):
        self.cache_dir = Path(cache_dir)
        files = sorted(self.cache_dir.glob("pair*.npz"))
        if not files:
            raise FileNotFoundError(f"no teacher cache npz files under {self.cache_dir}")
        if use_all:
            self.files = files
            return
        n_val = max(1, int(len(files) * val_fraction)) if len(files) > 1 else 0
        if split == "train":
            self.files = files[:-n_val] if n_val else files
        elif split == "val":
            self.files = files[-n_val:] if n_val else files
        else:
            raise ValueError(split)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        doc_ids = data["doc_ids"].astype(np.int64)
        query_ids = data["query_ids"].astype(np.int64)
        scores = data["teacher_scores"].astype(np.float32)
        top_idx = data["teacher_top_idx"].astype(np.int64)
        labels = np.zeros_like(scores, dtype=np.float32)
        labels[top_idx] = 1.0
        return {
            "input_ids": np.concatenate([doc_ids, query_ids]).astype(np.int64),
            "doc_len": np.asarray(len(doc_ids), dtype=np.int64),
            "teacher_scores": scores,
            "teacher_top_labels": labels,
        }


def collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    max_doc = max(int(x["doc_len"]) for x in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    doc_mask = torch.zeros(len(batch), max_doc, dtype=torch.bool)
    teacher_scores = torch.zeros(len(batch), max_doc, dtype=torch.float)
    teacher_top_labels = torch.zeros(len(batch), max_doc, dtype=torch.float)
    doc_lens = torch.zeros(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        ids = torch.from_numpy(item["input_ids"])
        dlen = int(item["doc_len"])
        input_ids[i, : ids.numel()] = ids
        attention_mask[i, : ids.numel()] = True
        doc_mask[i, :dlen] = True
        teacher_scores[i, :dlen] = torch.from_numpy(item["teacher_scores"])
        teacher_top_labels[i, :dlen] = torch.from_numpy(item["teacher_top_labels"])
        doc_lens[i] = dlen
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "doc_mask": doc_mask,
        "teacher_scores": teacher_scores,
        "teacher_top_labels": teacher_top_labels,
        "doc_lens": doc_lens,
    }


class TinySelector(nn.Module):
    def __init__(self, vocab_size, max_len=1024, hidden=512, layers=4, heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.type_emb = nn.Embedding(2, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.score_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, doc_lens):
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        type_ids = torch.ones_like(input_ids)
        for i, dlen in enumerate(doc_lens.tolist()):
            type_ids[i, :dlen] = 0
        hidden = self.token_emb(input_ids) + self.pos_emb(pos) + self.type_emb(type_ids)
        key_padding_mask = ~attention_mask
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        logits = self.score_head(hidden).squeeze(-1)
        max_doc = int(doc_lens.max().item())
        return logits[:, :max_doc]


def masked_kl(student_logits, teacher_scores, doc_mask, temperature):
    neg_inf = torch.finfo(student_logits.dtype).min
    s_logp = F.log_softmax(student_logits.masked_fill(~doc_mask, neg_inf) / temperature, dim=-1)
    t_prob = F.softmax(teacher_scores.masked_fill(~doc_mask, neg_inf) / temperature, dim=-1)
    return F.kl_div(s_logp, t_prob, reduction="batchmean") * (temperature * temperature)


def masked_bce(student_logits, labels, doc_mask, pos_weight=1.0):
    pos_weight_t = torch.tensor(float(pos_weight), device=student_logits.device, dtype=student_logits.dtype)
    loss = F.binary_cross_entropy_with_logits(student_logits, labels, reduction="none", pos_weight=pos_weight_t)
    return (loss * doc_mask.float()).sum() / doc_mask.float().sum().clamp_min(1.0)


@torch.no_grad()
def eval_metrics(model, dataloader, accelerator, eval_ratios, temperature):
    model.eval()
    total_kl = total_bce = total_n = 0.0
    ratio_stats = {ratio: {"recall": 0.0, "jaccard": 0.0} for ratio in eval_ratios}
    for batch in dataloader:
        logits = model(batch["input_ids"], batch["attention_mask"], batch["doc_lens"])
        doc_mask = batch["doc_mask"]
        kl = masked_kl(logits, batch["teacher_scores"], doc_mask, temperature)
        bce = masked_bce(logits, batch["teacher_top_labels"], doc_mask, getattr(model, "pos_weight", 1.0))
        per_ratio = {ratio: {"recall": [], "jaccard": []} for ratio in eval_ratios}
        for i in range(logits.shape[0]):
            dlen = int(batch["doc_lens"][i].item())
            for ratio in eval_ratios:
                k = max(1, int(dlen * ratio))
                pred = torch.topk(logits[i, :dlen], k).indices
                teacher = torch.topk(batch["teacher_scores"][i, :dlen], k).indices
                pred_set = set(pred.detach().cpu().tolist())
                teacher_set = set(teacher.detach().cpu().tolist())
                inter = len(pred_set & teacher_set)
                union = len(pred_set | teacher_set)
                per_ratio[ratio]["recall"].append(inter / max(1, len(teacher_set)))
                per_ratio[ratio]["jaccard"].append(inter / max(1, union))
        n = logits.shape[0]
        total_kl += float(accelerator.gather_for_metrics(kl.detach()).mean().item()) * n
        total_bce += float(accelerator.gather_for_metrics(bce.detach()).mean().item()) * n
        for ratio in eval_ratios:
            ratio_stats[ratio]["recall"] += sum(per_ratio[ratio]["recall"])
            ratio_stats[ratio]["jaccard"] += sum(per_ratio[ratio]["jaccard"])
        total_n += n
    model.train()
    metrics = {
        "val_kl": total_kl / max(1, total_n),
        "val_bce": total_bce / max(1, total_n),
    }
    for ratio in eval_ratios:
        tag = str(ratio).replace(".", "p")
        metrics[f"val_recall_r{tag}"] = ratio_stats[ratio]["recall"] / max(1, total_n)
        metrics[f"val_jaccard_r{tag}"] = ratio_stats[ratio]["jaccard"] / max(1, total_n)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/teacher_cache")
    parser.add_argument("--val-cache-dir", default=None)
    parser.add_argument("--out-dir", default="MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/checkpoints/accelerate_tiny_selector")
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--bce-weight", type=float, default=0.0)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--top-ratio", type=float, default=0.15)
    parser.add_argument("--eval-ratios", default="0.05,0.10,0.15,0.30")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    eval_ratios = [float(x) for x in args.eval_ratios.split(",") if x.strip()]

    accelerator = Accelerator(mixed_precision="fp16")
    out_dir = Path(args.out_dir)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.val_cache_dir:
        train_ds = TeacherCacheDataset(args.cache_dir, use_all=True)
        val_ds = TeacherCacheDataset(args.val_cache_dir, use_all=True)
    else:
        train_ds = TeacherCacheDataset(args.cache_dir, "train", args.val_fraction)
        val_ds = TeacherCacheDataset(args.cache_dir, "val", args.val_fraction)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2)

    model = TinySelector(args.vocab_size, args.max_len, args.hidden, args.layers, args.heads)
    model.pos_weight = args.pos_weight
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model, optimizer, train_dl, val_dl = accelerator.prepare(model, optimizer, train_dl, val_dl)

    history = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_dl:
            logits = model(batch["input_ids"], batch["attention_mask"], batch["doc_lens"])
            kl = masked_kl(logits, batch["teacher_scores"], batch["doc_mask"], args.temperature)
            bce = masked_bce(logits, batch["teacher_top_labels"], batch["doc_mask"], args.pos_weight)
            loss = kl + args.bce_weight * bce
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach().item()) * logits.shape[0]
            seen += logits.shape[0]
        metrics = eval_metrics(model, val_dl, accelerator, eval_ratios, args.temperature)
        metrics.update({"epoch": epoch + 1, "train_loss": running / max(1, seen)})
        history.append(metrics)
        if accelerator.is_main_process:
            print(metrics, flush=True)

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save({"model": unwrapped.state_dict(), "args": vars(args), "history": history}, out_dir / "model.pt")
        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)


if __name__ == "__main__":
    main()
