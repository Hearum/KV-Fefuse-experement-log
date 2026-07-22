#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
sys.path.insert(0, str(ROOT / "MOTIVATION_EXPERIMENTS/native_initialized_layer4_distill/scripts"))
from train_native_layer4_selector import (  # noqa: E402
    NativeLayer4AttentionSelector,
    TeacherCacheDataset,
    collate,
    eval_metrics,
)


class SingleProcessAccelerator:
    def gather_for_metrics(self, x):
        return x


def load_cd_student_into_selector(selector, checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    student_state = state.get("student_state", state)
    embed = {}
    layers = {}
    for key, value in student_state.items():
        if key.startswith("model.embed_tokens."):
            embed[key[len("model.embed_tokens.") :]] = value
        elif key.startswith("model.layers."):
            layers[key[len("model.layers.") :]] = value
    missing_e, unexpected_e = selector.embed_tokens.load_state_dict(embed, strict=True)
    missing_l, unexpected_l = selector.layers.load_state_dict(layers, strict=True)
    return {
        "step": int(state.get("step", -1)),
        "epoch": int(state.get("epoch", -1)),
        "missing_embed": len(missing_e),
        "unexpected_embed": len(unexpected_e),
        "missing_layers": len(missing_l),
        "unexpected_layers": len(unexpected_l),
    }


def run_eval(model, cache_dir, limit, batch_size, ratios, temperature, device):
    ds = TeacherCacheDataset(cache_dir, limit)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=2)
    accelerator = SingleProcessAccelerator()

    class DeviceModel(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, doc_lens):
            scores = self.inner(input_ids.to(device), attention_mask.to(device), doc_lens.to(device))
            return scores.cpu()

        def train(self, mode=True):
            self.inner.train(mode)
            return self

        def eval(self):
            self.inner.eval()
            return self

    return eval_metrics(DeviceModel(model), dl, accelerator, ratios, temperature)


def metric_get(metrics, key):
    return float(metrics.get(key, 0.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--cd-checkpoint", required=True)
    parser.add_argument("--wiki-val-cache-dir", default="MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/teacher_cache_wikitext103_val_50k")
    parser.add_argument("--musique-val-cache-dir", default="MOTIVATION_EXPERIMENTS/native_initialized_layer4_distill/teacher_cache_musique_val")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--score-mode", choices=["attn_prob", "qk_logits"], default="attn_prob")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--wiki-limit", type=int, default=5000)
    parser.add_argument("--musique-limit", type=int, default=0)
    parser.add_argument("--eval-ratios", default="0.05,0.10,0.15,0.30")
    parser.add_argument("--temperature", type=float, default=2.0)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ratios = [float(x) for x in args.eval_ratios.split(",") if x.strip()]

    model = NativeLayer4AttentionSelector(args.model_path, args.layers, dtype=dtype, score_mode=args.score_mode)
    load_info = load_cd_student_into_selector(model, ROOT / args.cd_checkpoint)
    model.to(device)

    results = {
        "load_info": load_info,
        "config": vars(args),
        "score_note": "teacher_scores are cached full 3B draft-model scores from the HS/native layer4 pipeline; metrics compare CD 4-layer top-token sets against that full draft distribution.",
    }
    results["wiki"] = run_eval(model, ROOT / args.wiki_val_cache_dir, args.wiki_limit, args.batch_size, ratios, args.temperature, device)
    if args.musique_val_cache_dir:
        results["musique"] = run_eval(model, ROOT / args.musique_val_cache_dir, args.musique_limit, args.batch_size, ratios, args.temperature, device)

    (out_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for split in ["wiki", "musique"]:
        if split in results:
            rows.append({"split": split, **results[split]})
    fields = ["split"] + sorted({k for row in rows for k in row if k != "split"})
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# CD 4-layer vs Full DraftModel Validation\n\n",
        "评估目标：复用 HS/native layer4 训练 pipeline 的验证缓存，用完整 3B DraftModel 缓存分数作为 teacher，比较 CD 4-layer causal distillation checkpoint 的 token ranking/selection 与 full draft 的重合度。\n\n",
        f"- CD checkpoint: `{args.cd_checkpoint}`\n",
        f"- loaded step/epoch: step={load_info['step']}, epoch={load_info['epoch']}\n",
        f"- score mode: `{args.score_mode}`\n",
        f"- wiki val limit: {args.wiki_limit}\n",
        f"- musique val limit: {'all' if args.musique_limit == 0 else args.musique_limit}\n\n",
        "| split | KL | R@5% | J@5% | R@10% | J@10% | R@15% | J@15% | R@30% | J@30% |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for split in ["wiki", "musique"]:
        metrics = results.get(split)
        if not metrics:
            continue
        lines.append(
            f"| {split} | {metric_get(metrics, 'val_kl'):.4f} | "
            f"{metric_get(metrics, 'val_recall_r0p05'):.4f} | {metric_get(metrics, 'val_jaccard_r0p05'):.4f} | "
            f"{metric_get(metrics, 'val_recall_r0p1'):.4f} | {metric_get(metrics, 'val_jaccard_r0p1'):.4f} | "
            f"{metric_get(metrics, 'val_recall_r0p15'):.4f} | {metric_get(metrics, 'val_jaccard_r0p15'):.4f} | "
            f"{metric_get(metrics, 'val_recall_r0p3'):.4f} | {metric_get(metrics, 'val_jaccard_r0p3'):.4f} |\n"
        )
    (out_dir / "README.md").write_text("".join(lines), encoding="utf-8")
    print("".join(lines))


if __name__ == "__main__":
    main()
