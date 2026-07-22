#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

ROOT = Path("/raid/home/hming/FusionRAG-pca-analysis")
REAL_PROJ = "/mnt/qjhs-sh-lab-01/wjh/FusionRAG"
sys.path = [
    p
    for p in sys.path
    if p not in {REAL_PROJ, str(ROOT), str(ROOT / "ktransformers"), str(Path(REAL_PROJ) / "ktransformers")}
]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, REAL_PROJ)

from ktransformers.util.utils import _fusionrag_compute_draft_doc_attention_scores  # noqa: E402


def topk_indices(scores, ratio):
    k = max(1, int(len(scores) * ratio))
    idx = np.argpartition(-scores, k - 1)[:k]
    return np.asarray(sorted(idx), dtype=np.int32)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/data/wikitext_pairs.jsonl")
    parser.add_argument("--out-dir", default="MOTIVATION_EXPERIMENTS/predictor_distill_wikitext/teacher_cache")
    parser.add_argument("--draft-model-path", default="/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-ratio", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.draft_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    manifest = []
    with Path(args.pairs).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    end = args.end if args.end is not None else len(rows)
    rows = rows[args.start : end]
    if args.limit is not None and args.limit > 0:
        rows = rows[: args.limit]

    for row in rows:
        pair_id = int(row["pair_id"])
        doc_ids = torch.tensor(row["doc_ids"], dtype=torch.long)
        query_ids = torch.tensor(row["query_ids"], dtype=torch.long)
        input_ids = torch.cat([doc_ids, query_ids]).unsqueeze(0).to(args.device)
        scores = _fusionrag_compute_draft_doc_attention_scores(
            model,
            input_ids,
            query_start=int(doc_ids.numel()),
            doc_start=0,
            doc_len=int(doc_ids.numel()),
            device=args.device,
            rrf_k=18,
            score_layers=None,
            stop_after_score_layers=False,
        )
        scores_np = np.nan_to_num(scores.float().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        top_idx = topk_indices(scores_np, args.top_ratio)
        path = out_dir / f"pair{pair_id:05d}.npz"
        np.savez_compressed(
            path,
            pair_id=np.asarray(pair_id, dtype=np.int32),
            doc_ids=np.asarray(row["doc_ids"], dtype=np.int32),
            query_ids=np.asarray(row["query_ids"], dtype=np.int32),
            teacher_scores=scores_np,
            teacher_top_idx=top_idx,
            top_ratio=np.asarray(args.top_ratio, dtype=np.float32),
        )
        manifest.append({"pair_id": pair_id, "path": str(path), "doc_len": len(row["doc_ids"]), "query_len": len(row["query_ids"]), "top_count": len(top_idx)})
        print(f"teacher pair={pair_id} top={len(top_idx)} score_mean={scores_np.mean():.6g}", flush=True)
        del input_ids, scores
        torch.cuda.empty_cache()

    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(manifest)} teacher files to {out_dir}")


if __name__ == "__main__":
    main()
