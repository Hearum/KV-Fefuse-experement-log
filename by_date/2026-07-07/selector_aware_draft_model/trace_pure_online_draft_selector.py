#!/usr/bin/env python3
"""Trace pure online DraftModel token selection without loading the main LLM."""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ktransformers.util.utils import (
    _fusionrag_compute_draft_doc_attention_scores,
    _fusionrag_smart_query_selection,
)
from test_fusionrag_reflect_preprocess_exp import (
    PreprocessScope,
    RecallMethod,
    _rate_tag,
    prepare_reflect_data,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/mnt/qjhs-sh-lab-01/models/Qwen3-32B")
    parser.add_argument("--draft_model_path", default="/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--data_path", default="./data/result_reflect.json")
    parser.add_argument("--bge_model_path", default="/mnt/qjhs-sh-lab-01/models/bge-m3")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rate", type=float, default=0.10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--end_sample", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold_factor", type=float, default=0.5)
    parser.add_argument("--draft_score_layers", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    questions_data, system_tensor, _, _ = prepare_reflect_data(
        args.data_path,
        tokenizer,
        args.bge_model_path,
        model_type="qwen3",
        topk=args.topk,
        preprocess=False,
        recall_method=RecallMethod.BGE,
        preprocess_scope=PreprocessScope.GLOBAL,
    )

    device = torch.device(args.device)
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    draft_model.eval()

    system_len = int(system_tensor.shape[0])
    n_written = 0
    with torch.inference_mode():
        for example_id in range(args.start_sample, min(args.end_sample, len(questions_data))):
            q_data = questions_data[example_id]
            if not q_data.get("should_test", True):
                continue
            doc_tensors = q_data["doc_tensors"]
            for sub_q_idx, sub_q_info in enumerate(q_data["sub_questions"]):
                question_text = (
                    "<|im_end|>\n<|im_start|>user\n/no_think\n"
                    f"Question: {sub_q_info['query']}<|im_end|>\n"
                    "<|im_start|>assistant\nAnswer: "
                )
                question_tensor = torch.tensor(
                    tokenizer.encode(question_text, add_special_tokens=False),
                    dtype=torch.long,
                )
                doc_chunk_ids = [int(x) for x in sub_q_info["chunk_ids"]]
                sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]
                iter_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
                past_len = int(system_tensor.shape[0] + sum(int(t.shape[0]) for t in sub_q_doc_tensors))
                doc_len = int(past_len - system_len)
                full_input = torch.cat(iter_tokens).unsqueeze(0).to(device)

                draft_scores = _fusionrag_compute_draft_doc_attention_scores(
                    draft_model,
                    full_input,
                    query_start=past_len,
                    doc_start=system_len,
                    doc_len=doc_len,
                    device=device,
                    rrf_k=18,
                    score_layers=args.draft_score_layers,
                )
                selected = _fusionrag_smart_query_selection(
                    draft_scores,
                    doc_len,
                    args.rate,
                    system_len,
                    threshold_factor=args.threshold_factor,
                )
                selected_abs = [int(x) for x in selected.cpu().tolist()]
                payload = {
                    "example_id": int(example_id),
                    "sub_q_idx": int(sub_q_idx),
                    "method": "PureDraftModel",
                    "rate": float(args.rate),
                    "system_len": system_len,
                    "passages_len": [int(t.shape[0]) for t in iter_tokens],
                    "chunk_ids": [0] + doc_chunk_ids,
                    "selected_abs": selected_abs,
                    "selected_count": len(selected_abs),
                }
                out_file = out_dir / (
                    f"example{example_id:03d}_sub{sub_q_idx:02d}_"
                    f"PureDraftModel_rate{_rate_tag(args.rate)}.json"
                )
                out_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                n_written += 1
                if n_written % 25 == 0:
                    print(f"written={n_written}", flush=True)
    print(f"done written={n_written}")


if __name__ == "__main__":
    main()
