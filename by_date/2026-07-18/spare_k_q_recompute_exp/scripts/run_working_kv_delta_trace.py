#!/usr/bin/env python3
"""Run setup-v2 while sampling layer-wise candidate-minus-base KV deltas."""

from __future__ import annotations

import argparse
import atexit
import runpy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
RUNNER = ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py"
sys.path.insert(0, str(ROOT))

import models.modeling_qwen3 as modeling_qwen3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-output", required=True)
    parser.add_argument("--sample-tokens", type=int, default=32)
    args, runner_args = parser.parse_known_args()
    records = []

    def capture(layer, candidate_k, candidate_v, base_k, base_v):
        count = min(args.sample_tokens, candidate_k.shape[2])
        records.append({
            "layer": int(layer),
            "tokens": int(candidate_k.shape[2]),
            "delta_k": (candidate_k[:, :, :count] - base_k[:, :, :count]).detach().to("cpu", torch.float16),
            "delta_v": (candidate_v[:, :, :count] - base_v[:, :, :count]).detach().to("cpu", torch.float16),
        })

    def save_trace():
        output = Path(args.trace_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"sample_tokens": args.sample_tokens, "records": records}, output)

    modeling_qwen3.WORKING_KV_TRACE_CALLBACK = capture
    atexit.register(save_trace)
    sys.argv = [str(RUNNER), *runner_args]
    runpy.run_path(str(RUNNER), run_name="__main__")


if __name__ == "__main__":
    main()
