#!/usr/bin/env python3
"""Small regression check for deterministic setup-v2 GPU binding."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

RUNNER = ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py"
spec = importlib.util.spec_from_file_location("setup_v2_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)
configure_runtime_environment = runner.configure_runtime_environment


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    configure_runtime_environment(argparse.Namespace(gpu=3))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    configure_runtime_environment(argparse.Namespace(gpu=2))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    print("PASS setup-v2 deterministic GPU binding")


if __name__ == "__main__":
    main()
