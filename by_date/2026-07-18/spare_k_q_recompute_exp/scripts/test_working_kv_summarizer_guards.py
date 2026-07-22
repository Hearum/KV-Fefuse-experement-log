#!/usr/bin/env python3
"""Regression checks for full-row duplicate and question-set validation."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp/scripts"
VALIDATION_SCRIPT = SCRIPT_DIR / "summarize_working_kv_results.py"
TEST_SCRIPT = SCRIPT_DIR / "summarize_working_kv_test.py"
FIELDS = ["Question", "Real Answer", "Pred Answer", "em", "f1", "glm_correct", "glm_reason"]


def write_metric(
    root: Path, method: str, question: str, prediction: str, alpha: str = "alpha_0p0"
) -> None:
    path = root / method / "musique-v2" / "rate_0p15" / alpha / "seg" / "metrics" / "metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({
            "Question": question,
            "Real Answer": "answer",
            "Pred Answer": prediction,
            "em": "1",
            "f1": "1",
            "glm_correct": "True",
            "glm_reason": "same metric is insufficient for duplicate identity",
        })


def run_validation(roots: list[Path], output: Path) -> subprocess.CompletedProcess:
    command = [sys.executable, str(VALIDATION_SCRIPT)]
    for root in roots:
        command.extend(["--root", str(root)])
    command.extend(["--output", str(output), "--expected-n", "1"])
    return subprocess.run(command, text=True, capture_output=True)


def run_test(roots: list[Path], output: Path) -> subprocess.CompletedProcess:
    command = [sys.executable, str(TEST_SCRIPT)]
    for root in roots:
        command.extend(["--root", str(root)])
    command.extend([
        "--summary", str(output / "summary.csv"),
        "--paired", str(output / "paired.csv"),
        "--integrity", str(output / "integrity.json"),
        "--expected-n", "1",
    ])
    output.mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, text=True, capture_output=True)


def write_complete_test_methods(root: Path, question: str = "q1") -> None:
    write_metric(root, "dense_working_kv_raw", question, "dense0", "alpha_0p0")
    write_metric(root, "dense_working_kv_raw", question, "dense75", "alpha_0p75")
    write_metric(root, "sparse_working_kv_raw", question, "sparse75", "alpha_0p75")
    write_metric(root, "full_rate1", question, "full", "alpha_none")


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        root_a, root_b = base / "a", base / "b"
        write_metric(root_a, "dense", "q1", "p1")
        write_metric(root_b, "dense", "q1", "p1")
        assert run_validation([root_a, root_b], base / "ok.csv").returncode == 0

        write_metric(root_b, "dense", "q1", "different prediction")
        conflict = run_validation([root_a, root_b], base / "conflict.csv")
        assert conflict.returncode != 0 and "conflicting duplicate" in conflict.stderr

        root_c = base / "c"
        write_metric(root_c, "sparse", "q2", "p2")
        mismatch = run_validation([root_a, root_c], base / "mismatch.csv")
        assert mismatch.returncode != 0 and "question set mismatch" in mismatch.stderr

        test_a, test_b = base / "test-a", base / "test-b"
        write_complete_test_methods(test_a)
        assert run_test([test_a], base / "test-ok").returncode == 0

        write_metric(test_b, "dense_working_kv_raw", "q1", "conflict", "alpha_0p0")
        conflict = run_test([test_a, test_b], base / "test-conflict")
        assert conflict.returncode != 0 and "conflicting duplicate" in conflict.stderr

        mismatch_root = base / "test-mismatch"
        write_complete_test_methods(mismatch_root)
        write_metric(mismatch_root, "full_rate1", "q2", "full", "alpha_none")
        mismatch = run_test([mismatch_root], base / "test-mismatch-output")
        assert mismatch.returncode != 0 and "question set mismatch" in mismatch.stderr
    print("PASS validation/test summarizer full-row, N, and question-set guards")


if __name__ == "__main__":
    main()
