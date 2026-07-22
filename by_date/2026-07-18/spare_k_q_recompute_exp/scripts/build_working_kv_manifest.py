#!/usr/bin/env python3
"""Write a reproducibility manifest without hashing multi-terabyte KV tensors."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
EXP = ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-18/spare_k_q_recompute_exp"
CACHE = Path("/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2")
MODEL = Path("/mnt/qjhs-sh-lab-01/models/Qwen3-32B")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_metadata(path: Path) -> dict:
    entries = []
    total = 0
    for item in sorted(path.rglob("*")):
        if item.is_file():
            size = item.stat().st_size
            total += size
            entries.append(f"{item.relative_to(path)}\t{size}")
    digest = hashlib.sha256("\n".join(entries).encode()).hexdigest()
    return {"path": str(path), "files": len(entries), "bytes": total, "name_size_sha256": digest}


def sampled_tree_content(path: Path, samples: int = 8) -> dict:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        return {"sampling": "evenly spaced by sorted relative path", "files": []}
    indices = sorted({round(i * (len(files) - 1) / max(1, samples - 1)) for i in range(samples)})
    return {
        "sampling": "evenly spaced by sorted relative path; not a full cache content hash",
        "files": [
            {
                "relative_path": str(files[index].relative_to(path)),
                "bytes": files[index].stat().st_size,
                "sha256": sha256(files[index]),
            }
            for index in indices
        ],
    }


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    data = ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/data/musique-v2.jsonl"
    code_files = [
        ROOT / "models/modeling_qwen3.py",
        ROOT / "ktransformers/util/utils.py",
        ROOT / "ktransformers/operators/sparse_attention.py",
        ROOT / "ktransformers/models/custom_cache.py",
        ROOT / "ktransformers/unified_process_cache.py",
        ROOT / "MOTIVATION_EXPERIMENTS/by_date/2026-07-14/setup_standard_v2_cross_dataset/scripts/run_setup_v2_task.py",
        EXP / "scripts/summarize_working_kv_results.py",
        EXP / "scripts/summarize_working_kv_test.py",
        EXP / "scripts/freeze_working_kv_evidence.py",
        EXP / "scripts/test_working_kv_semantics.py",
        EXP / "scripts/test_setup_v2_runner_guards.py",
        EXP / "scripts/test_working_kv_summarizer_guards.py",
        EXP / "scripts/summarize_working_kv_alpha_kv.py",
        EXP / "scripts/compare_working_kv_alpha_kv.py",
        EXP / "scripts/cluster_bootstrap_working_kv.py",
        EXP / "scripts/launch_confirmatory_and_kv_ablation.py",
    ]
    cache_dirs = [
        CACHE / "data/musique-v2/Qwen3-32B",
        CACHE / "data/musique-v2-preprocess-10-revert_rope-True/Qwen3-32B",
    ]
    evidence_files = [
        EXP / "working_kv_fix2_validation_summary.csv",
        EXP / "working_kv_fix2_validation_summary_paired.csv",
        EXP / "working_kv_fix2_test_summary.csv",
        EXP / "working_kv_fix2_test_summary_paired.csv",
        EXP / "working_kv_fix2_test_integrity.json",
        EXP / "delta_trace/raw_delta_direction_summary.csv",
        EXP / "delta_trace/preprocess_delta_direction_summary.csv",
        EXP / "router_stats_fix2/five_example_summary.csv",
        EXP / "router_stats_fix2/five_example_aggregate.json",
        EXP / "split_covariates.json",
        EXP / "working_kv_semantics_test_output.txt",
        EXP / "setup_v2_runner_guards_test_output.txt",
        EXP / "working_kv_summarizer_guards_test_output.txt",
        EXP / "working_kv_alpha_kv_validation_summary.csv",
        EXP / "working_kv_alpha_kv_validation_paired.csv",
        EXP / "working_kv_confirmatory_preprocess_test_summary.csv",
        EXP / "working_kv_confirmatory_preprocess_test_paired_vs_alpha0.csv",
        EXP / "cluster_bootstrap_dense_raw075_vs_raw0.csv",
        EXP / "cluster_bootstrap_dense_pp_alpha1_vs_alpha0.csv",
        EXP / "cluster_bootstrap_sparse_raw075_vs_raw0.csv",
        EXP / "task.md",
        EXP / "delta_trace/raw_dense.pt",
        EXP / "delta_trace/raw_sparse.pt",
        EXP / "delta_trace/preprocess_dense.pt",
        EXP / "delta_trace/preprocess_sparse.pt",
        EXP / "router_stats_fix2/preprocess_alpha0p5_sample0_v3.jsonl",
        EXP / "router_stats_fix2/preprocess_alpha0p5_examples1_2.jsonl",
        EXP / "router_stats_fix2/preprocess_alpha0p5_examples3_4.jsonl",
    ]
    manifest = {
        "branch": git("branch", "--show-current"),
        "generated_from_head": git("rev-parse", "HEAD"),
        "self_reference_note": (
            "The commit containing this manifest is necessarily a descendant of generated_from_head; "
            "verify that the intervening commit changes only repro_manifest.json."
        ),
        "working_kv_semantic_guard_commit": "89b25a8422f6dc58267026c7e27c5ee3bb598db2",
        "runner_gpu_binding_commit": "8712c8a021ade9c3411448217ce31c86fe26ba16",
        "python": platform.python_version(),
        "validation_example_ids": list(range(1, 51)),
        "test_example_ids": list(range(51, 201)),
        "selection_rule": (
            "Actually applied to raw Dense/Sparse candidates only: validation primary=GLM, "
            "tie-break=F1 then lower alpha; test never changes alpha. Protocol deviation: early "
            "wording implied this covered preprocess too, but nonzero preprocess candidates were "
            "excluded by an undefined stability criterion and were not frozen-tested. Full rate=1 "
            "was a control and was not eligible for selection."
        ),
        "supplemental_protocol": (
            "Post-review frozen test added Dense-preprocess alpha=1.0 and Sparse-preprocess alpha=0.5; "
            "validation added one-axis K-only and V-only ablations around each shared alpha."
        ),
        "dataset": {"path": str(data), "sha256": sha256(data)},
        "model": {
            "path": str(MODEL),
            "config_sha256": sha256(MODEL / "config.json"),
            "tokenizer_config_sha256": sha256(MODEL / "tokenizer_config.json"),
        },
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in code_files},
        "cache_metadata": [
            {**tree_metadata(path), "sampled_content": sampled_tree_content(path)}
            for path in cache_dirs
        ],
        "evidence_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in evidence_files
        },
        "result_roots": [
            str(EXP / "results_working_kv_fix2_validation_50"),
            str(EXP / "results_working_kv_fix2_controls"),
            str(EXP / "results_working_kv_fix2_test_150"),
            str(EXP / "results_working_kv_fix2_test_150_full_shards"),
            str(EXP / "results_working_kv_confirmatory_preprocess_test"),
            str(EXP / "results_working_kv_alpha_kv_validation"),
        ],
    }
    supplemental_roots = [
        EXP / "results_working_kv_confirmatory_preprocess_test",
        EXP / "results_working_kv_alpha_kv_validation",
    ]
    supplemental_metrics = sorted(
        path for root in supplemental_roots for path in root.glob("**/metrics/metrics.csv")
    )
    manifest["supplemental_metrics_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path) for path in supplemental_metrics
    }
    frozen_evidence = sorted((EXP / "frozen_evidence").glob("**/*"))
    manifest["frozen_evidence_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path) for path in frozen_evidence if path.is_file()
    }
    output = EXP / "repro_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
