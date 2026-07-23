#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT_CANDIDATES = [Path("/raid/home/hming/FusionRAG-pca-analysis"), Path("/home/hming/FusionRAG-pca-analysis")]
ROOT = next((p for p in ROOT_CANDIDATES if p.exists()), ROOT_CANDIDATES[0])
EXP = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = Path("/mnt/qjhs-sh-lab-03/lpl/hming/fusionrag-kv-cache/Qwen3-32B/setup-v2")
MODEL = "/mnt/qjhs-sh-lab-01/models/Qwen3-32B"
DRAFT = "/mnt/qjhs-sh-lab-01/models/Qwen2.5-3B-Instruct"
BGE = "/mnt/qjhs-sh-lab-01/models/bge-m3"

METHODS = {
    "full_rate1": {"reprocess_method": "FusionRAG", "rate": 1.0, "preprocess": False, "draft": False},
    "raw_rate0": {"reprocess_method": "FusionRAG", "rate": 0.0, "preprocess": False, "draft": False},
    "preprocess_rate0": {"reprocess_method": "FusionRAG", "rate": 0.0, "preprocess": True, "draft": False},
    "position_adapter_raw_rate0": {"reprocess_method": "FusionRAG", "rate": 0.0, "preprocess": False, "draft": False, "static_bias": True},
    "position_adapter_preprocess_rate0": {"reprocess_method": "FusionRAG", "rate": 0.0, "preprocess": True, "draft": False, "static_bias": True},
    "online_qk": {"reprocess_method": "FusionRAG", "preprocess": True, "draft": False},
    "online_draft": {"reprocess_method": "DraftModel", "preprocess": True, "draft": True},
    "uniform_alpha0p1_draft": {"reprocess_method": "DraftModel", "preprocess": True, "draft": True, "ablation": "uniform", "alpha": "0.1"},
    "offline3b_mean": {
        "reprocess_method": "FusionRAG", "preprocess": True, "draft": False,
        "fixed_set_family": "3b", "fixed_set_method": "offline3b_mean_score_global",
    },
    "offline3b_freq_boundary2": {
        "reprocess_method": "FusionRAG", "preprocess": True, "draft": False,
        "fixed_set_family": "3b", "fixed_set_method": "offline3b_freq_boundary0p02_global",
    },
    "offline3b_top2": {
        "reprocess_method": "FusionRAG", "preprocess": True, "draft": False,
        "fixed_set_family": "3b", "fixed_set_method": "offline3b_top2_mean_global",
    },
    "offline32b_top2": {
        "reprocess_method": "FusionRAG", "preprocess": True, "draft": False,
        "fixed_set_family": "32b", "fixed_set_method": "offline32b_mean_score_global",
    },
    "sparse_block_preprocess": {
        "reprocess_method": "FusionRAG", "preprocess": True, "draft": False,
        "sparse_block": True,
    },
    "sparse_block_raw": {
        "reprocess_method": "FusionRAG", "preprocess": False, "draft": False,
        "sparse_block": True,
    },
    "dense_working_kv_preprocess": {
        "reprocess_method": "FusionRAG", "preprocess": True, "draft": False,
        "working_kv": True,
    },
    "dense_working_kv_raw": {
        "reprocess_method": "FusionRAG", "preprocess": False, "draft": False,
        "working_kv": True,
    },
    "sparse_working_kv_preprocess": {
        "reprocess_method": "FusionRAG", "preprocess": True, "draft": False,
        "sparse_block": True, "working_kv": True,
    },
    "sparse_working_kv_raw": {
        "reprocess_method": "FusionRAG", "preprocess": False, "draft": False,
        "sparse_block": True, "working_kv": True,
    },
    "sparse_delta_adapter_preprocess": {
        "reprocess_method": "FusionRAG", "preprocess": True, "draft": False,
        "sparse_block": True, "kv_adapter": True,
    },
    "sparse_delta_adapter_raw": {
        "reprocess_method": "FusionRAG", "preprocess": False, "draft": False,
        "sparse_block": True, "kv_adapter": True,
    },
}

FIXED_SET_DATASET = {
    "2wikimqa-v2": "2wikimqa",
    "hotpotqa-v2": "hotpotqa",
    "triviaqa-v2": "triviaqa",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run setup-standard v2 FusionRAG segment with shared KV cache.")
    p.add_argument("--dataset", required=True, choices=["musique-v2", "2wikimqa-v2", "hotpotqa-v2", "triviaqa-v2"])
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--rate", type=float, default=None)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    p.add_argument("--result-root", default=str(EXP / "results"))
    p.add_argument("--offline-fixed-set-dir", default=None)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--revert-rope", action="store_true", default=True)
    p.add_argument("--no-revert-rope", dest="revert_rope", action="store_false")
    p.add_argument("--static-key-bias-path", default=None)
    p.add_argument("--static-key-bias-scale", type=float, default=1.0)
    p.add_argument("--static-key-bias-require-all", action="store_true")
    p.add_argument("--static-value-bias-path", default=None)
    p.add_argument("--static-value-bias-scale", type=float, default=1.0)
    p.add_argument("--static-value-bias-require-all", action="store_true")
    p.add_argument("--sparse-block-size", type=int, default=int(os.environ.get("FUSIONRAG_SPARSE_BLOCK_SIZE", "64")))
    p.add_argument("--sparse-block-topk", type=int, default=int(os.environ.get("FUSIONRAG_SPARSE_BLOCK_TOPK", "8")))
    p.add_argument("--adapter-alpha", type=float, default=float(os.environ.get("FUSIONRAG_KV_ADAPTER_ALPHA", "0")))
    p.add_argument("--working-kv-alpha", type=float, default=1.0)
    p.add_argument("--working-kv-alpha-k", type=float, default=None)
    p.add_argument("--working-kv-alpha-v", type=float, default=None)
    return p.parse_args()


def configure_runtime_environment(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


def main() -> None:
    args = parse_args()
    cfg = dict(METHODS[args.method])
    rate = cfg.get("rate", args.rate if args.rate is not None else 0.15)
    if args.method == "full_rate1":
        rate = 1.0
    if rate is None:
        raise ValueError("rate is required for this method")

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ktransformers"))
    configure_runtime_environment(args)
    os.environ["ONLY_EXAMPLES"] = ",".join(str(i) for i in range(args.start + 1, args.end + 1))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if "ablation" in cfg:
        os.environ["FUSIONRAG_REPROCESS_ATTENTION_ABLATION"] = cfg["ablation"]
        os.environ["FUSIONRAG_REPROCESS_ATTENTION_ABLATION_ALPHA"] = cfg["alpha"]
        if cfg["ablation"] == "linear":
            # Keep the question prefill as dense MHA. The linear probe is
            # applied only during the separate document reprocess call.
            os.environ["FUSIONRAG_STRICT_REPROCESS_ABLATION"] = "1"
    if cfg.get("sparse_block"):
        os.environ["FUSIONRAG_SPARSE_BLOCK_SIZE"] = str(args.sparse_block_size)
        os.environ["FUSIONRAG_SPARSE_BLOCK_TOPK"] = str(args.sparse_block_topk)
    if cfg.get("working_kv"):
        alpha_k = args.working_kv_alpha if args.working_kv_alpha_k is None else args.working_kv_alpha_k
        alpha_v = args.working_kv_alpha if args.working_kv_alpha_v is None else args.working_kv_alpha_v
        for name, value in (("--working-kv-alpha-k", alpha_k), ("--working-kv-alpha-v", alpha_v)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        os.environ["FUSIONRAG_SPARSE_KV_ALPHA"] = str(args.working_kv_alpha)
        os.environ["FUSIONRAG_SPARSE_KV_ALPHA_K"] = str(alpha_k)
        os.environ["FUSIONRAG_SPARSE_KV_ALPHA_V"] = str(alpha_v)
        os.environ["FUSIONRAG_STRICT_REPROCESS_ABLATION"] = "1"
    if cfg.get("kv_adapter"):
        if not 0.0 <= args.adapter_alpha <= 1.0:
            raise ValueError("--adapter-alpha must be in [0, 1]")
        os.environ["FUSIONRAG_KV_ADAPTER_ALPHA"] = str(args.adapter_alpha)

    import ktransformers.util.utils as utils
    import ktransformers.unified_process_cache as upc

    orig_prepare = utils.prepare_data
    rate_tag = str(rate).replace(".", "p")
    result_dir = Path(args.result_root) / args.method / args.dataset / f"rate_{rate_tag}"
    if cfg.get("kv_adapter"):
        alpha_tag = str(args.adapter_alpha).replace(".", "p")
        result_dir = result_dir / f"alpha_{alpha_tag}"
    if cfg.get("working_kv"):
        alpha_k = args.working_kv_alpha if args.working_kv_alpha_k is None else args.working_kv_alpha_k
        alpha_v = args.working_kv_alpha if args.working_kv_alpha_v is None else args.working_kv_alpha_v
        if args.working_kv_alpha_k is None and args.working_kv_alpha_v is None:
            alpha_tag = str(args.working_kv_alpha).replace(".", "p")
            result_dir = result_dir / f"alpha_{alpha_tag}"
        else:
            alpha_k_tag = str(alpha_k).replace(".", "p")
            alpha_v_tag = str(alpha_v).replace(".", "p")
            result_dir = result_dir / f"alpha_k_{alpha_k_tag}__alpha_v_{alpha_v_tag}"
    result_dir = result_dir / f"seg_{args.start}_{args.end}"
    result_csv_dir = result_dir / "csv"
    result_csv_dir.mkdir(parents=True, exist_ok=True)

    def patched_prepare_data(*pargs, **kwargs):
        out = list(orig_prepare(*pargs, **kwargs))
        # Fields: prompt_data, tokens_data, question_list, real_answer_list, stop_token_id,
        # save_path, preprocess_path, csv_path, data_name_prefix, rouge_metrics, context_rank, corpus_lens.
        out[7] = str(result_csv_dir)
        return tuple(out)

    utils.prepare_data = patched_prepare_data
    upc.prepare_data = patched_prepare_data

    print(f"[setup-v2] dataset={args.dataset} method={args.method} rate={rate} start={args.start} end={args.end}")
    print(f"[setup-v2] shared_cache={args.cache_root}")
    print(f"[setup-v2] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"[setup-v2] result_dir={result_dir}")
    if cfg.get("static_bias") and not (args.static_key_bias_path or args.static_value_bias_path):
        raise ValueError(f"{args.method} requires --static-key-bias-path and/or --static-value-bias-path")
    if args.static_key_bias_path:
        print(f"[setup-v2] static_key_bias_path={args.static_key_bias_path} scale={args.static_key_bias_scale}")
    if args.static_value_bias_path:
        print(f"[setup-v2] static_value_bias_path={args.static_value_bias_path} scale={args.static_value_bias_scale}")

    offline_fixed_set_dir = None
    if "fixed_set_method" in cfg:
        if args.offline_fixed_set_dir:
            offline_fixed_set_dir = Path(args.offline_fixed_set_dir)
        else:
            if args.dataset not in FIXED_SET_DATASET:
                raise ValueError(f"{args.method} has no fixed-set artifact for dataset {args.dataset}; pass --offline-fixed-set-dir for setup-v2 fixed sets")
            old_dataset = FIXED_SET_DATASET[args.dataset]
            offline_fixed_set_dir = ROOT / "MOTIVATION_EXPERIMENTS/cross_dataset_offline_generalization" / f"fixed_sets_{old_dataset}_{cfg['fixed_set_family']}" / "chunk_fixed_sets_npz"
        if not offline_fixed_set_dir.exists():
            raise FileNotFoundError(offline_fixed_set_dir)
        print(f"[setup-v2] offline_fixed_set_dir={offline_fixed_set_dir}")
        print(f"[setup-v2] offline_fixed_set_method={cfg['fixed_set_method']}")

    kwargs_main = dict(
        model_type="qwen3",
        model_path=MODEL,
        model_name="Qwen3-32B",
        data_name=f"{args.dataset}.jsonl",
        data_path=str(EXP / "data") + "/",
        cache_path=str(Path(args.cache_root)) + "/",
        rate=rate,
        preprocess=bool(cfg["preprocess"]),
        reprocess_method=cfg["reprocess_method"],
        revert_rope=bool(args.revert_rope),
        topk=args.topk,
        use_sparse_attention=False,
        bge_model_path=BGE,
        device="cuda:0",
        offline_fixed_set_dir=str(offline_fixed_set_dir) if offline_fixed_set_dir else None,
        offline_fixed_set_method=cfg.get("fixed_set_method"),
        offline_fixed_set_rate=rate if "fixed_set_method" in cfg else None,
        static_key_bias_path=args.static_key_bias_path,
        static_key_bias_scale=args.static_key_bias_scale,
        static_key_bias_require_all=bool(args.static_key_bias_require_all),
        static_value_bias_path=args.static_value_bias_path,
        static_value_bias_scale=args.static_value_bias_scale,
        static_value_bias_require_all=bool(args.static_value_bias_require_all),
    )
    if cfg.get("sparse_block"):
        kwargs_main["use_sparse_attention"] = True
    if cfg.get("draft"):
        kwargs_main["draft_model_path"] = DRAFT

    inline_judge = None
    if os.environ.get("SETUP_V2_INLINE_GLM", "1").lower() not in {"0", "false", "no"}:
        from judge_task_csv import InlineJudge
        inline_judge = InlineJudge(str(result_dir / "metrics" / "metrics.csv"), MODEL)
        kwargs_main["sample_callback"] = inline_judge

    upc.main(**kwargs_main)
    if inline_judge is None and os.environ.get("SETUP_V2_AUTO_GLM", "1").lower() not in {"0", "false", "no"}:
        csv_files = sorted(result_csv_dir.glob("*.csv"))
        if csv_files:
            metrics_path = result_dir / "metrics" / "metrics.csv"
            judge_cmd = [
                sys.executable,
                str(Path(__file__).with_name("judge_task_csv.py")),
                "--csv", str(csv_files[0]),
                "--output", str(metrics_path),
            ]
            print("[setup-v2] auto GLM/EM/F1: " + " ".join(judge_cmd), flush=True)
            subprocess.run(judge_cmd, cwd=str(ROOT), check=True)
        else:
            raise FileNotFoundError(f"No result CSV found for automatic metrics in {result_csv_dir}")


if __name__ == "__main__":
    main()
