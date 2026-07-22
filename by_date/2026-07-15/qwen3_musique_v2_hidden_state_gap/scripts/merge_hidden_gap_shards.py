#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge hidden-state gap shard summaries.")
    parser.add_argument("--shard-root", required=True, help="Directory containing shard output directories.")
    parser.add_argument("--output-dir", required=True, help="Merged output directory.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(layer_rows: list[dict], fig_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"raw": "#4C78A8", "preprocess": "#F58518"}
    for metric, ylabel, filename in [
        ("relative_l2", "Relative L2 to Full Hidden", "hidden_layer_relative_l2_50.png"),
        ("delta_energy_share", "Hidden Delta Energy Share", "hidden_layer_energy_share_50.png"),
    ]:
        plt.figure(figsize=(9.5, 4.8))
        for source in ("raw", "preprocess"):
            sub = sorted([r for r in layer_rows if r["source"] == source], key=lambda r: int(r["layer"]))
            plt.plot(
                [int(r["layer"]) for r in sub],
                [float(r[metric]) for r in sub],
                label=source,
                color=colors[source],
                linewidth=2,
            )
        plt.xlabel("Layer")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.25)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    shard_root = Path(args.shard_root)
    out = Path(args.output_dir)
    shard_dirs = sorted(p for p in shard_root.iterdir() if (p / "results" / "hidden_layer_summary.csv").exists())
    if not shard_dirs:
        raise FileNotFoundError(f"No shard summaries found under {shard_root}")

    accum: dict[tuple[str, int], dict[str, float]] = {}
    source_totals = {"raw": {"examples": 0, "chunks": 0, "tokens": 0}, "preprocess": {"examples": 0, "chunks": 0, "tokens": 0}}
    shard_payloads = []
    skipped = []

    for shard_dir in shard_dirs:
        global_rows = read_csv(shard_dir / "results" / "hidden_global_summary.csv")
        layer_rows = read_csv(shard_dir / "results" / "hidden_layer_summary.csv")
        summary_path = shard_dir / "results" / "hidden_summary.json"
        if summary_path.exists():
            payload = json.loads(summary_path.read_text())
            shard_payloads.append({
                "shard": shard_dir.name,
                "args": payload.get("args", {}),
                "processed_examples": payload.get("processed_examples"),
                "processed_chunks": payload.get("processed_chunks"),
                "processed_tokens": payload.get("processed_tokens"),
            })
            for row in payload.get("skipped", []):
                row = dict(row)
                row["shard"] = shard_dir.name
                skipped.append(row)
        for row in global_rows:
            source = row["source"]
            source_totals[source]["examples"] += int(float(row["num_examples"]))
            source_totals[source]["chunks"] += int(float(row["num_chunks"]))
            source_totals[source]["tokens"] += int(float(row["num_tokens"]))
        for row in layer_rows:
            key = (row["source"], int(row["layer"]))
            slot = accum.setdefault(key, {"diff_sq": 0.0, "full_sq": 0.0, "source_sq": 0.0, "dot": 0.0})
            slot["diff_sq"] += float(row["diff_norm"]) ** 2
            slot["full_sq"] += float(row["full_norm"]) ** 2
            slot["source_sq"] += float(row["source_norm"]) ** 2
            slot["dot"] += float(row["cosine_source_full"]) * float(row["source_norm"]) * float(row["full_norm"])

    global_rows = []
    layer_rows = []
    for source in ("raw", "preprocess"):
        keys = sorted(k for k in accum if k[0] == source)
        total_diff = sum(accum[k]["diff_sq"] for k in keys)
        total_full = sum(accum[k]["full_sq"] for k in keys)
        total_source = sum(accum[k]["source_sq"] for k in keys)
        total_dot = sum(accum[k]["dot"] for k in keys)
        global_rows.append({
            "source": source,
            "kind": "hidden_layer_input",
            "relative_l2": math.sqrt(total_diff / total_full) if total_full else float("nan"),
            "diff_norm": math.sqrt(total_diff),
            "full_norm": math.sqrt(total_full),
            "source_norm": math.sqrt(total_source),
            "cosine_source_full": total_dot / math.sqrt(total_source * total_full) if total_source and total_full else float("nan"),
            "num_examples": source_totals[source]["examples"],
            "num_chunks": source_totals[source]["chunks"],
            "num_tokens": source_totals[source]["tokens"],
        })
        for _, layer in keys:
            slot = accum[(source, layer)]
            diff = slot["diff_sq"]
            full = slot["full_sq"]
            src = slot["source_sq"]
            dot = slot["dot"]
            layer_rows.append({
                "source": source,
                "kind": "hidden_layer_input",
                "layer": layer,
                "relative_l2": math.sqrt(diff / full) if full else float("nan"),
                "diff_norm": math.sqrt(diff),
                "full_norm": math.sqrt(full),
                "source_norm": math.sqrt(src),
                "cosine_source_full": dot / math.sqrt(src * full) if src and full else float("nan"),
                "delta_energy_share": diff / total_diff if total_diff else float("nan"),
            })

    write_csv(out / "hidden_global_summary.csv", global_rows)
    write_csv(out / "hidden_layer_summary.csv", layer_rows)
    payload = {
        "shard_root": str(shard_root),
        "shards": shard_payloads,
        "skipped": skipped,
        "global_summary": global_rows,
        "top_layers_by_energy": {
            source: sorted([r for r in layer_rows if r["source"] == source], key=lambda r: float(r["delta_energy_share"]), reverse=True)[:10]
            for source in ("raw", "preprocess")
        },
    }
    (out / "hidden_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    plot(layer_rows, out.parent / "figures")
    print(json.dumps(global_rows, indent=2))
    print(f"merged_shards={len(shard_dirs)} skipped={len(skipped)}")


if __name__ == "__main__":
    main()
